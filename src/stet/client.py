"""Talking to the Stet server: accounts, entries, the leaderboard, and verifying."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__
from .data import USER_AGENT, FetchError, api_base, cache_dir
from .keys import Key, config_dir, create_key, key_path, load_key
from .project import MAX_BUNDLE_BYTES, NAME_RE, unzstd
from .spec import CRAWL_ID_RE, SHA256_RE

JSON_LIMIT_BYTES = 16 << 20


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class WrongBytes(ApiError):
    """What the server sent is not what its hash says it is. Asking again would fetch the same bytes."""

    def __init__(self, message: str):
        super().__init__(0, message)


def clean(text, limit: int = 400) -> str:
    """Make text from the server, or from another entrant, safe to print.

    A terminal obeys control characters: they move the cursor, retitle the window, and on
    some terminals write to the clipboard. An editor's description is 120 bytes of anything.
    """
    return "".join(ch if ch.isprintable() else " " for ch in str(text)[:limit])


def request(method: str, path: str, body=None, key: Key | None = None, raw: bool = False, timeout: float = 120, limit: int = JSON_LIMIT_BYTES):
    """One API call. `body` may be bytes or anything JSON-serialisable.

    The answer is read up to `limit` bytes and no further: what a server sends is not
    allowed to decide how much memory this process uses.
    """
    data = b""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if body is not None:
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
            headers["Content-Type"] = "application/octet-stream"
        else:
            data = json.dumps(body, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
    if key is not None:
        headers.update(key.headers(method, path, data))
    req = urllib.request.Request(api_base() + path, data=data if method != "GET" else None, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read(limit + 1)
            promised = str((getattr(r, "headers", None) or {}).get("Content-Length") or "")
    except urllib.error.HTTPError as e:
        text = e.read(1 << 16).decode("utf-8", "replace")
        try:
            message = json.loads(text).get("error") or text
        except (ValueError, AttributeError):
            message = text.strip()[:300] or e.reason
        raise ApiError(e.code, clean(message)) from None
    except urllib.error.URLError as e:
        raise ApiError(0, f"cannot reach {api_base()}: {e.reason}") from None
    except (TimeoutError, ConnectionError) as e:  # while reading the body
        raise ApiError(0, f"lost the connection to {api_base()}: {e}") from None
    if len(payload) > limit:
        raise ApiError(0, f"{api_base()} sent more than {limit / 1e6:.0f} MB in answer to {method} {path.split('?')[0]}; refusing it")
    if promised.isdigit() and len(payload) < int(promised):
        # A connection that closes early hands over what had arrived, without a word. Those
        # are not the bytes that were asked for, and asking again may well get them.
        raise ApiError(0, f"the download was cut short: {api_base()} promised {int(promised):,} bytes for {method} {path.split('?')[0]} and {len(payload):,} arrived")
    if raw:
        return payload
    try:
        return json.loads(payload or b"{}")
    except ValueError:
        raise ApiError(0, f"{api_base()} answered {method} {path.split('?')[0]} with something that is not JSON") from None


def report(path: str, payload: dict, key: Key, log=None) -> dict:
    """Send a finished job's result, and keep trying while the server is away.

    A result is minutes of GPU time. The server answers 503 when it could not make a write
    durable, in which case the write may or may not have happened; sending the same report
    again is always safe, and "this job was already done" in answer to a second try means
    the first one did land. Nothing else does: "already refused" or "already released" is
    the server saying no, and is raised like any other refusal.

    Raises ApiError with the server's status when it refuses, and with status 0 when the
    report could not be delivered at all.
    """
    delay = 5
    for attempt in range(40):
        try:
            return request("POST", path, payload, key=key, timeout=600)
        except ApiError as e:
            if e.status == 409 and attempt > 0 and "already done" in str(e):
                return {"status": "recorded"}
            if e.status not in (0, 429, 500, 502, 503, 504):
                raise
            if log:
                log(f"the server is not answering ({e}); trying again in {delay}s")
            time.sleep(delay)
            delay = min(120, delay * 2)
    raise ApiError(0, "gave up reporting after many attempts")


def need_key() -> Key:
    key = load_key()
    if key is None or not key.handle:
        sys.stderr.write("stet: no account on this machine yet. Run `stet login` first.\n")
        sys.exit(1)
    return key


# --- rounds ---------------------------------------------------------------------------------


def verify_beacon(season, beacon: dict) -> None:
    """Check a round's randomness against drand itself, not against what the server says.

    A relay that cannot be reached, or whose answer cannot be read as this round's beacon
    (not JSON, an error object, fields missing or that do not add up), has said nothing,
    and the next relay is asked. The round is refused (ValueError) only when a relay
    returns a well-formed beacon for this round whose randomness is not the server's. If
    no relay gives a readable answer, that is a download that failed (FetchError), to be
    waited out like any other: it says nothing against the round.
    """
    cfg = season.raw["beacon"]
    number = int(beacon["round"])
    last: Exception | None = None
    for relay in cfg["relays"]:
        url = f"{relay}/{cfg['chain_hash']}/public/{number}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as r:
                got = json.loads(r.read(1 << 16))
            randomness = str(got["randomness"]).lower()
            if int(got["round"]) != number or hashlib.sha256(bytes.fromhex(got["signature"])).hexdigest() != randomness:
                raise ValueError("the beacon it returned does not add up")
        except Exception as e:  # this relay has told us nothing; try the next
            last = e
            continue
        if randomness != str(beacon["randomness"]).lower():
            raise ValueError(
                f"round randomness does not match drand round {number}: the server says "
                f"{str(beacon['randomness'])[:16]}…, drand says {randomness[:16]}…"
            )
        return
    raise FetchError(f"no drand relay gave a readable answer about round {number}, so the round's randomness could not be checked ({last})")


def round_from_api(season, description: dict):
    """Rebuild a round from the server's description without trusting the server about it."""
    from .spec import round_from_description

    rnd = round_from_description(season, description)
    if rnd.kind == "official" and os.environ.get("STET_DEV") != "1" and season.draw_slot(rnd.beacon["round"]) is None:
        raise ValueError(
            f"round {rnd.id} was drawn from drand round {rnd.beacon['round']}, which is not on the season's draw grid "
            f"(every {season.raw['draw_grid_seconds']} seconds). This verifier will not score it. Please report this."
        )
    if rnd.kind == "official":
        marker = cache_dir() / "rounds" / rnd.id / f"beacon-{rnd.seed.hex()[:16]}.ok"
        if not marker.exists():
            verify_beacon(season, rnd.beacon)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("verified against drand\n")
    return rnd


def fetch_round(season, index: int):
    try:
        d = request("GET", f"/api/v1/rounds/{season.id}-o{index:05d}")
    except ApiError as e:
        sys.stderr.write(f"stet: {e}\n")
        sys.exit(1)
    if not d["round"].get("seed"):
        sys.stderr.write(f"stet: round {clean(d['round']['id'], 40)} has not been drawn yet\n")
        sys.exit(1)
    return round_from_api(season, d["round"])


# --- small output helpers -------------------------------------------------------------------


def table(rows: list[list[str]], header: list[str], right: tuple[int, ...] = ()) -> str:
    rows = [[clean(c, 160) for c in row] for row in rows]  # most cells are someone else's words
    cols = list(zip(*([header] + rows), strict=True)) if rows else [[h] for h in header]
    widths = [max(len(str(c)) for c in col) for col in cols]
    out = []
    for r, row in enumerate([header] + rows):
        cells = [str(c).rjust(w) if i in right else str(c).ljust(w) for i, (c, w) in enumerate(zip(row, widths, strict=True))]
        out.append("  ".join(cells).rstrip())
        if r == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def fmt_gain(v) -> str:
    return "" if v is None else f"{v:+.4f}"


def fmt_mult(v, capped=False) -> str:
    if v is None:
        return ""
    return f">{v:g}x" if capped else f"{v:.2f}x"


def ago(ts) -> str:
    if not ts:
        return ""
    s = int(time.time() - ts)
    future = s < 0
    s = abs(s)
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= n:
            text = f"{s // n}{unit}"
            break
    else:
        text = f"{s}s"
    return f"in {text}" if future else f"{text} ago"


def editor_label(e: dict) -> str:
    owner = e.get("owner") or ""
    return clean(f"{owner + '/' if owner else ''}{e.get('name')} v{e.get('version')}", 100)


# --- accounts -------------------------------------------------------------------------------


def cmd_login(args):
    key = load_key()
    if key is None:
        key = create_key()
        key.save()
        sys.stderr.write(f"Created a new key at {key_path()}\n")
    if args.gist:
        r = request("POST", "/api/v1/accounts/github", {"gist": args.gist}, key=need_key())
        sys.stderr.write(f"Linked GitHub account @{clean(r['account']['github'])} to {clean(r['account']['handle'])}.\n")
        return
    try:
        me = request("GET", "/api/v1/me", key=key)
        key.handle = clean(me["account"]["handle"], 40)
        key.save()
        sys.stderr.write(f"Logged in as {key.handle}.\n")
    except ApiError as e:
        if e.status != 401:
            raise
        handle = args.handle
        if not handle:
            if not sys.stdin.isatty():
                sys.stderr.write("stet: choose a handle with `stet login --handle <name>`\n")
                sys.exit(1)
            handle = input("Choose a handle (3-24 lowercase letters, digits, hyphens): ").strip()
        me = request("POST", "/api/v1/accounts", {"handle": handle}, key=key)
        key.handle = clean(me["account"]["handle"], 40)
        key.save()
        sys.stderr.write(f"Welcome, {key.handle}. Your key is at {key_path()}; back it up, it cannot be recovered.\n")
    if args.github:
        sys.stderr.write(
            "\nEntering the competition needs a GitHub account at least 30 days old, linked to this key.\n"
            "(Practice runs, forking and verifying do not.)\n"
            f"  1. On that account, create a public gist containing exactly this line:\n\n       stet-key:{key.public_hex}\n\n"
            "  2. Run: stet login --gist <the gist's URL>\n"
        )


def cmd_whoami(args):
    key = need_key()
    me = request("GET", "/api/v1/me", key=key)
    if args.json:
        json.dump(me, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return
    a, v = me["account"], me["verifier"]
    flags = [f for f, on in (("an organisers' account", a.get("anchor")), ("given no more work for now", v.get("muted"))) if on]
    sys.stderr.write(clean(a["handle"]) + (f"  (@{clean(a['github'])} on GitHub)" if a.get("github") else "") + "\n")
    sys.stderr.write(f"key       {key.public_hex}\n")
    sys.stderr.write(f"verified  {int(v['validated'])} grades agreed ({int(v['recent'])} in the last 14 days), streak {int(v['streak'])}" + (f"  [{', '.join(flags)}]" if flags else "") + "\n")
    sys.stderr.write(f"standing  {standing_line(a, v)}\n")
    sys.stderr.write(f"editors   {len(me['editors'])}\n")


def standing_line(a: dict, v: dict) -> str:
    """Whether this account's work counts towards scores yet, and what is missing if not."""

    def count(x) -> int:
        return x if isinstance(x, int) and not isinstance(x, bool) else 0

    ran = f"; {count(v.get('runs'))} editors run" if count(v.get("runs")) else ""
    if v.get("established"):
        return "established: its runs and grades count towards scores" + ran
    missing = []
    need = count(v.get("establish_after"))
    if need and count(v.get("validated")) < need:
        missing.append(f"{need - count(v.get('validated'))} more grades that agree")
    if v.get("needs_github"):
        missing.append("a linked GitHub account of the age entering asks for (stet login --github)")
    return "not yet established: its grades are published and compared, and do not set scores" + (". Still to come: " + "; ".join(missing) if missing else "")


def cmd_key(args):
    """Print this machine's public key, making one if there is none. Contacts nobody."""
    key = load_key()
    if key is None:
        key = create_key()
        key.save()
        sys.stderr.write(f"Created a new key at {key_path()}\n")
    print(key.public_hex)


def cmd_draw(args):
    """Anchors only: bring the next draw forward to the next moment on the draw grid."""
    r = request("POST", "/api/v1/admin/draw", {}, key=need_key(), timeout=300)
    rd = r["round"]
    sys.stderr.write(f"{clean(rd['id'])}: {clean(rd['status'])}, draw {ago(rd['draw_at'])}\n")


def cmd_publish(args):
    """Anchors only: publish a download, which is how a season's exam pack reaches the server."""
    from .spec import load_season

    path = Path(args.file)
    blob = path.read_bytes()
    name = args.name or path.name
    season = load_season(args.season)
    if hashlib.sha256(blob).hexdigest() == (season.raw.get("heldout_exam") or {}).get("sha256") and args.close_season != season.id:
        # The one file that must not be published by a slip of the hand: a command from
        # the shell's history, a tab completion in the config directory, an agent following
        # the runbook a season early.
        sys.stderr.write(
            f"stet: {path.name} is season {season.id}'s held-out exam. Nothing was sent.\n"
            "  Publishing it cannot be taken back: from that moment anyone can fit an editor to it, and the season's\n"
            "  final table is worth only what was measured before. It is published once, when the season has closed\n"
            f"  and its last round has settled:  stet publish {args.file} --close-season {season.id}\n"
        )
        sys.exit(1)
    sys.stderr.write(f"publishing {name} ({len(blob) / 1e6:.1f} MB) ...\n")
    r = request("PUT", f"/api/v1/admin/files/{name}", blob, key=need_key(), timeout=1800)
    if r["sha256"] != hashlib.sha256(blob).hexdigest():
        sys.stderr.write("stet: the server stored something else; publish it again\n")
        sys.exit(1)
    print(f"{clean(r['url'])}\nsha256 {r['sha256']}")


# --- entries --------------------------------------------------------------------------------


def _listing(project, files, skipped, everything: bool) -> str:
    """What `stet submit` would publish, and what it leaves at home."""
    rows = [(p.relative_to(project.path).as_posix(), p.stat().st_size) for p in files]
    show = rows if everything or len(rows) <= 40 else sorted(rows, key=lambda r: -r[1])[:30]
    out = [f"  {size / 1e3:10,.1f} KB  {name}" for name, size in show]
    if len(show) < len(rows):
        out.append(f"  ... and {len(rows) - len(show)} smaller files (`stet submit --dry-run` lists every one)")
    if skipped:
        out.append("Left out (not uploaded, so not there when the editor runs):")
        out += [f"  {name}  ({why})" for name, why in (skipped if everything else skipped[:15])]
        if not everything and len(skipped) > 15:
            out.append(f"  ... and {len(skipped) - 15} more")
    return "\n".join(out) + "\n"


def cmd_submit(args):
    from .project import ProjectError, load_project, pack, survey

    try:
        project = load_project(args.path)
        files, skipped = survey(project)
        blob, editor_id = pack(project)
    except ProjectError as e:
        sys.stderr.write(f"stet: {e}\n")
        sys.exit(1)
    # A bundle is public for good from the moment it is first drawn into an official round,
    # whether or not it then runs, so the entrant sees every file first.
    sys.stderr.write(f"{project.name}: {len(files)} files, {len(blob) / 1e3:.1f} KB packed, id {editor_id[:12]}\n")
    sys.stderr.write(_listing(project, files, skipped, args.dry_run))
    if args.dry_run:
        sys.stderr.write("Nothing was sent.\n")
        return
    key = need_key()
    runs = sorted((project.path / ".stet" / "runs").glob("*/result.json"))
    if not any(json.loads(p.read_text()).get("valid") for p in runs[-5:]):
        sys.stderr.write("warning: no recent valid `stet run` for this project. An editor that fails scores a gain of 0.\n")
    if not args.yes:
        if not sys.stdin.isatty():
            sys.stderr.write("stet: pass --yes to submit without a prompt\n")
            sys.exit(1)
        sys.stderr.write(
            f"\nThis enters {project.name!r} in the next official round as {key.handle}.\n"
            "The files listed above become public under the MIT licence from the moment the editor is first drawn into an\n"
            "official round, and so does every edit script it writes.\n"
        )
        if input("Submit? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.stderr.write("Not submitted.\n")
            sys.exit(1)
    r = request("POST", "/api/v1/editors", blob, key=key, timeout=600)
    if args.json:
        json.dump(r, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return
    e = r["editor"]
    sys.stderr.write(f"\nSubmitted {clean(e['name'])} v{clean(e['version'])}  ({clean(e['id'])[:12]})\n")
    if r.get("replaced"):
        sys.stderr.write(f"It takes the place of your earlier entry {clean(r['replaced'])[:12]}, which will not run again.\n")
    if isinstance(r.get("in_play"), int) and isinstance(r.get("in_play_limit"), int) and r["in_play_limit"] > 1:
        sys.stderr.write(f"You have {r['in_play']} of {r['in_play_limit']} entries in play. A new version of an entry replaces the old one; another name is another entry.\n")
    # `enters` when the server expects it in that round; `earliest` when the queue is long or
    # the draw too close for that to be a promise.
    for field, words in (("enters", "It enters round"), ("earliest", "The earliest round it can enter is")):
        when = r.get(field)
        if isinstance(when, dict) and when.get("round"):
            drawn = ago(when.get("draw_at"))
            sys.stderr.write(f"{words} {clean(when['round'], 40)}" + (f", drawn {drawn}" if drawn else "") + ".\n")
            break
    if r.get("message"):
        sys.stderr.write(clean(r["message"], 1000) + "\n")
    sys.stderr.write(f"{clean(r.get('url', ''))}\n\nFollow it with `stet status`.\n")


def cmd_status(args):
    from .spec import load_season

    key = need_key()
    me = request("GET", "/api/v1/me", key=key)
    try:
        check_heldout_pin(load_season(), request("GET", "/api/v1/season"))
    except ApiError:
        pass
    if args.json:
        json.dump(me, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return
    if not me["editors"]:
        sys.stderr.write("No entries yet. `stet submit` enters the editor in this directory.\n")
        return
    rows = []
    for e in me["editors"]:
        last = e.get("last") or {}
        rows.append([
            f"{e['name']} v{e['version']}", e["id"][:10], e["status"], str(e.get("rounds") or 0),
            fmt_gain(e.get("rating")), fmt_mult(e.get("multiplier"), e.get("capped")),
            f"#{e['rank']}" if e.get("rank") else "",
            (f"{last.get('round', '')} {fmt_gain(last.get('gain')) if last.get('valid', True) else 'invalid'}").strip(),
        ])  # fmt: skip
    print(table(rows, ["editor", "id", "status", "rounds", "rating", "multiplier", "rank", "last round"], right=(3, 4, 5)))
    nxt = me.get("next_round")
    if nxt:
        sys.stderr.write(f"\nNext round {clean(nxt['id'])} is drawn {ago(nxt['draw_at'])}.\n")


def cmd_leaderboard(args):
    r = request("GET", "/api/v1/leaderboard")
    if args.json:
        json.dump(r, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return
    rows = [
        [
            ("★" if e.get("record") else "") + str(e["rank"]), editor_label(e), fmt_mult(e.get("multiplier"), e.get("capped")),
            fmt_gain(e.get("rating")), f"±{e['se']:.4f}" if e.get("se") is not None else "", str(e["rounds"]), e.get("technique") or "",
        ]
        for e in r["editors"][: args.top]
    ]  # fmt: skip
    if rows:
        print(table(rows, ["#", "editor", "multiplier", "rating", "", "rounds", "technique"], right=(0, 2, 3, 5)))
        sys.stderr.write("\n★ record holder   > beyond the end of the measured reference curve\n")
    qualifying = [
        [editor_label(e), fmt_gain(e.get("rating")), f"{e['rounds']} of {r.get('rated_rounds', 3)}", e.get("technique") or ""]
        for e in r.get("provisional", [])[: args.top]
    ]
    if qualifying:
        sys.stderr.write(("\n" if rows else "") + "Qualifying (ranked once they have enough rounds):\n")
        print(table(qualifying, ["editor", "rating so far", "rounds", "technique"], right=(1, 2)))
    if not rows and not qualifying:
        sys.stderr.write("No editors yet. Be the first: `stet submit`.\n")


def cmd_rounds(args):
    r = request("GET", f"/api/v1/rounds?limit={args.limit}")
    if args.json:
        json.dump(r, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return
    rows = [
        [x["id"], x["status"], ago(x.get("drawn_at") or x.get("draw_at")), str(x.get("entries", "")),
         f"{x['baseline_loss']:.4f}" if x.get("baseline_loss") else "", x.get("crawl_id") or ""]
        for x in r["rounds"]
    ]  # fmt: skip
    print(table(rows, ["round", "status", "drawn", "entries", "baseline", "crawl"], right=(3, 4)))
    sys.stderr.write("\nReplay any drawn round locally: stet run --round <id> --scale " + clean(r.get("scale", "micro"), 20) + "\n")


def cmd_fork(args):
    from .project import ProjectError, unpack

    key = load_key()
    info = request("GET", f"/api/v1/editors/{urllib.parse.quote(args.editor, safe='')}")
    e = info["editor"]
    # The server's answer names a directory and a download. Neither is taken on trust.
    if not isinstance(e.get("id"), str) or not SHA256_RE.fullmatch(e["id"]) or not isinstance(e.get("name"), str) or not NAME_RE.fullmatch(e["name"]):
        sys.stderr.write("stet: the server described that editor in a way this client does not accept\n")
        sys.exit(1)
    asked = args.editor.lower()
    if re.fullmatch(r"[0-9a-f]{8,64}", asked) and not e["id"].startswith(asked):
        sys.stderr.write(f"stet: asked for editor {asked[:12]}, but the server answered with {e['id'][:12]}\n")
        sys.exit(1)
    dest = Path(args.dest or e["name"]).resolve()
    if dest.exists() and any(dest.iterdir()):
        sys.stderr.write(f"stet: {dest} exists and is not empty\n")
        sys.exit(1)
    blob = request("GET", f"/api/v1/editors/{e['id']}/bundle", key=key, raw=True, limit=MAX_BUNDLE_BYTES)
    if hashlib.sha256(blob).hexdigest() != e["id"]:
        sys.stderr.write("stet: the downloaded bundle does not match the editor's id\n")
        sys.exit(1)
    try:
        left_out = unpack(blob, dest)
    except ProjectError as err:
        sys.stderr.write(f"stet: {clean(err)}\n")
        sys.exit(1)
    toml = dest / "stet.toml"
    if not toml.is_file():
        sys.stderr.write("stet: that bundle has no stet.toml\n")
        sys.exit(1)
    lines = toml.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln for ln in lines if not ln.strip().startswith("parent")]
    at = next((i for i, ln in enumerate(lines) if ln.strip().startswith("[")), len(lines))
    lines.insert(at, f'parent = "{e["id"]}"   # forked from {editor_label(e)}')
    toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from importlib import resources

    for name, template in (("AGENTS.md", "AGENTS.md"), (".gitignore", "gitignore")):
        if not (dest / name).exists():
            (dest / name).write_text(resources.files("stet.templates").joinpath(template).read_text(encoding="utf-8"), encoding="utf-8")
    sys.stderr.write(f"Forked {editor_label(e)} into {dest}\n")
    if left_out:
        sys.stderr.write(f"Left out {len(left_out)} files no bundle should hold: {clean(', '.join(left_out[:5]))}\n")
    sys.stderr.write(
        "This is someone else's program. Read it before you run it, or run it boxed in: stet run --sandbox (Linux).\n"
        f"Give it a new name in stet.toml, then: cd {dest.name} && stet run\n"
    )


# --- verifying ------------------------------------------------------------------------------
#
# Two kinds of job. A *grade* job is "here is an edit script (plain data) for this round:
# what does the student score on it?" It runs no entrant code, so any machine with a
# suitable GPU can do it. An *edit* job runs an entrant's program in the sandbox to produce
# that script; only established accounts that asked to run editors are given those.


def slim_result(result: dict, base: dict) -> dict:
    """What a grader reports: everything needed to judge the run, nothing bulky."""
    keys = ("harness", "harness_sha256", "season", "season_digest", "scale", "corpus_sha256", "exam_sha256", "script_sha256",
            "valid", "invalid_reason", "hardware", "started_at", "finished_at")  # fmt: skip
    out = {k: result.get(k) for k in keys}
    out["round"] = result["round"]["id"]
    out["baseline"] = {"loss": base["exam"]["loss"], "by_source": {k: v["loss"] for k, v in base["exam"]["by_source"].items()}}
    if result.get("valid"):
        s = result["script"]
        out["script"] = {k: s.get(k) for k in ("script_sha256", "tokens_sha256", "script_lines", "out_docs", "out_bytes", "src_docs", "ink_tokens", "ink_frac", "kept_doc_frac", "kept_byte_frac", "tokens_one_pass", "passes", "truncated")}
        out["exam"] = {"loss": result["exam"]["loss"], "by_source": {k: v["loss"] for k, v in result["exam"]["by_source"].items()}}
        t = result["train"]
        out["train"] = {k: t.get(k) for k in ("seconds", "tokens_per_second", "final_train_loss", "init_sha256", "precision")}
        out["train"]["curve"] = t["curve"][:: max(1, len(t["curve"]) // 60)]
    return out


def _announcements() -> Path:
    return config_dir() / "announced.json"


ANNOUNCE_MARGIN = 30  # seconds before a beacon is emitted after which an announcement proves nothing
ANNOUNCE_HORIZON = 8 * 86400  # no draw is announced further ahead than this
ANNOUNCE_KEEP = 30 * 86400


def _announcement_log() -> dict:
    path = _announcements()
    try:
        log = json.loads(path.read_text())
        if not isinstance(log, dict):
            raise ValueError("not an object")
        return log
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        # A damaged log is evidence of something. Set it aside; do not quietly start again.
        aside = path.with_name(path.name + ".damaged")
        try:
            path.replace(aside)
            sys.stderr.write(f"stet: {path} could not be read and was moved to {aside.name}; starting a new record of announcements\n")
        except OSError:
            pass
        return {}


def remember_announcement(season, a: dict | None, now: float | None = None) -> None:
    """Keep our own record of what the server said a draw would use.

    The beacon round a draw uses is fixed by its time. The crawl is not: it is named when
    the draw is scheduled, before the beacon value exists. Every verifier that is online
    keeps this log, so a server that changed the crawl after seeing the randomness would be
    contradicting records it does not control.

    The log is this verifier's, so the server gets no say in what it holds. Only a draw on
    the season's grid, still in the future and not absurdly far off, is written down: there
    are few enough of those that nothing honest can be crowded out, and entries leave by
    age alone. The server may change its mind about a draw as often as it likes before the
    beacon exists (the last word counts, as it would for anyone listening), and never after.
    """
    now = time.time() if now is None else now
    try:
        slot = int(a["drand_round"])
        entry = {"id": str(a["id"]), "crawl": str(a["crawl"]["crawl"]), "paths_sha256": str(a["crawl"]["paths_sha256"]).lower()}
        at = season.beacon_time(slot)
    except (KeyError, TypeError, ValueError):
        return
    if season.draw_slot(slot) is None and os.environ.get("STET_DEV") != "1":
        return
    if not (now + ANNOUNCE_MARGIN <= at <= now + ANNOUNCE_HORIZON):
        return
    if not CRAWL_ID_RE.fullmatch(entry["crawl"]) or not SHA256_RE.fullmatch(entry["paths_sha256"]) or len(entry["id"]) > 40:
        return
    log = _announcement_log()
    old = log.get(str(slot))
    if isinstance(old, dict) and all(old.get(k) == v for k, v in entry.items()):
        return
    log[str(slot)] = {**entry, "seen_at": int(now), "changed": (old.get("changed", 0) + 1) if isinstance(old, dict) else 0}
    for k in [k for k in log if not k.isdigit() or season.beacon_time(int(k)) < now - ANNOUNCE_KEEP]:
        del log[k]
    path = _announcements()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(log, indent=1))
    tmp.replace(path)


def check_announcement(season, rnd, warn=None) -> None:
    """Refuse a round that was not drawn the way we were told it would be.

    If we hold an announcement for this draw from before its beacon existed, the round must
    match it, or this verifier will not score it. If we hold none, though we heard of other
    draws in the day before, we cannot tell a draw moved in good time from one chosen after
    the fact. That is said out loud (`warn`) and the round is scored: the server announces
    a moved draw at least six minutes ahead, but a verifier in the middle of a job asks for
    work less often than that, so refusing would stop honest verifiers every time one was
    busy when a draw moved.
    """
    if rnd.kind != "official":
        return
    log = {k: v for k, v in _announcement_log().items() if isinstance(v, dict) and isinstance(v.get("seen_at"), int)}
    drawn = season.beacon_time(rnd.beacon["round"])
    seen = log.get(str(int(rnd.beacon["round"])))
    if seen and seen["seen_at"] + ANNOUNCE_MARGIN <= drawn:
        told = (seen.get("id"), seen.get("crawl"), seen.get("paths_sha256"))
        got = (rnd.id, rnd.crawl["crawl"], rnd.crawl["paths_sha256"])
        if told != got:
            raise RoundRefused(
                f"the draw at drand round {rnd.beacon['round']} was announced as {told[0]} on {told[1]}, but the server now says {got[0]} on {got[1]}. "
                "This verifier will not score it. Please report this."
            )
        return
    before = [(k, v) for k, v in log.items() if drawn - 86400 <= v["seen_at"] <= drawn - ANNOUNCE_MARGIN]
    if before and warn:
        slot, last = max(before, key=lambda kv: kv[1]["seen_at"])
        if last.get("id") == rnd.id and slot.isdigit():  # the same round, moved to another draw
            at = lambda t: time.strftime("%H:%M UTC", time.gmtime(t))  # noqa: E731
            what = (
                f"this verifier was told that round {rnd.id} would be drawn at {at(season.beacon_time(int(slot)))}. "
                f"It was drawn at {at(drawn)}, and word of the change did not reach this verifier before then."
            )
        else:
            what = (
                f"this verifier was not told of the draw of round {rnd.id} before it happened "
                f"(the last it had heard of, {(drawn - last['seen_at']) // 60} minutes earlier, was {clean(last.get('id'), 40)})."
            )
        warn(what + " It cannot vouch for how this draw was chosen.")


class RoundRefused(ValueError):
    """This verifier will not score the round: it was not drawn the way the rules say."""


_unannounced: set[str] = set()  # rounds already warned about


def _job_round(season, job: dict, out):
    """The round a job is for, rebuilt and checked. Raises RoundRefused if it does not hold up."""
    from . import data

    try:
        rnd = round_from_api(season, job["round"])
    except (ValueError, KeyError, TypeError) as err:  # the seed does not follow, the beacon is off the grid, ...
        raise RoundRefused(clean(err)) from None

    def warn(message):
        if rnd.id not in _unannounced:  # once a round, not once a job
            _unannounced.add(rnd.id)
            out.line("", out.bad(message))

    check_announcement(season, rnd, warn=warn)
    data.mark_verified(rnd)
    data.prune_verified(rnd)
    return rnd


def check_job(job) -> int:
    """A job as the server described it, checked before any of it reaches a path or a URL.

    Returns the job's id. The server is the only thing that hands out jobs, which is no
    reason to let a field of one name a directory to delete.
    """
    ok = (
        isinstance(job, dict)
        and isinstance(job.get("id"), int) and not isinstance(job["id"], bool) and 0 < job["id"] < 1 << 62
        and job.get("kind") in ("grade", "edit")
        and isinstance(job.get("round"), dict)
        and isinstance(job.get("editor"), dict)
    )  # fmt: skip
    if ok and job["kind"] == "edit":
        ok = isinstance(job["editor"].get("id"), str) and bool(SHA256_RE.fullmatch(job["editor"]["id"]))
    if ok and job["kind"] == "grade":
        ok = isinstance(job.get("script"), dict) and isinstance(job["script"].get("sha256"), str) and bool(SHA256_RE.fullmatch(job["script"]["sha256"]))
    if not ok:
        raise ValueError("the server sent a job this client does not understand")
    return job["id"]


def claim_workdir() -> tuple[Path, object]:
    """A directory under the cache that this process alone works in, and the lock that says so.

    Two verifiers on one machine (one per GPU, each with a key and an account of its own:
    see claim_key) must not clear each other's work, so each holds a lock on a numbered
    slot for as long as it runs. Holding the lock also means whatever is in the slot was
    left by a dead process.
    """
    from .runner import remove_tree

    root = cache_dir() / "verify"
    root.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # Windows: one verifier per cache
        return root / "slot0", None
    for n in range(64):
        lock = open(root / f"slot{n}.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            continue
        for p in root.iterdir():  # what earlier releases left behind, in directories named by job
            if not p.name.startswith("slot"):
                remove_tree(p)
        return root / f"slot{n}", lock
    raise OSError(f"64 verifiers already hold {root}")


def _grade_job(job, season, scale, device, key, out, work: Path) -> tuple[dict, dict]:
    import zstandard

    from .pipeline import baseline, grade_round

    rnd = _job_round(season, job, out)
    base = baseline(rnd, scale, device, progress=out.progress("baseline"), log=lambda m: out.line("", out.dim("baseline " + m)))
    want = job["script"]["sha256"]
    limit = int(season.raw["script"]["max_bytes"])
    # Signed: until its round settles, a script is given only to whoever was handed it to grade.
    blob = request("GET", f"/api/v1/scripts/{want}", key=key, raw=True, timeout=600, limit=limit + (1 << 20))
    try:
        raw = unzstd(blob, limit)
    except zstandard.ZstdError:
        raise WrongBytes("the script the server sent is damaged") from None
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != want:
        raise WrongBytes("the script the server sent does not match its hash")
    work.mkdir(parents=True, exist_ok=True)
    path = work / "script.jsonl"
    path.write_bytes(raw)
    result = grade_round(rnd, scale, path, device=device, progress=out.progress("training"), log=lambda m: out.line("", out.dim(m)))
    if result.get("heldout"):
        try:
            keep_heldout(season, job, result, base)
        except OSError as e:
            # The grade is finished and it is the public one that is owed to the server.
            # What the held-out exam said about it is lost; the entry shows as a gap in the
            # season's held-out table, with the command that measures it again.
            out.line("", out.bad(
                f"WHAT THE HELD-OUT EXAM SAID ABOUT THIS GRADE COULD NOT BE WRITTEN DOWN ({heldout_log(season)}: {clean(e.strerror or e)}). "
                "The public grade is reported as usual. Put the record right before the next job."
            ))  # fmt: skip
    elif result.get("heldout_error"):
        out.line("", out.bad(f"The held-out exam could not be sat for this grade ({clean(result['heldout_error'])}). The public grade is reported as usual."))
    return result, base


def heldout_log(season) -> Path:
    return config_dir() / "heldout" / f"{season.id}.jsonl"


def keep_heldout(season, job: dict, result: dict, base: dict) -> None:
    """This machine's record of what the held-out exam said about one official grade.

    It is appended here and reported to nobody: `slim_result` names every field a grader
    sends, and this is not one of them. The file is published, with the exam, when the
    season closes.

    Until then held-out numbers are in two places on the machine that holds the pack, both private to its
    account: this file (mode 0600, in the config directory), and each official round's
    cached baseline (~/.cache/stet/rounds/<round>/<scale>/baseline-*.json, mode 0600),
    which holds the unedited corpus's held-out loss beside its public one. `stet run`,
    `stet score` and `stet baseline` leave them out of what they print and save unless
    `--heldout` is given.
    """
    held, held_base = result["heldout"], base.get("heldout") or {}
    editor = job.get("editor") if isinstance(job.get("editor"), dict) else {}
    line = {
        "at": int(time.time()),
        "round": result["round"]["id"],
        "scale": result["scale"],
        "harness": result["harness"],
        "editor": {k: editor.get(k) for k in ("id", "owner", "name", "version")},
        "script_sha256": result.get("script_sha256"),
        "exam_sha256": result.get("exam_sha256"),
        "heldout_exam_sha256": held.get("exam_sha256"),
        "public": {"loss": result["exam"]["loss"], "baseline": base["exam"]["loss"]},
        "heldout": {
            "loss": held["loss"],
            "baseline": held_base.get("loss"),
            "by_source": {k: v["loss"] for k, v in held["by_source"].items()},
            "baseline_by_source": {k: v["loss"] for k, v in (held_base.get("by_source") or {}).items()},
        },
    }
    path = heldout_log(season)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def heldout_log_trouble(season) -> str:
    """Why the held-out record cannot be added to, or "" if it can. Makes the file if it is not there."""
    path = heldout_log(season)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.close(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600))
    except OSError as e:
        return clean(e.strerror or e)
    return ""


MACHINE_ERRNOS = (errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EIO, errno.EMFILE, errno.ENFILE, errno.ENOMEM)


def _edit_job(job, season, scale, key, out, work: Path, sandbox: str, gpu: int = 0) -> dict:
    import base64

    import zstandard

    from .runner import EditRun, HarnessError, release_gpu_memory

    # Checked here as well as where the flag is read: this is the function that runs a
    # stranger's program, and nothing that calls it should be able to get that wrong.
    if sandbox != "bwrap" and os.environ.get("STET_DEV") != "1":
        raise HarnessError("refusing to run an entrant's editor outside the sandbox")
    from . import data
    from .project import load_project, unpack
    from .spec import HARNESS_VERSION

    rnd = _job_round(season, job, out)
    e = job["editor"]
    blob = request("GET", f"/api/v1/editors/{e['id']}/bundle", key=key, raw=True, timeout=600, limit=MAX_BUNDLE_BYTES)
    if hashlib.sha256(blob).hexdigest() != e["id"]:
        raise WrongBytes("the bundle the server sent does not match the editor's id")
    corpus = data.corpus(rnd, scale)
    payload = {
        "harness": HARNESS_VERSION, "season": season.id, "season_digest": season.digest, "round": rnd.id, "scale": scale.name,
        "corpus_sha256": corpus.meta["sha256"], "script_sha256": None, "started_at": int(time.time()),
    }  # fmt: skip
    # From here on the bundle's bytes are what the entrant submitted, so whatever is wrong
    # with them is the entrant's failure and is reported as one. Handing the job back
    # instead would have the next verifier meet the same bundle, for ever.
    bundle_dir = work / "bundle"
    try:
        unpack(blob, bundle_dir)
        project = load_project(bundle_dir)
    except Exception as err:
        if isinstance(err, MemoryError) or (isinstance(err, OSError) and err.errno in MACHINE_ERRNOS):
            raise HarnessError(f"could not unpack a bundle on this machine: {err}") from None
        payload["edit"] = EditRun("error", 0.0, float(scale.edit_seconds), None, None, "", sandbox, reason=clean(f"the bundle could not be loaded: {err}", 300)).to_dict()
        payload["finished_at"] = int(time.time())
        return payload
    from .pipeline import edit_round

    # This process trained the last script it graded, and PyTorch keeps that memory. The
    # editor is promised its share of the card: let go of ours before it starts.
    release_gpu_memory()
    edit = edit_round(
        bundle_dir, project.entry, rnd, scale, work, sandbox=sandbox, log=lambda m: out.line("", out.dim(m)),
        limits={"required": sandbox == "bwrap"}, assets=project.assets, verify_assets=True, gpu=gpu,
    )  # fmt: skip
    payload["edit"] = edit.to_dict()
    payload["finished_at"] = int(time.time())
    if edit.script is not None:
        raw = edit.script.read_bytes()
        payload["script_sha256"] = hashlib.sha256(raw).hexdigest()
        payload["script_size"] = len(raw)
        payload["script_zst"] = base64.b64encode(zstandard.ZstdCompressor(level=10).compress(raw)).decode()
    return payload


def _forget_round(job: dict, season, scale) -> None:
    """Drop what is cached for a job's round, so that a second try starts from nothing.

    If the server says our corpus or baseline differs from the round's, the copy on disk is
    the first suspect, and it would otherwise be used again unchanged.
    """
    from . import data

    try:
        rnd = round_from_api(season, job["round"])
    except Exception:
        return
    data.set_aside(data.round_dir(rnd, scale))  # not deleted: another verifier here may be reading it


def _pins() -> Path:
    return config_dir() / "pins.json"


def check_heldout_pin(season, served: dict | None = None, warn=None, now: float | None = None) -> None:
    """Hold the organisers to the held-out exam they committed to.

    The season file names the held-out exam by its hash, and the rules promise that hash
    will not change while the season runs. A promise is only as good as somebody's record
    of it, so the first hash this machine is shown is written down here, and anything that
    later says otherwise (a new release's season file, or the server) is said out loud.
    It is a warning, never a refusal: a volunteer's machine has no way to know which of the
    two is the true one, only that the organisers have gone back on their word.
    """
    warn = warn or (lambda message: sys.stderr.write(message + "\n"))
    now = time.time() if now is None else now
    said = {
        "this copy of stet": (season.raw.get("heldout_exam") or {}).get("sha256"),
        f"the server at {api_base()}": (served or {}).get("heldout_exam_sha256") if (served or {}).get("season", season.id) == season.id else None,
    }
    path = _pins()
    try:
        pins = json.loads(path.read_text())
        pinned = pins["heldout_exam"].get(season.id) if isinstance(pins.get("heldout_exam"), dict) else None
    except FileNotFoundError:
        pins, pinned = {}, None
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        # A record that cannot be read is replaced, and that is said too: it is the one
        # file a changed commitment would have to get past.
        warn(f"stet: {path} could not be read; starting a new record of the season's commitments")
        pins, pinned = {}, None
    if not isinstance(pinned, dict) or not SHA256_RE.fullmatch(str(pinned.get("sha256"))):
        pinned = None
    for source, sha in said.items():
        if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
            if pinned and source == "this copy of stet":
                warn(
                    f"WARNING: season {season.id} committed to a held-out exam, {pinned['sha256']}, and {source} no longer names one.\n"
                    "  The organisers promised it would not change while the season runs. Please report this."
                )
            continue
        if pinned is None:
            pinned = {"sha256": sha, "seen_at": int(now), "from": source}
            if not isinstance(pins.get("heldout_exam"), dict):
                pins["heldout_exam"] = {}
            pins["heldout_exam"][season.id] = pinned
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(pins, f, indent=1)
            tmp.replace(path)
        elif pinned["sha256"] != sha:
            when = time.strftime("%d %B %Y", time.gmtime(int(pinned.get("seen_at") or 0)))
            warn(
                f"WARNING: season {season.id}'s held-out exam has changed.\n"
                f"  On {when} this machine was shown its hash as {pinned['sha256']} (by {clean(pinned.get('from'), 80)}).\n"
                f"  Now {source} says it is {sha}.\n"
                "  The organisers committed to one held-out exam for the whole season and promised it would not change.\n"
                f"  Please report this. (This machine's record is {path}.)"
            )


def claim_key():
    """One verifier per key, for as long as it runs. Returns the lock that says so.

    The server gives an account one job at a time, and a verifier that asks for work while
    its account holds a job is handed that job again: that is how a verifier that was
    restarted picks up where it stopped. Two verifiers on one key would therefore both be
    given the first one's job, and either could hand it back from under the other.
    """
    try:
        import fcntl
    except ImportError:  # Windows
        return None
    path = config_dir() / "verify.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(path, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.seek(0)
        pid = clean(lock.read(20).strip() or "unknown", 20)
        lock.close()
        sys.stderr.write(
            f"stet: another `stet verify` (process {pid}) is already running with this key.\n"
            "  The server gives one account one job at a time, so a second verifier on the same key would be handed\n"
            "  the first one's job. To put a second GPU to work, give it an account of its own:\n"
            "    STET_CONFIG=<another directory> stet login --handle <another name>\n"
            "    STET_CONFIG=<that directory> stet verify\n"
        )
        sys.exit(1)
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    return lock


def _stop_like_ctrl_c() -> dict:
    """Have a volunteer's verifier treat being told to stop (a closed terminal, `kill`, a
    service manager) as it treats Ctrl-C: hand the job back, then go. A job that is merely
    abandoned keeps its place on an entry until its time runs out, and the server then
    rests the account for a day. Returns what to put back afterwards.

    An anchor does not do this. Its service manager restarts it, the server hands it the
    same job again, and nothing is lost; handing the job back would count as one of the
    few attempts an entry gets.
    """
    import signal
    import threading

    def stop(signum, frame):
        raise KeyboardInterrupt

    old = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                try:
                    old[getattr(signal, name)] = signal.signal(getattr(signal, name), stop)
                except (ValueError, OSError):
                    pass
    return old


RETRY_PAUSES = (60, 300)  # a job that fails on this machine is tried again after these waits, then handed back
STOP_AFTER_FAILURES = 3 * (len(RETRY_PAUSES) + 1)  # three jobs' worth of failures in a row: it is the machine
FETCH_PATIENCE = 30 * 60  # how long a job is kept while what it needs cannot be downloaded


def cmd_upgrade(args):
    from . import __version__, upgrade
    from .cli import Out
    from .spec import load_season

    out = Out()
    info = request("GET", "/api/v1/season")
    info = info if isinstance(info, dict) else {}
    theirs = str((info.get("cli") or {}).get("version") or "")
    if not upgrade.newer(theirs):
        out.line("", f"stet {__version__} is the newest.")
        return
    if args.check:
        out.line("", f"stet {theirs} is out; this is {__version__}. Install it: stet upgrade")
        return
    ok, why = upgrade.can()
    site = str(info.get("site") or "https://www.stetrun.com")
    if not ok:
        sys.stderr.write(f"stet: {theirs} is out, and it cannot be installed from here: {why}.\n  Install it with: curl -fsSL {site}/install.sh | sh\n")
        sys.exit(1)
    out.line("", f"stet {theirs} is out; this is {__version__}. Upgrading.")
    try:
        season_env = load_season().raw.get("environment") or {}
    except Exception:
        season_env = {}
    worked, tail = upgrade.install(site, season_env, lambda m: out.line("", out.dim(m)))
    have = upgrade.installed_version(upgrade.launcher() or "stet") if worked else ""
    if not worked or not upgrade.newer(have):
        sys.stderr.write(f"stet: the upgrade did not work ({tail or 'the installer said nothing'}). This is still {__version__}.\n")
        sys.exit(1)
    out.line("", out.good(f"Now stet {have}."))


def cmd_service(args):
    from . import service
    from .cli import Out

    out = Out()
    name = args.name
    try:
        if args.action == "install":
            need_key()
            if args.edits:
                from . import runner

                for ok, why in (runner.bwrap_available(), runner.scope_available()):
                    if not ok:
                        sys.stderr.write(f"stet: this machine cannot run editors yet: {why}\n  `stet service install` without --edits sets it up to grade.\n")
                        sys.exit(1)
            done = service.install(name, edits=args.edits, no_upgrade=args.no_upgrade, device=args.device)
            out.line("", out.good(f"{name} is installed and running."))
            out.line("", f"It runs `stet verify{' --edits' if args.edits else ''}`, starts with this machine, and starts again if it stops.")
            out.line("", "It installs new versions of stet by itself, between jobs." if not args.no_upgrade else "It will not upgrade itself: run `stet upgrade` when a new version is out.")
            if done["note"]:
                out.line("", out.bad(done["note"]))
            out.line("", "")
            out.line("", "  stet service status       how it is doing")
            out.line("", "  stet service logs         what it has been saying")
            out.line("", "  stet service uninstall    stop it and remove it")
            out.line("", out.dim(f"  log: {done['log']}"))
        elif args.action == "uninstall":
            out.line("", f"{name} is stopped and removed." if service.uninstall(name) else f"There is no {name} service here.")
        elif args.action == "status":
            st = service.status(name)
            if not st["installed"]:
                out.line("", f"There is no {name} service here. Make one: stet service install")
                return
            out.line("", f"{name}: {st['active']}, {'starts with the machine' if st['enabled'] == 'enabled' else 'does not start with the machine'}")
            for line in st["tail"]:
                out.line("", out.dim("  " + line[:200]))
        elif args.action == "logs":
            for line in service.tail(name, args.lines):
                print(line)
    except service.ServiceError as e:
        sys.stderr.write(f"stet: {e}\n")
        sys.exit(1)


def cmd_verify(args):
    import signal

    from .cli import need_torch

    need_torch()
    key = need_key()
    lock = claim_key()
    # An organisers' machine runs as a service: stopped, it is started again and given the
    # job it held. Anyone else's hands its job back when it stops, however it is stopped.
    me = request("GET", "/api/v1/me", key=key)
    me = me if isinstance(me, dict) else {}
    organiser = bool((me.get("account") or {}).get("anchor"))
    handlers = {} if organiser else _stop_like_ctrl_c()
    try:
        return _verify(args, organiser, me)
    finally:
        for sig, old in handlers.items():
            signal.signal(sig, old)
        if lock is not None:
            lock.close()


def _verify(args, organiser: bool = False, me: dict | None = None):
    from .cli import Out
    from .data import heldout_pack_file, heldout_pack_path
    from .runner import bwrap_available, remove_tree, sandbox_selftest, scope_available
    from .spec import load_season
    from .train import hardware_info, pick_device

    out = Out()
    key = need_key()
    season = load_season(args.season)
    info = request("GET", "/api/v1/season")
    # A verifier left running keeps itself current: see upgrade.py. First of all here, since
    # a newer stet is the usual reason this one's season or libraries would differ below.
    from . import upgrade

    upgrader = upgrade.Upgrader(args, season.raw.get("environment") or {}, say=lambda m: out.line("", out.dim(m)), fetch_info=lambda: request("GET", "/api/v1/season"))
    upgrader.consider(info if isinstance(info, dict) else {})
    check_heldout_pin(season, info, warn=lambda message: out.line("", out.bad(message)))
    scale = season.scale(info["scale"])
    device = pick_device(args.device)
    hw = hardware_info(device)
    allowed = season.raw.get("grade_devices", ["cuda"])
    if device.type not in allowed and not args.any_device:
        sys.stderr.write(
            "stet: grading for the leaderboard needs an NVIDIA (CUDA) or AMD (ROCm) GPU, so that every grade is\n"
            "  computed with the same arithmetic. You can still build, run and submit editors on this machine.\n"
        )
        sys.exit(1)
    from . import bench, env

    dev = os.environ.get("STET_DEV") == "1"
    # Every grade is trained by the same PyTorch. A different version is a different trainer.
    if (wrong := env.problems(season, only=("torch",))) and not dev and not args.any_device:
        sys.stderr.write(f"stet: grading for the leaderboard uses the season's PyTorch, and this machine's is {'; '.join(wrong)}.\n  The installer puts it right: curl -fsSL https://www.stetrun.com/install.sh | sh\n")
        sys.exit(1)
    sandbox = "none"
    if args.edits:
        sandbox = "none" if args.no_sandbox else "bwrap"
        if sandbox == "none" and not dev:
            sys.stderr.write("stet: edit jobs run other people's code, and only ever in the sandbox\n")
            sys.exit(1)
        if sandbox == "bwrap":
            for ok, why in (bwrap_available(), scope_available(), sandbox_selftest()):
                if not ok:
                    sys.stderr.write(f"stet: edit jobs run other people's code and need the sandbox: {why}\n")
                    sys.exit(1)
        # An editor may import any of the season's libraries and is promised a machine no
        # slower than the floor. A machine that cannot keep either promise takes no edit jobs.
        if (wrong := env.problems(season)) and not dev:
            sys.stderr.write("stet: editors are promised the season's Python and libraries, and this machine's differ:\n  " + "\n  ".join(w[:400] for w in wrong)
                             + "\n  The installer puts it right: curl -fsSL https://www.stetrun.com/install.sh | sh\n")  # fmt: skip
            sys.exit(1)
        measured = bench.saved()
        if measured is None:
            out.line("", out.dim("measuring this machine's speed (about half a minute; it is kept for a week) ..."))
            measured = bench.measure(season)
            bench.save(measured)
        if (short := bench.shortfalls(season, measured)) and not dev:
            sys.stderr.write(
                "stet: this machine cannot run editors for the leaderboard: " + "; ".join(short) + ".\n"
                "  If the GPU or the CPU was busy when that was measured, measure again with nothing else running: stet bench\n"
                "  It can still grade: stet verify\n"
            )
            sys.exit(1)
        hw = {**hw, "bench": {k: measured.get(k) for k in ("gpu_tokens_per_second", "cpu_mb_per_second", "cores", "memory_gb", "gpu_memory_gb")}}
    try:
        slot, lock = claim_workdir()  # `lock` is held until this process ends
    except OSError as e:
        sys.stderr.write(f"stet: {e}\n")
        sys.exit(1)
    work = slot / "job"
    out.line("", out.bold(f"stet verify · {key.handle} · {hw.get('gpu', device.type)}"))
    out.line("", out.dim("Grading edit scripts with the fixed trainer. No one else's code runs on this machine." if not args.edits
                         else "Running entrants' editors in the sandbox, and grading."))  # fmt: skip
    try:
        if heldout_pack_path(season) is not None:
            trouble = heldout_log_trouble(season)
            if trouble:
                sys.stderr.write(
                    f"stet: this machine holds the season's held-out exam, and what it says about each official grade is added to\n"
                    f"  {heldout_log(season)}\n"
                    f"  That file cannot be written ({trouble}). Every grade made without it is a gap in the season's final table,\n"
                    "  so this verifier will not start until it can be.\n"
                )
                sys.exit(1)
            out.line("", out.dim(f"This machine holds the season's held-out exam. What it says is kept in {heldout_log(season)} and sent nowhere."))
        elif organiser and (season.raw.get("heldout_exam") or {}).get("sha256") and os.environ.get("STET_DEV") != "1":
            sys.stderr.write(
                f"stet: season {season.id} has a held-out exam, and the organisers' machine sits every student it grades on it.\n"
                f"  This machine does not hold it. It belongs at {heldout_pack_file(season)}\n"
                "  (or wherever STET_HELDOUT_EXAM points). Every official grade made without it is a gap in the season's\n"
                "  final table that has to be measured again, so this machine will not start until the pack is in place.\n"
            )
            sys.exit(1)
    except RuntimeError as e:  # a pack that is not the committed one: nothing is graded until that is put right
        sys.stderr.write(f"stet: {e}\n")
        sys.exit(1)
    hello = {"harness": __version__, "season": season.id, "season_digest": season.digest, "hardware": hw, "edits": bool(args.edits), "sandbox": sandbox}
    done = failures = waits = 0
    refused: dict[str, int] = {}  # round id -> how many times this machine and the server disagreed about it
    tries: dict[int, int] = {}  # job id -> how many times it has failed on this machine
    stuck: dict[int, int] = {}  # job id -> seconds spent waiting for a download it needs
    idle = False
    holding: int | None = None  # the job this verifier has taken and has neither reported nor handed back
    held_until = 0  # when the server stops waiting for that job, if it said
    last_job: int | None = None  # the last job worked on, to tell a new one from the same one again
    leave_held = False  # set on the ways out that must not hand the job back
    told_log = False
    no_editors: str | None = None  # why the server gives this machine no editors to run, as last said

    def stop_if_once():
        if args.once:
            sys.exit(1)

    def patience(why: str, keeping: int | None = None) -> int:
        """Wait out something that is not this machine's fault, a little longer each time.

        While a job is being kept for it (`keeping`), never past what is left of the time a
        job is kept: the server stops waiting for a job it has heard nothing about, and a
        volunteer whose job is given up on is offered no work for a day.
        """
        nonlocal waits
        waits += 1
        pause = min(3600, 60 * 2 ** min(waits - 1, 6))
        if keeping is not None:
            pause = max(1, min(pause, FETCH_PATIENCE - stuck.get(keeping, 0)))
        out.line("", out.dim(f"{why}; trying again in {max(1, round(pause / 60))} min"))
        stop_if_once()
        time.sleep(pause)
        return pause

    def still_held() -> str:
        until = f" until {time.strftime('%H:%M UTC', time.gmtime(held_until))}" if held_until > time.time() else ""
        if organiser:
            return f"The server could not be told, so the job is still this machine's{until}. It is handed the same job the next time it asks."
        return (
            f"The server could not be told, so the job is still held{until}. Run `stet verify` again before then and it is picked up "
            "where it was. A job left to run out costs this account a day without work."
        )

    def hand_back(reason: str) -> bool:
        """Give the job back. False if the server could not be told, in which case it is still ours."""
        nonlocal holding
        if _release(holding, key, reason):
            holding = None
            return True
        out.line("", out.bad(still_held()))
        return False

    def handed(back: bool) -> str:
        return "handed it back" if back else "it could not be handed back"

    try:
        while True:
            if holding is None:
                upgrader.look()  # never with a job in hand
            try:
                r = request("POST", "/api/v1/jobs/claim", hello, key=key)
            except ApiError as e:
                if e.status == 426 and holding is None:
                    upgrader.now()  # does not come back if it worked
                    if upgrader.on and upgrade.can()[0] and not args.once:
                        # Too old to be given work, and the upgrade did not go through this
                        # time (the network, most likely). Someone who left this running
                        # wants it to keep trying, not to find it stopped.
                        out.line("", out.bad(f"{e}. Waiting to upgrade; trying again in 10 minutes."))
                        time.sleep(600)
                        continue
                if e.status in (400, 401, 403, 426):
                    sys.stderr.write(f"stet: {e}\n")
                    sys.exit(1)
                out.line("", out.dim(f"server unavailable ({e}); retrying in 60s"))
                time.sleep(60)
                continue
            if not isinstance(r, dict):
                r = {}
            job = r.get("job")
            # Before anything else that can fail: from this answer on, the server counts the
            # job as ours, and whatever stops this verifier must know there is one to hand back.
            if isinstance(job, dict) and isinstance(job.get("id"), int) and not isinstance(job["id"], bool) and 0 < job["id"] < 1 << 62:
                holding = job["id"]
                held_until = job["expires_at"] if isinstance(job.get("expires_at"), int) and not isinstance(job.get("expires_at"), bool) else 0
            elif not job:
                holding = None  # the server gives a verifier the job it holds each time it asks: no job means none is held
            # Whether this account, on this machine, is handed editors to run is the server's
            # to say, with each answer. Said here once, and again when it changes.
            runner = r.get("runner")
            if args.edits and isinstance(runner, dict):
                why = "" if runner.get("eligible") is True else clean(runner.get("reason") or "the server did not say why", 300)
                if why and why != no_editors:
                    out.line("", out.dim(f"Not given editors to run: {why}. Given grading meanwhile."))
                elif not why and no_editors:
                    out.line("", out.dim("This machine is now given editors to run."))
                no_editors = why
            try:
                remember_announcement(season, r.get("announced"))
            except OSError as e:
                # This verifier's own record of what was announced. Without it the next draw
                # cannot be checked against what was said beforehand, which is worth saying,
                # and no reason to stop grading.
                if not told_log:
                    told_log = True
                    out.line("", out.bad(f"could not write down what the server announced for the next draw ({_announcements()}: {clean(e.strerror or e)}). Carrying on without that record."))
            if not job:
                if args.once or (args.max_jobs and done >= args.max_jobs):
                    out.line("", clean(r.get("message") or "No work right now."))
                    return
                if not idle:
                    idle = True
                    out.line("", out.dim(clean(r.get("message") or "No work right now.") + " Waiting; Ctrl-C to stop."))
                try:
                    pause = int(r.get("retry_after") or 60)
                except (TypeError, ValueError):
                    pause = 60
                time.sleep(min(300, max(5, pause)))
                continue
            idle = False
            try:
                job_id = check_job(job)
            except ValueError as err:
                if holding is not None:
                    hand_back("this client does not understand the job")
                patience(str(err))
                continue
            holding = job_id
            round_id = clean(job["round"].get("id"), 40)
            out.line("", "")
            out.line(job["kind"], f"job {job_id} · {editor_label(job['editor'])} · round {round_id}")
            if job["kind"] == "edit" and not args.edits:
                # An honest server never does this. Whatever it is, it is not going to run here.
                hand_back("this verifier does not run edit jobs")
                sys.stderr.write(
                    "stet: the server handed this verifier an edit job, which would run an entrant's program on this machine.\n"
                    "  It was refused, and this verifier has stopped. Please report this.\n"
                )
                sys.exit(1)
            # A volunteer steps away from a round it and the server disagree about. The
            # organisers' machine does not: early in a season it may be the only one there
            # is, each job it took and handed back would be one of the few attempts an entry
            # gets, and a whole round's entries would be void for one disagreement.
            if refused.get(round_id, 0) >= 2 and not organiser:
                hand_back("this verifier and the server disagree about this round")
                patience(f"not taking more work from round {round_id}, which the server and this machine disagree about")
                continue
            if job_id != last_job:
                waits, last_job = 0, job_id  # a new job starts with short waits, whatever the last one met
            if not remove_tree(work):
                leave_held = True
                sys.stderr.write(f"stet: could not clear {work}. Something an editor left there cannot be removed; stopping.\n")
                sys.exit(1)
            result = base = None
            try:
                if job["kind"] == "edit":
                    payload = _edit_job(job, season, scale, key, out, work, sandbox, gpu=getattr(device, "index", None) or 0)
                    path = f"/api/v1/jobs/{job_id}/script"
                else:
                    result, base = _grade_job(job, season, scale, device, key, out, work)
                    payload = slim_result(result, base)
                    path = f"/api/v1/jobs/{job_id}/result"
            except RoundRefused as err:
                remove_tree(work)
                if organiser:
                    # Not handed back, and not skipped either. The service manager starts this
                    # machine again, the server gives it the same job, and the round is checked
                    # afresh: nothing is spent while someone finds out what is wrong.
                    sys.stderr.write(
                        f"stet: {err}\n"
                        "  The organisers' machine does not work on a round it cannot vouch for, and it does not hand the job back\n"
                        "  either: a job handed back is one of the few attempts an entry gets. Stopping with the job still held.\n"
                        "  Started again, it is given the same job and checks the round afresh.\n"
                    )
                    sys.exit(1)
                hand_back(clean(f"round refused: {err}", 300))
                refused[round_id] = 2  # not this session
                out.line("", out.bad(str(err)))
                stop_if_once()
                continue
            except (FetchError, ApiError) as err:  # a download failed: the network's trouble, or the host's
                remove_tree(work)
                waited = stuck.get(job_id, 0)
                # Bytes that do not match their hash, or an answer that says no, will be the
                # same next time. Anything else may pass.
                final = isinstance(err, WrongBytes) or (isinstance(err, ApiError) and 400 <= err.status < 500 and err.status not in (408, 429))
                if final or args.once or waited >= FETCH_PATIENCE:
                    back = hand_back(clean(f"{type(err).__name__}: {err}", 300))
                    stuck.pop(job_id, None)
                    patience(f"could not fetch what this job needs ({clean(err)}); {handed(back)}")
                else:
                    # Kept, not handed back. A job handed back is not free: an entry gets only
                    # a few attempts, and a verifier is offered an entry once. The server
                    # gives this job back to us the next time we ask.
                    stuck[job_id] = waited + patience(f"could not fetch what this job needs ({clean(err)}). Keeping the job", keeping=job_id)
                continue
            except BaseException as err:  # this machine's trouble, not the editor's
                # A panic in a compiled library is not an Exception. It is a failed job all
                # the same, and must not be what stops this machine.
                if isinstance(err, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                remove_tree(work)
                failures += 1
                tries[job_id] = tries.get(job_id, 0) + 1
                what = f"could not run this job ({type(err).__name__}: {clean(err)})"
                # Three in a row on three jobs is this machine, not the jobs. Stop rather than churn through the queue.
                stopping = args.once or failures >= STOP_AFTER_FAILURES
                if tries[job_id] <= len(RETRY_PAUSES) and not stopping:
                    # A fault that passes (a full disk someone is clearing, a driver that
                    # recovers) should cost a few minutes, not one of the entry's attempts.
                    pause = RETRY_PAUSES[tries[job_id] - 1]
                    out.line("", out.bad(f"{what}; keeping it and trying again in {pause // 60} min"))
                    time.sleep(pause)
                    continue
                back = hand_back(clean(f"{type(err).__name__}: {err}", 300))
                tries.pop(job_id, None)
                out.line("", out.bad(f"{what}; {handed(back)}"))
                if stopping:
                    sys.stderr.write("stet: stopping, because jobs keep failing on this machine. `stet doctor` checks the setup.\n")
                    sys.exit(1)
                time.sleep(30)
                continue
            failures = waits = 0
            tries.pop(job_id, None)
            stuck.pop(job_id, None)
            try:
                ack = report(path, payload, key, log=lambda m: out.line("", out.dim(m)))
            except ApiError as err:
                if err.status == 0 or err.status >= 500 or err.status in (408, 425, 426, 429):
                    # Not a refusal: the server never said no to this result, it could not be
                    # given it (it is away, its file store is failing, this client is out of
                    # date). The work is good, and a job handed back spends an attempt.
                    # Any other answer in the 400s is a no, and is treated as one below: a
                    # machine that kept a job the server will never take would do it again
                    # for ever, and the round would never settle.
                    remove_tree(work)
                    if organiser:
                        sys.stderr.write(
                            f"stet: the result of job {job_id} could not be delivered ({err}).\n"
                            "  That is not a refusal, so the job is not handed back: handing it back would spend one of the entry's\n"
                            "  few attempts. Stopping with the job still held. Started again, this machine is given the same job and\n"
                            "  does it again.\n"
                        )
                        sys.exit(1)
                    back = hand_back("the result could not be delivered")
                    out.line("", out.bad(f"the result could not be delivered ({err}); the job was {'handed back' if back else 'not handed back either'}"))
                    stop_if_once()
                    continue
                # The server will not have this result. Training the same thing again would get
                # the same answer, so say so, let go of the job, and do not take it up again.
                out.line("", out.bad(f"the server did not accept the result: {err}"))
                hand_back("result refused")
                remove_tree(work)
                said = str(err)
                # A result that merely arrived too late is nobody's fault. "Already refused" is
                # not that: it is the refusal itself, heard on a second try.
                late = err.status == 409 and "already refused" not in said and any(w in said for w in ("already", "ended before", "settled before"))
                if err.status in (400, 409) and not late:  # a disagreement about the round, not some other no
                    refused[round_id] = refused.get(round_id, 0) + 1
                    if err.status == 409 and refused[round_id] == 1:
                        # Our corpus or baseline differs from the round's. The copy on disk is the first suspect.
                        _forget_round(job, season, scale)
                        out.line("", out.dim(f"dropped this machine's copy of round {round_id}; it is rebuilt once, in case the copy was at fault"))
                stop_if_once()
                continue
            holding = None
            if not remove_tree(work):
                sys.stderr.write(f"stet: could not clear {work}. Something an editor left there cannot be removed; stopping.\n")
                sys.exit(1)
            done += 1
            if job["kind"] == "edit":
                ed = payload["edit"]
                out.line("result", f"{ed['status']} in {ed['seconds']:.0f}s" + (f" · script {payload['script_sha256'][:12]} ({payload['script_size'] / 1e6:.1f} MB)" if payload.get("script_sha256") else f" · no script: {clean(ed.get('reason', ''))}"))
            elif result["valid"]:
                gain = base["exam"]["loss"] - result["exam"]["loss"]
                out.line("result", f"exam {result['exam']['loss']:.4f} · baseline {base['exam']['loss']:.4f} · gain {out.bold(f'{gain:+.4f}')}")
            else:
                out.line("result", f"invalid: {clean(result['invalid_reason'])}")
            if ack:
                v = ack.get("verifier") or {}
                out.line("server", clean(ack.get("status", "")) + (f" · {clean(ack['message'])}" if ack.get("message") else "") + (f" · {clean(v.get('validated', 0))} of your grades have agreed so far" if v and job["kind"] == "grade" else ""))
            if args.once or (args.max_jobs and done >= args.max_jobs):
                return
    except KeyboardInterrupt:
        # Wherever it lands: mid-job, between two tries at one, or while a result is on its way.
        if holding is None:
            raise
        leave_held = True  # said once, here
        back = hand_back("interrupted")
        remove_tree(work)
        out.line("", "Stopped; the job was handed back." if back else "Stopped.")
        return
    finally:
        # Every other way out. A volunteer's job that is merely left keeps its place on the
        # entry until its time runs out, and the account is then offered no work for a day.
        # The organisers' machine's is left where it is: started again, it is given the same
        # job, and a job handed back would be one of the entry's few attempts.
        if holding is not None and not organiser and not leave_held:
            if _release(holding, key, "the verifier stopped"):
                out.line("", out.dim("The job in hand was handed back."))
            else:
                out.line("", out.bad(still_held()))


RELEASE_PAUSES = (2, 5)  # seconds between tries at handing a job back


def _release(job_id: int, key: Key, reason: str) -> bool:
    """Hand a job back. True if the server heard, or says the job is not ours to hand back.

    Asked up to three times: saying it twice costs nothing, and a hand-back that is lost
    leaves the job held until its time runs out.
    """
    for pause in (*RELEASE_PAUSES, None):
        try:
            request("POST", f"/api/v1/jobs/{int(job_id)}/release", {"reason": reason}, key=key, timeout=20)
            return True
        except ApiError as e:
            if e.status in (404, 409):  # finished, settled or long gone: nothing is held
                return True
        except Exception:
            pass
        if pause is not None:
            time.sleep(pause)
    return False


# --- wiring ---------------------------------------------------------------------------------


def add_commands(sub) -> None:
    p = sub.add_parser("login", help="create your key and account")
    p.add_argument("--handle")
    p.add_argument("--github", action="store_true", help="show how to link a GitHub account")
    p.add_argument("--gist", help="URL of the gist that proves your GitHub account")
    p.set_defaults(fn=cmd_login)

    p = sub.add_parser("whoami", help="show your account")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_whoami)

    p = sub.add_parser("submit", help="enter your editor in the competition")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    p.add_argument("--dry-run", action="store_true", help="list every file that would be published, and send nothing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("status", help="how your entries are doing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("leaderboard", help="the current standings")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_leaderboard)

    p = sub.add_parser("rounds", help="recent official rounds (replay one with `stet run --round <id>`)")
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_rounds)

    p = sub.add_parser("fork", help="start from any public editor")
    p.add_argument("editor", help="an editor id, or handle/name")
    p.add_argument("dest", nargs="?")
    p.set_defaults(fn=cmd_fork)

    p = sub.add_parser("key", help="print this machine's public key")
    p.set_defaults(fn=cmd_key)

    p = sub.add_parser("draw")  # organisers only; not listed in the help
    p.set_defaults(fn=cmd_draw)

    p = sub.add_parser("publish")  # organisers only; not listed in the help
    p.add_argument("file")
    p.add_argument("--name", help="the name to publish it under (default: the file's own)")
    p.add_argument("--close-season", metavar="SEASON", help="needed to publish a season's held-out exam: the id of the season being closed")
    p.add_argument("--season", default="s0", help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("verify", help="lend your GPU: grade edit scripts for the leaderboard, and (with --edits) run editors")
    p.add_argument("--once", action="store_true", help="do one job and exit")
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--season", default="s0", help=argparse.SUPPRESS)
    p.add_argument("--edits", "--run-editors", dest="edits", action="store_true", help="also run entrants' editors, in the sandbox (Linux; `stet bench` says whether this machine is fast enough)")
    p.add_argument("--no-upgrade", action="store_true", help="do not install newer versions of stet automatically (the default is to, between jobs; STET_AUTO_UPGRADE=0 does the same)")
    p.add_argument("--no-sandbox", action="store_true", help=argparse.SUPPRESS)  # local testing only
    p.add_argument("--any-device", action="store_true", help=argparse.SUPPRESS)  # local testing only
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("service", help="keep `stet verify` running on this machine: start with it, restart, upgrade (Linux)")
    p.add_argument("action", choices=["install", "status", "logs", "uninstall"])
    p.add_argument("--edits", "--run-editors", dest="edits", action="store_true", help="also run entrants' editors, in the sandbox")
    p.add_argument("--no-upgrade", action="store_true", help="do not install newer versions of stet automatically")
    p.add_argument("--device", default=None, help=argparse.SUPPRESS)
    p.add_argument("--name", default="stet-verify", help="the service's name, for a second verifier on one machine (default: stet-verify)")
    p.add_argument("--lines", type=int, default=40, help="with logs: how many lines")
    p.set_defaults(fn=cmd_service)

    p = sub.add_parser("upgrade", help="install the newest stet now (a running `stet verify` does this by itself)")
    p.add_argument("--check", action="store_true", help="only say whether a newer version is out")
    p.set_defaults(fn=cmd_upgrade)

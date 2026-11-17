"""Talking to the Stet server: accounts, entries, the leaderboard, and verifying."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__
from .data import USER_AGENT, api_base, cache_dir
from .keys import Key, config_dir, create_key, key_path, load_key

ASSET_HOSTS = ("huggingface.co", "github.com", "objects.githubusercontent.com", "raw.githubusercontent.com", "files.pythonhosted.org")
MAX_ASSET_BYTES = 2 << 30


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def request(method: str, path: str, body=None, key: Key | None = None, raw: bool = False, timeout: float = 120):
    """One API call. `body` may be bytes or anything JSON-serialisable."""
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
            payload = r.read()
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        try:
            message = json.loads(text).get("error") or text
        except ValueError:
            message = text.strip()[:300] or e.reason
        raise ApiError(e.code, message) from None
    except urllib.error.URLError as e:
        raise ApiError(0, f"cannot reach {api_base()}: {e.reason}") from None
    return payload if raw else json.loads(payload or b"{}")


def need_key() -> Key:
    key = load_key()
    if key is None or not key.handle:
        sys.stderr.write("stet: no account on this machine yet. Run `stet login` first.\n")
        sys.exit(1)
    return key


# --- rounds ---------------------------------------------------------------------------------


def verify_beacon(season, beacon: dict) -> None:
    """Check a round's randomness against drand itself, not against what the server says."""
    cfg = season.raw["beacon"]
    last: Exception | None = None
    for relay in cfg["relays"]:
        url = f"{relay}/{cfg['chain_hash']}/public/{int(beacon['round'])}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as r:
                got = json.loads(r.read())
        except Exception as e:  # try the next relay
            last = e
            continue
        derived = hashlib.sha256(bytes.fromhex(got["signature"])).hexdigest()
        if int(got["round"]) != int(beacon["round"]) or derived != got["randomness"]:
            raise ValueError(f"drand relay {relay} returned an inconsistent beacon")
        if got["randomness"].lower() != str(beacon["randomness"]).lower():
            raise ValueError(
                f"round randomness does not match drand round {beacon['round']}: the server says "
                f"{beacon['randomness'][:16]}…, drand says {got['randomness'][:16]}…"
            )
        return
    raise OSError(f"could not reach any drand relay to check the round's randomness ({last})")


def round_from_api(season, description: dict):
    """Rebuild a round from the server's description without trusting the server about it."""
    from .spec import round_from_description

    rnd = round_from_description(season, description)
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
        sys.stderr.write(f"stet: round {d['round']['id']} has not been drawn yet\n")
        sys.exit(1)
    return round_from_api(season, d["round"])


# --- small output helpers -------------------------------------------------------------------


def table(rows: list[list[str]], header: list[str], right: tuple[int, ...] = ()) -> str:
    cols = list(zip(*([header] + rows))) if rows else [[h] for h in header]
    widths = [max(len(str(c)) for c in col) for col in cols]
    out = []
    for r, row in enumerate([header] + rows):
        cells = [str(c).rjust(w) if i in right else str(c).ljust(w) for i, (c, w) in enumerate(zip(row, widths))]
        out.append("  ".join(cells).rstrip())
        if r == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def fmt_gain(v) -> str:
    return "" if v is None else f"{v:+.4f}"


def fmt_mult(v, extrapolated=False) -> str:
    return "" if v is None else f"{v:.2f}x" + ("*" if extrapolated else "")


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
    return f"{owner + '/' if owner else ''}{e['name']} v{e['version']}"


# --- accounts -------------------------------------------------------------------------------


def cmd_login(args):
    key = load_key()
    if key is None:
        key = create_key()
        key.save()
        sys.stderr.write(f"Created a new key at {key_path()}\n")
    if args.gist:
        r = request("POST", "/api/v1/accounts/github", {"gist": args.gist}, key=need_key())
        sys.stderr.write(f"Linked GitHub account @{r['account']['github']} to {r['account']['handle']}.\n")
        return
    try:
        me = request("GET", "/api/v1/me", key=key)
        key.handle = me["account"]["handle"]
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
        key.handle = me["account"]["handle"]
        key.save()
        sys.stderr.write(f"Welcome, {key.handle}. Your key is at {key_path()}; back it up, it cannot be recovered.\n")
    if args.github:
        sys.stderr.write(
            "\nTo show a verified GitHub name on the leaderboard:\n"
            f"  1. Create a public gist containing exactly this line:\n\n       stet-key:{key.public_hex}\n\n"
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
    flags = [f for f, on in (("anchor", a.get("anchor")), ("trusted verifier", v.get("trusted"))) if on]
    sys.stderr.write(f"{a['handle']}" + (f"  (@{a['github']} on GitHub)" if a.get("github") else "") + "\n")
    sys.stderr.write(f"key       {key.public_hex}\n")
    sys.stderr.write(f"verified  {v['validated']} jobs ({v['recent']} in the last 14 days), streak {v['streak']}" + (f"  [{', '.join(flags)}]" if flags else "") + "\n")
    sys.stderr.write(f"editors   {len(me['editors'])}\n")


# --- entries --------------------------------------------------------------------------------


def cmd_submit(args):
    from .project import ProjectError, bundle_files, load_project, pack

    key = need_key()
    try:
        project = load_project(args.path)
        for a in project.assets:
            host = urllib.parse.urlparse(a.url).hostname or ""
            if host not in ASSET_HOSTS:
                raise ProjectError(f"asset host {host!r} is not allowed; use one of: {', '.join(ASSET_HOSTS)}")
        blob, editor_id = pack(project)
    except ProjectError as e:
        sys.stderr.write(f"stet: {e}\n")
        sys.exit(1)
    files = bundle_files(project)
    sys.stderr.write(f"{project.name}: {len(files)} files, {len(blob) / 1e3:.1f} KB packed, id {editor_id[:12]}\n")
    runs = sorted((project.path / ".stet" / "runs").glob("*/result.json"))
    if not any(json.loads(p.read_text()).get("valid") for p in runs[-5:]):
        sys.stderr.write("warning: no recent valid `stet run` for this project. An editor that fails scores a gain of 0.\n")
    if not args.yes:
        if not sys.stdin.isatty():
            sys.stderr.write("stet: pass --yes to submit without a prompt\n")
            sys.exit(1)
        sys.stderr.write(
            f"\nThis enters {project.name!r} in the next official round as {key.handle}.\n"
            "Its code becomes public under the MIT licence once it has run.\n"
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
    sys.stderr.write(f"\nSubmitted {e['name']} v{e['version']}  ({e['id'][:12]})\n")
    if r.get("replaced"):
        sys.stderr.write(f"It replaces your earlier entry {r['replaced'][:12]}, which will not run again.\n")
    if r.get("enters"):
        sys.stderr.write(f"It enters round {r['enters']['round']}, drawn {ago(r['enters']['draw_at'])}.\n")
    if r.get("message"):
        sys.stderr.write(r["message"] + "\n")
    sys.stderr.write(f"{r.get('url', '')}\n\nFollow it with `stet status`.\n")


def cmd_status(args):
    key = need_key()
    me = request("GET", "/api/v1/me", key=key)
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
            fmt_gain(e.get("rating")), fmt_mult(e.get("multiplier"), e.get("extrapolated")),
            f"#{e['rank']}" if e.get("rank") else "",
            (f"{last.get('round', '')} {fmt_gain(last.get('gain')) if last.get('valid', True) else 'invalid'}").strip(),
        ])  # fmt: skip
    print(table(rows, ["editor", "id", "status", "rounds", "mean gain", "multiplier", "rank", "last round"], right=(3, 4, 5)))
    nxt = me.get("next_round")
    if nxt:
        sys.stderr.write(f"\nNext round {nxt['id']} is drawn {ago(nxt['draw_at'])}.\n")


def cmd_leaderboard(args):
    r = request("GET", "/api/v1/leaderboard")
    if args.json:
        json.dump(r, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return
    rows = [
        [
            ("★" if e.get("record") else "") + str(e["rank"]), editor_label(e), fmt_mult(e.get("multiplier"), e.get("extrapolated")),
            fmt_gain(e.get("rating")), f"±{e['se']:.4f}" if e.get("se") is not None else "", str(e["rounds"]), e.get("technique") or "",
        ]
        for e in r["editors"][: args.top]
    ]  # fmt: skip
    if not rows:
        sys.stderr.write("No rated editors yet. Be the first: `stet submit`.\n")
        return
    print(table(rows, ["#", "editor", "multiplier", "mean gain", "", "rounds", "technique"], right=(0, 2, 3, 5)))
    sys.stderr.write("\n★ record holder   * multiplier extrapolated beyond the measured reference curve\n")


def cmd_rounds(args):
    r = request("GET", f"/api/v1/rounds?limit={args.limit}")
    if args.json:
        json.dump(r, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return
    rows = [
        [x["id"], x["status"], ago(x.get("drawn_at") or x.get("draw_at")), str(x.get("entries", "")),
         f"{x['baseline_loss']:.4f}" if x.get("baseline_loss") else "", x.get("crawl") or ""]
        for x in r["rounds"]
    ]  # fmt: skip
    print(table(rows, ["round", "status", "drawn", "entries", "baseline", "crawl"], right=(3, 4)))
    sys.stderr.write("\nReplay any drawn round locally: stet run --round <id> --scale " + r.get("scale", "micro") + "\n")


def cmd_fork(args):
    from .project import unpack

    key = load_key()
    info = request("GET", f"/api/v1/editors/{urllib.parse.quote(args.editor, safe='')}")
    e = info["editor"]
    dest = Path(args.dest or e["name"]).resolve()
    if dest.exists() and any(dest.iterdir()):
        sys.stderr.write(f"stet: {dest} exists and is not empty\n")
        sys.exit(1)
    blob = request("GET", f"/api/v1/editors/{e['id']}/bundle", key=key, raw=True)
    if hashlib.sha256(blob).hexdigest() != e["id"]:
        sys.stderr.write("stet: the downloaded bundle does not match the editor's id\n")
        sys.exit(1)
    unpack(blob, dest)
    toml = dest / "stet.toml"
    lines = toml.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if not ln.strip().startswith("parent")]
    at = next((i for i, ln in enumerate(lines) if ln.strip().startswith("[")), len(lines))
    lines.insert(at, f'parent = "{e["id"]}"   # forked from {editor_label(e)}')
    toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    agents = dest / "AGENTS.md"
    if not agents.exists():
        from importlib import resources

        agents.write_text(resources.files("stet.templates").joinpath("AGENTS.md").read_text(encoding="utf-8"), encoding="utf-8")
    sys.stderr.write(f"Forked {editor_label(e)} into {dest}\nGive it a new name in stet.toml, then: cd {dest.name} && stet run\n")


# --- verifying ------------------------------------------------------------------------------


def fetch_assets(project, bundle_dir: Path) -> None:
    """Fetch an editor's declared assets, by hash, before the sandbox closes the network."""
    from .data import fetch

    total = 0
    for a in project.assets:
        host = urllib.parse.urlparse(a.url).hostname or ""
        if host not in ASSET_HOSTS:
            raise OSError(f"asset host {host!r} is not on the allowlist")
        cached = fetch([a.url], a.sha256, cache_dir() / "assets" / a.sha256, a.path, quiet=True)
        total += cached.stat().st_size
        if total > MAX_ASSET_BYTES:
            raise OSError("the editor's assets exceed the size limit")
        target = (bundle_dir / a.path).resolve()
        if not str(target).startswith(str(bundle_dir.resolve()) + os.sep):
            raise OSError(f"asset path escapes the bundle: {a.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(cached, target)
        except OSError:
            shutil.copyfile(cached, target)


def slim_result(result: dict, base: dict) -> dict:
    """What a verifier reports: everything needed to judge the run, nothing bulky."""
    out = {k: result.get(k) for k in ("harness", "season", "season_digest", "scale", "corpus_sha256", "exam_sha256", "edit", "valid", "invalid_reason", "hardware", "started_at", "finished_at")}
    out["round"] = result["round"]["id"]
    out["baseline"] = {"loss": base["exam"]["loss"], "by_source": {k: v["loss"] for k, v in base["exam"]["by_source"].items()}}
    if result.get("valid"):
        s = result["script"]
        out["script"] = {k: s.get(k) for k in ("script_sha256", "script_lines", "out_docs", "out_bytes", "src_docs", "ink_frac", "kept_doc_frac", "kept_byte_frac", "tokens_one_pass", "passes", "truncated")}
        out["exam"] = {"loss": result["exam"]["loss"], "by_source": {k: v["loss"] for k, v in result["exam"]["by_source"].items()}}
        t = result["train"]
        out["train"] = {"seconds": t["seconds"], "tokens_per_second": t["tokens_per_second"], "final_train_loss": t["final_train_loss"], "curve": t["curve"][:: max(1, len(t["curve"]) // 60)]}
    return out


def calibration(season, scale, device) -> dict:
    """This machine's speed relative to the reference, measured once and remembered."""
    from .cli import calibrate
    from .train import hardware_info

    hw = hardware_info(device)
    tag = hashlib.sha256(json.dumps([hw, scale.name, __version__], sort_keys=True).encode()).hexdigest()[:16]
    path = config_dir() / f"calibration-{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    cal = calibrate(season, scale, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal))
    return cal


def time_factor(cal: dict) -> float:
    """Slower machines get proportionally longer edit limits; nobody gets less than the nominal limit."""
    f = cal.get("factor") or 1.0
    return max(1.0, min(10.0, f))


def cmd_verify(args):
    from .cli import Out, need_torch

    need_torch()
    from .pipeline import baseline, run_round
    from .project import load_project, unpack
    from .runner import bwrap_available
    from .spec import load_season
    from .train import hardware_info, pick_device

    out = Out()
    key = need_key()
    ok, why = bwrap_available()
    if not ok and not args.no_sandbox:
        sys.stderr.write(
            f"stet: verifying runs other people's code, and the sandbox is unavailable: {why}\n"
            "  Install bubblewrap (apt install bubblewrap / dnf install bubblewrap / pacman -S bubblewrap).\n"
        )
        sys.exit(1)
    sandbox = "none" if args.no_sandbox else "bwrap"
    season = load_season(args.season)
    info = request("GET", "/api/v1/season")
    scale = season.scale(info["scale"])
    device = pick_device(args.device)
    hw = hardware_info(device)
    out.line("", out.bold(f"stet verify · {key.handle} · {hw.get('gpu', device.type)}"))
    if device.type != "cuda":
        out.line("", "warning: no CUDA or ROCm GPU found. Verifying on this machine will be very slow.")
    cal = calibration(season, scale, device)
    factor = time_factor(cal)
    out.line("speed", f"{cal['tokens_per_second']:,} tokens/s on the {scale.name} student · edit time limit x{factor:.2f}")
    hello = {
        "harness": __version__, "season": season.id, "season_digest": season.digest, "hardware": hw,
        "tokens_per_second": cal["tokens_per_second"], "sandbox": sandbox,
    }  # fmt: skip
    done = 0
    idle_since = None
    while True:
        try:
            r = request("POST", "/api/v1/jobs/claim", hello, key=key)
        except ApiError as e:
            if e.status in (400, 401, 403, 426):
                sys.stderr.write(f"stet: {e}\n")
                sys.exit(1)
            out.line("", out.dim(f"server unavailable ({e}); retrying in 60s"))
            time.sleep(60)
            continue
        job = r.get("job")
        if not job:
            if args.once or (args.max_jobs and done >= args.max_jobs):
                out.line("", r.get("message") or "No work right now.")
                return
            if idle_since is None:
                idle_since = time.time()
                out.line("", out.dim((r.get("message") or "No work right now.") + " Waiting; Ctrl-C to stop."))
            time.sleep(min(300, max(5, int(r.get("retry_after") or 60))))
            continue
        idle_since = None
        work = cache_dir() / "verify" / str(job["id"])
        shutil.rmtree(work, ignore_errors=True)
        e = job["editor"]
        out.line("", "")
        out.line("job", f"{job['id']} · {editor_label(e)} · round {job['round']['id']}")
        try:
            rnd = round_from_api(season, job["round"])
            base = baseline(rnd, scale, device, progress=out.progress("baseline"), log=lambda m: out.line("", out.dim("baseline " + m)))
            blob = request("GET", f"/api/v1/editors/{e['id']}/bundle", key=key, raw=True)
            if hashlib.sha256(blob).hexdigest() != e["id"]:
                raise OSError("the bundle does not match the editor's id")
            bundle_dir = work / "bundle"
            unpack(blob, bundle_dir)
            project = load_project(bundle_dir)
            fetch_assets(project, bundle_dir)
            result = run_round(
                bundle_dir, project.entry, rnd, scale, work, sandbox=sandbox, device=device, time_factor=factor,
                progress=out.progress("training"), log=lambda m: out.line("", out.dim(m)),
            )  # fmt: skip
        except KeyboardInterrupt:
            _release(job, key, "interrupted")
            shutil.rmtree(work, ignore_errors=True)
            out.line("", "Stopped; the job was handed back.")
            return
        except Exception as err:  # our problem, not the editor's: hand the job back untouched
            _release(job, key, f"{type(err).__name__}: {err}"[:300])
            shutil.rmtree(work, ignore_errors=True)
            out.line("", out.bad(f"could not run this job ({err}); handed it back"))
            time.sleep(30)
            continue
        payload = slim_result(result, base)
        try:
            ack = request("POST", f"/api/v1/jobs/{job['id']}/result", payload, key=key)
        except ApiError as err:
            out.line("", out.bad(f"the server did not accept the result: {err}"))
            ack = {}
        shutil.rmtree(work, ignore_errors=True)
        done += 1
        if result["valid"]:
            gain = base["exam"]["loss"] - result["exam"]["loss"]
            out.line("result", f"exam {result['exam']['loss']:.4f} · baseline {base['exam']['loss']:.4f} · gain {out.bold(f'{gain:+.4f}')}")
        else:
            out.line("result", f"invalid: {result['invalid_reason']}")
        if ack:
            v = ack.get("verifier") or {}
            out.line("server", f"{ack.get('status', '')}" + (f" · {ack['message']}" if ack.get("message") else "") + (f" · {v.get('validated', 0)} validated, streak {v.get('streak', 0)}" if v else ""))
        if args.once or (args.max_jobs and done >= args.max_jobs):
            return


def _release(job: dict, key: Key, reason: str) -> None:
    try:
        request("POST", f"/api/v1/jobs/{job['id']}/release", {"reason": reason}, key=key, timeout=20)
    except Exception:
        pass


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

    p = sub.add_parser("verify", help="lend your GPU: run and score other people's editors")
    p.add_argument("--once", action="store_true", help="do one job and exit")
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--season", default="s0", help=argparse.SUPPRESS)
    p.add_argument("--no-sandbox", action="store_true", help=argparse.SUPPRESS)  # local testing only
    p.set_defaults(fn=cmd_verify)

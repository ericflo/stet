"""The `stet` command."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from importlib import resources
from pathlib import Path

from . import __version__
from .spec import CURRENT_SEASON

OK, WARN, BAD = "ok", "warn", "fail"


# --- output ---------------------------------------------------------------------------------


class Out:
    """Human output goes to stderr so that --json on stdout stays clean."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.tty = sys.stderr.isatty() and not os.environ.get("NO_COLOR")

    def c(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.tty else s

    def dim(self, s):
        return self.c("2", s)

    def bold(self, s):
        return self.c("1", s)

    def good(self, s):
        return self.c("32", s)

    def bad(self, s):
        return self.c("31", s)

    def line(self, label: str = "", text: str = "") -> None:
        if not self.quiet:
            sys.stderr.write(f"{self.dim(f'{label:<10}')}{text}\n" if label else f"{text}\n")
            sys.stderr.flush()

    def progress(self, label: str):
        last = [0.0]

        def cb(step, steps, loss, seconds):
            if self.quiet or not self.tty:
                return
            if time.time() - last[0] < 0.2 and step < steps:
                return
            last[0] = time.time()
            width = 24
            fill = int(width * step / steps)
            eta = seconds / step * (steps - step)
            sys.stderr.write(
                f"\r{self.dim(f'{label:<10}')}{'█' * fill}{self.dim('░' * (width - fill))} {step}/{steps} · train loss {loss:.3f} · {eta:.0f}s left "
            )
            if step == steps:
                sys.stderr.write("\r" + " " * 100 + "\r")
            sys.stderr.flush()

        return cb


def die(msg: str, code: int = 1):
    sys.stderr.write(f"stet: {msg}\n")
    sys.exit(code)


# --- helpers --------------------------------------------------------------------------------


TORCH_HELP = (
    "  The installer brings PyTorch with it, in the build for this machine's GPU (NVIDIA or AMD).\n"
    "  Run it again:\n"
    "    curl -fsSL https://www.stetrun.com/install.sh | sh\n"
    "  If it picks the wrong build, name the wheel index yourself (they are listed at\n"
    "  https://pytorch.org/get-started/locally/), for example:\n"
    "    curl -fsSL https://www.stetrun.com/install.sh | STET_TORCH_INDEX=https://download.pytorch.org/whl/rocm7.14 sh\n"
    "  If you installed stet with pip instead, install PyTorch into the same environment."
)


def gpu_vendors() -> set[str]:
    """Which makers' GPUs this Linux machine has, whatever PyTorch can or cannot see."""
    found = set()
    for vendor in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            found.add({"0x1002": "amd", "0x10de": "nvidia"}.get(vendor.read_text().strip(), ""))
        except OSError:
            pass
    return found - {""}


def need_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        die("this command trains a model and needs PyTorch.\n" + TORCH_HELP)


def resolve_round(args, offset: int = 0):
    """--round accepts 3 (practice round 3), g3 (gym round 3), or a full id like s2-o00012."""
    import re

    from .spec import load_season, parse_round_id, practice_round

    season = load_season(args.season)
    r = str(args.round)
    try:
        m = re.fullmatch(r"([pg]?)(\d+)", r)
        if m:
            kind, index = {"": "practice", "p": "practice", "g": "gym"}[m.group(1)], int(m.group(2))
        else:
            try:
                _, kind, index = parse_round_id(r)
            except Exception:
                die(f"cannot parse round {r!r}; use 3 (practice), g3 (gym), or an id like {season.id}-o00003")
        if kind == "official":
            from .client import fetch_round

            # Replays of official rounds default to the scale they were scored at.
            return season, season.scale(args.scale or season.official_scale), fetch_round(season, index + offset)
        return season, season.scale(args.scale or "nano"), practice_round(season, index + offset, kind)
    except ValueError as e:
        die(str(e))


def report(out: Out, result: dict, base: dict | None, season) -> None:
    from .score import format_multiplier, multiplier

    edit = result.get("edit")
    if edit:
        status = out.good(edit["status"]) if edit["status"] == "ok" else out.bad(edit["status"])
        out.line("editing", f"{status} in {edit['seconds']:.1f}s of {edit['time_limit']:.0f}s")
    if not result["valid"]:
        out.line("result", out.bad("invalid") + f": {result['invalid_reason']}")
        if edit and edit.get("log_tail"):
            out.line("", out.dim("--- last lines of the curator's output ---"))
            out.line("", edit["log_tail"].rstrip()[-1500:])
        out.line("", "An invalid run scores as the unedited corpus: gain 0.")
        return
    s = result["script"]
    out.line(
        "script",
        f"{s['out_docs']:,} documents from {s['src_docs']:,} sources ({100 * s['kept_doc_frac']:.1f}% of the corpus) · "
        f"{s['out_bytes'] / 1e6:.1f} MB · ink {s['ink_tokens']:,}/{s['ink_budget']:,} tokens · {s['passes']:.2f} passes"
        + (" · truncated at the budget" if s["truncated"] else ""),
    )
    t = result["train"]
    out.line("training", f"{t['tokens']:,} tokens in {t['seconds']:.0f}s ({t['tokens_per_second']:,} tok/s) · final train loss {t['final_train_loss']:.3f}")
    e = result["exam"]
    out.line("exam", f"{e['items']:,} items · mean {e['mean']:.4f} bits per byte" + (f"   baseline {base['exam']['mean']:.4f}" if base else ""))
    if base and "gain" in result:
        out.line("", "          " + " · ".join(f"{k} {v:+.3f}" for k, v in (result.get("pillars") or {}).items()))
        text = out.bold(f"{result['gain']:+.4f}") + " bits per byte: writing three quarters, skills one, of those above"
        m = multiplier(season, result["round"]["kind"], result["scale"], result["gain"])
        if m:
            text += f"   →   {out.bold(format_multiplier(m))} data efficiency"
        noise = ((season.raw.get("noise") or {}).get(result["scale"]) or {}).get("gain_sd")
        if noise:
            text += out.dim(f"   (training again moves this by about ±{noise:.3f})")
        out.line("gain", text)


# --- commands -------------------------------------------------------------------------------


def cmd_init(args):
    dest = Path(args.path).resolve()
    name = args.name or dest.name.lower().replace("_", "-").replace(" ", "-")
    tpl = resources.files("stet.templates")
    dest.mkdir(parents=True, exist_ok=True)
    wrote = []
    for src, dst in (("stet.toml", "stet.toml"), ("curator.py", "curator.py"), ("AGENTS.md", "AGENTS.md"), ("gitignore", ".gitignore")):
        target = dest / dst
        if target.exists() and not args.force:
            continue
        text = tpl.joinpath(src).read_text(encoding="utf-8")
        if src == "stet.toml":
            text = text.replace("{name}", name)
        target.write_text(text, encoding="utf-8")
        wrote.append(dst)
    out = Out()
    out.line("", f"Curator project ready in {dest}" + (f" ({', '.join(wrote)})" if wrote else " (nothing overwritten)"))
    out.line("", "")
    out.line("", f"  cd {args.path}" if args.path != "." else "")
    out.line("", "  stet run            score it on a practice round")
    out.line("", "  stet submit         enter it")
    out.line("", "")
    out.line("", "AGENTS.md has everything a coding agent needs to work on it.")


def _with_baseline(result: dict, rnd, scale, device, out: Out, skip: bool):
    if not result["valid"]:
        result["gain"] = 0.0  # by the rules, and so that whatever reads the JSON finds a number there
    if skip:
        return None
    from .pipeline import baseline

    base = baseline(rnd, scale, device, progress=out.progress("baseline"), log=lambda m: out.line("", out.dim("baseline " + m)))
    # The gain is the exam's aggregate of the item-by-item differences, with the languages
    # weighed by the published table (a round's own corpus is measured only for official rounds).
    from . import data
    from .exam.aggregate import gains, round_score

    result["baseline"] = {"mean": base["exam"]["mean"]}
    if result["valid"]:
        agg = round_score(data.round_exam(rnd, scale), gains(result["exam"]["item_scores"], base["exam"]["item_scores"]))
        result["gain"] = round(agg["score"], 5)
        result["pillars"] = {k: v["score"] for k, v in agg["pillars"].items()}
    return base


def _emit(args, result: dict):
    if args.json:
        slim = dict(result)
        if "train" in slim:
            slim["train"] = {k: v for k, v in slim["train"].items() if k != "curve"}
        json.dump(slim, sys.stdout, indent=1)
        sys.stdout.write("\n")


def _summary(out: Out, results: list[dict]) -> dict:
    """Mean gain over several rounds, with its standard error."""
    import math

    gains = [r.get("gain", 0.0) if r["valid"] else 0.0 for r in results]
    mean = sum(gains) / len(gains)
    se = math.sqrt(sum((g - mean) ** 2 for g in gains) / (len(gains) - 1) / len(gains)) if len(gains) > 1 else None
    out.line("", "")
    out.line("mean gain", out.bold(f"{mean:+.4f}") + (f" ± {se:.4f} (standard error over {len(gains)} rounds)" if se is not None else ""))
    return {"rounds": [r["round"]["id"] for r in results], "gains": gains, "mean_gain": round(mean, 5), "se": None if se is None else round(se, 5)}


def _edit_time_line(out: Out, result: dict, season) -> None:
    """How long the edit took here, and what that means on the slowest machine that runs
    curators. A time limit in seconds is only as fair as the comparison is plain."""
    from . import bench

    edit = result.get("edit") or {}
    took, limit = edit.get("seconds"), edit.get("time_limit")
    if not took or not limit:
        return
    m = bench.saved()
    if m is None:
        out.line("edit", f"{took:.0f}s of {limit:.0f}s here. `stet bench` says how this machine compares with the slowest that runs curators")
        return
    g, c = bench.ratios(season, m)
    if g <= 0 or c <= 0:
        out.line("edit", f"{took:.0f}s of {limit:.0f}s here")
        return
    lo, hi = sorted((took * min(g, c), took * max(g, c)))
    lo, hi = max(lo, took) if min(g, c) >= 1 else lo, hi
    verdict = "over the limit if most of that time is spent on the faster part" if hi > limit >= lo else ("over the limit" if lo > limit else "inside the limit")
    out.line("edit", f"{took:.0f}s of {limit:.0f}s here. This machine's GPU is {g:.1f}x and its CPU {c:.1f}x the slowest that runs curators: allow {lo:.0f}-{hi:.0f}s there, {verdict}")


def cmd_run(args):
    need_torch()
    from .pipeline import run_round
    from .project import ProjectError, load_project, pack, unpack
    from .runner import remove_tree
    from .train import pick_device

    out = Out(quiet=args.quiet)
    try:
        project = load_project(args.path)
    except ProjectError as e:
        die(str(e))
    device = pick_device(args.device)
    sandbox = "bwrap" if args.sandbox is True else (args.sandbox or "none")  # --sandbox alone is bwrap
    results = []
    for k in range(max(1, args.rounds)):
        season, scale, rnd = resolve_round(args, offset=k)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        work = project.path / ".stet" / "runs" / f"{stamp}-{rnd.id}-{scale.name}"
        work.mkdir(parents=True, exist_ok=True)
        out.line("", out.bold(f"stet · {project.name} · {rnd.kind} round {rnd.id} · {scale.name} · {device.type}"))
        bundle = project.path
        if sandbox != "none":
            # Run what `stet submit` would send, unpacked as a verifier unpacks it: a file the
            # bundle leaves out is missing here too, not first in an official round.
            try:
                blob, _ = pack(project)
                unpack(blob, work / "bundle")
            except ProjectError as e:
                die(str(e))
            bundle = work / "bundle"
        try:
            result = run_round(
                bundle, project.entry, rnd, scale, work, sandbox=sandbox, device=device,
                echo=args.verbose, progress=out.progress("training"), log=lambda m: out.line("", out.dim(m)),
                assets=project.assets, time_limit=getattr(args, "time_limit", None),
            )  # fmt: skip
        finally:
            if bundle != project.path:
                remove_tree(bundle)
        result["editor"] = {"name": project.name}
        base = _with_baseline(result, rnd, scale, device, out, args.no_baseline)
        (work / "result.json").write_text(json.dumps(result, indent=1))
        report(out, result, base, season)
        _edit_time_line(out, result, season)
        out.line("saved", str((work / "result.json").relative_to(Path.cwd()) if work.is_relative_to(Path.cwd()) else work / "result.json"))
        results.append(result)
    if len(results) == 1:
        _emit(args, results[0])
    else:
        summary = _summary(out, results)
        if args.json:
            json.dump(summary, sys.stdout, indent=1)
            sys.stdout.write("\n")
    sys.exit(0 if all(r["valid"] for r in results) else 2)


def script_file(ref: str, season, rnd) -> Path:
    """The edit script `stet score` was asked to grade, as a file of JSON lines.

    `ref` is a file (plain, or zstd-compressed as the site serves them), the sha256 of a
    published script, or the name of an entry in the official round being replayed
    (`tidy`, `stet/tidy`, or `stet/tidy@v2` when the round holds more than one version of
    it). Anything fetched is checked against its hash.
    """
    import hashlib
    import re

    import zstandard

    from .data import cache_dir
    from .project import unzstd

    limit = int(season.raw["script"]["max_bytes"])

    def unpacked(blob: bytes, want: str | None) -> Path:
        try:
            raw = unzstd(blob, limit)
        except zstandard.ZstdError:
            die(f"{'the script the server sent' if want else ref} is zstd-compressed, but damaged or cut short")
        sha = hashlib.sha256(raw).hexdigest()
        if len(raw) > limit:
            die(f"that script is over the {limit >> 20} MB limit")
        if want and sha != want:
            die(f"the script the server sent does not match its hash ({want[:12]})")
        dest = cache_dir() / "scripts" / f"{sha}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
        tmp.write_bytes(raw)
        tmp.replace(dest)
        return dest

    path = Path(ref)
    if path.is_file():
        with path.open("rb") as f:
            packed = f.read(4) == b"\x28\xb5\x2f\xfd"  # zstd's magic number
        return unpacked(path.read_bytes(), None) if packed else path
    from .client import ApiError, request

    sha = ref.lower()
    if rnd.kind != "official":
        # A published script edits one official round's corpus. On any other round its line
        # numbers mean other documents, and the score would be a number about nothing.
        hint = f"`stet rounds` lists them, e.g. --round {season.id}-o00001"
        if re.fullmatch(r"[0-9a-f]{64}", sha):
            die(f"a published script belongs to the official round it was written for. Say which: stet score {sha[:12]}... --round <official round id> ({hint})")
        die(f"no file called {ref!r}. To grade a published script, give its sha256 or its entry's name, with --round <official round id> ({hint})")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        try:
            entries = request("GET", f"/api/v1/rounds/{rnd.id}").get("entries") or []
        except ApiError as e:
            die(str(e))
        from .client import clean

        owner, _, name = ref.rpartition("/")
        name, at, version = name.partition("@")
        wanted = re.fullmatch(r"v?([0-9]{1,6})", version.lower())
        if at and not wanted:
            die(f"{ref!r}: a version is written like {owner + '/' if owner else ''}{name}@v2")
        scripted = [e for e in entries if isinstance(e, dict) and (e.get("script") or {}).get("sha256")]
        found = [
            e for e in scripted
            if e.get("name") == name and (not owner or e.get("owner") == owner) and (not wanted or str(e.get("version")) == str(int(wanted.group(1))))
        ]  # fmt: skip
        full = lambda e: clean(f"{e.get('owner')}/{e.get('name')}@v{e.get('version')}", 100)  # noqa: E731
        if len(found) > 1:
            # Two versions of one curator share a round while the newer one qualifies, and two
            # people may use one name.
            die(
                f"round {rnd.id} has more than one entry called {name!r} with a script: {', '.join(sorted(full(e) for e in found))}. "
                f"Say which: stet score {full(found[0])} --round {rnd.id}"
            )
        if not found:
            names = ", ".join(sorted(full(e) for e in scripted))
            die(f"no file called {ref!r}, and round {rnd.id} has no entry of that name with a script. Its entries: {names or 'none yet'}")
        sha = str(found[0]["script"]["sha256"]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            die("the server described that script in a way this client does not accept")
    try:
        blob = request("GET", f"/api/v1/scripts/{sha}", raw=True, timeout=600, limit=limit + (1 << 20))
    except ApiError as e:
        die(f"could not fetch script {sha[:12]}: {e}")
    return unpacked(blob, sha)


def cmd_score(args):
    """Grade an edit script you already have, skipping the curator."""
    need_torch()
    from .pipeline import grade_round
    from .train import pick_device

    out = Out(quiet=args.quiet)
    season, scale, rnd = resolve_round(args)
    script = script_file(args.script, season, rnd)
    device = pick_device(args.device)
    out.line("", out.bold(f"stet score · {args.script} · {rnd.kind} round {rnd.id} · {scale.name} · {device.type}"))
    result = grade_round(rnd, scale, script, device=device, progress=out.progress("training"), log=lambda m: out.line("", out.dim(m)))
    base = _with_baseline(result, rnd, scale, device, out, args.no_baseline)
    report(out, result, base, season)
    _emit(args, result)
    sys.exit(0 if result["valid"] else 2)


def cmd_baseline(args):
    need_torch()
    from .pipeline import baseline
    from .train import pick_device

    out = Out(quiet=args.quiet)
    season, scale, rnd = resolve_round(args)
    base = baseline(rnd, scale, pick_device(args.device), progress=out.progress("training"), log=lambda m: out.line("", out.dim(m)))
    out.line("round", f"{rnd.id} · {scale.name}")
    out.line("baseline", out.bold(f"{base['exam']['mean']:.4f}") + f" bits per byte, the mean over {base['exam']['items']:,} items")
    if args.json:
        json.dump(base, sys.stdout, indent=1)
        sys.stdout.write("\n")


def cmd_examine(args):
    """The organisers' examiner: harvest every hour of the clock as it settles, and make each
    official round's exam at its draw from the hours the draw chose."""
    if args.follow:
        return _examine_follow(args)
    if not args.round:
        die("name a round, or pass --follow")
    _examine_one(args, args.round)


def _hours_dir() -> Path:
    from .data import cache_dir

    return cache_dir() / "examine" / "hours"


def _hour_name(hour: int) -> str:
    return time.strftime("%Y-%m-%d %H:00 UTC", time.gmtime(hour))


def _harvest_due(season, now: float, skip=()) -> int | None:
    """The newest settled hour a round could still be given that this machine has not harvested.
    Newest first: after a stop, the hours the sources still hold whole come before the old ones."""
    from .exam.fresh.harvest import HOUR, settled

    last = settled(season, now)
    first = int(now) - int(season.raw["exam"]["fresh"]["reach_seconds"])
    for hour in range(last, first - 1, -HOUR):
        if hour >= first and hour not in skip and not (_hours_dir() / f"{hour}.jsonl").exists():
            return hour
    return None


def _harvest_one(season, hour: int, key, log) -> dict:
    """Harvest an hour and tell the server its hash, at once: a round drawn later can be given it."""
    from .exam.fresh.harvest import harvest_hour

    dest = _hours_dir() / f"{hour}.jsonl"
    part = dest.with_name(f"{hour}.part.jsonl")  # under its own name only once it is whole
    meta, sha = harvest_hour(season, hour, part, log=log)
    part.replace(dest)
    return _commit_hour(hour, dest, key) if key else {"hour": hour, "sha256": sha, "documents": meta.get("documents")}


def _commit_hour(hour: int, path: Path, key) -> dict:
    import hashlib

    from .client import request
    from .exam.fresh import read_snapshot

    meta, docs = read_snapshot(path)
    body = {"hour": hour, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "documents": len(docs),
            "sources": {k: v for k, v in (meta.get("sources") or {}).items() if isinstance(v, int)}}
    request("POST", "/api/v1/admin/harvests", body, key=key, timeout=120)
    return body


def _harvest_follow(season, key, out, stop) -> None:
    """Every hour, as soon as it has settled. Runs beside the exams, so that a round drawn while
    an hour is being read does not wait for it."""
    from .exam.fresh.harvest import HOUR

    keep = int(season.raw["exam"]["fresh"]["reach_seconds"]) + 7 * 86400
    failed: dict[int, int] = {}
    while not stop.is_set():
        hour = _harvest_due(season, time.time(), skip=[h for h, n in failed.items() if n >= 3])
        if hour is None:
            stop.wait(30)
            continue
        try:
            got = _harvest_one(season, hour, key, lambda m: out.line("", out.dim(m)))
            out.line("", f"harvested {_hour_name(hour)}: {got['documents']} documents ({got['sha256'][:12]})")
        except (Exception, SystemExit) as e:  # tried three times; an hour that will not come does not hold up the others
            failed[hour] = failed.get(hour, 0) + 1
            out.line("", out.bad(f"harvest of {_hour_name(hour)}: {e}"))
            stop.wait(30)
        for old in _hours_dir().glob("*.jsonl"):  # a round's exam can be made again for a week
            if old.stem.isdigit() and int(old.stem) + HOUR < time.time() - keep:
                old.unlink(missing_ok=True)


def _round_hours(season, d: dict, seed: bytes) -> tuple[list[tuple[int, Path]], bool]:
    """The harvests a round's exam is made from, and whether they were chosen late. The server
    chose them at the draw from those it had been told of (the round's `exam_hours`). If it had
    been told of none, they are chosen now, the same way, from those this machine holds."""
    import hashlib

    from .exam.fresh.harvest import choose_hours

    named = d.get("exam_hours") or []
    late = not named
    if late:
        first = int(d["draw_at"]) - int(season.raw["exam"]["fresh"]["reach_seconds"])
        held = [int(p.stem) for p in _hours_dir().glob("*.jsonl") if p.stem.isdigit() and int(p.stem) >= first]
        named = [{"hour": h} for h in choose_hours(seed, held, int(season.raw["exam"]["fresh"]["hours"]))]
    out = []
    for x in named:
        path = _hours_dir() / f"{int(x['hour'])}.jsonl"
        if not path.exists():
            raise RuntimeError(f"the harvest of {_hour_name(int(x['hour']))} is not on this machine")
        if x.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != x["sha256"]:
            raise RuntimeError(f"the harvest of {_hour_name(int(x['hour']))} on this machine is not the one the server was told of")
        out.append((int(x["hour"]), path))
    return out, late


def _examine_one(args, round_id: str) -> None:
    from .client import fetch_round, need_key, request
    from .data import cache_dir
    from .exam.examine import examine
    from .spec import load_season, parse_round_id

    season = load_season(args.season)
    _, kind, index = parse_round_id(round_id)
    if kind != "official":
        die("only an official round has an exam to make")
    rnd = fetch_round(season, index)
    d = request("GET", f"/api/v1/rounds/{rnd.id}")["round"]
    out = Out(quiet=args.quiet)
    if d.get("exam_digest"):
        out.line("", f"{rnd.id} has its exam already ({d['exam_digest'][:12]})")
        return
    key = need_key()
    snapshots, late = _round_hours(season, d, rnd.seed)
    if not snapshots:
        die(f"{rnd.id} has no harvest to make its exam from yet: none of the hours it could be given has been harvested")
    out.line("", out.bold(f"stet examine · {rnd.id} · {d.get('scale')}"))
    log = lambda m: out.line("", out.dim(m))  # noqa: E731
    log(f"  its writing: {', '.join(_hour_name(h) for h, _ in snapshots)}" + (", chosen after the draw: the server had been told of no harvest" if late else ""))
    if late:
        for hour, path in snapshots:
            _commit_hour(hour, path, key)
    examine(rnd, season.scale(d.get("scale") or season.official_scale), snapshots, cache_dir() / "examine" / rnd.id, key, log=log)


def _examine_follow(args) -> None:
    """Every hour's harvest as it settles, and every open round's exam as soon as it is drawn."""
    import threading

    from .client import need_key, request
    from .spec import load_season

    season = load_season(args.season)
    out = Out(quiet=args.quiet)
    out.line("", out.bold("stet examine --follow"))
    stop = threading.Event()
    threading.Thread(target=_harvest_follow, args=(season, need_key(), out, stop), daemon=True).start()
    failed: dict[str, int] = {}
    while True:
        try:
            rounds = request("GET", "/api/v1/rounds?limit=20").get("rounds") or []
        except Exception as e:
            out.line("", out.dim(f"the server did not answer ({e}); trying again in a minute"))
            rounds = []
        for x in sorted(rounds, key=lambda x: x["id"]):
            if x.get("status") != "open" or x.get("exam_digest") or failed.get(x["id"], 0) >= 20:
                continue
            try:
                _examine_one(args, x["id"])
            except (Exception, SystemExit) as e:  # tried again in a moment: a harvest may be on its way
                failed[x["id"]] = failed.get(x["id"], 0) + 1
                out.line("", out.bad(f"{x['id']}: {e}"))
        time.sleep(20)


def cmd_fetch(args):
    from . import data

    season, scale, rnd = resolve_round(args)
    corpus, exam = data.materialize(rnd, scale)
    Out().line("ready", f"{rnd.id} · {scale.name}: {len(corpus):,} documents, {corpus.total_bytes / 1e6:.0f} MB at {corpus.path}")


def size_text(n: int) -> str:
    return f"{n / 2**30:.1f} GB" if n >= 1 << 30 else f"{n / 2**20:.0f} MB"


def cmd_clean(args):
    """Delete what stet downloaded to this machine."""
    from . import data
    from .client import verifiers_running

    out = Out()
    cache = data.cache_dir()
    rows = data.cache_report()
    if not rows:
        out.line("", f"Nothing to clear: stet has nothing downloaded in {cache}.")
        return
    out.line("", out.bold(f"{cache} · {size_text(sum(size for _, _, size in rows))}"))
    for _, what, size in rows:
        out.line("", f"  {size_text(size):>8}  {what}")
    days = data.cache_days()
    if args.dry_run:
        out.line("", "")
        out.line("", "Nothing was deleted. `stet clean` deletes all of it; it is fetched again when it is next needed.")
        if days:
            out.line("", out.dim(f"Left alone, stet deletes what it has not used for {days:g} days (STET_CACHE_DAYS)."))
        return
    busy = verifiers_running()
    freed = data.clear_cache(in_use=bool(busy))
    out.line("", "")
    out.line("", f"Cleared {size_text(freed)}. It is fetched again when it is next needed.")
    if busy:
        out.line("", "A `stet verify` is running here, so what it used in the last few hours was left for it.")
    out.line("", out.dim("Not touched: your key and account, your own projects, and the packages stet was installed from"))
    out.line("", out.dim("(those are in uv's cache, shared with anything else uv installs: `uv cache clean` frees it)."))


def cmd_doctor(args):
    from . import bench, data, env
    from .runner import bwrap_available, scope_available
    from .spec import load_season

    out = Out()
    season = load_season()
    grade_devices = season.raw.get("grade_devices", ["cuda"])
    rows: list[tuple[str, str, str]] = []
    rows.append((OK, "stet", f"{__version__} on Python {sys.version.split()[0]} ({sys.platform})"))
    try:
        import torch

        from .train import hardware_info, pick_device

        dev = pick_device()
        hw = hardware_info(dev)
        if dev.type == "cuda":
            rows.append((OK, "gpu", f"{hw['gpu']} · {hw['vram_gb']} GB · {hw['backend']} · torch {torch.__version__}"))
        elif gpu_vendors():
            # The GPU is there and this PyTorch build cannot use it: the commonest way to end up slow.
            maker = "an AMD" if "amd" in gpu_vendors() else "an NVIDIA"
            rows.append((WARN, "gpu", f"this machine has {maker} GPU, but this PyTorch build ({torch.__version__}) cannot use it; training will run on {dev.type}.\n" + TORCH_HELP))
        else:
            rows.append((WARN, "gpu", f"no CUDA or ROCm GPU; training will run on {dev.type} (fine for --scale nano)"))
        if dev.type in grade_devices:
            rows.append((OK, "verify", "this machine can grade for the leaderboard: `stet verify`"))
        else:
            rows.append((WARN, "verify", "official grading needs an NVIDIA or AMD GPU, so every grade uses the same arithmetic"))
    except ImportError:
        rows.append((WARN, "torch", "not installed: `stet run` and `stet verify` need it.\n" + TORCH_HELP))
    ok, why = bwrap_available() if sys.platform == "linux" else (False, "")
    if ok:
        scoped, _ = scope_available()
        rows.append((OK, "sandbox", f"bubblewrap at {why}" + (", with memory and process limits" if scoped else "") + " (`stet run --sandbox`)"))
    elif sys.platform == "linux":
        from .cage import available as cage_available

        caged, cage_why = cage_available()
        if caged:
            rows.append((OK, "sandbox", "the container sandbox: this is a container of its own, as root (`stet run --sandbox container`, `stet verify --edits --sandbox container`)"))
        else:
            rows.append((WARN, "sandbox", why + ". Optional: only `stet run --sandbox` uses it."))
    # The libraries a curator may import. `stet run` uses whatever is installed; a machine
    # that runs curators for others must have exactly the season's.
    differs = env.problems(season)
    if not differs:
        rows.append((OK, "libraries", env.summary(season)))
    else:
        rows.append((WARN, "libraries", "not the season's set: " + "; ".join(d[:300] for d in differs) + ".\n  A curator that imports one of these may behave differently here than in an official run. The installer puts it right:\n    curl -fsSL https://www.stetrun.com/install.sh | sh"))
    m = bench.saved()
    if m is None:
        rows.append((WARN, "speed", "not measured yet: `stet bench` (half a minute) compares this machine with the slowest that runs curators"))
    else:
        g, c = bench.ratios(season, m)
        short = bench.shortfalls(season, m)
        text = f"GPU {g:.1f}x and CPU {c:.1f}x the slowest machine that runs curators"
        rows.append((OK, "speed", text) if not short else (WARN, "speed", text + ". It cannot run curators for the leaderboard: " + "; ".join(short)))
    cache = data.cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free / 2**30
    held = sum(size for _, _, size in data.cache_report())
    days = data.cache_days()
    keeps = f"what goes unused for {days:g} days is deleted" if days else "nothing is deleted (STET_CACHE_DAYS=0)"
    rows.append((OK if free > 10 else WARN, "cache", f"{cache} · {size_text(held)} downloaded, {free:.0f} GB free" + ("" if free > 10 else " (a round needs a few GB)")
                 + f" · {keeps}; `stet clean` clears it now"))  # fmt: skip
    for status, name, text in rows:
        mark = {OK: out.good("✓"), WARN: out.c("33", "!"), BAD: out.bad("✗")}[status]
        out.line("", f" {mark} {name:<8} {text}")
    sys.exit(1 if any(s == BAD for s, _, _ in rows) else 0)


def cmd_bench(args):
    """Measure this machine against the slowest that may run curators."""
    need_torch()
    from . import bench
    from .spec import load_season

    out = Out()
    season = load_season()
    out.line("", out.dim("measuring CPU and GPU (about half a minute; close other GPU work first) ..."))
    m = bench.measure(season)
    bench.save(m)
    need = bench.floor(season)
    g, c = bench.ratios(season, m)
    machine = season.raw.get("machine") or {}
    out.line("gpu", f"{m['gpu'] or 'none'} · {m['gpu_tokens_per_second']:,.0f} tokens a second · {g:.1f}x the floor of {need['gpu_tokens_per_second']:,.0f} · {m['gpu_memory_gb']:.0f} GB")
    out.line("cpu", f"{m['cpu_mb_per_second']:.0f} MB a second on {min(int(need['cores']), m['cores'])} cores · {c:.1f}x the floor of {need['cpu_mb_per_second']:.0f} · {m['cores']} cores, {m['memory_gb']:.0f} GB")
    out.line("", "")
    out.line("", f"Every curator is given {machine.get('cores', '?')} cores, {machine.get('memory_gb', '?')} GB of memory and {machine.get('gpu_memory_gb', '?')} GB of GPU memory,")
    out.line("", "on a machine at least as fast as the floor. A curator that fits its time limit on the")
    out.line("", "floor fits it everywhere, so plan for the floor:")
    scale = season.scale(season.official_scale)
    if g > 0 and c > 0:
        out.line("", f"  the {scale.edit_seconds}s limit at {scale.name} is worth {scale.edit_seconds / max(g, 1e-9):.0f}s of this GPU and {scale.edit_seconds / max(c, 1e-9):.0f}s of this CPU.")
    short = bench.shortfalls(season, m)
    out.line("", "")
    if short:
        out.line("", "This machine cannot run curators for the leaderboard: " + "; ".join(short) + ".")
        out.line("", "It can still grade (`stet verify`) if it has an NVIDIA or AMD GPU, and run your own curator (`stet run`).")
    else:
        out.line("", "This machine is fast enough to run curators for the leaderboard: `stet verify --edits`, on Linux.")
    if args.json:
        json.dump(m, sys.stdout, indent=1)
        sys.stdout.write("\n")


def _manifest_without_assets(text: str) -> str:
    """stet.toml with its [[assets]] tables taken out, so that they can be written again."""
    keep, skipping = [], False
    for line in text.splitlines():
        head = line.strip()
        if head.startswith("[[assets]]"):
            skipping = True
            continue
        if skipping and head.startswith("["):
            skipping = False
        if not skipping:
            keep.append(line)
    return "\n".join(keep).rstrip() + "\n"


def cmd_asset(args):
    """Name the files a curator needs that are too big for its bundle: model weights."""
    from . import assets as assets_
    from .project import ProjectError, load_project
    from .spec import load_season

    out = Out()
    season = load_season()
    lim = assets_.limits(season)
    try:
        project = load_project(args.path)
    except ProjectError as e:
        die(str(e))
    have: list = list(project.assets)

    def show(items):
        for a in items:
            mark = "cached" if assets_.cached(a).exists() else "not fetched"
            out.line("", f"  {a.size / 1e6:9.1f} MB  {a.path}  {out.dim(mark)}")
        total = sum(a.size for a in items)
        out.line("", f"  {len(items)} of {lim['max_files']} files, {total / 2**30:.2f} of {lim['max_bytes'] / 2**30:.0f} GiB")

    if args.action == "list":
        if not have:
            out.line("", "This curator names no assets. `stet asset add <owner>/<repository>` adds a Hugging Face repository's files.")
            return
        show(have)
        return
    if args.action == "fetch":
        assets_.fetch(have)
        show(have)
        return
    if args.action == "remove":
        prefix = (args.what or "").strip("/")
        if not prefix:
            die("say which to remove: a path, or the directory they are under")
        keep = [a for a in have if a.path != prefix and not a.path.startswith(prefix + "/")]
        if len(keep) == len(have):
            die(f"no asset at or under {prefix!r}")
        have = keep
    elif args.action == "add":
        if not args.what:
            die("say which repository: stet asset add <owner>/<repository>")
        try:
            commit, found, skipped = assets_.lookup(args.what, args.revision or "", args.include or None, args.under or "")
            if not found:
                die("nothing in that repository matches. See its file list on Hugging Face, and name files with --include")
            out.line("", out.dim(f"{args.what} at commit {commit[:12]}"))
            # Before anything is downloaded to learn its hash: does it fit at all?
            taken = {a.path for a in found}
            together = [a for a in have if a.path not in taken] + found
            if len(together) > lim["max_files"] or sum(a.size for a in together) > lim["max_bytes"]:
                die(
                    f"that would be {len(together)} files and {sum(a.size for a in together) / 2**30:.2f} GiB; a curator may name "
                    f"{lim['max_files']} files and {lim['max_bytes'] / 2**30:.0f} GiB. Pick the files it needs with --include (for example --include '*.safetensors' --include '*.json')"
                )
            found = assets_.fill_hashes(found, quiet=True)
        except assets_.AssetError as e:
            die(str(e))
        taken = {a.path for a in found}
        have = [a for a in have if a.path not in taken] + found
        if skipped:
            out.line("", out.dim(f"left out (name them with --include if the curator needs them): {', '.join(skipped[:12])}" + (" ..." if len(skipped) > 12 else "")))
    try:
        assets_.parse([{"path": a.path, "url": a.url, "sha256": a.sha256, "size": a.size} for a in have], season)
    except assets_.AssetError as e:
        die(str(e))
    toml = project.path / "stet.toml"
    text = _manifest_without_assets(toml.read_text(encoding="utf-8"))
    toml.write_text(text + ("\n" + assets_.to_toml(have) if have else ""), encoding="utf-8")
    show(have)
    if args.action == "add":
        out.line("", "")
        out.line("", f"Written to stet.toml. In the curator: task.asset(\"{found[0].path.split('/')[0]}\") is the directory holding them.")
        out.line("", "They are downloaded the first time `stet run` needs them (or now: stet asset fetch).")


def cmd_agents(args):
    sys.stdout.write(resources.files("stet.templates").joinpath("AGENTS.md").read_text(encoding="utf-8"))


# --- entry ----------------------------------------------------------------------------------


def add_round_args(p, default_round="0"):
    p.add_argument("--round", default=default_round, help="3 = practice round 3, g3 = gym round 3, or a round id like s2-o00012 (default: 0)")
    p.add_argument("--scale", default=None, help="nano, micro or standard (default: nano)")
    p.add_argument("--season", default=CURRENT_SEASON, help=argparse.SUPPRESS)
    p.add_argument("--device", default=None, help="torch device (default: best available)")
    p.add_argument("--json", action="store_true", help="print the result as JSON on stdout")
    p.add_argument("-q", "--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="stet",
        description="Stet: a competition for agents that edit pretraining data. The model is fixed. The trainer is fixed. The only thing you can touch is the data.",
    )
    ap.add_argument("--version", action="version", version=f"stet {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="command")

    p = sub.add_parser("init", help="create a curator project")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--name")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("run", help="run your curator on a round and score it")
    p.add_argument("path", nargs="?", default=".")
    add_round_args(p)
    p.add_argument("--rounds", type=int, default=1, help="run this many consecutive rounds and report the mean gain with its standard error")
    p.add_argument("--sandbox", nargs="?", const="bwrap", choices=["bwrap", "container"],
                   help="run the curator in the sandbox official runs use (Linux): bwrap, or container on a container of its own, as root")  # fmt: skip
    p.add_argument("--no-baseline", action="store_true", help="skip the baseline comparison")
    p.add_argument("--time-limit", type=float, default=None, help="give the curator this many seconds instead of the round's limit (see `stet bench`)")
    p.add_argument("-v", "--verbose", action="store_true", help="show the curator's output as it runs")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("asset", help="name model weights and other large files your curator needs (fetched by reference, not bundled)")
    p.add_argument("action", choices=["add", "list", "fetch", "remove"])
    p.add_argument("what", nargs="?", help="add: a Hugging Face repository, owner/name. remove: a path or directory")
    p.add_argument("--path", default=".", help="the curator project (default: here)")
    p.add_argument("--revision", help="a branch, tag or commit (default: the repository's latest). It is pinned to a commit either way")
    p.add_argument("--include", action="append", help="only files matching this pattern (may be given more than once), e.g. --include '*.safetensors'")
    p.add_argument("--under", help="the directory the files appear under for the curator (default: the repository's name, lowercased)")
    p.set_defaults(fn=cmd_asset)

    p = sub.add_parser("bench", help="measure this machine against the slowest that runs curators")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("score", help="grade an edit script directly, without running a curator")
    p.add_argument("script", help="a script file (.jsonl, or .zst as downloaded), a published script's sha256, or an entry's name in the official round given by --round")
    add_round_args(p)
    p.add_argument("--no-baseline", action="store_true", help="skip the baseline comparison")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("baseline", help="score the unedited corpus for a round")
    add_round_args(p)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("examine", help=argparse.SUPPRESS)  # organisers: harvest each hour, and make each official round's exam at its draw
    p.add_argument("--round", help="the official round (default with --follow: every open one)")
    p.add_argument("--follow", action="store_true", help="every hour as it settles, and every open round as soon as it is drawn")
    p.add_argument("--season", default=CURRENT_SEASON, help=argparse.SUPPRESS)
    p.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_examine)

    p = sub.add_parser("fetch", help="download and build a round's corpus ahead of time")
    add_round_args(p)
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("doctor", help="check this machine")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("clean", help="delete what stet downloaded to this machine (it is fetched again when needed)")
    p.add_argument("--dry-run", action="store_true", help="show what is there and delete nothing")
    p.set_defaults(fn=cmd_clean)

    p = sub.add_parser("agents", help="print the instructions for coding agents (AGENTS.md)")
    p.set_defaults(fn=cmd_agents)

    from . import client

    client.add_commands(sub)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 0
    from .assets import AssetError, Unavailable
    from .client import ApiError
    from .data import ExamNotReady
    from .project import ProjectError
    from .runner import HarnessError

    try:
        return args.fn(args) or 0
    except KeyboardInterrupt:
        die("interrupted", 130)
    except ApiError as e:
        die(str(e))
    except (OSError, HarnessError, ExamNotReady, ProjectError, AssetError, Unavailable) as e:
        die(str(e))


if __name__ == "__main__":
    sys.exit(main())

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


def need_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        die(
            "this command trains a model and needs PyTorch.\n"
            "  Install the build for your machine from https://pytorch.org/get-started/locally/\n"
            "  (or `pip install 'stet-cli[train]'` for the default build), then try again."
        )


def resolve_round(args, offset: int = 0):
    """--round accepts 3 (practice round 3), g3 (gym round 3), or a full id like s0-o00012."""
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
            out.line("", out.dim("--- last lines of the editor's output ---"))
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
    out.line("exam", out.bold(f"{e['loss']:.4f}") + (f"   baseline {base['exam']['loss']:.4f}" if base else ""))
    out.line("", "          " + " · ".join(f"{k} {v['loss']:.3f}" for k, v in e["by_source"].items()))
    if base:
        gain = base["exam"]["loss"] - e["loss"]
        m = multiplier(season, result["round"]["kind"], result["scale"], gain)
        text = out.bold(f"{gain:+.4f}") + " nats"
        if m:
            text += f"   →   {out.bold(format_multiplier(m))} data efficiency"
        noise = ((season.raw.get("noise") or {}).get(result["scale"]) or {}).get("gain_sd")
        if noise:
            text += out.dim(f"   (run-to-run noise is about ±{noise:.3f}; smaller differences mean nothing)")
        out.line("gain", text)


# --- commands -------------------------------------------------------------------------------


def cmd_init(args):
    dest = Path(args.path).resolve()
    name = args.name or dest.name.lower().replace("_", "-").replace(" ", "-")
    tpl = resources.files("stet.templates")
    dest.mkdir(parents=True, exist_ok=True)
    wrote = []
    for src, dst in (("stet.toml", "stet.toml"), ("editor.py", "editor.py"), ("AGENTS.md", "AGENTS.md"), ("gitignore", ".gitignore")):
        target = dest / dst
        if target.exists() and not args.force:
            continue
        text = tpl.joinpath(src).read_text(encoding="utf-8")
        if src == "stet.toml":
            text = text.replace("{name}", name)
        target.write_text(text, encoding="utf-8")
        wrote.append(dst)
    out = Out()
    out.line("", f"Editor project ready in {dest}" + (f" ({', '.join(wrote)})" if wrote else " (nothing overwritten)"))
    out.line("", "")
    out.line("", f"  cd {args.path}" if args.path != "." else "")
    out.line("", "  stet run            score it on a practice round")
    out.line("", "  stet submit         enter it")
    out.line("", "")
    out.line("", "AGENTS.md has everything a coding agent needs to work on it.")


def _with_baseline(result: dict, rnd, scale, device, out: Out, skip: bool):
    from .pipeline import baseline

    if skip:
        return None
    base = baseline(rnd, scale, device, progress=out.progress("baseline"), log=lambda m: out.line("", out.dim("baseline " + m)))
    result["baseline"] = {"loss": base["exam"]["loss"], "by_source": {k: v["loss"] for k, v in base["exam"]["by_source"].items()}}
    if result["valid"]:
        result["gain"] = round(base["exam"]["loss"] - result["exam"]["loss"], 5)
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


def cmd_run(args):
    need_torch()
    from .pipeline import run_round
    from .project import ProjectError, load_project
    from .train import pick_device

    out = Out(quiet=args.quiet)
    try:
        project = load_project(args.path)
    except ProjectError as e:
        die(str(e))
    device = pick_device(args.device)
    sandbox = "bwrap" if args.sandbox else "none"
    results = []
    for k in range(max(1, args.rounds)):
        season, scale, rnd = resolve_round(args, offset=k)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        work = project.path / ".stet" / "runs" / f"{stamp}-{rnd.id}-{scale.name}"
        work.mkdir(parents=True, exist_ok=True)
        out.line("", out.bold(f"stet · {project.name} · {rnd.kind} round {rnd.id} · {scale.name} · {device.type}"))
        result = run_round(
            project.path, project.entry, rnd, scale, work, sandbox=sandbox, device=device,
            echo=args.verbose, progress=out.progress("training"), log=lambda m: out.line("", out.dim(m)),
        )  # fmt: skip
        result["editor"] = {"name": project.name}
        base = _with_baseline(result, rnd, scale, device, out, args.no_baseline)
        (work / "result.json").write_text(json.dumps(result, indent=1))
        report(out, result, base, season)
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


def cmd_score(args):
    """Grade an edit script you already have, skipping the editor."""
    need_torch()
    from .pipeline import grade_round
    from .train import pick_device

    out = Out(quiet=args.quiet)
    season, scale, rnd = resolve_round(args)
    device = pick_device(args.device)
    out.line("", out.bold(f"stet score · {args.script} · {rnd.kind} round {rnd.id} · {scale.name} · {device.type}"))
    result = grade_round(rnd, scale, Path(args.script), device=device, progress=out.progress("training"), log=lambda m: out.line("", out.dim(m)))
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
    out.line("baseline", out.bold(f"{base['exam']['loss']:.4f}") + "   " + " · ".join(f"{k} {v['loss']:.3f}" for k, v in base["exam"]["by_source"].items()))
    if args.json:
        json.dump(base, sys.stdout, indent=1)
        sys.stdout.write("\n")


def cmd_fetch(args):
    from . import data

    season, scale, rnd = resolve_round(args)
    corpus, exam = data.materialize(rnd, scale)
    Out().line("ready", f"{rnd.id} · {scale.name}: {len(corpus):,} documents, {corpus.total_bytes / 1e6:.0f} MB at {corpus.path}")


def cmd_doctor(args):
    from . import data
    from .runner import bwrap_available, scope_available
    from .spec import load_season

    out = Out()
    grade_devices = load_season().raw.get("grade_devices", ["cuda"])
    rows: list[tuple[str, str, str]] = []
    rows.append((OK, "stet", f"{__version__} on Python {sys.version.split()[0]} ({sys.platform})"))
    try:
        import torch

        from .train import hardware_info, pick_device

        dev = pick_device()
        hw = hardware_info(dev)
        if dev.type == "cuda":
            rows.append((OK, "gpu", f"{hw['gpu']} · {hw['vram_gb']} GB · {hw['backend']} · torch {torch.__version__}"))
        else:
            rows.append((WARN, "gpu", f"no CUDA or ROCm GPU; training will run on {dev.type} (fine for --scale nano)"))
        if dev.type in grade_devices:
            rows.append((OK, "verify", "this machine can grade for the leaderboard: `stet verify`"))
        else:
            rows.append((WARN, "verify", "official grading needs an NVIDIA or AMD GPU, so every grade uses the same arithmetic"))
    except ImportError:
        rows.append((WARN, "torch", "not installed: `stet run` and `stet verify` need it (https://pytorch.org/get-started/locally/)"))
    ok, why = bwrap_available() if sys.platform == "linux" else (False, "")
    if ok:
        scoped, _ = scope_available()
        rows.append((OK, "sandbox", f"bubblewrap at {why}" + (", with memory and process limits" if scoped else "") + " (`stet run --sandbox`)"))
    elif sys.platform == "linux":
        rows.append((WARN, "sandbox", why + ". Optional: only `stet run --sandbox` uses it."))
    cache = data.cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free / 2**30
    rows.append((OK if free > 10 else WARN, "cache", f"{cache} · {free:.0f} GB free" + ("" if free > 10 else " (a round needs a few GB)")))
    for status, name, text in rows:
        mark = {OK: out.good("✓"), WARN: out.c("33", "!"), BAD: out.bad("✗")}[status]
        out.line("", f" {mark} {name:<8} {text}")
    sys.exit(1 if any(s == BAD for s, _, _ in rows) else 0)


def cmd_agents(args):
    sys.stdout.write(resources.files("stet.templates").joinpath("AGENTS.md").read_text(encoding="utf-8"))


# --- entry ----------------------------------------------------------------------------------


def add_round_args(p, default_round="0"):
    p.add_argument("--round", default=default_round, help="3 = practice round 3, g3 = gym round 3, or a round id like s0-o00012 (default: 0)")
    p.add_argument("--scale", default=None, help="nano, micro or standard (default: nano)")
    p.add_argument("--season", default="s0", help=argparse.SUPPRESS)
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

    p = sub.add_parser("init", help="create an editor project")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--name")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("run", help="run your editor on a round and score it")
    p.add_argument("path", nargs="?", default=".")
    add_round_args(p)
    p.add_argument("--rounds", type=int, default=1, help="run this many consecutive rounds and report the mean gain with its standard error")
    p.add_argument("--sandbox", action="store_true", help="run the editor in the same sandbox official runs use (Linux)")
    p.add_argument("--no-baseline", action="store_true", help="skip the baseline comparison")
    p.add_argument("-v", "--verbose", action="store_true", help="show the editor's output as it runs")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("score", help="grade an edit script directly, without running an editor")
    p.add_argument("script", help="path to a script.jsonl")
    add_round_args(p)
    p.add_argument("--no-baseline", action="store_true", help="skip the baseline comparison")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("baseline", help="score the unedited corpus for a round")
    add_round_args(p)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("fetch", help="download and build a round's corpus ahead of time")
    add_round_args(p)
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("doctor", help="check this machine")
    p.set_defaults(fn=cmd_doctor)

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
    try:
        return args.fn(args) or 0
    except KeyboardInterrupt:
        die("interrupted", 130)
    except OSError as e:
        die(str(e))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""The scale pilot: does a curator's place hold as the student grows, and which exam agrees
with what is already known about data at scale?

Runs curators (project directories, or settled entries by name) on one official round at
several scales, on this machine, and has the organisers' machines train every student
(stet verify takes them as shadow tasks). Each student sits the season's exam and the next
exam. The reference curators (heuristic, dclm, fineweb-edu) rebuild published recipes whose
order at 1B-7B is known: dclm and fineweb-edu above heuristic, heuristic above raw crawl.

    python tools/scale_pilot.py --round s0-o00012 --scales micro,standard,large \\
        --curators examples/identity,examples/english,examples/tidy,examples/heuristic,examples/dclm,examples/fineweb-edu,stet/compass@v1,stet/compass@v2 \\
        --items DIR/items.jsonl --shares DIR/languages.json --out ~/stet-pilot

A scale's students are posted as one batch. The large scale's take hours each, so they go
only to machines set aside for them (stet verify with STET_SHADOW_MAX_MINUTES=600).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stet import data
from stet.client import fetch_round, need_key, request
from stet.newexam.aggregate import gains, round_score
from stet.newexam.farm import farm_round, scale_of
from stet.newexam.items import Item
from stet.spec import load_season

# About ten times standard's training compute: 54M parameters, 640M tokens, on a pool four
# times standard's, so the pool is five times the budget, as at the season's scales.
SPECS = {
    "large": {"corpus_units": 32, "train_tokens": 640_000_000, "n_layer": 10, "n_head": 10, "d_model": 640, "ctx": 512,
              "batch_size": 32, "lr": 0.0008, "exam_items_per_source": 800, "edit_seconds": 4800},
}
LEASES = {"large": 5 * 3600}
ENTRIES: dict[str, str] = {}  # the round's entries, owner/name@vN -> editor id
# The order large models trained on these curators' data came out in, from the literature
# (DCLM 2024; FineWeb 2024): each group better than the next.
KNOWN = [{"dclm", "fineweb-edu"}, {"heuristic"}, {"identity"}]


def curator_dir(ref: str, season, work: Path) -> Path:
    """A project directory: as given, or an entry's bundle fetched and unpacked."""
    from stet.project import unpack

    if Path(ref).is_dir():
        return Path(ref)
    dest = work / "bundles" / ref.replace("/", "-").replace("@", "-")
    if not dest.exists():
        editor = ENTRIES.get(ref)
        if not editor:
            raise ValueError(f"{ref} is not an entry of the round (owner/name@vN)")
        blob = request("GET", f"/api/v1/editors/{editor}/bundle", key=need_key(), raw=True, timeout=600, limit=256 << 20)
        unpack(blob, dest)
    return dest


def run_curator(project_dir: Path, rnd, scale, work: Path, sandbox: str, log, label: str | None = None) -> Path | None:
    from stet.pipeline import edit_round
    from stet.project import load_project

    project = load_project(project_dir)
    label = label or project.name
    out = work / "scripts" / f"{label}.jsonl"
    if out.exists():
        return out
    run_dir = work / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    edit = edit_round(project_dir, project.entry, rnd, scale, run_dir, sandbox=sandbox, log=lambda m: None, assets=project.assets,
                      time_limit=float(scale.edit_seconds) * 4)
    if edit.script is None:
        log(f"  {label}: no script ({edit.status}: {edit.reason})")
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(Path(edit.script).read_bytes())
    log(f"  {label}: script in {time.time() - t0:.0f}s, {out.stat().st_size / 1e6:.1f} MB")
    return out


def upload(script: Path, key) -> str:
    import zstandard

    raw = script.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    request("PUT", f"/api/v1/admin/shadow/scripts/{sha}", zstandard.ZstdCompressor(level=10).compress(raw), key=key, timeout=900)
    return sha


def report(results: dict[str, dict], out: Path) -> str:
    lines = []
    names = sorted({r["name"] for s in results.values() for r in s["entries"]})
    lines.append("next exam score (bits per byte over the unedited corpus, averaged over its trainings), by scale:")
    lines.append(f"  {'curator':16s}" + "".join(f"{s:>14s}" for s in results))
    for n in names:
        row = {s: next((r for r in res["entries"] if r["name"] == n), None) for s, res in results.items()}
        lines.append(f"  {n:16s}" + "".join(f"{(r['next_score'] if r and r['valid'] else float('nan')):+14.4f}" for r in row.values()))
    lines.append("\nseason exam gain (nats over the unedited corpus), by scale:")
    for n in names:
        row = {s: next((r for r in res["entries"] if r["name"] == n), None) for s, res in results.items()}
        lines.append(f"  {n:16s}" + "".join(f"{(r['season_gain'] if r and r['valid'] else float('nan')):+14.4f}" for r in row.values()))
    for scale, res in results.items():
        rows = {r["name"]: r for r in res["entries"] if r["valid"]}
        for exam, key in (("next exam", "next_score"), ("season exam", "season_gain")):
            ok, total = 0, 0
            for hi, lo in zip(KNOWN, KNOWN[1:], strict=False):
                for a in hi & rows.keys():
                    for b in lo & rows.keys():
                        total += 1
                        ok += rows[a][key] > rows[b][key]
            if total:
                lines.append(f"{scale}: the {exam} agrees with the published order in {ok} of {total} pairs")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    return text


def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
    names = sorted(set(a) & set(b))
    if len(names) < 3:
        return float("nan")

    def ranks(d):
        order = sorted(names, key=lambda n: d[n])
        return {n: i for i, n in enumerate(order)}

    ra, rb = ranks(a), ranks(b)
    n = len(names)
    return 1 - 6 * sum((ra[x] - rb[x]) ** 2 for x in names) / (n * (n * n - 1))


def agreement(results: dict[str, dict]) -> str:
    """How far each exam's order at one scale holds at the next."""
    lines = ["\nrank agreement between scales (Spearman):"]
    scales = list(results)
    for key, exam in (("next_score", "next exam"), ("season_gain", "season exam")):
        vals = {s: {r["name"]: r[key] for r in results[s]["entries"] if r["valid"] and r["name"] != "identity"} for s in scales}
        for a, b in zip(scales, scales[1:], strict=False):
            lines.append(f"  {exam}: {a} vs {b}: {_spearman(vals[a], vals[b]):+.2f}")
    for s in scales:
        vals = {r["name"]: r for r in results[s]["entries"] if r["valid"] and r["name"] != "identity"}
        rho = _spearman({n: r["next_score"] for n, r in vals.items()}, {n: r["season_gain"] for n, r in vals.items()})
        lines.append(f"  {s}: next exam vs season exam: {rho:+.2f}")
    return "\n".join(lines)


def english_slice(out: Path, scale: str) -> dict[str, float]:
    """Each curator's next-exam score on English items only: what the published order speaks to."""
    d = out / scale
    items = [Item.from_dict(json.loads(line)) for line in (d / "items.jsonl").read_text().splitlines()]
    keep = [i for i, it in enumerate(items) if it.lang == "en"]
    from stet.newexam.items import digest

    tag = digest(items)[:12]
    metas = {}
    for m in d.glob("????????????????.json"):
        meta = json.loads(m.read_text())
        s = d / f"{m.stem}-{tag}.json"
        if meta.get("valid") and s.exists():
            metas[meta["name"]] = json.loads(s.read_text())
    base = metas.pop("(unedited)")
    sub = [items[i] for i in keep]
    return {n: round_score(sub, [g for i, g in enumerate(gains(sc, base)) if i in set(keep)])["score"] for n, sc in metas.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--scales", default="micro,standard,large")
    ap.add_argument("--curators", required=True, type=lambda s: [c for c in s.split(",") if c])
    ap.add_argument("--items", required=True, help="the next exam's items for the round (items.jsonl of its shadow run)")
    ap.add_argument("--shares", help="the corpus's languages (languages.json of its shadow run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sandbox", default="none", help="none for the organisers' own curators; bwrap for anyone else's")
    ap.add_argument("--edit-only", action="store_true", help="run the curators and keep their scripts; train nothing")
    ap.add_argument("--report-only", action="store_true", help="report on the scales already done, from their summaries")
    args = ap.parse_args()
    if args.report_only:
        out = Path(args.out).expanduser()
        results = {name: json.loads((out / name / "summary.json").read_text()) for name in args.scales.split(",")
                   if (out / name / "summary.json").exists()}
        print(report(results, out))
        print(agreement(results))
        for scale in results:
            eng = english_slice(out, scale)
            print(f"{scale}, English items only: " + ", ".join(f"{n} {v:+.4f}" for n, v in sorted(eng.items(), key=lambda x: -x[1])))
        return
    season = load_season("s0")
    from stet.spec import parse_round_id

    rnd = fetch_round(season, parse_round_id(args.round)[2])
    for e in request("GET", f"/api/v1/rounds/{rnd.id}").get("entries") or []:
        ENTRIES[f"{e['owner']}/{e['name']}@v{e['version']}"] = e["editor"]
    items = [Item.from_dict(json.loads(line)) for line in Path(args.items).read_text().splitlines()]
    shares = json.loads(Path(args.shares).read_text()) if args.shares else None
    key = need_key()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def log(m):
        with lock:
            print(time.strftime("%H:%M:%S"), m, flush=True)

    results: dict[str, dict] = {}
    threads = []
    for name in args.scales.split(","):
        # Curators run here one scale at a time; the training of each scale's students waits on
        # the machines, so a scale's batch is followed in a thread while the next one is edited.
        holder: dict = {}
        res = one_scale_posted(name, args, season, rnd, items, shares, key, log, holder)
        if res is not None:
            threads.append((name, res, holder))
    for name, t, holder in threads:
        t.join()
        if "summary" in holder:
            results[name] = holder["summary"]
    if results:
        print(report(results, out))
        for scale in results:
            try:
                eng = english_slice(out, scale)
                print(f"{scale}, English items only: " + ", ".join(f"{n} {v:+.4f}" for n, v in sorted(eng.items(), key=lambda x: -x[1])))
            except Exception as e:
                print(f"{scale}: no English slice ({e})")


def one_scale_posted(name, args, season, rnd, items, shares, key, log, holder):
    """Run a scale's curators now; follow its batch in a thread."""
    scale = scale_of(season, f"pilot-{name}" if name in SPECS else name, SPECS.get(name))
    work = Path(args.out).expanduser() / name
    work.mkdir(parents=True, exist_ok=True)
    corpus = data.corpus(rnd, scale)
    log(f"{name}: {len(corpus):,} documents, {corpus.total_bytes / 1e9:.1f} GB; running {len(args.curators)} curators")
    shas = {}
    for ref in args.curators:
        try:
            label = None if Path(ref).is_dir() else ref.split("/")[-1].replace("@v", "-v")  # stet/compass@v2 -> compass-v2
            script = run_curator(curator_dir(ref, season, Path(args.out).expanduser()), rnd, scale, work, args.sandbox, log, label=label)
        except Exception as e:  # one curator's trouble is not the pilot's
            log(f"  {ref}: {type(e).__name__}: {e}")
            continue
        if script is not None and not args.edit_only:
            shas[script.stem] = upload(script, key)
    if args.edit_only:
        return None

    def follow():
        try:
            holder["summary"] = farm_round(rnd, scale, shas, work, items, len(corpus), key, shares=shares, log=lambda m: log(f"{name}: {m}"),
                                           spec=SPECS.get(name), lease=LEASES.get(name, 0), patience=48 * 3600)
        except Exception as e:
            log(f"{name}: {type(e).__name__}: {e}")

    t = threading.Thread(target=follow)
    t.start()
    return t


if __name__ == "__main__":
    main()

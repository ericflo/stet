#!/usr/bin/env python3
"""What a shadow run says, and how sure it is: per family, every curator's gain over the unedited
corpus, and for each pair of curators the difference with a bootstrap standard error over items.

    python tools/next_exam_report.py ~/.cache/stet/shadow/s0-o00012/micro
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stet.newexam.aggregate import gains, round_score
from stet.newexam.items import Item, digest


def load(d: Path):
    items = [Item.from_dict(json.loads(line)) for line in (d / "items.jsonl").read_text().splitlines()]
    tag = digest(items)[:12]
    students, when = {}, {}
    for meta_path in d.glob("????????????????.json"):  # a curator run again leaves its older file: the newest is its
        meta = json.loads(meta_path.read_text())
        scores = d / f"{meta_path.stem}-{tag}.json"
        if meta.get("valid") and scores.exists() and meta_path.stat().st_mtime >= when.get(meta["name"], 0):
            students[meta["name"]], when[meta["name"]] = (meta, json.loads(scores.read_text())), meta_path.stat().st_mtime
    shares = json.loads((d / "languages.json").read_text()) if (d / "languages.json").exists() else None
    return items, students, shares


def boot(items, a, b, shares, n=300, seed=0):
    """Bootstrap SE of the round-score difference between two curators, resampling items."""
    rng = random.Random(seed)
    k = len(items)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        sub = [items[i] for i in idx]
        diffs.append(round_score(sub, [a[i] for i in idx], shares=shares)["score"]
                     - round_score(sub, [b[i] for i in idx], shares=shares)["score"])
    return st.pstdev(diffs)


def main():
    d = Path(sys.argv[1]).expanduser()
    items, students, shares = load(d)
    base = students.pop("(unedited)", None)
    if base is None:
        sys.exit("no unedited baseline has sat this exam yet")
    g = {n: gains(students[n][1], base[1]) for n in students}
    r = {n: round_score(items, g[n], shares=shares) for n in students}
    names = sorted(students, key=lambda n: -r[n]["score"])
    print(f"{len(items)} items, {len({i.corner for i in items})} corners; {len(names)} curators beside the unedited corpus")
    print("fresh items weighed by " + ("the corpus's measured languages" if shares else "the published table of the web's languages") + "\n")
    print(f"{'curator':32s} {'next':>8s} {'gen':>8s} {'fresh':>8s} {'season gain':>12s}")
    for n in names:
        gs = r[n]["generated"]["score"] if r[n]["generated"] else float("nan")
        fs = r[n]["fresh"]["score"] if r[n]["fresh"] else float("nan")
        season = base[0]["exam_loss"] - students[n][0]["exam_loss"]
        print(f"{n:32s} {r[n]['score']:+8.4f} {gs:+8.4f} {fs:+8.4f} {season:+12.4f}")
    pillars = sorted({p for n in names for p in r[n]["pillars"]})
    print(f"\nby pillar (each counts the same in the score){'':4s}" + "".join(f"{n.split('/')[-1][:12]:>14s}" for n in names))
    for p in pillars:
        print(f"  {p:34s}" + "".join(f"{r[n]['pillars'][p]['score']:+14.4f}" for n in names))
    fams = sorted({it.family for it in items})
    print(f"\nby family (mean gain, bits per byte){'':6s}" + "".join(f"{n.split('/')[-1][:12]:>14s}" for n in names))
    for f in fams:
        idx = [i for i, it in enumerate(items) if it.family == f]
        print(f"  {f:34s}" + "".join(f"{st.mean(g[n][i] for i in idx):+14.4f}" for n in names))
    print("\npairs: difference in the next exam's score, with its bootstrap standard error over items")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = r[a]["score"] - r[b]["score"]
            se = boot(items, g[a], g[b], shares)
            print(f"  {a:30s} − {b:30s} {diff:+.4f} ± {se:.4f}  ({abs(diff) / se if se else float('inf'):.1f} SE)")
    langs = (r[names[0]]["fresh"] or {}).get("languages", {}) if names else {}
    if langs:
        print("\nfresh items by language, heaviest first (weight in the fresh score; each curator's balanced score)")
        for lang, v in sorted(langs.items(), key=lambda x: -x[1]["weight"])[:15]:
            print(f"  {lang:6s} {v['weight']:6.1%} {v['items']:4d} items" + "".join(f"{r[n]['fresh']['languages'][lang]['score']:+14.4f}" for n in names))


if __name__ == "__main__":
    main()

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
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stet.newexam.aggregate import gains, round_score
from stet.newexam.items import Item, digest


def load(d: Path):
    items = [Item.from_dict(json.loads(line)) for line in (d / "items.jsonl").read_text().splitlines()]
    tag = digest(items)[:12]
    students = {}
    for meta_path in d.glob("????????????????.json"):
        meta = json.loads(meta_path.read_text())
        scores = d / f"{meta_path.stem}-{tag}.json"
        if meta.get("valid") and scores.exists():
            students[meta["name"]] = (meta, json.loads(scores.read_text()))
    return items, students


def boot(items, a, b, n=300, seed=0):
    """Bootstrap SE of the round-score difference between two curators, resampling items."""
    rng = random.Random(seed)
    k = len(items)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        sub = [items[i] for i in idx]
        diffs.append(round_score(sub, [a[i] for i in idx])["score"] - round_score(sub, [b[i] for i in idx])["score"])
    return st.pstdev(diffs)


def main():
    d = Path(sys.argv[1]).expanduser()
    items, students = load(d)
    base = students.pop("(unedited)", None)
    if base is None:
        sys.exit("no unedited baseline has sat this exam yet")
    names = sorted(students, key=lambda n: -round_score(items, gains(students[n][1], base[1]))["score"])
    g = {n: gains(students[n][1], base[1]) for n in names}
    print(f"{len(items)} items, {len({i.corner for i in items})} corners; {len(names)} curators beside the unedited corpus\n")
    print(f"{'curator':32s} {'next':>8s} {'gen':>8s} {'fresh':>8s} {'season gain':>12s}")
    for n in names:
        r = round_score(items, g[n])
        gen = [(it, x) for it, x in zip(items, g[n], strict=True) if it.source.startswith("gen:")]
        fr = [(it, x) for it, x in zip(items, g[n], strict=True) if not it.source.startswith("gen:")]
        gs = round_score([p[0] for p in gen], [p[1] for p in gen])["score"] if gen else float("nan")
        fs = round_score([p[0] for p in fr], [p[1] for p in fr])["score"] if fr else float("nan")
        season = base[0]["exam_loss"] - students[n][0]["exam_loss"]
        print(f"{n:32s} {r['score']:+8.4f} {gs:+8.4f} {fs:+8.4f} {season:+12.4f}")
    fams = sorted({it.family for it in items})
    print(f"\nby family (mean gain, bits per byte){'':6s}" + "".join(f"{n.split('/')[-1][:12]:>14s}" for n in names))
    for f in fams:
        idx = [i for i, it in enumerate(items) if it.family == f]
        print(f"  {f:34s}" + "".join(f"{st.mean(g[n][i] for i in idx):+14.4f}" for n in names))
    print("\npairs: difference in the next exam's score, with its bootstrap standard error over items")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = round_score(items, g[a])["score"] - round_score(items, g[b])["score"]
            se = boot(items, g[a], g[b])
            print(f"  {a:30s} − {b:30s} {diff:+.4f} ± {se:.4f}  ({abs(diff) / se if se else float('inf'):.1f} SE)")
    by_lang = defaultdict(list)
    for i, it in enumerate(items):
        if not it.source.startswith("gen:"):
            by_lang[it.lang].append(i)
    if by_lang:
        print("\nfresh items by language (top 10): mean gain")
        for lang, idx in sorted(by_lang.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {lang:6s} {len(idx):4d} items" + "".join(f"{st.mean(g[n][i] for i in idx):+14.4f}" for n in names))


if __name__ == "__main__":
    main()

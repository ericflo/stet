"""Keeping the round's corpus out of its exam.

The exam's fresh text was published in the day before the draw, which a crawl gathered before
that day cannot hold. But text can be copied: an addition to an article may quote a page
the crawl has. So an item whose scored text has a sentence opening that appears anywhere in the
round's corpus is dropped from the exam.

Openings are normalised (lowercase, runs of whitespace collapsed) and cut to the season's key
length, keeping only those with enough letters to be distinctive (`decontam` in the season).

There is no other way in. A script is copies of the round's corpus and holds no text of its
own (stet.script), so nothing an author has seen published, or means to publish, can be
written into one.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..data import normalise, sentence_starts, window_hashes
from .items import Item


def scored_text(it: Item) -> str:
    return {"passage": it.text, "answer": it.cont}.get(it.kind, "")


def seen(items: list[Item], corpus, season, log=None) -> set[int]:
    """Indices of the items whose scored text appears in the corpus."""
    cfg = season.raw["decontam"]
    k, need = int(cfg["key_bytes"]), int(cfg["min_letters"])
    owners: dict[int, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        for key in sentence_starts(scored_text(it), k, need):
            owners[int(window_hashes(key, k)[0])].append(i)
    if not owners:
        return set()
    keys = np.array(sorted(owners), dtype=np.uint64)
    hit: set[int] = set()
    for d in range(len(corpus)):
        h = window_hashes(normalise(corpus.text(d)), k)
        if not len(h):
            continue
        pos = np.searchsorted(keys, h)
        pos[pos == len(keys)] = 0
        for v in np.unique(h[keys[pos] == h]):
            hit.update(owners[int(v)])
        if log and d and d % 50_000 == 0:
            log(f"  decontamination: {d:,} of {len(corpus):,} documents, {len(hit)} items seen so far")
    return hit

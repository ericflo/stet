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


def scored_texts(it: Item) -> list[str]:
    """Every independent text span whose familiarity in the training corpus would bias this item's score."""
    from .aggregate import REASONING_KINDS

    if it.kind == "context":
        return [t for t in (it.ctx, it.cont, *it.wrongs) if t]
    if it.kind == "answer" and it.source.split(":", 1)[0] in REASONING_KINDS:
        # Fresh harvest-derived reasoning items embed an excerpt in `ctx`; drop the item if
        # either the source excerpt or the response appears in the round's training corpus.
        return [t for t in (it.ctx, it.cont) if t]
    t = {"passage": it.text, "answer": it.cont}.get(it.kind, "")
    return [t] if t else []


def scored_text(it: Item) -> str:
    return "\n".join(scored_texts(it))


def seen(items: list[Item], corpus, season, log=None) -> set[int]:
    """Indices of the items whose scored text appears in the corpus."""
    cfg = season.raw["decontam"]
    k, need = int(cfg["key_bytes"]), int(cfg["min_letters"])
    owners: dict[int, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        for text in scored_texts(it):
            for key in sentence_starts(text, k, need):
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

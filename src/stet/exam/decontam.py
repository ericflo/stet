"""Keeping the round's corpus out of its exam.

The exam is made after the corpus, from text published after the draw, which a crawl made
before the draw cannot hold. But text can be copied: an addition to an article may quote a page
the crawl has. So an item whose scored text has a sentence opening that appears anywhere in the
round's corpus is dropped from the exam.

Openings are normalised (lowercase, runs of whitespace collapsed) and cut to the season's key
length, keeping only those with enough letters to be distinctive (`decontam` in the season).

The other way in is the curator's own text. An author could put a passage they have seen
published into a curator, or write one in and publish it afterwards, hoping a harvest takes
it. So a script whose own text holds a sentence opening of a fresh item is refused
(`written`): the grader has both in hand.
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


def written(items: list[Item], windows: list[bytes], season) -> list[int]:
    """Indices of the fresh items a sentence opening of which stands in the curator's own text:
    `windows` are the pieces of ink as the student reads them (script.Stats.ink_windows)."""
    cfg = season.raw["decontam"]
    k, need = int(cfg["key_bytes"]), int(cfg["min_letters"])
    owners: dict[int, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        if it.source.startswith("gen:"):
            continue  # made from the seed after the curator ran: nothing of it can have been written in
        for key in sentence_starts(scored_text(it), k, need):
            owners[int(window_hashes(key, k)[0])].append(i)
    if not owners or not windows:
        return []
    keys = np.array(sorted(owners), dtype=np.uint64)
    hit: set[int] = set()
    for w in windows:
        h = window_hashes(normalise(w.decode("utf-8", errors="replace")), k)
        if not len(h):
            continue
        pos = np.searchsorted(keys, h)
        pos[pos == len(keys)] = 0
        for v in np.unique(h[keys[pos] == h]):
            hit.update(owners[int(v)])
    return sorted(hit)

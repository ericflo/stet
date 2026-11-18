"""From per-item scores to a round's score.

A curator's gain on an item is its score less the unedited baseline's, on the same item, from
the same initial weights. Gains are put into common units (doublings of raw data, once a
season's slopes are measured), clipped to a few robust widths within their stratum (a format
and a kind of task), averaged by a trimmed mean within each stratum, and the strata are
averaged with equal weight. A curator that gains hugely on a few items and loses on the rest
gets little; one that gains a little everywhere wins.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict

from .items import Item

CLIP = 4.0  # robust widths
TRIM = 0.1  # from each end, within a stratum


def gains(scores: list[float], baseline: list[float]) -> list[float]:
    if len(scores) != len(baseline):
        raise ValueError("a curator and its baseline must sit the same items")
    return [s - b for s, b in zip(scores, baseline, strict=True)]


def stratum_key(it: Item) -> str:
    return f"{it.fmt}/{it.kind}"


def _clip(vals: list[float], k: float) -> list[float]:
    if len(vals) < 3:
        return list(vals)
    med = st.median(vals)
    mad = st.median(abs(v - med) for v in vals)
    width = 1.4826 * mad
    if width <= 0:
        return list(vals)
    lo, hi = med - k * width, med + k * width
    return [min(hi, max(lo, v)) for v in vals]


def trimmed_mean(vals: list[float], trim: float) -> float:
    vals = sorted(vals)
    cut = int(len(vals) * trim)
    kept = vals[cut : len(vals) - cut] if len(vals) - 2 * cut >= 1 else vals
    return sum(kept) / len(kept)


def round_score(items: list[Item], item_gains: list[float], slopes: dict[str, float] | None = None,
                clip: float = CLIP, trim: float = TRIM) -> dict:
    """The round's score and what it is made of."""
    if len(items) != len(item_gains):
        raise ValueError("one gain per item")
    slopes = slopes or {}
    by: dict[str, list[float]] = defaultdict(list)
    fam: dict[str, list[float]] = defaultdict(list)
    for it, g in zip(items, item_gains, strict=True):
        key = stratum_key(it)
        s = slopes.get(key)
        g = g / s if s else g
        by[key].append(g)
        fam[it.family].append(g)
    strata = {k: trimmed_mean(_clip(v, clip), trim) for k, v in sorted(by.items())}
    return {
        "score": sum(strata.values()) / len(strata) if strata else 0.0,
        "units": "doublings" if slopes else "bits per byte",
        "strata": {k: round(v, 5) for k, v in strata.items()},
        "families": {k: round(sum(v) / len(v), 5) for k, v in sorted(fam.items())},
        "items": len(items),
        "corners": len({it.corner for it in items}),
    }

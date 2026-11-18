"""From per-item scores to a round's score.

A curator's gain on an item is its score less the unedited baseline's, on the same item, from
the same initial weights. Gains are put into common units (doublings of raw data, once a
season's slopes are measured), clipped to a few robust widths within their stratum (a format
and a kind of task), averaged by a trimmed mean within each stratum, and the strata are
averaged with equal weight; fresh text is first shared among languages as the round's corpus
is, and balanced by format and kind inside each. A curator that gains hugely on a few items
and loses on the rest gets little; one that gains a little everywhere wins.
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


def language_weights(langs, shares: dict[str, float] | None = None) -> dict[str, float]:
    """Each language's weight among the fresh items: its share of the round's own corpus (the
    published table when that was not measured), over the languages the exam has."""
    from .fresh.select import OTHER, SHARES

    shares = shares or SHARES
    w = {lang: max(float(shares.get(lang, OTHER)), 0.0) or OTHER for lang in sorted(set(langs))}
    total = sum(w.values())
    return {lang: v / total for lang, v in w.items()}


def _balanced(pairs: list[tuple[Item, float]], clip: float, trim: float) -> tuple[float, dict[str, float]]:
    """Formats and kinds of task with equal weight, each a trimmed mean of its clipped gains."""
    by: dict[str, list[float]] = defaultdict(list)
    for it, g in pairs:
        by[stratum_key(it)].append(g)
    strata = {k: trimmed_mean(_clip(v, clip), trim) for k, v in sorted(by.items())}
    return sum(strata.values()) / len(strata), strata


def is_generated(it: Item) -> bool:
    return it.source.startswith("gen:")


def round_score(items: list[Item], item_gains: list[float], slopes: dict[str, float] | None = None,
                clip: float = CLIP, trim: float = TRIM, shares: dict[str, float] | None = None) -> dict:
    """The round's score and what it is made of.

    Generated items: formats and kinds of task with equal weight. Fresh items: each language
    with its share of the round's corpus, and formats and kinds with equal weight inside each
    language, because what a source offers depends on its language (StackExchange is mostly
    English), and balancing formats first would let a few English posts outweigh everything
    else. The two parts count by how many items each has."""
    if len(items) != len(item_gains):
        raise ValueError("one gain per item")
    slopes = slopes or {}
    gen: list[tuple[Item, float]] = []
    fresh: dict[str, list[tuple[Item, float]]] = defaultdict(list)
    fam: dict[str, list[float]] = defaultdict(list)
    for it, g in zip(items, item_gains, strict=True):
        s = slopes.get(stratum_key(it))
        g = g / s if s else g
        (gen if is_generated(it) else fresh[it.lang]).append((it, g))
        fam[it.family].append(g)
    out: dict = {"units": "doublings" if slopes else "bits per byte", "items": len(items),
                 "corners": len({it.corner for it in items}), "strata": {}, "generated": None, "fresh": None}
    parts = []
    if gen:
        score, strata = _balanced(gen, clip, trim)
        out["generated"] = {"score": score, "items": len(gen)}
        out["strata"] = {k: round(v, 5) for k, v in strata.items()}
        parts.append((score, len(gen)))
    if fresh:
        weights = language_weights(fresh, shares)
        langs = {lang: {"weight": round(weights[lang], 5), "score": round(_balanced(pairs, clip, trim)[0], 5), "items": len(pairs)}
                 for lang, pairs in sorted(fresh.items())}
        score = sum(weights[lang] * _balanced(pairs, clip, trim)[0] for lang, pairs in fresh.items())
        n = sum(len(p) for p in fresh.values())
        out["fresh"] = {"score": score, "items": n, "languages": langs}
        parts.append((score, n))
    out["score"] = sum(s * n for s, n in parts) / sum(n for _, n in parts) if parts else 0.0
    out["families"] = {k: round(sum(v) / len(v), 5) for k, v in sorted(fam.items())}
    return out

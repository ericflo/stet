"""From per-item scores to a round's score.

A curator's gain on an item is its score less the unedited baseline's, on the same item, from
the same initial weights. Gains are put into common units (doublings of raw data, once a
season's slopes are measured) and clipped to a few robust widths.

The score is the mean of the pillars, what pretrained models are for, each counting the same:
knowledge, reading in context, reasoning, code, agents and tools, writing, and conversation.

Inside a pillar:
- Each sub-skill counts the same. For generated items that is a family of generators; for
  fresh text, a kind of writing.
- Inside a sub-skill, each language counts by its share of the round's own corpus (programming
  languages count together as one).
- Inside a language, formats and kinds of task count the same, each a trimmed mean.

So no generator, source or language weighs more than its declared share. A curator that gains
hugely in one place and loses everywhere else gets little; one that gains a little everywhere
wins. See docs/research/exam-first-principles.md for why these are the weights.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict

from .items import Item

CLIP = 4.0  # robust widths
TRIM = 0.1  # from each end, within a stratum

PILLARS = {
    "knowledge": "facts the web knows, in many languages",
    "reading": "reading and learning from the context",
    "reasoning": "mathematics and logic",
    "code": "programs and structured data",
    "agents": "agents at work with tools",
    "writing": "human writing of every kind, in every language",
    "conversation": "replies in discussions",
}
FAMILY_PILLAR = {
    "knowledge": "knowledge",
    "context": "reading",
    "arith": "reasoning", "word": "reasoning", "units": "reasoning", "algebra": "reasoning", "logic": "reasoning",
    "algo": "reasoning", "multi": "reasoning",
    "prog": "code", "transform": "code", "struct": "code",
    "agent": "agents",
    "grammar": "writing",
}
# Fresh text: the kind of writing each source gives. Discussion is the conversation pillar; every
# other kind is writing. A source not listed is a kind of its own.
SOURCE_KIND = {
    "wikipedia": "reference", "wikipedia-added": "reference", "wikivoyage": "reference",
    "wikinews": "news", "ccnews": "news",
    "wikibooks": "instructional", "wikiversity": "instructional",
    "github": "technical",
    "bluesky": "social", "mastodon": "social",
    "wikipedia-talk": "discussion", "hackernews": "discussion", "lemmy": "discussion", "github-comment": "discussion",
    "bluesky-reply": "discussion", "mastodon-reply": "discussion", "stackexchange": "discussion",
}
CODE = "code"  # every programming language, as one language


def gains(scores: list[float], baseline: list[float]) -> list[float]:
    if len(scores) != len(baseline):
        raise ValueError("a curator and its baseline must sit the same items")
    return [s - b for s, b in zip(scores, baseline, strict=True)]


def stratum_key(it: Item) -> str:
    return f"{it.fmt}/{it.kind}"


def is_generated(it: Item) -> bool:
    return it.source.startswith("gen:")


def kind_of(it: Item) -> str:
    """A fresh item's kind of writing."""
    src = it.source.split(":", 1)[0]
    return SOURCE_KIND.get(src, src)


def pillar_of(it: Item) -> tuple[str, str]:
    """(pillar, sub-skill) of an item."""
    if is_generated(it):
        return FAMILY_PILLAR[it.family], it.family
    kind = kind_of(it)
    return ("conversation" if kind == "discussion" else "writing"), kind


def lang_group(it: Item) -> str:
    return CODE if it.lang.startswith("code:") else it.lang


def item_table(items: list[Item]) -> list[dict]:
    """All the score needs to know of each item: its pillar, its part (a family, or a kind of
    writing), its language (programming languages as one) and its stratum. The server scores a
    round from this table and the gains alone (server/aggregate.go)."""
    rows = []
    for it in items:
        pillar, part = pillar_of(it)
        rows.append({"pillar": pillar, "part": part, "lang": lang_group(it), "stratum": stratum_key(it)})
    return rows


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
    """Each language's weight among some items: its share of the round's own corpus (the published
    table when that was not measured), over the languages those items have. Programming languages
    together, beside natural languages, take half."""
    from .fresh.select import OTHER, SHARES

    shares = shares or SHARES
    natural = sorted(set(langs) - {CODE})
    w = {lang: max(float(shares.get(lang, OTHER)), 0.0) or OTHER for lang in natural}
    total = sum(w.values())
    out = {lang: v / total for lang, v in w.items()}
    if CODE in set(langs):
        if not out:
            return {CODE: 1.0}
        out = {lang: v / 2 for lang, v in out.items()}
        out[CODE] = 0.5
    return out


Pair = tuple[dict, float]  # an item's row of item_table, and its gain


def _balanced(pairs: list[Pair], clip: float, trim: float) -> tuple[float, dict[str, float]]:
    """Formats and kinds of task with equal weight, each a trimmed mean of its clipped gains."""
    by: dict[str, list[float]] = defaultdict(list)
    for row, g in pairs:
        by[row["stratum"]].append(g)
    strata = {k: trimmed_mean(_clip(v, clip), trim) for k, v in sorted(by.items())}
    return sum(strata.values()) / len(strata), strata


def _by_language(pairs: list[Pair], clip: float, trim: float, shares) -> float:
    langs: dict[str, list[Pair]] = defaultdict(list)
    for row, g in pairs:
        langs[row["lang"]].append((row, g))
    w = language_weights(langs, shares)
    return sum(w[lang] * _balanced(p, clip, trim)[0] for lang, p in langs.items())


def _pillars(pairs: list[Pair], clip: float, trim: float, shares) -> dict[str, dict]:
    tree: dict[str, dict[str, list[Pair]]] = defaultdict(lambda: defaultdict(list))
    for row, g in pairs:
        tree[row["pillar"]][row["part"]].append((row, g))
    out = {}
    for p in sorted(tree):
        subs = {sub: _by_language(v, clip, trim, shares) for sub, v in sorted(tree[p].items())}
        out[p] = {"score": sum(subs.values()) / len(subs), "items": sum(len(v) for v in tree[p].values()),
                  "parts": {k: round(v, 5) for k, v in subs.items()}}
    return out


def _mean_of(pillars: dict[str, dict]) -> float:
    return sum(v["score"] for v in pillars.values()) / len(pillars) if pillars else 0.0


def score_from_table(table: list[dict], item_gains: list[float], shares: dict[str, float] | None,
                     clip: float = CLIP, trim: float = TRIM) -> dict:
    """The round's score and each pillar's, unrounded, from item_table and the gains alone: what
    round_score gives (without slopes), by the same code, and what the server computes. The server
    is always given the round's shares (and refuses none); here None or {} means the published SHARES."""
    if len(table) != len(item_gains):
        raise ValueError("one gain per item")
    pillars = _pillars(list(zip(table, item_gains, strict=True)), clip, trim, shares)
    return {"score": _mean_of(pillars), "pillars": {p: v["score"] for p, v in pillars.items()}}


def round_score(items: list[Item], item_gains: list[float], slopes: dict[str, float] | None = None,
                clip: float = CLIP, trim: float = TRIM, shares: dict[str, float] | None = None) -> dict:
    """The round's score (the mean of its pillars) and what it is made of: each pillar and its
    parts, the generated and fresh items each scored the same way on their own, the fresh
    items' languages, and every family's plain mean."""
    if len(items) != len(item_gains):
        raise ValueError("one gain per item")
    slopes = slopes or {}
    pairs: list[Pair] = []
    fam: dict[str, list[float]] = defaultdict(list)
    for it, row, g in zip(items, item_table(items), item_gains, strict=True):
        s = slopes.get(row["stratum"])
        g = g / s if s else g
        pairs.append((row, g))
        fam[it.family].append(g)
    pillars = _pillars(pairs, clip, trim, shares)
    out: dict = {"units": "doublings" if slopes else "bits per byte", "items": len(items),
                 "corners": len({it.corner for it in items}), "score": _mean_of(pillars),
                 "pillars": {p: {"score": round(v["score"], 5), "items": v["items"], "parts": v["parts"]} for p, v in pillars.items()},
                 "strata": {}, "generated": None, "fresh": None}
    gen = [p for it, p in zip(items, pairs, strict=True) if is_generated(it)]
    fresh = [p for it, p in zip(items, pairs, strict=True) if not is_generated(it)]
    if gen:
        out["generated"] = {"score": _mean_of(_pillars(gen, clip, trim, shares)), "items": len(gen)}
        out["strata"] = {k: round(v, 5) for k, v in _balanced(gen, clip, trim)[1].items()}
    if fresh:
        by_lang: dict[str, list[Pair]] = defaultdict(list)
        for it, p in zip(items, pairs, strict=True):
            if not is_generated(it):
                by_lang[it.lang].append(p)
        weights = language_weights(by_lang, shares)
        out["fresh"] = {"score": _mean_of(_pillars(fresh, clip, trim, shares)), "items": len(fresh),
                        "languages": {lang: {"weight": round(weights[lang], 5), "score": round(_balanced(p, clip, trim)[0], 5), "items": len(p)}
                                      for lang, p in sorted(by_lang.items())}}
    out["families"] = {k: round(sum(v) / len(v), 5) for k, v in sorted(fam.items())}
    return out

"""From per-item scores to a round's score.

A curator's gain on an item is its score less the unedited baseline's, on the same item, from
the same initial weights.

The season declares what the score is made of (its file's exam.groups and exam.pillars): groups,
each a fixed share of the score, and in each group pillars that count the same. Season 2 has two
groups. Writing, three quarters: how well the student predicts human writing published in the
day before the draw, one pillar for each kind. Skills, one quarter: generated tasks, each pillar
a family of generators or a few that ask one kind of thing. A pillar that a round's exam does not
have gives its share to the others of its group, never to the other group.

Inside a pillar:
- Each part counts the same. A part is a kind of writing, or a family of generators.
- Inside a part, each language counts by its share of the round's own corpus (programming
  languages count together as one, and beside natural languages take half). A natural language
  with fewer than SMALL items in the part is measured together with the part's other small ones,
  as "other", with their shares together.
- Inside a language, formats and kinds of task count the same, each a trimmed mean of its clipped
  gains.
- A language or a format with fewer than SMALL items counts in proportion to how many it has, so
  that no share of the score rests on one or two items.

The season names every part. A family of generators it does not name is not generated, a kind
of writing it does not name is not harvested or selected, and an item of such a part is an error
here: nothing on the exam counts for nothing.

So no generator, source or language weighs more than its declared share. A curator that gains
hugely in one place and loses everywhere else gets little; one that gains a little everywhere
wins. See docs/research/exam-first-principles.md for why these are the weights.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from functools import cache

from .items import Item

CLIP = 4.0  # robust widths
TRIM = 0.1  # from each end, within a stratum
SMALL = 8  # items a language or a format needs, in a part, to count in full
OTHER_LANG = "other"  # a part's small languages, measured together

# Fresh text: the kind of writing each source gives, for every source a harvester can give
# (fresh/harvest.py). A source not listed is a kind of its own. A kind the season does not name
# is no part of the exam: its sources are not harvested and its documents are not selected
# (fresh/select.py).
SOURCE_KIND = {
    "wikipedia": "reference", "wikipedia-added": "reference",
    "ccnews": "news",
    "github": "technical", "lore": "technical",
    "bluesky": "social", "mastodon": "social",
    "wikipedia-talk": "discussion", "hackernews": "discussion", "lemmy": "discussion", "github-comment": "discussion",
    "bluesky-reply": "discussion", "mastodon-reply": "discussion", "discourse": "discussion",
}
# Fresh harvest-derived comprehension & reasoning kinds (Season 3+, exam/fresh/reason.py).
REASONING_KINDS = ("explain", "infer", "synthesis")
CODE = "code"  # every programming language, as one language


@dataclass(frozen=True)
class Plan:
    """What a season's score is made of."""

    groups: dict[str, float]  # each group's share of the score
    pillars: dict[str, str]  # each pillar's group, in the order they are shown
    parts: dict[str, str]  # each part's pillar
    clip: float = CLIP
    trim: float = TRIM
    small: int = SMALL

    def pillar_of(self, part: str) -> str:
        """The pillar a part counts in. A part the season does not name is not on the exam."""
        if part not in self.parts:
            raise ValueError(f"{part!r} is not a part of the season's exam")
        return self.parts[part]

    def weights(self, present) -> dict[str, float]:
        """Each pillar's share of the score, among the pillars an exam has."""
        present = set(present)
        by_group: dict[str, list[str]] = defaultdict(list)
        for p, g in self.pillars.items():
            if p in present:
                by_group[g].append(p)
        total = sum(self.groups[g] for g in by_group)
        return {p: self.groups[g] / total / len(ps) for g, ps in by_group.items() for p in ps}


def plan_of(season) -> Plan:
    """The season's plan, from its file (exam.groups, exam.pillars, exam.aggregate). The file is
    the one list of what is on the exam, so it is held to what this copy can make: every part it
    names is a family of generators (gen.FAMILIES) or a kind of writing a source gives
    (SOURCE_KIND), and only a generated part it names is given more items (exam.more)."""
    from .gen import FAMILIES

    cfg = season.raw["exam"]
    pillars = {p: v["group"] for p, v in cfg["pillars"].items()}
    named = [part for v in cfg["pillars"].values() for part in v["parts"]]
    parts = {part: p for p, v in cfg["pillars"].items() for part in v["parts"]}
    agg = cfg.get("aggregate", {})
    plan = Plan(groups={g: float(s) for g, s in cfg["groups"].items()}, pillars=pillars, parts=parts,
                clip=float(agg.get("clip", CLIP)), trim=float(agg.get("trim", TRIM)), small=int(agg.get("small", SMALL)))
    if set(pillars.values()) != set(plan.groups) or len(named) != len(parts) or abs(sum(plan.groups.values()) - 1) > 1e-9:
        raise ValueError(f"season {season.id}: the exam's groups, pillars and parts do not fit together")
    families = {f.NAME for f in FAMILIES}
    if unknown := sorted(set(parts) - families - set(SOURCE_KIND.values()) - set(REASONING_KINDS)):
        raise ValueError(f"season {season.id}: the exam names {', '.join(unknown)}: no generator makes that and no source gives it")
    if stray := sorted(set(cfg.get("more", {})) - (set(parts) & families)):
        raise ValueError(f"season {season.id}: exam.more gives more items to {', '.join(stray)}, which the exam does not name as a generated part")
    return plan


@cache
def current_plan() -> Plan:
    from ..spec import load_season

    return plan_of(load_season())


def gains(scores: list[float], baseline: list[float]) -> list[float]:
    if len(scores) != len(baseline):
        raise ValueError("a curator and its baseline must sit the same items")
    return [s - b for s, b in zip(scores, baseline, strict=True)]


def stratum_key(it: Item) -> str:
    return f"{it.fmt}/{it.kind}"


def is_generated(it: Item) -> bool:
    return it.source.startswith("gen:")


def source_kind(source: str) -> str:
    """The kind of writing a source gives."""
    return SOURCE_KIND.get(source, source)


def kind_of(it: Item) -> str:
    """A fresh item's kind of writing."""
    return source_kind(it.source.split(":", 1)[0])


def part_of(it: Item) -> str:
    """An item's part: its family of generators, or its kind of writing."""
    return it.family if is_generated(it) else kind_of(it)


def lang_group(it: Item) -> str:
    return CODE if it.lang.startswith("code:") else it.lang


def item_table(items: list[Item], plan: Plan | None = None) -> list[dict]:
    """All the score needs to know of each item: its pillar, its part (a family, or a kind of
    writing), its language (programming languages as one) and its stratum. The server scores a
    round from this table and the gains alone (server/aggregate.go). An item of a part the season
    does not name is refused: it should not be on the exam."""
    plan = plan or current_plan()
    rows = []
    for it in items:
        part = part_of(it)
        rows.append({"pillar": plan.pillar_of(part), "part": part, "lang": lang_group(it), "stratum": stratum_key(it)})
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


def _share(lang: str, shares: dict[str, float]) -> float:
    from .fresh.select import OTHER

    return max(float(shares.get(lang, OTHER)), 0.0) or OTHER


def language_cells(counts: dict[str, int], shares: dict[str, float] | None = None, small: int = SMALL) -> tuple[dict[str, str], dict[str, float]]:
    """Where each language of a part is measured, and each cell's weight.

    counts is how many items each language has in the part. A natural language with at least
    `small` is a cell of its own; the others share the cell "other". A cell's weight is its share
    of the round's own corpus (the published table when that was not measured), a pooled cell's
    the sum of its languages', and a cell with fewer than `small` items counts in proportion.
    Programming languages together, beside natural languages, take half."""
    from .fresh.select import SHARES

    shares = shares or SHARES
    natural = sorted(set(counts) - {CODE})
    cell = {lang: (lang if counts[lang] >= small else OTHER_LANG) for lang in natural}
    w: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)
    for lang in natural:
        w[cell[lang]] += _share(lang, shares)
        n[cell[lang]] += counts[lang]
    full = {c: min(n[c], small) / small for c in w}
    total = sum(w[c] * full[c] for c in sorted(w))
    out = {c: w[c] * full[c] / total for c in sorted(w)}
    if CODE in counts:
        cell[CODE] = CODE
        if not out:
            return cell, {CODE: 1.0}
        half = 0.5 * min(counts[CODE], small) / small
        out = {c: v * (1 - half) for c, v in out.items()}
        out[CODE] = half
    return cell, out


Pair = tuple[dict, float]  # an item's row of item_table, and its gain


def _balanced(
    pairs: list[Pair], plan: Plan, item_weights: list[float] | None = None
) -> tuple[float, dict[str, float]]:
    """Formats and kinds of task with equal weight, each a trimmed mean of its clipped gains; one
    with fewer than `small` items in proportion to how many it has. When `item_weights` is given
    (the pooled small-language cell 'other'), each item's clipped gain is weighted by its
    language's crawl share so tiny languages never outweigh larger pooled languages."""
    by: dict[str, list[float]] = defaultdict(list)
    by_w: dict[str, list[float]] = defaultdict(list)
    for idx, (row, g) in enumerate(pairs):
        st_key = row["stratum"]
        by[st_key].append(g)
        if item_weights is not None:
            by_w[st_key].append(item_weights[idx])
    strata: dict[str, float] = {}
    for k, v in sorted(by.items()):
        cl = _clip(v, plan.clip)
        if item_weights is None:
            strata[k] = trimmed_mean(cl, plan.trim)
        else:
            ws = by_w[k]
            strata[k] = sum(w * x for w, x in zip(ws, cl, strict=True)) / sum(ws)
    w = {k: min(len(by[k]), plan.small) for k in strata}
    return sum(w[k] * strata[k] for k in strata) / sum(w.values()), strata


def _part(pairs: list[Pair], plan: Plan, shares) -> float:
    from .fresh.select import SHARES

    shares = shares or SHARES
    counts: dict[str, int] = defaultdict(int)
    for row, _ in pairs:
        counts[row["lang"]] += 1
    cell, w = language_cells(counts, shares, plan.small)
    cells: dict[str, list[Pair]] = defaultdict(list)
    for row, g in pairs:
        cells[cell[row["lang"]]].append((row, g))
    total = 0.0
    for c in sorted(cells):
        iw = [_share(row["lang"], shares) for row, _ in cells[c]] if c == OTHER_LANG else None
        total += w[c] * _balanced(cells[c], plan, iw)[0]
    return total


def _parts(pairs: list[Pair], plan: Plan, shares) -> dict[str, dict]:
    """Every pillar an exam has: its score, its items and its parts."""
    tree: dict[str, dict[str, list[Pair]]] = defaultdict(lambda: defaultdict(list))
    for row, g in pairs:
        tree[row["pillar"]][row["part"]].append((row, g))
    out = {}
    for p in sorted(tree):
        subs = {sub: _part(v, plan, shares) for sub, v in sorted(tree[p].items())}
        out[p] = {"score": sum(subs.values()) / len(subs), "items": sum(len(v) for v in tree[p].values()),
                  "parts": {k: round(v, 5) for k, v in subs.items()}}
    return out


def _score(pillars: dict[str, dict], plan: Plan) -> float:
    w = plan.weights(pillars)
    return sum(w[p] * pillars[p]["score"] for p in sorted(w))


def _check(table: list[dict], plan: Plan):
    for row in table:
        if plan.pillar_of(row["part"]) != row["pillar"]:
            raise ValueError(f"the season's exam has no part {row['part']!r} in a pillar {row['pillar']!r}")


def score_from_table(table: list[dict], item_gains: list[float], shares: dict[str, float] | None, plan: Plan | None = None) -> dict:
    """The round's score and each pillar's, unrounded, from item_table and the gains alone: what
    round_score gives, by the same code, and what the server computes. A row whose part the
    season does not name, or names under another pillar, is refused. The server is always given
    the round's shares (and refuses none); here None or {} means the published SHARES."""
    plan = plan or current_plan()
    if len(table) != len(item_gains):
        raise ValueError("one gain per item")
    _check(table, plan)
    pillars = _parts(list(zip(table, item_gains, strict=True)), plan, shares)
    return {"score": _score(pillars, plan), "pillars": {p: v["score"] for p, v in pillars.items()}}


def round_score(items: list[Item], item_gains: list[float], shares: dict[str, float] | None = None, plan: Plan | None = None) -> dict:
    """The round's score and what it is made of: each pillar with its share and its parts, the
    generated and fresh items each scored the same way on their own, the fresh items' languages,
    and every family's plain mean. An item of a part the season does not name is refused."""
    plan = plan or current_plan()
    if len(items) != len(item_gains):
        raise ValueError("one gain per item")
    rows = item_table(items, plan)
    pairs: list[Pair] = list(zip(rows, item_gains, strict=True))
    fam: dict[str, list[float]] = defaultdict(list)
    for it, g in zip(items, item_gains, strict=True):
        fam[it.family].append(g)
    pillars = _parts(pairs, plan, shares)
    weights = plan.weights(pillars)
    out: dict = {"units": "bits per byte", "items": len(items),
                 "corners": len({it.corner for it in items}), "score": _score(pillars, plan),
                 "pillars": {p: {"score": round(pillars[p]["score"], 5), "items": pillars[p]["items"], "parts": pillars[p]["parts"],
                                 "group": plan.pillars[p], "weight": round(weights[p], 5)} for p in plan.pillars if p in pillars},
                 "strata": {}, "generated": None, "fresh": None}
    gen = [p for it, p in zip(items, pairs, strict=True) if is_generated(it)]
    fresh = [p for it, p in zip(items, pairs, strict=True) if not is_generated(it)]
    if gen:
        out["generated"] = {"score": _score(_parts(gen, plan, shares), plan), "items": len(gen)}
        out["strata"] = {k: round(v, 5) for k, v in _balanced(gen, plan)[1].items()}
    if fresh:
        by_lang: dict[str, list[Pair]] = defaultdict(list)
        for it, p in zip(items, pairs, strict=True):
            if not is_generated(it):
                by_lang[it.lang].append(p)
        cell, weight = language_cells({k: len(v) for k, v in by_lang.items()}, shares, plan.small)
        out["fresh"] = {"score": _score(_parts(fresh, plan, shares), plan), "items": len(fresh),
                        "languages": {lang: {"cell": cell[lang], "weight": round(weight[cell[lang]], 5),
                                             "score": round(_balanced(p, plan)[0], 5), "items": len(p)}
                                      for lang, p in sorted(by_lang.items())}}
    out["families"] = {k: round(sum(v) / len(v), 5) for k, v in sorted(fam.items())}
    return out

"""What the organisers measure on settled rounds beyond their grades, as arithmetic.

Everything here starts from what the farm sends back (farm.py): for each student trained, one
score for every item of a round's exam. Three things are made of them.

**The unedited curve.** The competition's goal is stated as a multiplier: a curated corpus doing
as well as so many times more raw text. So the unedited corpus is trained at several multiples of
the budget, and each of those students is given a gain exactly as a curator is: its item scores
less the mean 1x unedited student's, added up by the season's score (aggregate.py). The curve is
those gains against the multiple, 0 at 1x by construction. Written into the season as
[multiple, minus the gain], it is what score.multiplier and the server read a curator's gain off.

**The scale audit.** A settled round is trained again one size up, and the question is whether
its curators kept their order. For the score and each pillar: how far curators differ (spread)
against how far two trainings of one script do (noise) at each size, and the rank agreement
between the sizes. One round has a handful of curators, so its rank agreement is rough: several
rounds' audits are pooled (`pool`) for a number worth reading.

**What a training costs.** The season names the slowest GPU that may run curators. What a
student's training takes there follows from the student's shape and budget, so a plan can say
what it asks for in GPU hours before anything is rented.

Nothing here trains a model, reads a corpus or asks the server anything: stages.py does those.
"""

from __future__ import annotations

import dataclasses
import math

from .. import bench
from .aggregate import gains, score_from_table

SCORE = "score"  # the round's score, among the things a gain is reported on
PILLAR = "pillar:"  # and the others, each pillar: "pillar:news"


# --- small statistics ------------------------------------------------------------------------------


def mean(vals) -> float:
    vals = list(vals)
    return sum(vals) / len(vals)


def sd(vals) -> float | None:
    """The sample standard deviation, or None of fewer than two values."""
    vals = list(vals)
    if len(vals) < 2:
        return None
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def pooled_sd(groups) -> float | None:
    """How far values of one group lie from their group's mean, over every group with two or
    more: the noise between trainings of the same script, from several scripts trained twice."""
    num, df = 0.0, 0
    for g in groups:
        g = list(g)
        if len(g) > 1:
            m = mean(g)
            num += sum((v - m) ** 2 for v in g)
            df += len(g) - 1
    return math.sqrt(num / df) if df else None


def rms(vals) -> float | None:
    """Standard deviations from several rounds as one: the root of their mean square."""
    vals = [v for v in vals if v is not None]
    return math.sqrt(mean(v * v for v in vals)) if vals else None


def ranks(vals: list[float]) -> list[float]:
    """Each value's rank, 1 for the smallest; equal values share the mean of their ranks."""
    order = sorted(range(len(vals)), key=vals.__getitem__)
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Rank agreement between two orders of the same names: 1 when they agree, -1 when one is
    the other reversed. None with fewer than three names in both, or when either is all ties:
    there is no order to agree with."""
    names = sorted(set(a) & set(b))
    if len(names) < 3:
        return None
    ra, rb = ranks([a[n] for n in names]), ranks([b[n] for n in names])
    ma, mb = mean(ra), mean(rb)
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True)) / math.sqrt(va * vb)


def mean_items(trainings: list[list[float]]) -> list[float]:
    """Several trainings of one script as one student: the mean of their scores, item by item."""
    if not trainings:
        raise ValueError("no training to take the mean of")
    return [sum(col) / len(trainings) for col in zip(*trainings, strict=True)]


def scores_of(table: list[dict], item_gains: list[float], shares, plan=None) -> dict[str, float]:
    """One student's gain on everything the season reports: the score, and each pillar
    ("pillar:news")."""
    agg = score_from_table(table, item_gains, shares, plan)
    return {SCORE: agg["score"], **{PILLAR + p: v for p, v in agg["pillars"].items()}}


# --- the unedited curve ----------------------------------------------------------------------------


def multiple_tag(m: float) -> str:
    """A multiple as it is written in a scale's name: 2, 0p5."""
    return f"{m:g}".replace(".", "p")


def at_multiple(scale, m: float):
    """The scale with `m` times its training budget and nothing else changed: the same student,
    the same corpus, the same batch and peak learning rate. What follows the budget does so by
    itself: the number of steps, and with it the learning-rate schedule, which warms up and
    decays over the run's own steps (train.lr_at), so each student is a whole training of its
    length and not a longer one cut short. The corpus does not grow: the unedited script is
    dealt in the corpus's order until the budget is full (script.materialize), so the student
    at half the budget trains on the first half of what the 1x student sees, and a budget past
    the end of the corpus starts again from the top (see `passes`)."""
    if m == 1:
        return scale
    return dataclasses.replace(scale, name=f"{scale.name}-x{multiple_tag(m)}", train_tokens=int(round(scale.train_tokens * m)))


def spec_of(scale) -> dict:
    """A scale as the farm's machines are told one the season does not name (farm.scale_of)."""
    return {f.name: getattr(scale, f.name) for f in dataclasses.fields(scale) if f.name != "name"}


def passes(scale, corpus_tokens: int) -> float:
    """How many times the unedited corpus is read to fill the scale's budget. Above 1, the
    student has seen some of it twice, and is no longer "that much more raw data"."""
    return scale.train_tokens / max(1, corpus_tokens)


def curve(table: list[dict], shares, points: dict[float, list[list[float]]], plan=None) -> list[dict]:
    """The unedited curve of one round. `points` gives, for each multiple of the budget, every
    training's item scores; 1x must be among them. Each point is the season's score of that
    student's item scores less the mean 1x student's: what a curator's gain is, for the unedited
    corpus at another budget. At 1x it is 0, and `sd` there is how far one unedited training
    lies from the mean of them."""
    if not points.get(1.0):
        raise ValueError("the curve is measured from the unedited student at 1x, and it has no training")
    base = mean_items(points[1.0])
    out = []
    for m in sorted(points):
        each = [score_from_table(table, gains(s, base), shares, plan)["score"] for s in points[m]]
        if each:
            out.append({"multiple": float(m), "trainings": len(each), "gain": 0.0 if m == 1 else mean(each), "sd": sd(each), "each": each})
    return out


def slopes(points: list[dict]) -> list[dict]:
    """Between each pair of neighbouring points, the gain a doubling of the data is worth."""
    return [{"from": a["multiple"], "to": b["multiple"], "per_doubling": (b["gain"] - a["gain"]) / math.log2(b["multiple"] / a["multiple"])}
            for a, b in zip(points, points[1:], strict=False)]


def rises(points: list[dict]) -> bool:
    """Whether more data always did better. A curve that does not rise cannot be read both ways:
    one gain would stand for two multiples."""
    return all(b["gain"] > a["gain"] for a, b in zip(points, points[1:], strict=False))


def as_reference(points: list[dict]) -> list[list[float]]:
    """The curve as the season file holds one (`reference_curves`): [multiple, loss], where only
    differences from the loss at 1x are ever read (score.read_off, server/season.go). Minus the
    gain is such a loss: 0 at 1x, and a curator's gain g is looked up at 0 - g."""
    return [[p["multiple"], round(-p["gain"], 5) or 0.0] for p in points]


def average(curves: list[list[dict]]) -> list[dict]:
    """Several rounds' curves as one: at each multiple, the mean of the rounds that measured it,
    with how far they spread (`sd`) and how well that pins the mean (`se`)."""
    by: dict[float, list[float]] = {}
    for c in curves:
        for p in c:
            by.setdefault(p["multiple"], []).append(p["gain"])
    out = []
    for m in sorted(by):
        g = by[m]
        spread = sd(g)
        out.append({"multiple": m, "rounds": len(g), "gain": 0.0 if m == 1 else mean(g), "sd": spread,
                    "se": None if spread is None else spread / math.sqrt(len(g)), "each": g})
    return out


# --- the scale audit -------------------------------------------------------------------------------


def size_stats(table: list[dict], shares, unedited: list[list[float]], curators: dict[str, list[list[float]]], plan=None) -> dict:
    """One size of student on one round. `unedited` and each curator's entry are the item scores
    of every training. For each thing the season reports: every curator's gain (the mean of its
    trainings), the noise (how far trainings of one script differ, pooled over the curators
    trained more than once) and the spread (how far curators' gains differ)."""
    base = mean_items(unedited)
    per = {name: [scores_of(table, gains(s, base), shares, plan) for s in runs] for name, runs in curators.items() if runs}
    measures = list(next(iter(per.values()))[0]) if per else []
    out: dict = {"unedited": len(unedited), "trainings": {name: len(runs) for name, runs in per.items()}, "gains": {}, "measures": {}}
    for m in measures:
        each = {name: [r[m] for r in runs] for name, runs in per.items()}
        got = {name: mean(v) for name, v in each.items()}
        noise, spread = pooled_sd(each.values()), sd(got.values())
        out["gains"][m] = got
        out["measures"][m] = {"noise": noise, "spread": spread, "spread_over_noise": spread / noise if noise and spread is not None else None}
    return out


def audit(sizes: dict[str, dict]) -> dict:
    """A round at several sizes of student, the rounds' own first (each from size_stats). For
    each thing reported: its noise and spread at every size, and whether the order of curators
    carried from each size to the next (`carried`)."""
    names = list(sizes)
    own = sizes[names[0]]
    out: dict = {"sizes": names, "measures": {}, "curators": {}}
    for m in own["measures"]:
        row: dict = {"sizes": {s: sizes[s]["measures"][m] for s in names if m in sizes[s]["measures"]}, "carried": {}}
        for a, b in zip(names, names[1:], strict=False):
            if m in sizes[a]["gains"] and m in sizes[b]["gains"]:
                row["carried"][f"{a}>{b}"] = spearman(sizes[a]["gains"][m], sizes[b]["gains"][m])
        out["measures"][m] = row
    for s in names:
        for name, g in sizes[s]["gains"].get(SCORE, {}).items():
            out["curators"].setdefault(name, {})[s] = g
    return out


def pool(reports: list[dict]) -> dict:
    """Several rounds' audits as one, for the score and each pillar. Noise and spread are pooled
    as variances; rank agreements are averaged."""
    out: dict = {"rounds": len(reports), "measures": {}}
    order: list[str] = []
    for r in reports:
        order += [m for m in r["measures"] if m not in order]
    for m in order:
        have = [r["measures"][m] for r in reports if m in r["measures"]]
        row: dict = {"rounds": len(have), "sizes": {}, "carried": {}}
        for size in dict.fromkeys(s for h in have for s in h["sizes"]):
            got = [h["sizes"][size] for h in have if size in h["sizes"] and h["sizes"][size]["noise"] and h["sizes"][size]["spread"] is not None]
            noise, spread = rms(g["noise"] for g in got), rms(g["spread"] for g in got)
            row["sizes"][size] = {"rounds": len(got), "noise": noise, "spread": spread, "spread_over_noise": spread / noise if noise else None}
        for pair in dict.fromkeys(p for h in have for p in h["carried"]):
            got = [h["carried"][pair] for h in have if h["carried"].get(pair) is not None]
            row["carried"][pair] = {"rounds": len(got), "mean": mean(got) if got else None}
        out["measures"][m] = row
    return out


# --- what a training costs -------------------------------------------------------------------------

BACKWARD = 3.0  # a training step is a forward pass and a backward one: about three forward passes of arithmetic
SERVER_HOLD = 45 * 60  # how long the server lets a machine hold a farm student unless its batch says (server/farm.go)
LONGEST_HOLD = 24 * 3600  # and the longest a batch may say
TOKEN_BUFFERS = 3  # first-pass arrays, lane matrix, and hash copy held simultaneously by script.materialize
TOKEN_BYTES = 2 * TOKEN_BUFFERS  # uint16 default (vocab_size <= 65535); prefer token_bytes(season)


def work_per_token(layers: int, width: int, length: int, vocab: int = 0) -> float:
    """The arithmetic of one forward pass over one token, in multiply-adds: each layer's
    matrices (12 x width^2 weights, each used once), attention over a window of `length` tokens
    (the dot products with them and the weighted sum of them), and the output layer of a model
    that has one."""
    return layers * (24.0 * width * width + 4.0 * length * width) + 2.0 * vocab * width


def train_seconds(season, scale) -> float:
    """About how long the student's training takes on the slowest GPU that may run curators.

    The season's floor (`runner.gpu_tokens_per_second`) is tokens a second through the
    benchmark's model, forward only (bench.py). Times that model's arithmetic per token it is
    arithmetic a second, and the student's training needs its own arithmetic per token, three
    times over, for every token of the budget. That comes to 37 minutes for season 2's standard
    student and four and a half hours for its large one. A faster card takes less in about the
    proportion of its `stet bench` to the floor's: an RTX 3090 trains the standard student in
    about 18 minutes, and an RTX 4090 the large one in about an hour (docs/EXPERIMENTS.md)."""
    g = bench.GPU_MODEL
    floor = float(bench.floor(season)["gpu_tokens_per_second"]) * work_per_token(g["layers"], g["width"], g["length"])
    if floor <= 0:
        raise ValueError(f"season {season.id} names no floor for a runner's GPU: there is nothing to estimate from")
    return scale.train_tokens * BACKWARD * work_per_token(scale.n_layer, scale.d_model, scale.ctx, season.vocab_size) / floor


def hold_seconds(season, scale) -> int:
    """How long a machine is let hold one of these students (a farm batch's `lease`): 0, the
    server's usual hold, when the floor trains it within that; otherwise the floor's time and a
    quarter, and an hour to build the corpus. A batch held for more than an hour is handed only
    to machines set aside for long students (STET_FARM_MAX_MINUTES), which must allow this much."""
    t = train_seconds(season, scale)
    if t <= SERVER_HOLD:
        return 0
    return min(LONGEST_HOLD, math.ceil((1.25 * t + 3600) / 300) * 300)


def token_bytes(season=None) -> int:
    """Peak host memory bytes per token across script.materialize's three simultaneous buffers:
    uint16 (2 B/token -> 6 B/token) when vocab_size <= 65535, uint32 (4 B/token -> 12 B/token)
    otherwise (matching script.materialize)."""
    width = 2 if (season is None or int(season.vocab_size) <= 65535) else 4
    return TOKEN_BUFFERS * width


def memory_gb(scale, season=None) -> float:
    """Memory a machine needs to train the student: script.materialize holds the first pass's
    tokens, the lanes they are dealt into and, while it hashes them, a copy."""
    return token_bytes(season) * scale.train_tokens / 2**30

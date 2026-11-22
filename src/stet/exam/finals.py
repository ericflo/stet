"""A season's finals, as arithmetic: who is in them, the lot, and the table.

The rules are in docs/RULES.md ("Finals"). The rounds choose who is looked at closely: the eight
best curators on the board, one per person, and one more drawn by lot from the next eight. Each
is run on two fresh rounds at the large size, its student trained twice on each script, and the
table is made from those gains alone.

Everything here is a function of what is published: the board as it stood, a beacon value, and
the students' scores item by item. stages.py gathers those.
"""

from __future__ import annotations

import hashlib
import math

from ..score import read_off
from ..spec import prf_u64
from . import measure
from .aggregate import gains, score_from_table

SCALE = "large"  # the size the finals' students are
PLACES = 8  # finalists
LOT_PLACES = 8  # the lot is drawn among the next eight on the board: those ranked 9th to 16th
LOT_ROUNDS = 5  # that have run at least this many rounds
ROUNDS = 2  # finals rounds
TRAININGS = 2  # of the student, on each script
BASELINES = 4  # of the unedited corpus's student, in each round
MULTIPLES = (0.5, 2.0, 4.0)  # the unedited curve's other points, each trained once a round
TIE = 2.0  # standard errors of a difference within which two curators share a place
FINALIST, LOT, YARDSTICK = "finalist", "lot", "yardstick"


def person(row: dict) -> str:
    """Who a curator belongs to, for "one per person": the GitHub identity it was entered under,
    and the account where the board gives none."""
    return str(row.get("github") or row.get("owner") or "")


def slim(row: dict) -> dict:
    """What the finals keep of a row of the board."""
    return {k: row.get(k) for k in ("id", "name", "version", "owner", "github", "rank", "rounds", "rating")}


def choose(board: list[dict], organisers, places: int = PLACES, lot_places: int = LOT_PLACES, lot_rounds: int = LOT_ROUNDS) -> dict:
    """Who the finals look at, from the board's ranked curators in the board's order.

    The organisers' own curators are yardsticks: all of them are run, and none takes a place or
    counts toward one. Of the rest, each person's best is kept. The first `places` are the
    finalists. The lot is drawn among the next `lot_places` of that same list, the 9th to the
    16th, and only those that have run `lot_rounds` rounds: a curator with fewer is on the board
    by a round or two of luck."""
    organisers = set(organisers)
    yardsticks = [slim(r) for r in board if r.get("owner") in organisers]
    entrants, people = [], set()
    for r in board:
        if r.get("owner") in organisers or person(r) in people:
            continue
        people.add(person(r))
        entrants.append(slim(r))
    pool = [r for r in entrants[places : places + lot_places] if int(r.get("rounds") or 0) >= lot_rounds]
    return {"finalists": entrants[:places], "pool": pool, "yardsticks": yardsticks}


def lot_seed(season_id: str, beacon: dict) -> bytes:
    """What the lot is drawn from: the season and one round of the public beacon, named before
    it was emitted. Written as a round's seed is (spec.official_seed)."""
    return hashlib.sha256(f"stet|{season_id}|finals|lot|{int(beacon['round'])}|{str(beacon['randomness']).lower()}".encode()).digest()


def draw_lot(season_id: str, pool: list[dict], beacon: dict) -> dict | None:
    """The lot's pick: one of the pool, in the board's order, by the beacon's value. Anyone with
    the published pool and the beacon round works out the same one. None of an empty pool."""
    if not pool:
        return None
    return pool[prf_u64(lot_seed(season_id, beacon), "lot", 0) % len(pool)]


def table(rounds: list[dict], roles: dict[str, str], tie: float = TIE, plan=None) -> dict:
    """The finals' table, from every student's scores item by item.

    `rounds` has, for each finals round: `table` and `shares` (what each item counts toward, and
    the corpus's languages: aggregate.item_table), `unedited` (the item scores of each training
    of the unedited corpus), `students` (for each curator, those of each training of its script;
    none for a curator that wrote no usable script there, whose gain in that round is the
    lowest any curator got there, or zero if that is lower, as in any round), and `curve` (for each other multiple of the budget, the unedited trainings
    there). `roles` names every curator run: finalist, lot or yardstick.

    A curator's gain is the mean of its rounds, and a round's the mean of its trainings there.

    Its standard error is that of the mean of its rounds, from how far they differ, and never
    less than the noise between two trainings of one script allows (pooled over every script
    trained more than once: with two rounds, two that happen to agree would otherwise claim a
    precision nobody measured).

    Places are decided by differences, and the standard error of a difference is taken from the
    curators together: how far their gains moved against each other from one round to the other,
    once each curator's own level and each round's are taken out, which is what a difference
    between two curators is uncertain by. With two rounds one pair's own two differences say
    almost nothing; all the pairs together say it with as many degrees of freedom as there are
    curators, less one. It is never less than training noise allows either. Yardsticks are left
    out of it, as they are of the places.

    From the top, a curator shares the place of the best curator not yet separated from it:
    those within `tie` standard errors of a difference of that one. The next place counts
    everyone above it (1, 1, 3).

    The multiplier is read off the finals' own curve: the mean, over the rounds, of the unedited
    corpus's gain at each multiple of the budget."""
    names = list(roles)
    cells: dict[str, list[list[float]]] = {n: [] for n in names}
    curves = []
    for r in rounds:
        base = measure.mean_items(r["unedited"])
        for n in names:
            cells[n].append([score_from_table(r["table"], gains(s, base), r["shares"], plan)["score"] for s in (r["students"].get(n) or [])])
        if r.get("curve"):
            curves.append(measure.curve(r["table"], r["shares"], {1.0: r["unedited"], **r["curve"]}, plan))
    n_rounds = len(rounds)
    by_round: dict[str, list[float]] = {n: [] for n in names}
    for k in range(n_rounds):
        finished = {n: measure.mean(cells[n][k]) for n in names if cells[n][k]}
        lowest = min([0.0, *finished.values()])  # failing is never better than finishing
        for n in names:
            by_round[n].append(finished.get(n, lowest))
    gain = {n: measure.mean(by_round[n]) for n in names}
    noise = measure.pooled_sd(c for n in names for c in cells[n])

    def of_training(n: str) -> float:
        """The variance training noise alone gives one round's gain of a curator."""
        fewest = min((len(c) for c in cells[n] if c), default=0)
        return noise**2 / fewest if noise is not None and fewest else 0.0

    def se(n: str) -> float | None:
        v = max((measure.sd(by_round[n]) or 0.0) ** 2, of_training(n))
        return math.sqrt(v / n_rounds) if v > 0 else None

    placed = [n for n in names if roles[n] != YARDSTICK]
    whole = [n for n in placed if all(cells[n])]  # a round with no script is a 0 by rule, not a measurement
    against = 0.0
    if n_rounds >= 2 and len(whole) >= 2:
        level = [measure.mean(by_round[n][r] for n in whole) for r in range(n_rounds)]
        grand = measure.mean(level)
        ss = sum((by_round[n][r] - gain[n] - level[r] + grand) ** 2 for n in whole for r in range(n_rounds))
        against = ss / ((len(whole) - 1) * (n_rounds - 1))
    v = max(against, max((of_training(n) for n in whole), default=0.0))
    se_diff = math.sqrt(2 * v / n_rounds) if v > 0 else None

    order = sorted(placed, key=lambda n: (-gain[n], n))
    place: dict[str, int] = {}
    lead = None
    for i, n in enumerate(order):
        apart = gain[lead] - gain[n] if lead is not None else None
        if lead is None or (apart >= tie * se_diff if se_diff is not None else apart > 0):
            lead, place[n] = n, i + 1
        else:
            place[n] = place[lead]

    points = measure.average(curves) if curves else []
    ref = measure.as_reference(points) if points else None
    rows = []
    for n in sorted(names, key=lambda n: (-gain[n], n)):
        m = read_off(ref, gain[n]) if ref else None
        rows.append({"name": n, "role": roles[n], "place": place.get(n),
                     "shares_place_with": [o for o in order if o != n and n in place and place[o] == place[n]],
                     "gain": gain[n], "se": se(n), "rounds": by_round[n], "trainings": [len(c) for c in cells[n]],
                     "multiplier": m[0] if m else None, "capped": bool(m and m[1])})
    out: dict = {"rows": rows, "se_difference": se_diff, "training_noise": noise, "tie": tie, "curve": ref,
                 "curve_rises": measure.rises(points) if points else None, "lot": None}
    lot = next((n for n in names if roles[n] == LOT), None)
    if lot is not None:
        finalists = [n for n in order if roles[n] == FINALIST]
        out["lot"] = {"name": lot, "place": place[lot], "ahead_of": [n for n in finalists if place[n] > place[lot]],
                      "level_with": [n for n in finalists if place[n] == place[lot]]}
    return out

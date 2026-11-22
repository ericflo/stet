"""A season's finals as arithmetic (exam/finals.py): who is in them, the lot, and the table, on a
board and on students made up so that the answers are known."""

import hashlib
import math

import pytest

from stet.exam import finals
from stet.exam.aggregate import Plan

PLAN = Plan(groups={"writing": 0.75, "skills": 0.25}, pillars={"news": "writing", "logic": "skills"}, parts={"news": "news", "logic": "logic"})
SHARES = {"en": 100.0}
TABLE = [{"pillar": p, "part": p, "lang": "en", "stratum": "prose/passage"} for p in ("news", "logic") for _ in range(10)]
BASE = [0.5 + 0.01 * i for i in range(len(TABLE))]


def student(gain: float) -> list[float]:
    """A student that scores `gain` more than the unedited one on every item: its gain under the season's score is that."""
    return [b + gain for b in BASE]


def twice(gain: float, noise: float = 0.001) -> list[list[float]]:
    return [student(gain + noise), student(gain - noise)]


UNEDITED = [student(d) for d in (0.003, -0.003, 0.001, -0.001)]  # four trainings whose mean is BASE itself


def row(rank: int, owner: str, name: str = "c", rounds: int = 9, github: str | None = None) -> dict:
    return {"id": f"{rank:064x}", "name": name, "version": 1, "owner": owner, "github": github or owner, "rank": rank, "rounds": rounds, "rating": 1.0 - rank / 100,
            "mean_gain": 0.01}


def test_eight_finalists_one_per_person_and_the_lots_pool_is_the_next_eight_with_five_rounds():
    board = [row(1, "ann"), row(2, "stet", "reference"), row(3, "bob"), row(4, "ann", "second"), row(5, "cy"), row(6, "ann2", github="ann")]
    board += [row(7 + i, f"p{i}") for i in range(5)]  # p0..p4: with ann, bob and cy, the eight
    board += [row(12, "q0"), row(13, "q1", rounds=4), row(14, "stet", "lead"), row(15, "q2", rounds=5), row(16, "bob", "again")]
    board += [row(17 + i, f"r{i}") for i in range(7)]  # r0..r4 fill the next eight; r5 and r6 are past it
    got = finals.choose(board, {"stet"})
    assert [r["owner"] for r in got["finalists"]] == ["ann", "bob", "cy", "p0", "p1", "p2", "p3", "p4"]
    assert [r["rank"] for r in got["finalists"]] == [1, 3, 5, 7, 8, 9, 10, 11]  # each person's best, in the board's order
    # The next eight people are q0, q1, q2 and r0 to r4. q1 has run four rounds, and is not drawn among.
    assert [r["owner"] for r in got["pool"]] == ["q0", "q2", "r0", "r1", "r2", "r3", "r4"]
    assert [r["name"] for r in got["yardsticks"]] == ["reference", "lead"]  # the organisers' own: all run, none placed
    assert "mean_gain" not in got["finalists"][0] and set(got["finalists"][0]) == {"id", "name", "version", "owner", "github", "rank", "rounds", "rating"}
    few = finals.choose(board[:6], {"stet"})
    assert [r["owner"] for r in few["finalists"]] == ["ann", "bob", "cy"] and few["pool"] == []


def test_the_lot_is_anyones_to_work_out_from_the_pool_and_the_beacon():
    pool = [row(12 + i, f"q{i}") for i in range(7)]
    beacon = {"round": 6_000_123, "randomness": "ab" * 32}
    seed = hashlib.sha256(f"stet|s2|finals|lot|6000123|{'ab' * 32}".encode()).digest()
    want = int.from_bytes(hashlib.sha256(seed + b"|lot|0").digest()[:8], "big") % 7
    assert finals.draw_lot("s2", pool, beacon) == pool[want]
    assert finals.draw_lot("s2", pool, {"round": "6000123", "randomness": "AB" * 32}) == pool[want]  # however the value is spelt
    assert finals.draw_lot("s2", pool, beacon) == finals.draw_lot("s2", list(pool), dict(beacon))
    picks = {finals.draw_lot("s2", pool, {"round": 6_000_123, "randomness": f"{i:064x}"})["owner"] for i in range(40)}
    assert len(picks) > 3  # the value decides it
    assert finals.draw_lot("s3", pool, beacon) == pool[int.from_bytes(hashlib.sha256(
        hashlib.sha256(f"stet|s3|finals|lot|6000123|{'ab' * 32}".encode()).digest() + b"|lot|0").digest()[:8], "big") % 7]
    assert finals.draw_lot("s2", [], beacon) is None


# Two rounds. Each curator's gain in the first round is its level and something, in the second its
# level less the same: A and C go up then down by 0.002, B and D the other way, the lot's pick stays.
LEVEL = {"A": 0.050, "B": 0.046, "L": 0.040, "C": 0.036, "D": 0.020, "Y": 0.045}
MOVE = {"A": 0.002, "B": -0.002, "L": 0.0, "C": 0.002, "D": -0.002, "Y": 0.001}
ROLES = {"A": finals.FINALIST, "B": finals.FINALIST, "C": finals.FINALIST, "D": finals.FINALIST, "L": finals.LOT, "Y": finals.YARDSTICK}


def _rounds(names=tuple(LEVEL)) -> list[dict]:
    first = {"table": TABLE, "shares": SHARES, "unedited": UNEDITED, "students": {n: twice(LEVEL[n] + MOVE[n]) for n in names},
             "curve": {0.5: [student(-0.04)], 2.0: [student(0.04)], 4.0: [student(0.08)]}}
    second = {"table": TABLE, "shares": SHARES, "unedited": UNEDITED, "students": {n: twice(LEVEL[n] - MOVE[n]) for n in names},
              "curve": {0.5: [student(-0.04)], 2.0: [student(0.03)], 4.0: [student(0.07)]}}
    return [first, second]


def test_the_table_ties_curators_within_two_standard_errors_and_places_the_lots_pick():
    out = finals.table(_rounds(), ROLES, plan=PLAN)
    rows = {r["name"]: r for r in out["rows"]}
    assert [r["name"] for r in out["rows"]] == ["A", "B", "Y", "L", "C", "D"]  # by gain, the yardstick where its gain puts it
    for n in LEVEL:
        assert rows[n]["gain"] == pytest.approx(LEVEL[n]) and rows[n]["rounds"] == pytest.approx([LEVEL[n] + MOVE[n], LEVEL[n] - MOVE[n]])
        assert rows[n]["trainings"] == [2, 2]
    # Training noise: every script's two trainings lie 0.001 either side of their mean.
    assert out["training_noise"] == pytest.approx(0.001 * math.sqrt(2))
    # A curator's own standard error: from its two rounds, 0.002 either side of their mean, and never
    # less than two trainings a round allow (the lot's pick, whose rounds agree).
    assert rows["A"]["se"] == pytest.approx(0.002) and rows["Y"]["se"] == pytest.approx(0.001)
    assert rows["L"]["se"] == pytest.approx(math.sqrt(0.001**2 / 2))
    # The standard error of a difference, from how the five placed curators moved against each
    # other between the rounds: half-differences of +2, -2, 0, +2, -2 thousandths have variance
    # 4e-6 over four degrees of freedom, a difference's standard error is the root of twice that.
    assert out["se_difference"] == pytest.approx(math.sqrt(8e-6))
    # So gains within 0.00566 share a place. B is 0.004 behind A. The lot's pick is 0.010 behind A:
    # third. C is 0.004 behind it. D is far behind both.
    assert {n: rows[n]["place"] for n in LEVEL} == {"A": 1, "B": 1, "L": 3, "C": 3, "D": 5, "Y": None}
    assert rows["A"]["shares_place_with"] == ["B"] and rows["C"]["shares_place_with"] == ["L"] and rows["D"]["shares_place_with"] == []
    assert out["lot"] == {"name": "L", "place": 3, "ahead_of": ["D"], "level_with": ["C"]}
    # The finals' own curve: the mean of the two rounds' at each budget, and each gain read off it.
    assert out["curve"] == [[0.5, 0.04], [1.0, 0.0], [2.0, -0.035], [4.0, -0.075]] and out["curve_rises"]
    assert rows["A"]["multiplier"] == pytest.approx(2 ** (1 + 0.015 / 0.04)) and not rows["A"]["capped"]
    assert rows["D"]["multiplier"] == pytest.approx(2 ** (0.020 / 0.035))
    assert rows["Y"]["multiplier"] == pytest.approx(2 ** (1 + 0.010 / 0.04)) and rows["Y"]["role"] == finals.YARDSTICK


def test_a_yardstick_takes_no_place_and_moves_nobody():
    with_it = finals.table(_rounds(), ROLES, plan=PLAN)
    without = finals.table(_rounds([n for n in LEVEL if n != "Y"]), {n: r for n, r in ROLES.items() if n != "Y"}, plan=PLAN)
    assert {r["name"]: r["place"] for r in without["rows"]} == {r["name"]: r["place"] for r in with_it["rows"] if r["name"] != "Y"}
    assert without["se_difference"] == pytest.approx(with_it["se_difference"])
    # A yardstick better than every finalist is still no place, and the best finalist is first.
    best = _rounds()
    for r in best:
        r["students"]["Y"] = twice(0.2)
    out = finals.table(best, ROLES, plan=PLAN)
    assert out["rows"][0]["name"] == "Y" and out["rows"][0]["place"] is None and out["rows"][1]["place"] == 1


def test_the_lots_pick_takes_whatever_place_its_gain_earns():
    top = _rounds()
    for r in top:
        r["students"]["L"] = twice(0.070)
    out = finals.table(top, ROLES, plan=PLAN)
    assert out["lot"] == {"name": "L", "place": 1, "ahead_of": ["A", "B", "C", "D"], "level_with": []}
    assert [(r["name"], r["place"]) for r in out["rows"] if r["place"]] == [("L", 1), ("A", 2), ("B", 2), ("C", 4), ("D", 5)]
    last = _rounds()
    for r in last:
        r["students"]["L"] = twice(0.001)
    assert finals.table(last, ROLES, plan=PLAN)["lot"] == {"name": "L", "place": 5, "ahead_of": [], "level_with": []}


def test_a_round_with_no_usable_script_is_a_gain_of_0_there():
    rounds = _rounds()
    del rounds[1]["students"]["D"]
    out = finals.table(rounds, ROLES, plan=PLAN)
    d = next(r for r in out["rows"] if r["name"] == "D")
    assert d["rounds"] == pytest.approx([0.018, 0.0]) and d["gain"] == pytest.approx(0.009) and d["trainings"] == [2, 0] and d["place"] == 5
    # A 0 by rule is not a measurement: the standard error of a difference is taken from the four
    # that have both rounds, whose half-differences of +2, -2, +2 and 0 thousandths have mean 0.5.
    assert out["se_difference"] == pytest.approx(math.sqrt(2 * (3 * 4e-6 - 4 * 0.0005**2) / 3))


def test_one_round_trained_once_places_by_gain_alone():
    only = {"table": TABLE, "shares": SHARES, "unedited": UNEDITED[:1], "students": {"A": [student(0.03)], "B": [student(0.03)], "C": [student(0.01)]}}
    out = finals.table([only], {"A": finals.FINALIST, "B": finals.FINALIST, "C": finals.FINALIST}, plan=PLAN)
    assert out["se_difference"] is None and out["training_noise"] is None and out["curve"] is None and out["lot"] is None
    assert [(r["name"], r["place"], r["se"], r["multiplier"]) for r in out["rows"]] == [("A", 1, None, None), ("B", 1, None, None), ("C", 3, None, None)]


def test_training_noise_is_the_least_a_difference_can_be_uncertain_by():
    """Two rounds in which nobody moved against anybody: the curators' agreement between rounds is
    not taken for more precision than two trainings a round can give."""
    rounds = _rounds()
    for r in rounds:
        r["students"] = {n: twice(LEVEL[n], noise=0.004) for n in LEVEL}
    out = finals.table(rounds, ROLES, plan=PLAN)
    assert out["training_noise"] == pytest.approx(0.004 * math.sqrt(2))
    assert out["se_difference"] == pytest.approx(0.004)  # each round's gain is uncertain by 0.004 (two trainings), a difference of two-round means by as much
    # Gains within 0.008 share a place: C, 0.014 behind A, would be level with it if agreement between rounds were believed.
    assert {r["name"]: r["place"] for r in out["rows"]} == {"A": 1, "B": 1, "L": 3, "C": 3, "D": 5, "Y": None}
    quiet = _rounds()
    for r in quiet:
        r["students"] = {n: twice(LEVEL[n], noise=0.0001) for n in LEVEL}
    assert {r["name"]: r["place"] for r in finals.table(quiet, ROLES, plan=PLAN)["rows"]} == {"A": 1, "B": 2, "L": 3, "C": 4, "D": 5, "Y": None}

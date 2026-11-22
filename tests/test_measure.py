"""The organisers' measurements as arithmetic (exam/measure.py): the unedited curve, the scale
audit and what a training costs, on students whose scores are made up so that the answers are known.

Every student here scores a fixed amount more than the unedited one on every counted item, so its
gain under the season's score is that amount, whatever the score is made of."""

import dataclasses
import json
import math
import re

import pytest

from stet import score
from stet.exam import farm, measure
from stet.exam.aggregate import Plan
from stet.spec import Scale, load_season

PLAN = Plan(groups={"writing": 0.75, "skills": 0.25}, pillars={"news": "writing", "logic": "skills"}, parts={"news": "news", "logic": "logic"})
SHARES = {"en": 100.0}
# Ten items of news and ten of logic, which count, and ten each of two parts that are only watched.
TABLE = [{"pillar": p, "part": part, "lang": "en", "stratum": "prose/passage"}
         for p, part in (("news", "news"), ("logic", "logic"), ("watch", "arith"), ("watch", "morph")) for _ in range(10)]
BASE = [0.5 + 0.01 * i for i in range(len(TABLE))]


def student(gain: float, arith: float = 0.0, morph: float = 0.0) -> list[float]:
    """A student's score on every item: `gain` over the unedited one on what counts, and what it is given on each watched part."""
    return [b + {"arith": arith, "morph": morph}.get(row["part"], gain) for b, row in zip(BASE, TABLE, strict=True)]


def twice(gain: float, noise: float = 0.001, **watched) -> list[list[float]]:
    """Two trainings of one script: `noise` either side of its gain, on everything."""
    return [student(gain + d, **{k: v + d for k, v in watched.items()}) for d in (noise, -noise)]


UNEDITED = [student(d, d, d) for d in (0.002, -0.002)]  # two trainings whose mean is BASE itself


def test_a_gain_is_reported_on_the_score_each_pillar_and_each_watched_part():
    got = measure.scores_of(TABLE, [s - b for s, b in zip(student(0.03, arith=0.01, morph=-0.02), BASE, strict=True)], SHARES, PLAN)
    assert got == pytest.approx({"score": 0.03, "pillar:news": 0.03, "pillar:logic": 0.03, "watch:arith": 0.01, "watch:morph": -0.02})


def test_the_curve_is_each_budgets_gain_over_the_1x_student_and_is_0_there():
    points = {0.5: [student(-0.04)], 1.0: UNEDITED, 2.0: [student(0.035)], 4.0: [student(0.06)]}
    curve = measure.curve(TABLE, SHARES, points, PLAN)
    assert [p["multiple"] for p in curve] == [0.5, 1.0, 2.0, 4.0] and [p["trainings"] for p in curve] == [1, 2, 1, 1]
    assert [p["gain"] for p in curve] == pytest.approx([-0.04, 0.0, 0.035, 0.06])
    assert curve[1]["gain"] == 0.0  # by construction, not by luck
    assert curve[1]["sd"] == pytest.approx(0.002 * math.sqrt(2)) and curve[0]["sd"] is None  # how far one unedited training lies from their mean
    assert [s["per_doubling"] for s in measure.slopes(curve)] == pytest.approx([0.04, 0.035, 0.025])
    assert measure.rises(curve)
    with pytest.raises(ValueError, match="1x"):
        measure.curve(TABLE, SHARES, {2.0: [student(0.035)]}, PLAN)


def test_the_curve_goes_into_the_season_as_it_is_and_reads_back_as_a_multiplier():
    """`reference_curves` holds [multiple, loss] and only differences from the loss at 1x are read:
    minus the gain is such a loss. The server reads it the same way (server/curve_test.go)."""
    curve = measure.curve(TABLE, SHARES, {0.5: [student(-0.04)], 1.0: UNEDITED, 2.0: [student(0.035)], 4.0: [student(0.06)]}, PLAN)
    ref = measure.as_reference(curve)
    assert ref == [[0.5, 0.04], [1.0, 0.0], [2.0, -0.035], [4.0, -0.06]]
    assert "-0.0]" not in json.dumps(ref)  # the point at 1x is written as 0
    season = load_season("s2")
    season = dataclasses.replace(season, raw={**season.raw, "reference_curves": {"official/standard": ref}})
    for gain, want in [(0.0, (1.0, False)), (0.035, (2.0, False)), (0.0175, (2**0.5, False)), (0.0475, (2**1.5, False)), (0.06, (4.0, False)),
                       (0.1, (4.0, True)), (-0.02, (2**-0.5, False)), (-0.04, (0.5, False)), (-0.09, (0.5, True))]:
        m, capped = score.multiplier(season, "official", "standard", gain)
        assert (m, capped) == (pytest.approx(want[0]), want[1]), gain
        assert score.read_off(ref, gain) == (m, capped)  # the finals read their own curve by the same rule
    assert score.multiplier(season, "official", "large", 0.01) is None  # no curve measured, no multiplier
    assert score.read_off([[1.0, 0.0]], 0.01) is None


def test_a_curve_where_more_data_did_worse_is_said_not_to_rise():
    curve = measure.curve(TABLE, SHARES, {1.0: UNEDITED, 2.0: [student(0.03)], 4.0: [student(0.02)]}, PLAN)
    assert not measure.rises(curve)


def test_several_rounds_curves_average_point_by_point_with_their_spread():
    a = measure.curve(TABLE, SHARES, {0.5: [student(-0.04)], 1.0: UNEDITED, 2.0: [student(0.03)]}, PLAN)
    b = measure.curve(TABLE, SHARES, {1.0: UNEDITED, 2.0: [student(0.05)], 4.0: [student(0.08)]}, PLAN)
    mean = {p["multiple"]: p for p in measure.average([a, b])}
    assert sorted(mean) == [0.5, 1.0, 2.0, 4.0]
    assert mean[2.0]["gain"] == pytest.approx(0.04) and mean[2.0]["rounds"] == 2
    assert mean[2.0]["sd"] == pytest.approx(0.02 / math.sqrt(2)) and mean[2.0]["se"] == pytest.approx(0.01)
    assert mean[1.0]["gain"] == 0.0 and mean[1.0]["sd"] == 0.0
    assert mean[4.0]["rounds"] == 1 and mean[4.0]["sd"] is None and mean[4.0]["se"] is None  # one round says nothing of the spread
    assert measure.as_reference(measure.average([a, b])) == [[0.5, 0.04], [1.0, 0.0], [2.0, -0.04], [4.0, -0.08]]


def test_a_budgets_multiple_is_a_whole_scale_with_only_the_budget_changed():
    season = load_season("s2")
    base = season.scale("standard")
    double, half = measure.at_multiple(base, 2), measure.at_multiple(base, 0.5)
    assert measure.at_multiple(base, 1) is base
    assert (double.name, half.name, measure.at_multiple(base, 1.5).name) == ("standard-x2", "standard-x0p5", "standard-x1p5")
    assert double.train_tokens == 2 * base.train_tokens and half.train_tokens == base.train_tokens // 2
    assert dataclasses.replace(double, name=base.name, train_tokens=base.train_tokens) == base  # nothing else moved
    assert double.steps == pytest.approx(2 * base.steps, abs=1)  # and the schedule follows the steps (train.lr_at)
    for s in (double, half, measure.at_multiple(season.scale("large"), 4)):
        assert re.fullmatch(r"[a-z][a-z0-9-]{0,23}", s.name)  # what the server takes for a scale's name (server/farm.go)
        spec = measure.spec_of(s)
        assert set(spec) == {f.name for f in dataclasses.fields(Scale)} - {"name"}
        # A machine of the farm builds the scale from the spec as the server hands it on: an object of numbers.
        assert farm.scale_of(season, s.name, {k: float(v) for k, v in json.loads(json.dumps(spec)).items()}) == s
        assert len(json.dumps(spec)) < 2048


def test_a_budget_the_corpus_cannot_fill_reads_it_more_than_once():
    base = load_season("s2").scale("standard")
    corpus_tokens = int(2.5 * base.train_tokens)
    assert measure.passes(base, corpus_tokens) == pytest.approx(0.4)
    assert measure.passes(measure.at_multiple(base, 2), corpus_tokens) == pytest.approx(0.8)
    assert measure.passes(measure.at_multiple(base, 4), corpus_tokens) == pytest.approx(1.6)  # not four times the data


def test_rank_agreement():
    assert measure.spearman({"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 10, "b": 20, "c": 30, "d": 40}) == pytest.approx(1.0)
    assert measure.spearman({"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 4, "b": 3, "c": 2, "d": 1}) == pytest.approx(-1.0)
    assert measure.spearman({"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 1, "b": 2, "c": 4, "d": 3}) == pytest.approx(0.8)  # 1 - 6*2/(4*15)
    assert measure.spearman({"a": 1, "b": 2, "c": 3, "x": 9}, {"a": 3, "b": 2, "c": 1, "y": 9}) == pytest.approx(-1.0)  # only names in both
    assert measure.ranks([0.3, 0.1, 0.3, 0.2]) == [3.5, 1.0, 3.5, 2.0]  # equal values share the mean of their ranks
    assert measure.spearman({"a": 1, "b": 1, "c": 2}, {"a": 1, "b": 2, "c": 3}) == pytest.approx(math.sqrt(3) / 2)
    assert measure.spearman({"a": 1, "b": 2}, {"a": 1, "b": 2}) is None  # two names are no order
    assert measure.spearman({"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 2, "c": 3}) is None  # nor are three level ones


def test_noise_is_between_trainings_of_one_script_and_spread_between_curators():
    assert measure.pooled_sd([[1.0, 3.0], [10.0, 10.0], [5.0]]) == pytest.approx(1.0)  # (2 + 0) over 2 degrees of freedom
    assert measure.pooled_sd([[1.0], [2.0]]) is None and measure.sd([1.0]) is None
    assert measure.rms([3.0, 4.0, None]) == pytest.approx(math.sqrt(12.5)) and measure.rms([None]) is None
    curators = {"a": twice(0.01), "b": twice(0.02), "c": twice(0.03)}
    got = measure.size_stats(TABLE, SHARES, UNEDITED, curators, PLAN)
    assert got["gains"]["score"] == pytest.approx({"a": 0.01, "b": 0.02, "c": 0.03})
    s = got["measures"]["score"]
    assert s["noise"] == pytest.approx(0.001 * math.sqrt(2)) and s["spread"] == pytest.approx(0.01)
    assert s["spread_over_noise"] == pytest.approx(0.01 / (0.001 * math.sqrt(2)))
    assert got["measures"]["pillar:news"]["spread"] == pytest.approx(0.01)
    assert got["trainings"] == {"a": 2, "b": 2, "c": 2} and got["unedited"] == 2
    # One training each: a spread, and no noise to put it over.
    once = measure.size_stats(TABLE, SHARES, UNEDITED, {k: v[:1] for k, v in curators.items()}, PLAN)["measures"]["score"]
    assert once["noise"] is None and once["spread_over_noise"] is None and once["spread"] == pytest.approx(0.01)


def _sizes(large_order=("a", "b", "d", "c")) -> dict:
    """Four curators at two sizes. At the rounds' own size they stand a, b, c, d from the worst;
    one size up, as `large_order` has them. The watched part "arith" orders them as the score
    does and spreads them well; "morph" has them backwards."""
    gain = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
    standard = {n: twice(g, arith=2 * g, morph=-g) for n, g in gain.items()}
    large = {n: twice(0.02 * (i + 1), noise=0.002, arith=0.04 * (i + 1), morph=-0.01 * (i + 1)) for i, n in enumerate(large_order)}
    return {"standard": measure.size_stats(TABLE, SHARES, UNEDITED, standard, PLAN), "large": measure.size_stats(TABLE, SHARES, UNEDITED, large, PLAN)}


def test_the_audit_says_whether_the_order_carried_to_the_next_size():
    rep = measure.audit(_sizes())
    assert rep["sizes"] == ["standard", "large"]
    s = rep["measures"]["score"]
    assert s["carried"] == {"standard>large": pytest.approx(0.8)}  # c and d changed places
    assert s["sizes"]["standard"]["spread_over_noise"] == pytest.approx(math.sqrt(5 / 3) * 0.01 / (0.001 * math.sqrt(2)))
    assert s["sizes"]["large"]["noise"] == pytest.approx(0.002 * math.sqrt(2))
    assert "with_score" not in s and "passes" not in s  # the test is for watched parts
    assert rep["curators"]["d"] == {"standard": pytest.approx(0.04), "large": pytest.approx(0.06)}
    arith, morph = rep["measures"]["watch:arith"], rep["measures"]["watch:morph"]
    assert arith["with_score"] == pytest.approx(1.0) and arith["passes"] is True
    assert morph["with_score"] == pytest.approx(-1.0) and morph["passes"] is False  # it spreads curators, but not as the score does
    assert morph["carried"] == {"standard>large": pytest.approx(0.8)}
    assert measure.audit(_sizes(("a", "b", "c", "d")))["measures"]["score"]["carried"] == {"standard>large": pytest.approx(1.0)}


def test_the_seasons_test_for_a_watched_part():
    assert measure.passes_test(1.6, 0.5) is True
    assert measure.passes_test(1.5, 0.9) is False  # above 1.5, not at it
    assert measure.passes_test(3.0, 0.49) is False
    assert measure.passes_test(None, 0.9) is None and measure.passes_test(2.0, None) is None


def test_a_watched_part_earns_a_place_only_over_twenty_rounds():
    one = {**measure.audit(_sizes()), "round": "s2-o00001"}
    nineteen = measure.pool([one] * 19)
    assert nineteen["rounds"] == 19
    assert nineteen["measures"]["watch:arith"]["verdict"] == "too few rounds: 19 of 20, passing so far"
    assert nineteen["measures"]["watch:morph"]["verdict"] == "too few rounds: 19 of 20, not passing so far"
    twenty = measure.pool([one] * 20)
    arith, morph, s = twenty["measures"]["watch:arith"], twenty["measures"]["watch:morph"], twenty["measures"]["score"]
    assert arith["verdict"] == "earns a place" and arith["measured"] == 20 and arith["with_score"] == pytest.approx(1.0)
    assert arith["spread_over_noise"] == pytest.approx(one["measures"]["watch:arith"]["sizes"]["standard"]["spread_over_noise"])
    assert morph["verdict"] == "stays watched"
    assert "verdict" not in s and s["carried"]["standard>large"] == {"rounds": 20, "mean": pytest.approx(0.8)}
    assert s["sizes"]["large"]["spread_over_noise"] == pytest.approx(one["measures"]["score"]["sizes"]["large"]["spread_over_noise"])


def test_pooled_rounds_weigh_noise_and_spread_as_variances_and_average_rank_agreement():
    a = {**measure.audit(_sizes()), "round": "s2-o00001"}
    b = {**measure.audit(_sizes(("a", "b", "c", "d"))), "round": "s2-o00002"}
    # A round where the part was too noisy to tell curators apart: it counts against it.
    gain = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
    noisy = {n: [student(g, arith=2 * g + d) for d in (0.1, -0.1)] for n, g in gain.items()}
    c = {**measure.audit({"standard": measure.size_stats(TABLE, SHARES, UNEDITED, noisy, PLAN)}), "round": "s2-o00003"}
    got = measure.pool([a, b, c])
    assert got["measures"]["score"]["carried"]["standard>large"] == {"rounds": 2, "mean": pytest.approx(0.9)}
    arith = got["measures"]["watch:arith"]
    quiet, loud = 0.001 * math.sqrt(2), 0.1 * math.sqrt(2)
    spread = 2 * math.sqrt(5 / 3) * 0.01
    assert arith["measured"] == 3 and arith["spread_over_noise"] == pytest.approx(spread / math.sqrt((2 * quiet**2 + loud**2) / 3))
    assert arith["spread_over_noise"] < measure.PROMOTE_SPREAD_OVER_NOISE < a["measures"]["watch:arith"]["sizes"]["standard"]["spread_over_noise"]
    assert arith["verdict"] == "too few rounds: 3 of 20, not passing so far"
    assert got["measures"]["score"]["sizes"]["large"]["rounds"] == 2  # the third round was audited at its own size only


def _season(tokens_per_second: int = 80_000, vocab: int = 8192):
    season = load_season("s2")
    return dataclasses.replace(season, raw={**season.raw, "runner": {**season.raw["runner"], "gpu_tokens_per_second": tokens_per_second},
                                            "tokenizer": {**season.raw["tokenizer"], "vocab_size": vocab}})


STUDENT = Scale(name="standard", corpus_units=8, train_tokens=160_000_000, n_layer=8, n_head=8, d_model=512, ctx=512, batch_size=32, lr=0.001, edit_seconds=1200)


def test_what_a_training_takes_follows_from_the_seasons_floor_and_the_students_shape():
    # The benchmark's model: 12 layers of 768 over 256 tokens, no output layer. The student: 8 of 512 over 512, and its 8192 words.
    bench = 12 * (24 * 768**2 + 4 * 256 * 768)
    own = 8 * (24 * 512**2 + 4 * 512 * 512) + 2 * 8192 * 512
    assert (bench, own) == (179_306_496, 67_108_864)
    assert measure.work_per_token(12, 768, 256) == bench and measure.work_per_token(8, 512, 512, 8192) == own
    want = 160_000_000 * 3 * own / (80_000 * bench)  # 37 minutes
    assert measure.train_seconds(_season(), STUDENT) == pytest.approx(want) and 2240 < want < 2250
    assert measure.train_seconds(_season(), measure.at_multiple(STUDENT, 4)) == pytest.approx(4 * want)
    assert measure.train_seconds(_season(160_000), STUDENT) == pytest.approx(want / 2)  # a floor twice as fast
    with pytest.raises(ValueError, match="floor"):
        measure.train_seconds(_season(0), STUDENT)


def test_a_student_the_floor_trains_in_time_is_held_as_usual_and_a_longer_one_for_as_long_as_it_needs():
    season = _season()
    assert measure.hold_seconds(season, STUDENT) == 0  # within the server's three quarters of an hour
    double = measure.hold_seconds(season, measure.at_multiple(STUDENT, 2))
    took = measure.train_seconds(season, measure.at_multiple(STUDENT, 2))
    assert double % 300 == 0 and 1.25 * took + 3600 <= double < 1.25 * took + 3900
    assert measure.hold_seconds(season, measure.at_multiple(STUDENT, 400)) == 24 * 3600  # the longest the server allows
    # Two bytes a token, three times over while the tokens are built: the 4x large student is 14 GB of them.
    assert measure.memory_gb(measure.at_multiple(dataclasses.replace(STUDENT, train_tokens=640_000_000), 4)) == pytest.approx(6 * 2.56e9 / 2**30)

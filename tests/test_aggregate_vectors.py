"""The server scores the exam from a table of the items and their gains (server/aggregate.go):
the table must give what round_score gives, and the server's test vectors must be what the harness
gives now, so the two cannot drift apart without a test failing."""

import json
import random
import sys
from pathlib import Path

import pytest

from stet.exam import gen
from stet.exam.aggregate import item_table, round_score, score_from_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def _same(table_score: dict, full: dict) -> None:
    assert table_score["score"] == full["score"]
    assert {p: round(v, 5) for p, v in table_score["pillars"].items()} == {p: v["score"] for p, v in full["pillars"].items()}


def test_the_score_from_the_table_is_round_scores():
    from aggregate_vectors import draw_gains, draw_shares, fresh_items, random_case

    for seed in range(300):
        items, g, shares = random_case(random.Random(seed))
        for sh in (shares, None):
            _same(score_from_table(item_table(items), g, sh), round_score(items, g, shares=sh))
    rng = random.Random(7)
    items = gen.generate(b"table", per_corner=1) + fresh_items(rng)
    for style in ("normal", "outliers", "ties", "heavy"):
        g = draw_gains(rng, len(items), style)
        shares = draw_shares(rng, [it.lang for it in items])
        _same(score_from_table(item_table(items), g, shares), round_score(items, g, shares=shares))
    assert item_table(items[:1])[0].keys() == {"pillar", "part", "lang", "stratum"}
    with pytest.raises(ValueError):
        score_from_table(item_table(items[:2]), [0.1], shares)


def test_the_servers_vectors_are_current():
    from aggregate_vectors import OUT, build

    committed, now = json.loads(OUT.read_text()), build()
    hint = "run: python tools/aggregate_vectors.py"
    assert committed["tables"] == now["tables"], hint
    assert [{k: v for k, v in c.items() if k not in ("score", "pillars")} for c in committed["cases"]] == \
        [{k: v for k, v in c.items() if k not in ("score", "pillars")} for c in now["cases"]], hint
    for old, new in zip(committed["cases"], now["cases"], strict=True):
        # the same to the last bit on Python 3.12 and later; before, sum() added a little differently
        assert old["score"] == pytest.approx(new["score"], rel=0, abs=1e-12), (old["name"], hint)
        assert old["pillars"] == pytest.approx(new["pillars"], rel=0, abs=1e-12), (old["name"], hint)
    assert len(now["cases"]) >= 30 and all(c["shares"] for c in now["cases"])  # the server is always given shares

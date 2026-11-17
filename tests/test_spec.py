import hashlib

import pytest

from stet.spec import (
    load_season,
    official_round,
    parse_round_id,
    practice_round,
    prf_u64,
    round_from_description,
    sample_without_replacement,
)

CRAWL = {"crawl": "CC-MAIN-2026-99", "paths_sha256": "66" * 32, "n_shards": 100000}


def test_prf_is_stable():
    # Pinned: any other implementation (the Go server) must reproduce these.
    assert prf_u64(b"seed", "shard", 0) == int.from_bytes(hashlib.sha256(b"seed|shard|0").digest()[:8], "big")
    assert sample_without_replacement(b"x", "a", 10, 10) != list(range(10))
    assert sorted(sample_without_replacement(b"x", "a", 10, 10)) == list(range(10))


def test_practice_rounds_are_web_text_from_a_pinned_crawl():
    season = load_season("s0")
    scale = season.scale("micro")
    a, b = practice_round(season, 0), practice_round(season, 1)
    cfg = a.pool_cfg
    assert a.id == "s0-p00000" and cfg["kind"] == "cc-wet" and cfg["crawl"] == "CC-MAIN-2026-43"
    n = cfg["n_shards"]
    assert a.shard_indices(scale, n) == practice_round(season, 0).shard_indices(scale, n)
    assert a.shard_indices(scale, n) != b.shard_indices(scale, n)
    assert len(set(a.shard_indices(scale, n))) == scale.corpus_units * cfg["shards_per_unit"]
    assert a.split == "practice"


def test_gym_rounds_are_a_different_dataset():
    season = load_season("s0")
    g = practice_round(season, 0, "gym")
    assert g.id == "s0-g00000" and g.pool_cfg["kind"] == "dclm" and len(season.pool("gym")) == 1024
    assert g.seed != practice_round(season, 0).seed and g.split == "practice"
    assert parse_round_id("s0-g00012") == ("s0", "gym", 12)
    assert round_from_description(season, g.describe()).seed == g.seed


def test_official_rounds_depend_on_the_beacon_and_the_crawl():
    season = load_season("s0")
    x = official_round(season, 3, {"round": 100, "randomness": "ab" * 32}, CRAWL)
    y = official_round(season, 3, {"round": 100, "randomness": "cd" * 32}, CRAWL)
    z = official_round(season, 3, {"round": 100, "randomness": "ab" * 32}, {**CRAWL, "crawl": "CC-MAIN-2026-39"})
    assert len({x.seed, y.seed, z.seed}) == 3 and x.id == "s0-o00003"
    assert x.pool_cfg["kind"] == "cc-wet" and x.pool_cfg["crawl"] == "CC-MAIN-2026-99" and x.split == "official"
    scale = season.scale("micro")
    assert len(x.shard_indices(scale, CRAWL["n_shards"])) == scale.corpus_units * x.pool_cfg["shards_per_unit"]


def test_a_round_description_cannot_lie_about_its_seed():
    season = load_season("s0")
    x = official_round(season, 3, {"round": 100, "randomness": "ab" * 32}, CRAWL)
    d = x.describe()
    assert round_from_description(season, d).seed == x.seed
    d["beacon"] = {"round": 100, "randomness": "cd" * 32}
    with pytest.raises(ValueError):
        round_from_description(season, d)


def test_the_corpus_order_is_a_seeded_shuffle():
    season = load_season("s0")
    a = practice_round(season, 0)
    order = a.doc_order(1000)
    assert sorted(order) == list(range(1000)) and order != list(range(1000))
    assert order == practice_round(season, 0).doc_order(1000)
    assert order != practice_round(season, 1).doc_order(1000)
    # Pinned, so that another implementation can check itself against this one.
    keys = [prf_u64(a.seed, "order", i) for i in range(5)]
    assert a.doc_order(5) == sorted(range(5), key=keys.__getitem__)

import hashlib

import pytest

from stet.spec import (
    load_season,
    official_round,
    practice_round,
    prf_u64,
    round_from_description,
    sample_without_replacement,
)

CRAWL = {"crawl": "CC-MAIN-2026-43", "paths_sha256": "66" * 32, "n_shards": 100000}


def test_prf_is_stable():
    # Pinned: any other implementation (the Go server) must reproduce these.
    assert prf_u64(b"seed", "shard", 0) == int.from_bytes(hashlib.sha256(b"seed|shard|0").digest()[:8], "big")
    assert sample_without_replacement(b"x", "a", 10, 10) != list(range(10))
    assert sorted(sample_without_replacement(b"x", "a", 10, 10)) == list(range(10))


def test_practice_rounds_are_deterministic_and_distinct():
    season = load_season("s0")
    scale = season.scale("micro")
    a, b = practice_round(season, 0), practice_round(season, 1)
    n = len(season.pool("practice"))
    assert n == 1024
    assert a.shard_indices(scale, n) == practice_round(season, 0).shard_indices(scale, n)
    assert a.shard_indices(scale, n) != b.shard_indices(scale, n)
    assert len(set(a.shard_indices(scale, n))) == scale.corpus_units
    assert a.id == "s0-p00000" and a.pool_cfg["kind"] == "dclm"


def test_official_rounds_depend_on_the_beacon_and_the_crawl():
    season = load_season("s0")
    x = official_round(season, 3, {"round": 100, "randomness": "ab" * 32}, CRAWL)
    y = official_round(season, 3, {"round": 100, "randomness": "cd" * 32}, CRAWL)
    z = official_round(season, 3, {"round": 100, "randomness": "ab" * 32}, {**CRAWL, "crawl": "CC-MAIN-2026-39"})
    assert len({x.seed, y.seed, z.seed}) == 3 and x.id == "s0-o00003"
    # official rounds draw from a different dataset than practice rounds
    assert x.pool_cfg["kind"] == "cc-wet" and x.pool_cfg["crawl"] == "CC-MAIN-2026-43"
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

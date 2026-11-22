#!/usr/bin/env python3
"""Write the season facts the server needs (it is Go and does not import the harness).

    python tools/export_season.py > server/season.json

A test (tests/test_export.py) fails if the committed file is stale.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stet.spec import CURRENT_SEASON, GRADES_SINCE, load_season, official_round, prf_u64


def export(season_id: str = CURRENT_SEASON) -> dict:
    season = load_season(season_id)
    raw = season.raw
    # A fixed example, so the server's tests can prove it derives seeds exactly as the harness does.
    beacon = {"round": 123456, "randomness": "ab" * 32}
    crawl = {"crawl": "CC-MAIN-2026-43", "paths_sha256": "cd" * 32, "n_shards": 100000}
    rnd = official_round(season, 7, beacon, crawl)
    scale = season.scale(season.official_scale)
    grid_round = season.beacon_round_at(1_800_000_000 // raw["draw_grid_seconds"] * raw["draw_grid_seconds"])
    return {
        "id": season.id,
        "title": season.title,
        "digest": season.digest,
        "harness": GRADES_SINCE,  # the oldest stet the server takes work from
        "rules_version": raw["rules_version"],
        "official_scale": season.official_scale,
        "scales": raw["scales"],
        "ink_frac": season.ink_frac,
        "ink_tokens": {name: season.ink_tokens(season.scale(name)) for name in raw["scales"]},
        "min_copy_bytes": season.min_copy_bytes,
        "max_doc_bytes": season.max_doc_bytes,
        "script": raw["script"],
        "decontam": raw["decontam"],
        "exam_sources": season.exam_sources,
        # What the score is made of: server/aggregate.go scores every round from this.
        "exam": {
            "groups": raw["exam"]["groups"],
            "pillars": [{"name": p, "group": v["group"], "parts": v["parts"], "about": v["about"]} for p, v in raw["exam"]["pillars"].items()],
            "clip": raw["exam"]["aggregate"]["clip"], "trim": raw["exam"]["aggregate"]["trim"], "small": raw["exam"]["aggregate"]["small"],
            "answer_floor_bytes": season.answer_floor_bytes,
        },
        "beacon": raw["beacon"],
        "draw_grid_seconds": raw["draw_grid_seconds"],
        "pools": {k: {f: v for f, v in pool.items() if f != "manifest"} for k, pool in raw["pools"].items()},
        "grade_devices": raw["grade_devices"],
        "environment": raw.get("environment", {}),
        "assets": raw.get("assets", {}),
        "machine": raw.get("machine", {}),
        "runner": raw.get("runner", {}),
        "reference": raw.get("reference", {}),
        "reference_curves": raw.get("reference_curves", {}),
        "tolerance": raw.get("tolerance", {}),
        "noise": raw.get("noise", {}),
        "record": raw["record"],
        "train": raw["train"],
        "tokenizer": {k: raw["tokenizer"][k] for k in ("vocab_size",)},
        "parity": {
            "index": 7, "beacon": beacon, "crawl": crawl, "seed": rnd.seed.hex(),
            "shards": rnd.shard_indices(scale, crawl["n_shards"]),
            "order": rnd.doc_order(12),
            "prf": str(prf_u64(b"seed", "shard", 0)),
            "grid": {"round": grid_round, "slot": season.draw_slot(grid_round), "off": season.draw_slot(grid_round + 1)},
        },
    }


if __name__ == "__main__":
    json.dump(export(*sys.argv[1:2]), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")

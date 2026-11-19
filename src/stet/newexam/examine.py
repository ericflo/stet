"""The examiner: an official round's exam, made after its draw (season 1 on).

Once the round's harvest window has closed and its sources have published all of it, the
organisers' examiner:

1. harvests the fresh text written in the window;
2. builds the round's corpus and measures its languages;
3. chooses the fresh items, and drops those the corpus already holds;
4. generates the other items from the round's seed;
5. posts them to the server, with what each counts toward.

Grades start then. Everything here is a pure function of public inputs and the harvest, whose
sha256 is posted with the exam: anyone can make the same exam again from the same snapshot.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from .farm import pack_items
from .items import Item, digest


def build(rnd, scale, draw_at: int, work: Path, log=print) -> dict:
    """The round's exam, as the server takes it. Kept under work, so that a second try costs
    nothing it does not have to."""
    from .. import data
    from . import gen
    from .aggregate import item_table
    from .decontam import seen
    from .fresh import read_snapshot
    from .fresh.harvest import harvest_window
    from .fresh.langs import shares
    from .fresh.select import select

    season = rnd.season
    cfg = season.raw["exam"]
    if gen.VERSION != int(cfg["generator_version"]):
        raise RuntimeError(f"season {season.id} sets the generators at version {cfg['generator_version']}, this copy has {gen.VERSION}: upgrade stet")
    window = int(cfg["fresh"]["window_seconds"])
    work.mkdir(parents=True, exist_ok=True)
    snapshot = work / "fresh.jsonl"
    if not snapshot.exists():
        harvest_window(rnd.seed, draw_at, draw_at + window, snapshot, log=log)
    import hashlib

    snap_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    _, docs = read_snapshot(snapshot)
    corpus = data.corpus(rnd, scale)
    mix = shares(corpus, rnd.seed)  # every language in proportion to the round's own corpus
    log(f"  the corpus's languages: {', '.join(f'{k} {v:.1f}%' for k, v in list(mix.items())[:8])}, and {max(0, len(mix) - 8)} more")
    fresh = select(docs, rnd.seed, int(cfg["fresh"]["items"]), per_corner=int(cfg["fresh"]["per_corner"]), shares=mix)
    mark = work / f"seen-{digest(fresh)[:12]}.json"
    if mark.exists():
        dropped = set(json.loads(mark.read_text()))
    else:
        dropped = seen(fresh, corpus, season, log=log)
        mark.write_text(json.dumps(sorted(dropped)))
    if dropped:
        log(f"  {len(dropped)} fresh items are in the round's corpus already, and are dropped")
    fresh = [it for i, it in enumerate(fresh) if i not in dropped]
    items: list[Item] = gen.generate(rnd.seed, per_corner=int(cfg["per_corner"])) + fresh
    log(f"  the exam: {len(items)} items, {len(fresh)} of them fresh")
    return {"round": rnd.id, "items_digest": digest(items), "items_file": base64.b64encode(pack_items(items)).decode(),
            "table": item_table(items), "languages": dict(list(mix.items())[:500]), "snapshot_sha256": snap_sha, "fresh": len(fresh)}


def post(exam: dict, key) -> dict:
    from ..client import request

    return request("POST", f"/api/v1/admin/rounds/{exam['round']}/exam", exam, key=key, timeout=600)


def examine(rnd, scale, draw_at: int, work: Path, key, log=print) -> dict:
    """Make a round's exam and post it."""
    t0 = time.time()
    exam = build(rnd, scale, draw_at, work, log=log)
    out = post(exam, key)
    log(f"  posted {rnd.id}'s exam: {len(exam['table'])} items ({time.time() - t0:.0f}s)")
    return out

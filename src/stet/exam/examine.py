"""The examiner: an official round's exam, made at its draw.

The organisers' examiner harvests every hour of the clock as it settles (fresh/harvest.py). At
a round's draw the server names the hours the round's seed chose, and the examiner:

1. puts those hours' harvests together;
2. builds the round's corpus and measures its languages;
3. chooses the fresh items, and drops those the corpus already holds;
4. generates the other items from the round's seed;
5. posts them to the server, with what each counts toward.

Grades start then. Everything here is a pure function of public inputs and the harvests, whose
sha256s were posted before the draw: anyone can make the same exam again from the same
snapshots.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from .farm import pack_items
from .items import Item, digest

# Sources whose text is openly licensed, and is published with a settled round as it is. Every
# other source's items are published as their source, id and hash.
OPEN = ("gen:", "wikipedia:", "wikipedia-added:", "wikipedia-talk:")


def public_items(items: list[Item]) -> list[dict]:
    """A settled round's exam as it is published: the generated items and the openly licensed
    fresh text whole, every other item as what it was (source, kind, format, language, corner)
    and its id. The ids, in order, hash to the round's exam digest, so the published file can be
    checked against it."""
    out = []
    for it in items:
        if it.source.startswith(OPEN):
            out.append(it.to_dict())
        else:
            out.append({"withheld": True, "id": it.id, "source": it.source, "kind": it.kind, "fmt": it.fmt, "lang": it.lang, "corner": it.corner})
    return out


def read_public(raw: bytes, want: str) -> list[Item]:
    """The items of a published exam that it gives whole, checked against the round's digest."""
    import hashlib

    h = hashlib.sha256()
    kept = []
    for line in raw.decode().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("withheld"):
            h.update(str(row["id"]).encode())
        else:
            it = Item.from_dict(row)
            h.update(it.id.encode())
            kept.append(it)
    if h.hexdigest() != want:
        raise ValueError("the published exam does not hash to the round's exam digest")
    return kept


def assemble(rnd, scale, snapshots: list[tuple[int, Path]], work: Path, log=print, measure: bool = True):
    """A round's exam from the harvests of its hours, (hour, the snapshot's path) in any order:
    its items, how many of them are fresh, the languages' shares, and each hour's sha256 in the
    hours' order. An official round's languages are measured on its own corpus; a practice
    round's (measure=False) are the published table's. Kept under work, so that a second try
    costs nothing it does not have to."""
    import hashlib

    from .. import data
    from . import gen
    from .decontam import seen
    from .fresh import read_snapshot
    from .fresh.langs import shares
    from .fresh.select import SHARES, select

    season = rnd.season
    cfg = season.raw["exam"]
    if gen.VERSION != int(cfg["generator_version"]):
        raise RuntimeError(f"season {season.id} sets the generators at version {cfg['generator_version']}, this copy has {gen.VERSION}: upgrade stet")
    if not snapshots:
        raise ValueError(f"{rnd.id} has no harvest to make its exam from")
    work.mkdir(parents=True, exist_ok=True)
    snapshots = sorted(snapshots)
    docs, shas, had = [], [], set()
    for _, path in snapshots:
        shas.append(hashlib.sha256(path.read_bytes()).hexdigest())
        for d in read_snapshot(path)[1]:
            if (d.source, d.id) not in had:  # a thread that runs over two hours is one document
                had.add((d.source, d.id))
                docs.append(d)
    corpus = data.corpus(rnd, scale)
    mix = dict(SHARES)
    if measure:
        mix = shares(corpus, rnd.seed)  # every language in proportion to the round's own corpus
        log(f"  the corpus's languages: {', '.join(f'{k} {v:.1f}%' for k, v in list(mix.items())[:8])}, and {max(0, len(mix) - 8)} more")
    from .aggregate import SOURCE_KIND, WATCH, plan_of

    plan = plan_of(season)
    watched = {k for k in set(SOURCE_KIND.values()) if plan.pillar_of(k) == WATCH}  # kinds of writing that count for nothing
    fresh = select(docs, rnd.seed, int(cfg["fresh"]["items"]), per_corner=int(cfg["fresh"]["per_corner"]), shares=mix, watched=watched)
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
    return items, len(fresh), mix, shas


def snapshot_name(shas: list[str]) -> str:
    """One name for what an exam was made from: its hours' hashes in the hours' order, hashed."""
    import hashlib

    return hashlib.sha256("".join(shas).encode()).hexdigest()


def build(rnd, scale, snapshots: list[tuple[int, Path]], work: Path, log=print) -> dict:
    """The round's exam, as the server takes it, from the harvests of its hours."""
    import zstandard

    from .aggregate import item_table

    snapshots = sorted(snapshots)
    items, fresh, mix, shas = assemble(rnd, scale, snapshots, work, log=log)

    public = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public_items(items)).encode()
    return {"round": rnd.id, "items_digest": digest(items), "items_file": base64.b64encode(pack_items(items)).decode(),
            "public_file": base64.b64encode(zstandard.ZstdCompressor(level=10).compress(public)).decode(),
            "table": item_table(items), "languages": dict(list(mix.items())[:500]), "snapshot_sha256": snapshot_name(shas), "fresh": fresh,
            "hours": [{"hour": h, "sha256": sha} for (h, _), sha in zip(snapshots, shas, strict=True)]}


def post(exam: dict, key) -> dict:
    from ..client import request

    return request("POST", f"/api/v1/admin/rounds/{exam['round']}/exam", exam, key=key, timeout=600)


def examine(rnd, scale, snapshots: list[tuple[int, Path]], work: Path, key, log=print) -> dict:
    """Make a round's exam from its hours' harvests and post it."""
    t0 = time.time()
    exam = build(rnd, scale, snapshots, work, log=log)
    out = post(exam, key)
    log(f"  posted {rnd.id}'s exam: {len(exam['table'])} items ({time.time() - t0:.0f}s)")
    return out

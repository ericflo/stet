"""The examiner: an official round's exam, made at its draw.

The organisers' examiner harvests every hour of the clock as it settles (fresh/harvest.py). At
a round's draw the server names the hours the round's seed chose, and the examiner:

1. puts those hours' harvests together;
2. builds the round's corpus and measures its languages;
3. chooses the fresh twin-context passages and writes the reasoning items once with the pinned
   teacher model, then drops any item whose text the corpus already holds;
4. omits any pillar that falls below `min_pillar_items` (its share goes to the rest of its
   group, refusing only when an entire half is missing);
5. posts the exam to the server, with what each item counts toward.

Grades start then. Human-written twin-context items are derived deterministically from the
committed hourly harvests; model-written reasoning items are written once at the draw and
published in full at settlement (teacher lead-in and answer for every item, plus full excerpt for
openly licensed sources) so anyone can inspect every item and verify its digest against the
round's committed `exam_digest`.
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections import Counter
from pathlib import Path

from .farm import pack_items
from .fresh.reason import FatalExamError
from .items import Item, digest


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _pillar_counts(items: list[Item], plan) -> Counter[str]:
    from .aggregate import part_of

    counts: Counter[str] = Counter()
    for it in items:
        counts[plan.pillar_of(part_of(it))] += 1
    return counts


def _filter_pillar_support(items: list[Item], season, rnd_id: str, log=print) -> list[Item]:
    """Drop any pillar below `min_pillar_items` so its share transfers cleanly to the rest of its
    group (matching RULES.md), and refuse only when an entire declared group has zero items left."""
    from .aggregate import part_of, plan_of

    min_items = int((season.raw.get("exam") or {}).get("min_pillar_items", 0))
    plan = plan_of(season)
    counts = _pillar_counts(items, plan)
    omitted: list[tuple[str, int]] = []
    if min_items > 0:
        under = {p for p in plan.pillars if counts.get(p, 0) < min_items}
        if under:
            omitted = [(p, counts.get(p, 0)) for p in plan.pillars if p in under]
            items = [it for it in items if plan.pillar_of(part_of(it)) not in under]
            detail = ", ".join(f"{p}={n}" for p, n in omitted)
            log(f"  omitting under-supported pillars (<{min_items} items; share goes to group): {detail}")
    remaining = _pillar_counts(items, plan)
    empty_groups = [
        g
        for g, w in plan.groups.items()
        if w > 0 and not any(remaining.get(p, 0) > 0 for p, pg in plan.pillars.items() if pg == g)
    ]
    if empty_groups:
        detail = ", ".join(f"{p}={counts.get(p, 0)}" for p in plan.pillars)
        raise FatalExamError(
            f"{rnd_id}: exam group(s) {', '.join(empty_groups)} have no pillars meeting minimum of {min_items} items ({detail})"
        )
    return items


def _reasoning_cache_usable(reasoning: list[Item], season) -> bool:
    """Whether a cached reasoning synthesis result is complete enough to reuse.

    Matches `_filter_pillar_support`: a synthesis where at least one reasoning pillar meets
    `min_pillar_items` is valid to reuse (a legitimately short kind is omitted rather than
    discarding the entire 20-minute build on restart)."""
    from .aggregate import REASONING_KINDS, plan_of

    if not reasoning:
        return False
    min_items = int((season.raw.get("exam") or {}).get("min_pillar_items", 0))
    if min_items <= 0:
        return True
    plan = plan_of(season)
    counts = _pillar_counts(reasoning, plan)
    r_pillars = [plan.pillar_of(k) for k in REASONING_KINDS if k in plan.parts]
    return any(counts.get(p, 0) >= min_items for p in set(r_pillars))


def _round_init_sha256(rnd, scale) -> str:
    """Authoritative round initial weights SHA-256 on CPU when the season has a pinned student and torch is installed."""
    if not rnd.season.raw.get("student"):
        return ""
    try:
        import torch

        from ..train import build_model, weights_sha256

        model = build_model(scale, rnd.season, rnd.init_seed, torch.device("cpu"))
        sha = weights_sha256(model)
        del model
        return sha
    except Exception:
        return ""

# Sources whose text is openly licensed, and is published with a settled round as it is. Every
# other source's items withhold the third-party source excerpt while still publishing the
# organisers' own generated teacher lead-in and answer for every model-written reasoning item.
OPEN = (
    "gen:",
    "wikipedia:", "wikipedia-added:", "wikipedia-talk:",
    "explain:wikipedia", "infer:wikipedia", "synthesis:wikipedia",
)
REASONING_PREFIXES = ("explain:", "infer:", "synthesis:")


def public_items(items: list[Item]) -> list[dict]:
    """A settled round's exam as it is published:
    - generated items and openly licensed Wikipedia items whole;
    - third-party model-written reasoning items with the organisers' own generated lead-in (`prompt`),
      answer (`cont`), `wrapper_id`, and `excerpt_sha256` published in full (withholding only the
      third-party excerpt);
    - third-party human-written items as their metadata and item id.

    Every row's id, in order, hashes to the round's exam digest, so anyone can check the published
    file against `exam_digest` and rebuild any withheld reasoning item from its source excerpt."""
    import hashlib

    from .fresh.reason import split_wrapped_ctx

    out = []
    for it in items:
        if it.source.startswith(OPEN):
            out.append(it.to_dict())
        elif it.source.startswith(REASONING_PREFIXES):
            task_type = it.source.split(":", 1)[0]
            row = {
                "withheld": True,
                "id": it.id,
                "source": it.source,
                "kind": it.kind,
                "fmt": it.fmt,
                "lang": it.lang,
                "corner": it.corner,
                "cont": it.cont,
            }
            if split := split_wrapped_ctx(it.ctx, task_type):
                excerpt, prompt_line, wrapper_id = split
                row["withheld_excerpt"] = True
                row["prompt"] = prompt_line
                row["wrapper_id"] = wrapper_id
                row["excerpt_sha256"] = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
                row["excerpt_chars"] = len(excerpt)
            out.append(row)
        else:
            out.append({"withheld": True, "id": it.id, "source": it.source, "kind": it.kind, "fmt": it.fmt, "lang": it.lang, "corner": it.corner})
    return out


def rebuild_withheld_reasoning(row: dict, excerpt: str) -> Item:
    """Reconstruct a settled non-Wikipedia reasoning item from its published teacher text and
    source excerpt, verifying both `excerpt_sha256` and the exact item `id`."""
    import hashlib

    from .fresh.reason import wrap_prompt

    if "prompt" not in row or "cont" not in row or "wrapper_id" not in row:
        raise ValueError("published row does not carry withheld reasoning fields")
    got_sha = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    if row.get("excerpt_sha256") and got_sha != row["excerpt_sha256"]:
        raise ValueError(f"excerpt sha256 mismatch: got {got_sha}, want {row['excerpt_sha256']}")
    task_type = str(row["source"]).split(":", 1)[0]
    ctx = wrap_prompt(excerpt, str(row["prompt"]), task_type, int(row["wrapper_id"]))
    it = Item(
        kind=str(row["kind"]),
        corner=str(row["corner"]),
        fmt=str(row["fmt"]),
        lang=str(row["lang"]),
        source=str(row["source"]),
        ctx=ctx,
        cont=str(row["cont"]),
    )
    if it.id != row["id"]:
        raise ValueError(f"rebuilt item id {it.id} does not match published id {row['id']}")
    return it


def read_public_info(raw: bytes, want: str) -> tuple[list[Item], int]:
    """The items of a published exam that it gives whole, plus the total row count (including
    withheld third-party items), checked against the round's digest."""
    import hashlib

    h = hashlib.sha256()
    kept = []
    total = 0
    for line in raw.decode().split("\n"):  # the newline the exam was written with, and no other break
        if not line:
            continue
        total += 1
        row = json.loads(line)
        if row.get("withheld"):
            h.update(str(row["id"]).encode())
        else:
            it = Item.from_dict(row)
            h.update(it.id.encode())
            kept.append(it)
    if h.hexdigest() != want:
        raise ValueError("the published exam does not hash to the round's exam digest")
    return kept, total


def read_public(raw: bytes, want: str) -> list[Item]:
    """The items of a published exam that it gives whole, checked against the round's digest."""
    return read_public_info(raw, want)[0]


def assemble(
    rnd,
    scale,
    snapshots: list[tuple[int, Path]],
    work: Path,
    log=print,
    measure: bool = True,
    synthesize: bool = True,
    check_coverage: bool | None = None,
):
    """A round's exam from the harvests of its hours, (hour, the snapshot's path) in any order:
    its items, how many of them are fresh, the languages' shares, and each hour's sha256 in the
    hours' order. An official round's languages are measured on its own corpus; a practice
    round's (measure=False) are the published table's. Kept under work, so that a second try
    costs nothing it does not have to."""
    import hashlib

    import zstandard

    from .. import data
    from . import gen
    from .decontam import seen
    from .farm import unpack_items
    from .fresh import read_snapshot
    from .fresh.langs import shares
    from .fresh.reason import synthesize_reasoning
    from .fresh.select import SHARES, select, select_context

    season = rnd.season
    cfg = season.raw["exam"]
    if int(cfg.get("generator_version", 0)) > gen.VERSION:
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
    from .aggregate import plan_of

    # Only the kinds of writing the season names: a document of any other kind is not on the exam.
    chooser = select_context if cfg["fresh"].get("mode") == "context" else select
    fresh = chooser(docs, rnd.seed, int(cfg["fresh"]["items"]), plan_of(season).parts, per_corner=int(cfg["fresh"]["per_corner"]), shares=mix)

    # Fresh harvest-derived comprehension & reasoning (Season 3 Exam C v2): deterministic greedy
    # synthesis cached atomically under `work` by snapshot + seed + reasoning config digest so
    # retries cost zero GPU work and partial/interrupted cache files are automatically regenerated.
    if synthesize and (r_cfg := cfg.get("reasoning")):
        rcfg_sig = hashlib.sha256(json.dumps(r_cfg, sort_keys=True).encode()).hexdigest()[:8]
        cache_stem = f"{snapshot_name(shas)[:12]}-{rnd.seed.hex()[:8]}-{rcfg_sig}"
        rcache = work / f"reasoning-{cache_stem}.jsonl.zst"
        rdrops = work / f"reasoning-drops-{cache_stem}.jsonl"
        reasoning: list[Item] = []
        loaded = False
        if rcache.exists():
            try:
                cached = unpack_items(rcache.read_bytes())
                if _reasoning_cache_usable(cached, season):
                    reasoning = cached
                    loaded = True
                    log(f"  reasoning items: {len(reasoning)} loaded from cache")
                else:
                    log("  reasoning cache incomplete; regenerating")
                    rcache.unlink(missing_ok=True)
            except (OSError, ValueError, zstandard.ZstdError):
                log("  reasoning cache unreadable; regenerating")
                rcache.unlink(missing_ok=True)
        if not loaded:
            reasoning = synthesize_reasoning(docs, rnd.seed, r_cfg, shares=mix, log=log, drops_path=rdrops)
            if not reasoning:
                raise FatalExamError(f"{rnd.id}: reasoning synthesis produced zero items")
            _atomic_write_bytes(rcache, pack_items(reasoning))
        fresh = fresh + reasoning

    mark = work / f"seen-{digest(fresh)[:12]}.json"
    if mark.exists():
        try:
            dropped = set(json.loads(mark.read_text()))
        except (OSError, ValueError):
            mark.unlink(missing_ok=True)
            dropped = seen(fresh, corpus, season, log=log)
            _atomic_write_bytes(mark, json.dumps(sorted(dropped)).encode())
    else:
        dropped = seen(fresh, corpus, season, log=log)
        _atomic_write_bytes(mark, json.dumps(sorted(dropped)).encode())
    if dropped:
        log(f"  {len(dropped)} fresh items are in the round's corpus already, and are dropped")
    fresh = [it for i, it in enumerate(fresh) if i not in dropped]
    generated = gen.for_season(season, rnd.seed) if int(cfg.get("per_corner", 0)) > 0 else []
    items: list[Item] = generated + fresh
    enforce = bool(synthesize and getattr(rnd, "kind", "") == "official") if check_coverage is None else bool(check_coverage)
    if enforce:
        items = _filter_pillar_support(items, season, rnd.id, log=log)
        fresh_n = sum(1 for it in items if not it.source.startswith("gen:"))
    else:
        fresh_n = len(fresh)
    log(f"  the exam: {len(items)} items, {fresh_n} of them fresh")
    return items, fresh_n, mix, shas


def snapshot_name(shas: list[str]) -> str:
    """One name for what an exam was made from: its hours' hashes in the hours' order, hashed."""
    import hashlib

    return hashlib.sha256("".join(shas).encode()).hexdigest()


def build(rnd, scale, snapshots: list[tuple[int, Path]], work: Path, log=print) -> dict:
    """The round's exam, as the server takes it, from the harvests of its hours."""
    import zstandard

    from .aggregate import item_table, plan_of

    snapshots = sorted(snapshots)
    items, fresh, mix, shas = assemble(rnd, scale, snapshots, work, log=log, check_coverage=True)

    public = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public_items(items)).encode()
    out = {
        "round": rnd.id,
        "items_digest": digest(items),
        "items_file": base64.b64encode(pack_items(items)).decode(),
        "public_file": base64.b64encode(zstandard.ZstdCompressor(level=10).compress(public)).decode(),
        "table": item_table(items, plan_of(rnd.season)),
        "languages": dict(list(mix.items())[:500]),
        "snapshot_sha256": snapshot_name(shas),
        "fresh": fresh,
        "hours": [{"hour": h, "sha256": sha} for (h, _), sha in zip(snapshots, shas, strict=True)],
    }
    if init_sha := _round_init_sha256(rnd, scale):
        out["init_sha256"] = init_sha
    return out


def post(exam: dict, key) -> dict:
    from ..client import request

    return request("POST", f"/api/v1/admin/rounds/{exam['round']}/exam", exam, key=key, timeout=600)


KEPT_ITEMS, KEPT_LANGUAGES = "items.jsonl.zst", "languages.json"


def keep(exam: dict, work: Path) -> None:
    """Keep the whole exam and the corpus's languages beside the harvest. Once the round settles
    the server gives out only the published exam, and the organisers' own measurements of the
    round (the curve, the audit, the finals: stages.py) sit students on all of it."""
    (work / KEPT_ITEMS).write_bytes(base64.b64decode(exam["items_file"]))
    (work / KEPT_LANGUAGES).write_text(json.dumps(exam["languages"]))


def examine(rnd, scale, snapshots: list[tuple[int, Path]], work: Path, key, log=print) -> dict:
    """Make a round's exam from its hours' harvests and post it."""
    t0 = time.time()
    exam = build(rnd, scale, snapshots, work, log=log)
    keep(exam, work)
    out = post(exam, key)
    log(f"  posted {rnd.id}'s exam: {len(exam['table'])} items ({time.time() - t0:.0f}s)")
    return out

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
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from .farm import pack_items
from .fresh.reason import FatalExamError
from .items import Item, digest


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
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


def _student_window(season, scale, *, required: bool = False) -> tuple[object, int] | None:
    """Return `(student_encoding, ctx)` so reasoning synthesis can reject items whose wrapped
    prompt + answer would exceed the student's scoring window.

    When `required=True` (official builds on seasons with a pinned student tokenizer), any
    download, checksum, or parse error raises a retryable `RuntimeError` instead of silently
    disabling the student-window fit check."""
    seq_len = getattr(scale, "ctx", None)
    if not isinstance(seq_len, int) or seq_len <= 0:
        if required:
            raise RuntimeError(f"cannot enforce student window check: scale {scale!r} has invalid ctx={seq_len!r}")
        return None
    try:
        from ..tok import load_encoding

        return (load_encoding(season), seq_len)
    except Exception as exc:
        if required:
            raise RuntimeError(f"could not load pinned student tokenizer for window check: {exc}") from exc
        return None


def _reasoning_cache_usable(
    reasoning: list[Item],
    season,
    *,
    student_window: tuple[object, int] | None = None,
    require_langid: bool = False,
) -> bool:
    """Whether a cached reasoning synthesis result is complete and conforms to current filter rules.

    Matches `_filter_pillar_support` and fresh synthesis: every cached item must pass
    `cached_item_passes_filter` (authoritative prompt/wrapper metadata, quality/length/sentence
    checks, active language verification, zero control tokens, prompt/response overlap limits,
    and student scoring-window fit), and at least one reasoning pillar must meet `min_pillar_items`."""
    from .aggregate import REASONING_KINDS, plan_of
    from .fresh.reason import _ensure_langid_available, cached_item_passes_filter

    if require_langid:
        _ensure_langid_available()
    if not reasoning:
        return False
    r_cfg = (season.raw.get("exam") or {}).get("reasoning") or {}
    max_ov = float(r_cfg.get("max_ngram_overlap", 0.50))
    if not all(
        cached_item_passes_filter(
            it,
            max_ngram_overlap=max_ov,
            require_langid=require_langid,
            student_window=student_window,
        )
        for it in reasoning
    ):
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
    except Exception as exc:
        sys.stderr.write(f"stet: warning: could not compute authoritative round init_sha256 on CPU ({exc}); graders will establish consensus\n")
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

    from .fresh.reason import extract_wrapped_excerpt, ngram_overlap, prompt_copies_excerpt, split_wrapped_ctx

    out = []
    for it in items:
        if it.source.startswith(OPEN):
            row = it.to_dict()
            attrs = [
                entry
                for sid in (it.source, *it.wrong_sources)
                if (entry := wikipedia_attribution_entry(sid)) is not None
            ]
            if attrs:
                row["attribution"] = attrs
            out.append(row)
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
            split: tuple[str, str, int] | None = None
            if it.prompt and it.wrapper_id is not None:
                if (ex := extract_wrapped_excerpt(it.ctx, it.prompt, task_type, it.wrapper_id)) is not None:
                    w_mod = 3 if task_type == "synthesis" else 6
                    split = (ex, it.prompt, it.wrapper_id % w_mod)
            else:
                split = split_wrapped_ctx(it.ctx, task_type)
            if split is not None:
                excerpt, prompt_line, wrapper_id = split
                row["withheld_excerpt"] = True
                row["wrapper_id"] = wrapper_id
                row["excerpt_sha256"] = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
                row["excerpt_chars"] = len(excerpt)
                if prompt_copies_excerpt(prompt_line, excerpt):
                    row["withheld_prompt"] = True
                    row["prompt_sha256"] = hashlib.sha256(prompt_line.encode("utf-8")).hexdigest()
                    row["prompt_chars"] = len(prompt_line)
                else:
                    row["prompt"] = prompt_line
                if ngram_overlap(it.cont, excerpt, n=5) > 0.50:
                    row.pop("cont", None)
                    row["withheld_cont"] = True
                    row["cont_sha256"] = hashlib.sha256(it.cont.encode("utf-8")).hexdigest()
                    row["cont_chars"] = len(it.cont)
            out.append(row)
        else:
            out.append({"withheld": True, "id": it.id, "source": it.source, "kind": it.kind, "fmt": it.fmt, "lang": it.lang, "corner": it.corner})
    return out


def rebuild_withheld_reasoning(
    row: dict,
    excerpt: str,
    *,
    prompt: str | None = None,
    cont: str | None = None,
) -> Item:
    """Reconstruct a settled non-Wikipedia reasoning item from its published teacher text and
    source excerpt (plus `prompt`/`cont` if withheld due to source overlap), verifying every
    published SHA-256 and the exact item `id`."""
    import hashlib

    from .fresh.reason import wrap_prompt

    if "wrapper_id" not in row:
        raise ValueError("published row does not carry withheld reasoning fields")
    prompt_line = str(row["prompt"]) if "prompt" in row else prompt
    if prompt_line is None:
        raise ValueError("published row withheld prompt; pass prompt= to rebuild")
    if row.get("prompt_sha256"):
        got_p_sha = hashlib.sha256(prompt_line.encode("utf-8")).hexdigest()
        if got_p_sha != row["prompt_sha256"]:
            raise ValueError(f"prompt sha256 mismatch: got {got_p_sha}, want {row['prompt_sha256']}")
    cont_text = str(row["cont"]) if "cont" in row else cont
    if cont_text is None:
        raise ValueError("published row withheld cont; pass cont= to rebuild")
    if row.get("cont_sha256"):
        got_c_sha = hashlib.sha256(cont_text.encode("utf-8")).hexdigest()
        if got_c_sha != row["cont_sha256"]:
            raise ValueError(f"cont sha256 mismatch: got {got_c_sha}, want {row['cont_sha256']}")
    got_sha = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    if row.get("excerpt_sha256") and got_sha != row["excerpt_sha256"]:
        raise ValueError(f"excerpt sha256 mismatch: got {got_sha}, want {row['excerpt_sha256']}")
    task_type = str(row["source"]).split(":", 1)[0]
    w_mod = 3 if task_type == "synthesis" else 6
    w_id = int(row["wrapper_id"]) % w_mod
    ctx = wrap_prompt(excerpt, prompt_line, task_type, w_id)
    it = Item(
        kind=str(row["kind"]),
        corner=str(row["corner"]),
        fmt=str(row["fmt"]),
        lang=str(row["lang"]),
        source=str(row["source"]),
        ctx=ctx,
        cont=cont_text,
        prompt=prompt_line,
        wrapper_id=w_id,
    )
    if it.id == row["id"]:
        return it
    # Legacy items serialized before authoritative (prompt, wrapper_id) fields were added to Item.
    legacy = Item(
        kind=str(row["kind"]),
        corner=str(row["corner"]),
        fmt=str(row["fmt"]),
        lang=str(row["lang"]),
        source=str(row["source"]),
        ctx=ctx,
        cont=cont_text,
    )
    if legacy.id == row["id"]:
        return legacy
    raise ValueError(f"rebuilt item id {it.id} does not match published id {row['id']}")


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
    # synthesis cached atomically under `work` by snapshot + seed + filter version + reasoning
    # config digest so retries cost zero GPU work and partial/stale cache files regenerate.
    if synthesize and (r_cfg := cfg.get("reasoning")):
        from .fresh.reason import REASONING_FILTER_VERSION, _ensure_langid_available

        is_official = bool(getattr(rnd, "kind", "") == "official")
        if is_official:
            _ensure_langid_available()
        require_sw = bool(is_official and season.raw.get("student") and season.raw.get("tokenizer"))
        sw = _student_window(season, scale, required=require_sw)
        rcfg_sig = hashlib.sha256(
            json.dumps({"filter_version": REASONING_FILTER_VERSION, "cfg": r_cfg}, sort_keys=True).encode()
        ).hexdigest()[:8]
        cache_stem = f"{snapshot_name(shas)[:12]}-{rnd.seed.hex()[:8]}-{rcfg_sig}"
        rcache = work / f"reasoning-{cache_stem}.jsonl.zst"
        rdrops = work / f"reasoning-drops-{cache_stem}.jsonl"
        reasoning: list[Item] = []
        loaded = False
        if rcache.exists():
            cached: list[Item] | None = None
            try:
                cached = unpack_items(rcache.read_bytes())
            except (OSError, ValueError, zstandard.ZstdError):
                log("  reasoning cache unreadable; regenerating")
                rcache.unlink(missing_ok=True)
            if cached is not None:
                if _reasoning_cache_usable(cached, season, student_window=sw, require_langid=is_official):
                    reasoning = cached
                    loaded = True
                    log(f"  reasoning items: {len(reasoning)} loaded from cache")
                else:
                    log("  reasoning cache incomplete; regenerating")
                    rcache.unlink(missing_ok=True)
        if not loaded:
            import inspect

            synth_kw: dict = {"shares": mix, "log": log, "drops_path": rdrops}
            try:
                s_params = inspect.signature(synthesize_reasoning).parameters
                has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in s_params.values())
                if sw is not None and ("student_window" in s_params or has_varkw):
                    synth_kw["student_window"] = sw
                if is_official and ("require_langid" in s_params or has_varkw):
                    synth_kw["require_langid"] = True
            except (TypeError, ValueError):
                pass
            reasoning = synthesize_reasoning(docs, rnd.seed, r_cfg, **synth_kw)
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


def wikipedia_attribution_entry(source_id: str, *, title: str = "", url: str = "") -> dict | None:
    """Map a Wikipedia passage or reasoning source ID (`wikipedia:<lang>:<revid>`,
    `wikipedia-added:<lang>:<revid>`, `wikipedia-talk:<lang>:<revid>`, or
    `<task>:wikipedia*:<lang>:<revid>`) to its CC-BY-SA 4.0 revision & history URLs."""
    raw = source_id
    for pref in REASONING_PREFIXES:
        if raw.startswith(pref):
            raw = raw[len(pref) :]
            break
    parts = raw.split(":")
    if len(parts) < 3 or parts[0] not in ("wikipedia", "wikipedia-added", "wikipedia-talk"):
        return None
    kind_src, lang, revid = parts[0], parts[1], parts[2]
    if not lang or not revid.isdigit():
        return None
    base = f"https://{lang}.wikipedia.org/w/index.php?oldid={revid}"
    entry = {
        "source": f"{kind_src}:{lang}:{revid}",
        "lang": lang,
        "revid": revid,
        "revision_url": base,
        "history_url": f"{base}&action=history",
        "license": "CC-BY-SA-4.0",
    }
    if title:
        entry["title"] = title
    if url:
        entry["url"] = url
    return entry


def wikipedia_attribution_manifest(items: list[Item]) -> list[dict]:
    """Deduplicated passage-to-page attribution manifest covering every primary and donor
    (`wrong_sources`) Wikipedia passage appearing in `items`."""
    seen: dict[str, dict] = {}
    for it in items:
        for sid in (it.source, *it.wrong_sources):
            entry = wikipedia_attribution_entry(sid)
            if entry is not None and entry["source"] not in seen:
                seen[entry["source"]] = entry
    return [seen[k] for k in sorted(seen)]


def shares_digest(languages: dict) -> str:
    """Canonical SHA-256 digest of a round's corpus language shares (normalizing int/float numbers)."""
    import hashlib

    canon = {str(k): float(v) for k, v in languages.items()}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def candidate_dir(work: Path, items_digest: str, shares_sha: str | None = None) -> Path:
    """Immutable per-candidate staging directory under `work` keyed by `items_digest` and
    (when provided) `shares_digest`, so retries with identical items but different language
    shares never overwrite an accepted candidate's staged files."""
    if shares_sha:
        return work / f"candidate-{items_digest[:16]}-{shares_sha[:12]}"
    return work / f"candidate-{items_digest[:16]}"


def stage(exam: dict, work: Path) -> Path:
    """Atomically stage a candidate exam's full items and language shares under an immutable
    `(items_digest, shares_digest)` directory before posting, so a rejected or duplicate build
    never overwrites an already-accepted exam's files."""
    s_sha = shares_digest(exam["languages"]) if isinstance(exam.get("languages"), dict) else None
    dest = candidate_dir(work, str(exam["items_digest"]), s_sha)
    dest.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(dest / KEPT_ITEMS, base64.b64decode(exam["items_file"]))
    _atomic_write_bytes(dest / KEPT_LANGUAGES, json.dumps(exam["languages"]).encode("utf-8"))
    return dest


def keep(exam: dict, work: Path) -> None:
    """Keep the accepted whole exam and the corpus's languages beside the harvest. Once the
    round settles the server gives out only the published exam, and the organisers' own
    measurements of the round (the curve, the audit, the finals: stages.py) sit students on all
    of it."""
    work.mkdir(parents=True, exist_ok=True)
    if exam.get("items_digest"):
        stage(exam, work)
    _atomic_write_bytes(work / KEPT_ITEMS, base64.b64decode(exam["items_file"]))
    _atomic_write_bytes(work / KEPT_LANGUAGES, json.dumps(exam["languages"]).encode("utf-8"))


def examine(rnd, scale, snapshots: list[tuple[int, Path]], work: Path, key, log=print) -> dict:
    """Make a round's exam from its hours' harvests and post it.

    Stages each candidate under an immutable `(items_digest, shares_digest)` directory before
    posting and promotes it into `KEPT_ITEMS` / `KEPT_LANGUAGES` only when the server accepts
    it as the winning publication (`posted != False`), so an overlapping or retried build never
    overwrites the accepted exam or its language shares on disk."""
    t0 = time.time()
    exam = build(rnd, scale, snapshots, work, log=log)
    stage(exam, work)
    out = post(exam, key)
    if isinstance(out, dict) and out.get("posted") is False:
        if not ((work / KEPT_ITEMS).exists() and (work / KEPT_LANGUAGES).exists()):
            keep(exam, work)
    else:
        keep(exam, work)
    log(f"  posted {rnd.id}'s exam: {len(exam['table'])} items ({time.time() - t0:.0f}s)")
    return out

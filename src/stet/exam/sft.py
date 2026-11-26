"""Standardized disjoint multilingual SFT probe run immediately after pretraining.

Pretraining data that builds latent reasoning, reading, multilingual, and tool-use circuits
responds strongly to a brief fixed supervised fine-tuning probe; shallow style mimicry does not.
Every verifier runs the exact same deterministic probe derived solely from the round's seed
(before the official exam is even posted). In Season 3, the probe loads the bundled, sha256-pinned
commercial-permissive multilingual SFT pack (`s3-sft-ml.jsonl.zst`, 512 examples from Dolly-15k
and Aya/COHERE-permissive multilingual instruction splits), ordered deterministically from the
round's seed with zero overlap with any scored exam item. Earlier/unpacked configurations fall back
to disjoint generated examples (using a separate RNG domain and the odd-parity Wikidata split).
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources

from ..spec import prf_u64
from .gen import algebra, algo, arith, context, knowledge, logic, shell, shots, units, word
from .rng import Rng


def _load_sft_pack(pack_file: str, want_sha: str, seed: bytes | str) -> list[tuple[str, str]]:
    """Load a bundled, sha256-pinned multilingual SFT probe pack in deterministic seeded order."""
    import zstandard

    blob = resources.files("stet.exam").joinpath("data", pack_file).read_bytes()
    if want_sha and hashlib.sha256(blob).hexdigest() != want_sha:
        raise RuntimeError(f"SFT pack {pack_file} does not match the season's sha256")
    raw = zstandard.ZstdDecompressor().decompress(blob, max_output_size=32 << 20).decode("utf-8")
    rows = [json.loads(ln) for ln in raw.split("\n") if ln]
    seed_b = seed if isinstance(seed, bytes) else str(seed).encode()
    order = sorted(range(len(rows)), key=lambda i: (prf_u64(seed_b, "sft-pack", i), i))
    return [(rows[i]["prompt"], rows[i]["comp"]) for i in order if rows[i].get("prompt") and rows[i].get("comp")]


def _odd_knowledge(rng: Rng, corner: str) -> tuple[str, str]:
    """A factual QA pair from the odd-parity Wikidata split (disjoint from knowledge-even's even split)."""
    _, lang, fact = corner.split("/")
    pool, lang = knowledge.split_pool(fact, lang, parity=1)
    s, o = rng.choice(pool)
    if fact == "birthyear":
        o = str(int(o))
    ctx = knowledge.TEMPLATES[fact][lang].format(s=s)
    return ctx, (o if lang in knowledge.NO_SPACE else f" {o}")


def _tail_question(ctx: str) -> str:
    """The final question block of a prompt (strips preceding few-shot examples if any)."""
    return ctx.split("\n\n")[-1].strip()


def build_sft_examples(
    seed: bytes | str,
    n: int = 256,
    *,
    season=None,
    exclude=None,
) -> list[tuple[str, str]]:
    """Return `n` deterministic, disjoint multilingual (prompt, completion) SFT probe pairs.

    In Season 3+, draws from the season's sha256-pinned `sft.pack` (`s3-sft-ml.jsonl.zst`) in
    seeded order; when no pack is configured, falls back to generated multilingual reading, logic,
    shell, odd-parity factual knowledge, and quantitative tasks. Always excludes any scored exam
    `(ctx, cont)` pair and tail question so no exam question can appear in the SFT probe.
    """
    if exclude is None:
        if season is None:
            from ..spec import load_season

            season = load_season("s3")
        from . import gen

        raw_exclude = [(it.ctx, it.cont) for it in gen.for_season(season, seed) if it.kind == "answer"]
    else:
        raw_exclude = list(exclude)

    blocked_exact: set[tuple[str, str]] = set(raw_exclude)
    blocked_tail: set[tuple[str, str]] = {(_tail_question(p), c.strip()) for p, c in raw_exclude}
    blocked_qa: set[str] = {f"{_tail_question(p)}{c}".strip() for p, c in raw_exclude}

    # Season 3+: use the pinned commercial-permissive multilingual SFT dev/train pack when named.
    sft_cfg = (season.raw.get("sft") or {}) if season is not None else {}
    if pack_file := sft_cfg.get("pack"):
        packed = _load_sft_pack(pack_file, str(sft_cfg.get("pack_sha256") or ""), seed)
        out_p: list[tuple[str, str]] = []
        seen_p: set[tuple[str, str]] = set()
        for pair in packed:
            if pair in seen_p or pair in blocked_exact:
                continue
            blocks = [b.strip() for b in pair[0].split("\n\n")]
            if (blocks[-1], pair[1].strip()) in blocked_tail:
                continue
            seen_p.add(pair)
            out_p.append(pair)
            if len(out_p) >= n:
                break
        return out_p

    rng = Rng(seed, "sft-probe/v1")
    buckets: list[tuple[str, list[str], object]] = [
        ("context", context.corners(), context),
        ("logic", logic.corners(), logic),
        ("agent", [c for c in shell.corners() if not c.endswith("/session")], shell),
        ("knowledge", knowledge.corners(), None),
        ("arith", shots.Shots(arith).corners(), shots.Shots(arith)),
        ("word", shots.Shots(word).corners(), shots.Shots(word)),
        ("units", shots.Shots(units).corners(), shots.Shots(units)),
        ("algebra", shots.Shots(algebra).corners(), shots.Shots(algebra)),
        ("algo", shots.Shots(algo).corners(), shots.Shots(algo)),
    ]
    # Give heavier weight to multilingual reading, logic, knowledge, and agent tasks.
    weights = [3, 3, 2, 2, 1, 1, 1, 1, 1]
    schedule = [idx for idx, w in enumerate(weights) for _ in range(w)]

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    i = 0
    while len(out) < n and i < n * 16:
        b_idx = schedule[i % len(schedule)]
        name, corners, fam = buckets[b_idx]
        sub = rng.child(f"{name}#{i}")
        corner = sub.choice(corners)
        if fam is None:
            pair = _odd_knowledge(sub.child("item"), corner)
        else:
            it = fam.make(sub.child("item"), corner)
            pair = (it.ctx, it.cont)
        i += 1
        if not pair[0] or not pair[1] or pair in seen or pair in blocked_exact:
            continue
        blocks = [b.strip() for b in pair[0].split("\n\n")]
        if (blocks[-1], pair[1].strip()) in blocked_tail:
            continue
        if any(b in blocked_qa for b in blocks[:-1]):
            continue
        seen.add(pair)
        out.append(pair)
    return out

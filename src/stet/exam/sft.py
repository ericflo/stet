"""Standardized disjoint multilingual SFT probe run immediately after pretraining.

Pretraining data that builds latent reasoning, reading, multilingual, and tool-use circuits
responds strongly to a brief fixed supervised fine-tuning probe; shallow style mimicry does not.
Every verifier runs the exact same deterministic probe derived solely from the round's seed
(before the official exam is even posted), using a disjoint RNG domain and the odd-parity split
of Wikidata facts so there is zero overlap with any exam item.
"""

from __future__ import annotations

from .gen import algebra, algo, arith, context, knowledge, logic, shell, shots, units, word
from .rng import Rng


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

    Balanced across reading comprehension (11 languages), deductive logic (11 languages),
    shell/tool use (11 languages), odd-parity factual knowledge (10 languages), and
    quantitative/program reasoning. Excludes every generated exam `(ctx, cont)` pair and
    every final-question `(tail_question(ctx), cont.strip())` (including inside few-shot
    worked examples) for the same round seed so no scored exam question can appear in the
    SFT probe.
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

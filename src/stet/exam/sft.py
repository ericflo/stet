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
    quantitative/program reasoning. Excludes every generated exam `(ctx, cont)` pair for the
    same round seed so small finite generators (such as 3-term Booleans) can never overlap
    with scored exam questions.
    """
    if exclude is None:
        if season is None:
            from ..spec import load_season

            season = load_season("s3")
        from . import gen

        blocked: set[tuple[str, str]] = {(it.ctx, it.cont) for it in gen.for_season(season, seed) if it.kind == "answer"}
    else:
        blocked = set(exclude)

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
        if pair[0] and pair[1] and pair not in seen and pair not in blocked:
            seen.add(pair)
            out.append(pair)
    return out

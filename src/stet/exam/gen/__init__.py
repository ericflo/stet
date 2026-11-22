"""Generated families. Each module has a NAME, a list of corners, and make(rng, corner) -> Item.

FAMILIES is every family this copy can make. Which of them a round's exam has is the season's
to say: its file names the parts of the exam (exam.pillars), and a family it does not name is
not generated (for_season). A round's generated exam is a few items from every corner of every
family the season names, each made from its own stream of the round's seed, so adding a family
or a corner never changes what the others produce.

Agent, context, transform, logic and grammar ask in the web's main languages, each language's
questions in corners of their own (logic/ru/...), so that a language's share of the corpus is its
share of a skill, as it is of the fresh writing. What a small student cannot do cold can also be
asked few-shot (shots.py), each of those a family of its own (arith-shots beside arith).
"""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from . import (
    algebra,
    algo,
    arith,
    context,
    grammar,
    knowledge,
    logic,
    multi,
    programs,
    shell,
    shots,
    structured,
    transform,
    units,
    word,
)

FAMILIES = [arith, word, units, algebra, programs, algo, logic, structured, shell, grammar, transform, multi, context, knowledge]
# What a small student cannot do cold, asked again after solved examples of the same task; and
# where a near miss can be told from the right answer, as a pair. Knowledge is not: a fact is
# known or it is not, and other facts do not help.
FAMILIES += [shots.Shots(f) for f in (arith, word, units, algebra, programs, algo, structured, multi)]
FAMILIES += [shots.Pairs(arith), shots.Pairs(units, lambda corner: "/convert/" in corner)]
VERSION = 9  # bump when any generator changes what a seed produces


def all_corners() -> list[str]:
    return [c for fam in FAMILIES for c in fam.corners()]


def named(parts) -> list:
    """The families among the parts a season names, in FAMILIES' order."""
    parts = set(parts)
    return [fam for fam in FAMILIES if fam.NAME in parts]


def generate(seed: bytes | str, per_corner: int = 1, *, families, more: dict[str, int] | None = None) -> list[Item]:
    """These families' items for a seed, in a fixed order: per_corner from every corner, and
    more[name] times that from a family named in `more`."""
    root = Rng(seed, f"exam/v{VERSION}")
    more = more or {}
    out: list[Item] = []
    had: set[str] = set()
    for fam in families:
        for corner in fam.corners():
            for i in range(per_corner * int(more.get(fam.NAME, 1))):
                it = fam.make(root.child(f"{corner}#{i}"), corner)
                # A corner with few things to say can say one of them twice. An item sat twice
                # counts twice: draw again, a few times, for one the exam does not have yet.
                for again in range(1, 9):
                    if it.id not in had:
                        break
                    it = fam.make(root.child(f"{corner}#{i}/again{again}"), corner)
                had.add(it.id)
                out.append(it)
    return out


def for_season(season, seed: bytes | str) -> list[Item]:
    """A round's generated items: the families the season's exam names and no other, exam.per_corner
    from every corner, and more from the families the season says rest on more (exam.more)."""
    from ..aggregate import plan_of

    cfg = season.raw["exam"]
    return generate(seed, int(cfg["per_corner"]), families=named(plan_of(season).parts), more=cfg.get("more"))

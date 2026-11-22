"""Generated families. Each module has a NAME, a list of corners, and make(rng, corner) -> Item.

A round's generated exam is one item (or a few) from every corner of every family, each made
from its own stream of the round's seed, so adding a family or a corner never changes what
the others produce.

The five families that count (MORE) ask in the web's main languages, each language's questions in
corners of their own (logic/ru/...), so that a language's share of the corpus is its share of a
skill, as it is of the fresh writing. The families that do not count are also asked few-shot
(shots.py), each as a family of its own that is watched beside it.
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
# The families that count in the season's score make more items from each corner, so that each
# rests on enough of them. The rest are watched: one round's worth is plenty.
MORE = {"agent": 4, "context": 2, "transform": 3, "logic": 2, "grammar": 3}


def all_corners() -> list[str]:
    return [c for fam in FAMILIES for c in fam.corners()]


def generate(seed: bytes | str, per_corner: int = 1, families=None) -> list[Item]:
    """Every corner's items for a seed: per_corner of each (more for the families in MORE), in a fixed order."""
    root = Rng(seed, f"exam/v{VERSION}")
    out: list[Item] = []
    had: set[str] = set()
    for fam in families or FAMILIES:
        for corner in fam.corners():
            for i in range(per_corner * MORE.get(fam.NAME, 1)):
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

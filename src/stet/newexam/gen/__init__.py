"""Generated families. Each module has a NAME, a list of corners, and make(rng, corner) -> Item.

A round's generated exam is one item (or a few) from every corner of every family, each made
from its own stream of the round's seed, so adding a family or a corner never changes what
the others produce.
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
    structured,
    transform,
    units,
    word,
)

FAMILIES = [arith, word, units, algebra, programs, algo, logic, structured, shell, grammar, transform, multi, context, knowledge]
VERSION = 6  # bump when any generator changes what a seed produces
# Families that are a pillar (or most of one) on few corners make more items from each, so that
# every pillar rests on enough of them.
MORE = {"agent": 4, "context": 2}


def all_corners() -> list[str]:
    return [c for fam in FAMILIES for c in fam.corners()]


def generate(seed: bytes | str, per_corner: int = 1, families=None) -> list[Item]:
    """Every corner's items for a seed: per_corner of each (more for the families in MORE), in a fixed order."""
    root = Rng(seed, f"newexam/v{VERSION}")
    out: list[Item] = []
    for fam in families or FAMILIES:
        for corner in fam.corners():
            for i in range(per_corner * MORE.get(fam.NAME, 1)):
                out.append(fam.make(root.child(f"{corner}#{i}"), corner))
    return out

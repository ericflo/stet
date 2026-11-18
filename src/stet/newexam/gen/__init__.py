"""Generated families. Each module has a NAME, a list of corners, and make(rng, corner) -> Item.

A round's generated exam is one item (or a few) from every corner of every family, each made
from its own stream of the round's seed, so adding a family or a corner never changes what
the others produce.
"""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from . import algebra, algo, arith, grammar, logic, programs, shell, structured, transform, units, word

FAMILIES = [arith, word, units, algebra, programs, algo, logic, structured, shell, grammar, transform]
VERSION = 2  # bump when any generator changes what a seed produces


def all_corners() -> list[str]:
    return [c for fam in FAMILIES for c in fam.corners()]


def generate(seed: bytes | str, per_corner: int = 1, families=None) -> list[Item]:
    """Every corner's items for a seed: per_corner of each, in a fixed order."""
    root = Rng(seed, f"newexam/v{VERSION}")
    out: list[Item] = []
    for fam in families or FAMILIES:
        for corner in fam.corners():
            for i in range(per_corner):
                out.append(fam.make(root.child(f"{corner}#{i}"), corner))
    return out

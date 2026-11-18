"""Randomness that every machine agrees on.

A generated exam must come out byte for byte the same on every grader, whatever its Python.
`random.Random` promises that for `random()` but not for every method across versions, so
the generators use this: SplitMix64 from a SHA-256 of the seed and a label, and integer
methods only.
"""

from __future__ import annotations

import hashlib

MASK = (1 << 64) - 1


class Rng:
    def __init__(self, seed: bytes | str, label: str = ""):
        if isinstance(seed, str):
            seed = seed.encode()
        self._s = int.from_bytes(hashlib.sha256(seed + b"|" + label.encode()).digest()[:8], "big")
        self._seed, self._label = seed, label

    def next64(self) -> int:
        self._s = (self._s + 0x9E3779B97F4A7C15) & MASK
        z = self._s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def child(self, label: str) -> Rng:
        """An independent stream for a sub-task: the same label always gives the same stream."""
        return Rng(self._seed, f"{self._label}/{label}")

    def below(self, n: int) -> int:
        """A uniform integer in [0, n)."""
        if n <= 0:
            raise ValueError("below() needs a positive bound")
        limit = MASK - (MASK + 1) % n
        while True:
            x = self.next64()
            if x <= limit:
                return x % n

    def between(self, lo: int, hi: int) -> int:
        """A uniform integer in [lo, hi]."""
        return lo + self.below(hi - lo + 1)

    def choice(self, seq):
        return seq[self.below(len(seq))]

    def chance(self, num: int, den: int) -> bool:
        return self.below(den) < num

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def sample(self, seq, k: int) -> list:
        pool = list(seq)
        if k > len(pool):
            raise ValueError(f"cannot sample {k} from {len(pool)}")
        for i in range(k):
            j = i + self.below(len(pool) - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]

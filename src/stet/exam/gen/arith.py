"""Arithmetic with one right answer, asked in several ways."""

from __future__ import annotations

from ..items import Item
from ..rng import Rng

NAME = "arith"
OPS = ["add", "sub", "mul", "div", "mixed", "paren", "percent", "square"]
SIZES = ["small", "medium", "large"]
STYLES = ["qa", "equation", "words", "calc"]


def corners() -> list[str]:
    return [f"{NAME}/{op}/{size}/{style}" for op in OPS for size in SIZES for style in STYLES]


def num(n: int, commas: bool) -> str:
    return f"{n:,}" if commas else str(n)


def _digits(rng: Rng, size: str, lo_small=2, hi_small=99) -> int:
    if size == "small":
        return rng.between(lo_small, hi_small)
    if size == "medium":
        return rng.between(100, 999)
    return rng.between(10_000, 999_999)


def near_miss(rng: Rng, corner: str, item: Item) -> str:
    """An item's answer with one digit one off: as near to right as a wrong answer comes."""
    s = item.cont
    at = [i for i, ch in enumerate(s) if ch.isdigit()]
    i = rng.choice(at)
    was = int(s[i])
    step = 1 if rng.chance(1, 2) else -1
    least = 1 if i == at[0] and len(at) > 1 else 0  # a number of several digits does not begin with a zero
    if not least <= was + step <= 9:
        step = -step
    return f"{s[:i]}{was + step}{s[i + 1:]}"


def make(rng: Rng, corner: str) -> Item:
    _, op, size, style = corner.split("/")
    commas = style == "words"
    if op == "add":
        a, b = _digits(rng, size), _digits(rng, size)
        sym, words, ans = f"{a} + {b}", f"the sum of {num(a, commas)} and {num(b, commas)}", a + b
    elif op == "sub":
        a, b = _digits(rng, size), _digits(rng, size)
        if a < b and size != "large":
            a, b = b, a
        sym, words, ans = f"{a} - {b}", f"{num(a, commas)} minus {num(b, commas)}", a - b
    elif op == "mul":
        if size == "small":
            a, b = rng.between(2, 12), rng.between(2, 12)
        elif size == "medium":
            a, b = rng.between(13, 99), rng.between(3, 19)
        else:
            a, b = rng.between(100, 999), rng.between(11, 99)
        sym, words, ans = f"{a} * {b}", f"the product of {num(a, commas)} and {num(b, commas)}", a * b
    elif op == "div":
        b = rng.between(2, 12) if size != "large" else rng.between(13, 99)
        q = rng.between(2, 20) if size == "small" else rng.between(21, 300) if size == "medium" else rng.between(301, 9999)
        a = b * q
        sym, words, ans = f"{a} / {b}", f"{num(a, commas)} divided by {num(b, commas)}", q
    elif op == "mixed":
        hi = {"small": 9, "medium": 50, "large": 400}[size]
        a, b, c = rng.between(2, hi), rng.between(2, hi), rng.between(2, 12)
        sym, ans = f"{a} + {b} * {c}", a + b * c
        words = f"{num(a, commas)} plus {num(b, commas)} times {c}"
    elif op == "paren":
        hi = {"small": 9, "medium": 50, "large": 400}[size]
        a, b, c = rng.between(2, hi), rng.between(2, hi), rng.between(2, 12)
        sym, ans = f"({a} + {b}) * {c}", (a + b) * c
        words = f"the sum of {num(a, commas)} and {num(b, commas)}, times {c}"
    elif op == "percent":
        p = rng.choice([5, 10, 20, 25, 50, 75] if size == "small" else [12, 15, 30, 35, 40, 60, 80, 90])
        unit = {"small": 20, "medium": 100, "large": 1000}[size]
        n = unit * rng.between(1, 99)
        if (n * p) % 100:
            n = 100 * rng.between(1, 99)
        sym, words, ans = f"{p}% of {n}", f"{p} percent of {num(n, commas)}", n * p // 100
    elif op == "square":
        n = rng.between(2, 15) if size == "small" else rng.between(16, 99) if size == "medium" else rng.between(100, 999)
        sym, words, ans = f"{n}^2", f"{n} squared", n * n
    else:
        raise ValueError(corner)
    if style == "qa":
        ctx = f"Question: What is {sym}?\nAnswer:"
    elif style == "equation":
        ctx = f"{sym} ="
    elif style == "words":
        ctx = f"{words[0].upper()}{words[1:]} is"
    else:
        ctx = f"Calculate: {sym}\nResult:"
    return Item(kind="answer", corner=corner, fmt="math", lang="en", source=f"gen:{NAME}", ctx=ctx, cont=" " + num(ans, commas))

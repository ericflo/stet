"""Algebra and number patterns: equations to solve, functions to evaluate, sequences to continue."""

from __future__ import annotations

from ..items import Item
from ..rng import Rng

NAME = "algebra"
KINDS = ["one-step", "two-step", "both-sides", "evaluate", "system", "seq-arith", "seq-geom", "seq-square", "seq-fib", "seq-alt"]
STYLES = ["solve", "fill"]


def corners() -> list[str]:
    return [f"{NAME}/{k}/{s}" for k in KINDS for s in STYLES]


def _signed(n: int) -> str:
    return f"+ {n}" if n >= 0 else f"- {-n}"


def _term(a: int, var: str = "x") -> str:
    """A coefficient and its variable as they are written: x and -x, never 1x."""
    return var if a == 1 else f"-{var}" if a == -1 else f"{a}{var}"


def _plus(a: int, var: str) -> str:
    """A later term with its sign: "+ 3x", "- x"; nothing at all for 0x."""
    if a == 0:
        return ""
    return f" {'+' if a > 0 else '-'} {_term(abs(a), var)}"


def _nz(rng: Rng, lo: int, hi: int) -> int:
    while True:
        v = rng.between(lo, hi)
        if v:
            return v


def make(rng: Rng, corner: str) -> Item:
    _, kind, style = corner.split("/")
    if kind.startswith("seq-"):
        if kind == "seq-arith":
            a, d = rng.between(-20, 50), _nz(rng, -12, 15)
            terms = [a + d * i for i in range(6)]
        elif kind == "seq-geom":
            a, r = rng.between(1, 6), rng.choice([2, 3, 4, 5, -2])
            terms = [a * r**i for i in range(6)]
        elif kind == "seq-square":
            s = rng.between(1, 20)
            terms = [(s + i) ** 2 for i in range(6)]
        elif kind == "seq-fib":
            a, b = rng.between(1, 9), rng.between(1, 9)
            terms = [a, b]
            while len(terms) < 6:
                terms.append(terms[-1] + terms[-2])
        else:
            a, p, q = rng.between(1, 30), rng.between(2, 9), rng.between(1, 5)
            terms = [a]
            for i in range(5):
                terms.append(terms[-1] + (p if i % 2 == 0 else -q))
        shown, ans = ", ".join(map(str, terms[:5])), terms[5]
        ctx = f"What number comes next in the sequence {shown}?\nAnswer:" if style == "solve" else f"{shown},"
        return Item(kind="answer", corner=corner, fmt="math", lang="en", source=f"gen:{NAME}", ctx=ctx, cont=f" {ans}")
    if kind == "one-step":
        x, b = rng.between(-30, 60), rng.between(-40, 40)
        eq, ans = f"x {_signed(b)} = {x + b}", x
    elif kind == "two-step":
        x, a, b = rng.between(-15, 25), _nz(rng, -9, 12), rng.between(-30, 30)
        eq, ans = f"{_term(a)} {_signed(b)} = {a * x + b}", x
    elif kind == "both-sides":
        x, a, c = rng.between(-12, 15), rng.between(2, 12), rng.between(-6, 1)
        b = rng.between(-20, 20)
        d = a * x + b - c * x
        eq, ans = f"{_term(a)} {_signed(b)} = " + (f"{_term(c)} {_signed(d)}" if c else f"{d}"), x
    elif kind == "evaluate":
        a, b, c, x = rng.between(1, 6), rng.between(-9, 9), rng.between(-9, 9), rng.between(-6, 8)
        val = a * x * x + b * x + c
        fx = f"{_term(a, 'x^2')}{_plus(b, 'x')} {_signed(c)}"
        ctx = f"Let f(x) = {fx}. What is f({x})?\nAnswer:" if style == "solve" else f"If f(x) = {fx}, then f({x}) ="
        return Item(kind="answer", corner=corner, fmt="math", lang="en", source=f"gen:{NAME}", ctx=ctx, cont=f" {val}")
    elif kind == "system":
        x, y = rng.between(-9, 12), rng.between(-9, 12)
        a, b, c, d = rng.between(1, 5), rng.between(1, 5), rng.between(1, 5), _nz(rng, -5, 5)
        if a * d == b * c:
            d = d + 1 if d != -1 else 2
        one, two = f"{_term(a)}{_plus(b, 'y')} = {a * x + b * y}", f"{_term(c)}{_plus(d, 'y')} = {c * x + d * y}"
        ctx = f"Solve the system: {one} and {two}.\nAnswer:" if style == "solve" else f"If {one} and {two}, then"
        return Item(kind="answer", corner=corner, fmt="math", lang="en", source=f"gen:{NAME}", ctx=ctx, cont=f" x = {x}, y = {y}")
    else:
        raise ValueError(corner)
    ctx = f"Solve for x: {eq}\nAnswer:" if style == "solve" else f"If {eq}, then x ="
    cont = f" x = {ans}" if style == "solve" else f" {ans}"
    return Item(kind="answer", corner=corner, fmt="math", lang="en", source=f"gen:{NAME}", ctx=ctx, cont=cont)

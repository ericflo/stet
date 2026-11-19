"""Units, times and dates: conversions and calendar arithmetic with one right answer."""

from __future__ import annotations

import datetime as dt

from ..items import Item
from ..rng import Rng
from .lexicon import MONTHS, WEEKDAYS

NAME = "units"
# (corner, from-unit, to-unit, factor): value_in_to = value_in_from * factor
CONVERSIONS = [
    ("km-m", "kilometres", "metres", 1000), ("m-cm", "metres", "centimetres", 100), ("kg-g", "kilograms", "grams", 1000),
    ("l-ml", "litres", "millilitres", 1000), ("h-min", "hours", "minutes", 60), ("min-s", "minutes", "seconds", 60),
    ("day-h", "days", "hours", 24), ("week-day", "weeks", "days", 7), ("ft-in", "feet", "inches", 12),
    ("dozen", "dozen", "items", 12), ("year-month", "years", "months", 12),
]
DATES = ["weekday-after", "date-after", "days-between", "leap-year", "month-length", "weekday-before"]


def corners() -> list[str]:
    out = [f"{NAME}/convert/{c[0]}/{d}" for c in CONVERSIONS for d in ("up", "down")]
    return out + [f"{NAME}/date/{d}" for d in DATES]


def _date(rng: Rng) -> dt.date:
    return dt.date(rng.between(1900, 2099), 1, 1) + dt.timedelta(days=rng.below(365))


def _say(d: dt.date) -> str:
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def make(rng: Rng, corner: str) -> Item:
    parts = corner.split("/")
    if parts[1] == "convert":
        key, way = parts[2], parts[3]
        _, big, small, f = next(c for c in CONVERSIONS if c[0] == key)
        if way == "up":
            v = rng.between(2, 60)
            ctx, ans = f"Question: How many {small} are in {v} {big}?\nAnswer:", f" {v * f}"
        else:
            v = rng.between(2, 60)
            ctx, ans = f"Question: {v * f} {small} is how many {big}?\nAnswer:", f" {v}"
        return Item(kind="answer", corner=corner, fmt="math", lang="en", source=f"gen:{NAME}", ctx=ctx, cont=ans)
    kind = parts[2]
    if kind == "weekday-after":
        start, n = rng.below(7), rng.between(2, 60)
        ctx = f"If today is {WEEKDAYS[start]}, what day of the week will it be in {n} days?\nAnswer:"
        ans = f" {WEEKDAYS[(start + n) % 7]}"
    elif kind == "weekday-before":
        start, n = rng.below(7), rng.between(2, 60)
        ctx = f"If today is {WEEKDAYS[start]}, what day of the week was it {n} days ago?\nAnswer:"
        ans = f" {WEEKDAYS[(start - n) % 7]}"
    elif kind == "date-after":
        d, n = _date(rng), rng.between(3, 90)
        ctx, ans = f"What date is {n} days after {_say(d)}?\nAnswer:", f" {_say(d + dt.timedelta(days=n))}"
    elif kind == "days-between":
        d, n = _date(rng), rng.between(3, 200)
        ctx, ans = f"How many days are there from {_say(d)} to {_say(d + dt.timedelta(days=n))}?\nAnswer:", f" {n}"
    elif kind == "leap-year":
        y = rng.between(1600, 2400)
        if rng.chance(1, 2):
            y -= y % 4
        leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        ctx, ans = f"Is {y} a leap year?\nAnswer:", " Yes" if leap else " No"
    elif kind == "month-length":
        y, m = rng.between(1900, 2099), rng.between(1, 12)
        days = ((dt.date(y + (m == 12), m % 12 + 1, 1)) - dt.date(y, m, 1)).days
        ctx, ans = f"How many days are there in {MONTHS[m - 1]} {y}?\nAnswer:", f" {days}"
    else:
        raise ValueError(corner)
    return Item(kind="answer", corner=corner, fmt="math", lang="en", source=f"gen:{NAME}", ctx=ctx, cont=ans)

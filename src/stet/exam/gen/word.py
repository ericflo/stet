"""Word problems: a short situation, a question, a worked solution and one number."""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from .lexicon import NAMES, OBJECTS, PLACES

NAME = "word"
TEMPLATES = ["change", "travel", "share", "savings", "recipe", "ages", "work", "area", "perimeter", "resale", "clock", "pages"]
STYLES = ["solution", "final"]


def corners() -> list[str]:
    return [f"{NAME}/{t}/{s}" for t in TEMPLATES for s in STYLES]


def _problem(rng: Rng, t: str) -> tuple[str, list[str], int, str]:
    """(question, solution steps, answer, unit prefix such as '$')"""
    n1, n2 = rng.sample(NAMES, 2)
    one, many = rng.choice(OBJECTS)
    if t == "change":
        a, p = rng.between(2, 9), rng.between(2, 15)
        bill = next(b for b in (10, 20, 50, 100, 200) if b >= a * p)
        return (f"{n1} buys {a} {many} at ${p} each and pays with a ${bill} note. How much change does {n1} get back?",
                [f"{a} {many} cost {a} × ${p} = ${a * p}.", f"The change is ${bill} − ${a * p} = ${bill - a * p}."], bill - a * p, "$")
    if t == "travel":
        v, h = rng.between(3, 12) * 10, rng.between(2, 7)
        a, b = rng.sample(PLACES, 2)
        return (f"A train leaves {a} for {b} and travels at {v} km per hour for {h} hours. How many kilometres does it cover?",
                [f"Distance is speed × time: {v} × {h} = {v * h} km."], v * h, "")
    if t == "share":
        k, each = rng.between(3, 9), rng.between(4, 25)
        total = k * each
        return (f"{n1} has {total} {many} to share equally among {k} friends. How many {many} does each friend get?",
                [f"{total} ÷ {k} = {each}."], each, "")
    if t == "savings":
        has, per, weeks = rng.between(5, 60), rng.between(5, 40), rng.between(3, 12)
        return (f"{n1} has ${has} and saves ${per} every week. How much will {n1} have after {weeks} weeks?",
                [f"In {weeks} weeks {n1} saves {weeks} × ${per} = ${weeks * per}.", f"Then {n1} has ${has} + ${weeks * per} = ${has + weeks * per}."],
                has + weeks * per, "$")
    if t == "recipe":
        per, serves, want = rng.between(2, 6), rng.between(2, 4), rng.between(2, 5)
        cups = per * want
        return (f"A recipe uses {per} cups of flour to serve {serves} people. How many cups are needed to serve {serves * want} people?",
                [f"{serves * want} people is {want} times the recipe.", f"{want} × {per} = {cups} cups."], cups, "")
    if t == "ages":
        b, d = rng.between(5, 40), rng.between(2, 30)
        return (f"{n1} is {d} years older than {n2}. {n2} is {b}. What is the sum of their ages?",
                [f"{n1} is {b} + {d} = {b + d}.", f"Together they are {b + d} + {b} = {2 * b + d}."], 2 * b + d, "")
    if t == "work":
        w, h, r = rng.between(2, 9), rng.between(2, 8), rng.between(3, 15)
        return (f"Each of {w} workers packs {r} boxes an hour. How many boxes do they pack in {h} hours?",
                [f"In an hour they pack {w} × {r} = {w * r} boxes.", f"In {h} hours: {w * r} × {h} = {w * r * h}."], w * r * h, "")
    if t == "area":
        a, b = rng.between(3, 40), rng.between(3, 40)
        return (f"A rectangular garden is {a} metres long and {b} metres wide. What is its area in square metres?",
                [f"Area is length × width: {a} × {b} = {a * b}."], a * b, "")
    if t == "perimeter":
        a, b = rng.between(3, 60), rng.between(3, 60)
        return (f"A field is {a} metres long and {b} metres wide. How many metres of fence go around it?",
                [f"The perimeter is 2 × ({a} + {b}) = 2 × {a + b} = {2 * (a + b)}."], 2 * (a + b), "")
    if t == "resale":
        bought, cost = rng.between(10, 40), rng.between(2, 9)
        sold = rng.between(3, bought)
        price = cost + rng.between(1, 6)
        profit = sold * price - bought * cost
        return (f"{n1} buys {bought} {many} for ${cost} each and sells {sold} of them for ${price} each. What is {n1}'s profit, in dollars?",
                [f"Cost: {bought} × ${cost} = ${bought * cost}.", f"Sales: {sold} × ${price} = ${sold * price}.",
                 f"Profit: ${sold * price} − ${bought * cost} = {profit}."], profit, "")
    if t == "clock":
        h, m, d = rng.between(6, 18), rng.choice([0, 10, 15, 20, 30, 40, 45, 50]), rng.between(20, 200)
        end = h * 60 + m + d
        eh, em = (end // 60) % 24, end % 60
        q = f"A meeting starts at {h:02d}:{m:02d} and lasts {d} minutes. At what time does it end?"
        return q, [f"{h:02d}:{m:02d} plus {d} minutes is {eh:02d}:{em:02d}."], f"{eh:02d}:{em:02d}", ""
    if t == "pages":
        per, days = rng.between(5, 40), rng.between(3, 14)
        total = per * days + rng.between(10, 90)
        return (f"A book has {total} pages. {n1} reads {per} pages a day for {days} days. How many pages are left?",
                [f"{n1} reads {per} × {days} = {per * days} pages.", f"{total} − {per * days} = {total - per * days} pages are left."],
                total - per * days, "")
    raise ValueError(t)


def make(rng: Rng, corner: str) -> Item:
    _, t, style = corner.split("/")
    q, steps, ans, unit = _problem(rng, t)
    shown = ans if isinstance(ans, str) else f"{unit}{ans}" if ans >= 0 or not unit else f"-{unit}{-ans}"
    if style == "solution":
        return Item(kind="answer", corner=corner, fmt="math", lang="en", source=f"gen:{NAME}",
                    ctx=f"Question: {q}\nSolution:", cont=" " + " ".join(steps) + f" The answer is {shown}.")
    return Item(kind="answer", corner=corner, fmt="math", lang="en", source=f"gen:{NAME}",
                ctx=f"Question: {q}\nSolution: {' '.join(steps)}\nThe answer is", cont=f" {shown}.")

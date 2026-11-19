"""Reasoning in a few sentences: orderings, chains, syllogisms, truth values, counting."""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from .lexicon import COLORS, NAMES

NAME = "logic"
RELATIONS = [("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest"),
             ("heavier", "heaviest", "lightest")]
KINDS = [f"order{n}-{r[0]}-{end}" for n in (3, 4) for r in RELATIONS for end in ("top", "bottom")]
KINDS += ["chain", "syllogism-yes", "syllogism-no", "boolean", "counting", "between"]
NONCE = ["blicket", "dax", "wug", "fep", "toma", "zorb", "glim", "plon", "vosk", "trell"]
PROPS = ["shiny", "loud", "soft", "round", "cold", "striped", "hollow", "sweet"]
CONTAINERS = ["box", "drawer", "cupboard", "bag", "basket", "chest", "case", "crate"]
THINGS = ["key", "ring", "coin", "letter", "map", "button", "ticket", "stone"]


def corners() -> list[str]:
    return [f"{NAME}/{k}" for k in KINDS]


def make(rng: Rng, corner: str) -> Item:
    k = corner.split("/", 1)[1]
    if k.startswith("order"):
        n = int(k[5])
        rel = next(r for r in RELATIONS if r[0] == k.split("-")[1])
        people = rng.sample(NAMES, n)  # people[0] is the most
        facts = [f"{people[i]} is {rel[0]} than {people[i + 1]}." for i in range(n - 1)]
        rng.shuffle(facts)
        top = k.endswith("top")
        q = f"Who is the {rel[1] if top else rel[2]}?"
        ctx, ans = f"{' '.join(facts)} {q}\nAnswer:", people[0] if top else people[-1]
    elif k == "between":
        people = rng.sample(NAMES, 3)
        facts = [f"{people[0]} is older than {people[1]}.", f"{people[1]} is older than {people[2]}."]
        rng.shuffle(facts)
        ctx, ans = f"{' '.join(facts)} Who is younger than {people[0]} but older than {people[2]}?\nAnswer:", people[1]
    elif k == "chain":
        thing = rng.choice(THINGS)
        boxes = rng.sample(CONTAINERS, 3)
        facts = [f"The {thing} is in the {boxes[0]}.", f"The {boxes[0]} is in the {boxes[1]}.", f"The {boxes[1]} is in the {boxes[2]}."]
        rng.shuffle(facts)
        ctx, ans = f"{' '.join(facts)} What is the {thing} ultimately inside?\nAnswer: the", boxes[2]
    elif k in ("syllogism-yes", "syllogism-no"):
        x, prop, who = rng.choice(NONCE), rng.choice(PROPS), rng.choice(NAMES)
        if k == "syllogism-yes":
            ctx, ans = f"Every {x} is {prop}. {who} owns a {x}. Is {who}'s {x} {prop}?\nAnswer:", "Yes"
        else:
            ctx, ans = f"No {x} is {prop}. {who} owns a {x}. Is {who}'s {x} {prop}?\nAnswer:", "No"
    elif k == "boolean":
        vals = [rng.chance(1, 2) for _ in range(3)]
        ops = [rng.choice(["and", "or"]) for _ in range(2)]
        neg = [rng.chance(1, 3) for _ in range(3)]
        terms = [("not " if n else "") + ("True" if v else "False") for v, n in zip(vals, neg, strict=True)]
        expr = f"{terms[0]} {ops[0]} {terms[1]} {ops[1]} {terms[2]}"
        t = [v != n for v, n in zip(vals, neg, strict=True)]  # not binds tightest, then and, then or
        if ops == ["and", "and"]:
            out = t[0] and t[1] and t[2]
        elif ops == ["or", "or"]:
            out = t[0] or t[1] or t[2]
        elif ops == ["and", "or"]:
            out = (t[0] and t[1]) or t[2]
        else:
            out = t[0] or (t[1] and t[2])
        ans = str(out)
        ctx = f"Evaluate: {expr}\nAnswer:"
    elif k == "counting":
        who = rng.choice(NAMES)
        cols = rng.sample(COLORS, 3)
        counts = [rng.between(1, 9) for _ in cols]
        facts = ", ".join(f"{c} {col}" for c, col in zip(counts, cols, strict=True))
        ask = rng.choice(cols[:2])
        if rng.chance(1, 2):
            ctx, ans = f"{who} has {facts} marbles. How many marbles does {who} have in all?\nAnswer:", str(sum(counts))
        else:
            ctx, ans = f"{who} has {facts} marbles. How many marbles are not {ask}?\nAnswer:", str(sum(counts) - counts[cols.index(ask)])
    else:
        raise ValueError(corner)
    return Item(kind="answer", corner=corner, fmt="prose", lang="en", source=f"gen:{NAME}", ctx=ctx, cont=f" {ans}")

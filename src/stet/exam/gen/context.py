"""Reading and learning in context: the answer is in the item's own context, about things that
exist nowhere else, so it can be read but never remembered.

- A short account of an invented place, company, river or person, then a question about it,
  asked one of several ways; some need two facts joined.
- A few examples of an invented rule (a cipher, a suffix, a number function, words of an
  invented language, a made-up category), then a new case.
- An invented unit or container, and a count that follows from its definition.

What these reward is the ability pretraining is for and memory cannot fake: taking in
something new and using it.
"""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from .lexicon import COLORS, NAMES, OBJECTS

NAME = "context"
ONSETS = ["b", "br", "d", "dr", "f", "g", "gl", "h", "k", "kr", "l", "m", "n", "p", "pl", "r", "s", "st", "t", "tr", "v", "z"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ou", "ei"]
CODAS = ["", "", "n", "r", "l", "s", "th", "m", "k"]
KINDS_OF_THING = ["town", "company", "river", "festival", "school", "ship", "mountain", "bridge"]
ATTRS = ["founded", "place", "leader", "colour", "count", "material"]
STYLES = ["qa", "question", "cloze", "according"]
RULES = ["reverse", "suffix", "affine", "glossary", "category", "swap"]
MATERIALS = ["granite", "oak", "copper", "glass", "slate", "cedar", "brass", "marble"]


def _a(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def word(rng: Rng, syllables: int | None = None) -> str:
    """A word that means nothing, from the seed: nobody can have read it before."""
    n = syllables or rng.between(2, 3)
    return "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(n))


def name(rng: Rng) -> str:
    return word(rng).capitalize()


def corners() -> list[str]:
    out = [f"{NAME}/passage/{a}/{s}" for a in ATTRS for s in STYLES]
    out += [f"{NAME}/two-hop/{s}" for s in STYLES]
    out += [f"{NAME}/rule/{r}" for r in RULES]
    out += [f"{NAME}/definition/{k}" for k in ("container", "unit")]
    return out


def _ask(style: str, subject: str, question: str, cloze: str) -> str:
    """The same question, put one of several ways; the answer follows each."""
    return {"qa": f"Q: {question}\nA:", "question": f"Question: {question}\nAnswer:", "cloze": cloze,
            "according": f"According to the passage above, {cloze[0].lower()}{cloze[1:]}"}[style]


def _passage(rng: Rng, attr: str, style: str) -> tuple[str, str]:
    kind = rng.choice(KINDS_OF_THING)
    thing = name(rng)
    place, region = name(rng), name(rng)
    leader = rng.choice(NAMES)
    year = rng.between(1650, 2015)
    colour = rng.choice(COLORS)
    count = rng.between(3, 40)
    material = rng.choice(MATERIALS)
    facts = {
        "founded": (f"{thing} was founded in {year}.", f"When was {thing} founded?", f"{thing} was founded in", str(year)),
        "place": (f"{thing} lies in {place}, a small district of {region}.", f"Where does {thing} lie?", f"{thing} lies in", place),
        "leader": (f"Since its last election, {thing} has been led by {leader}.", f"Who leads {thing}?", f"{thing} is led by", leader),
        "colour": (f"Its emblem is {_a(colour)} circle on a white field.", f"What colour is the circle in {thing}'s emblem?",
                   f"The circle in {thing}'s emblem is", colour),
        "count": (f"It is known for its {count} {rng.choice(OBJECTS)[1]}, counted every spring.", f"How many are counted at {thing} each spring?",
                  f"The number counted at {thing} each spring is", str(count)),
        "material": (f"Its oldest building is made of {material}.", f"What is the oldest building at {thing} made of?",
                     f"The oldest building at {thing} is made of", material),
    }
    lines = [facts[a][0] for a in ATTRS]
    rng.shuffle(lines)
    text = f"{thing} is a {kind}. " + " ".join(lines)
    _, q, cloze, ans = facts[attr]
    return f"{text}\n\n{_ask(style, thing, q, cloze)}", f" {ans}"


def _two_hop(rng: Rng, style: str) -> tuple[str, str]:
    a, town, region = name(rng), name(rng), name(rng)
    other_town, other_region = name(rng), name(rng)
    lines = [f"The {rng.choice(KINDS_OF_THING)} {a} stands in {town}.", f"{town} is a town in the province of {region}.",
             f"{other_town} is a town in the province of {other_region}."]
    rng.shuffle(lines)
    return (f"{' '.join(lines)}\n\n" + _ask(style, a, f"In which province does {a} stand?", f"{a} stands in the province of"), f" {region}")


def _rule(rng: Rng, rule: str) -> tuple[str, str]:
    n = rng.between(3, 5)
    if rule == "affine":
        a, b = rng.between(2, 9), rng.between(1, 20)
        xs = rng.sample(list(range(1, 30)), n + 1)
        shots = "\n".join(f"{x} -> {a * x + b}" for x in xs[:n])
        return f"{shots}\n{xs[n]} ->", f" {a * xs[n] + b}"
    if rule == "glossary":
        meanings = rng.sample([*COLORS, *(o[0] for o in OBJECTS)], n)
        words = [word(rng, 2) for _ in range(n)]
        gloss = "\n".join(f'"{w}" means "{m}".' for w, m in zip(words, meanings, strict=True))
        k = rng.below(n)
        return f"In the language of {name(rng)}:\n{gloss}\n\nIn that language, the word for \"{meanings[k]}\" is \"", f"{words[k]}\""
    if rule == "category":
        cat = word(rng, 2)
        yes, no = rng.sample(COLORS, 2), [o[0] for o in rng.sample(OBJECTS, 2)]
        shots = [f"Is {w} a {cat}? Yes." for w in yes] + [f"Is {_a(w)} a {cat}? No." for w in no]
        rng.shuffle(shots)
        query = rng.choice([c for c in COLORS if c not in yes])
        return f"{chr(10).join(shots)}\nIs {query} a {cat}?", " Yes."
    words = [word(rng) for _ in range(n + 1)]
    suffix = word(rng, 1)
    f = {"reverse": lambda w: w[::-1], "suffix": lambda w: w + suffix, "swap": lambda w: w[-1] + w[1:-1] + w[0]}[rule]
    shots = "\n".join(f"{w} -> {f(w)}" for w in words[:n])
    return f"{shots}\n{words[n]} ->", f" {f(words[n])}"


def _definition(rng: Rng, kind: str) -> tuple[str, str]:
    term = word(rng, 2)
    per = rng.between(2, 12)
    k = rng.between(2, 15)
    thing = rng.choice(OBJECTS)[1]
    if kind == "container":
        return (f"A {term} is a box that holds exactly {per} {thing}. How many {term}s are needed to hold {per * k} {thing}?\nAnswer:", f" {k}")
    return (f"In the old markets of {name(rng)}, one {term} was equal to {per} coins. A cloth cost {k} {term}s. How many coins was that?\nAnswer:",
            f" {per * k}")


def make(rng: Rng, corner: str) -> Item:
    parts = corner.split("/")
    kind = parts[1]
    if kind == "passage":
        ctx, cont = _passage(rng, parts[2], parts[3])
    elif kind == "two-hop":
        ctx, cont = _two_hop(rng, parts[2])
    elif kind == "rule":
        ctx, cont = _rule(rng, parts[2])
    else:
        ctx, cont = _definition(rng, parts[2])
    fmt = "list" if kind == "rule" else "prose"
    return Item(kind="answer", corner=corner, fmt=fmt, lang="en", source=f"gen:{NAME}", ctx=ctx, cont=cont)

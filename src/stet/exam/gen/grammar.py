"""Grammatical minimal pairs, made from templates: two sentences that differ in one place, one
acceptable and one not. The score is how many bits the student prefers the acceptable one by."""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from .lexicon import NAMES

NAME = "grammar"
NOUNS = [("doctor", "doctors"), ("teacher", "teachers"), ("child", "children"), ("woman", "women"), ("man", "men"),
         ("student", "students"), ("dog", "dogs"), ("artist", "artists"), ("farmer", "farmers"), ("pilot", "pilots"),
         ("baker", "bakers"), ("singer", "singers"), ("mouse", "mice"), ("person", "people"), ("goose", "geese"),
         ("nurse", "nurses"), ("lawyer", "lawyers"), ("driver", "drivers"), ("neighbour", "neighbours"), ("guest", "guests")]
INTRANSITIVE = [("sleeps", "sleep"), ("laughs", "laugh"), ("waits", "wait"), ("sings", "sing"), ("works", "work"),
                ("smiles", "smile"), ("arrives", "arrive"), ("leaves", "leave"), ("complains", "complain"), ("rests", "rest")]
TRANSITIVE = [("sees", "see", "saw"), ("likes", "like", "liked"), ("helps", "help", "helped"), ("knows", "know", "knew"),
              ("visits", "visit", "visited"), ("admires", "admire", "admired"), ("thanks", "thank", "thanked"),
              ("praises", "praise", "praised")]
PARTICIPLES = [("taken", "took"), ("written", "wrote"), ("eaten", "ate"), ("broken", "broke"), ("chosen", "chose"),
               ("given", "gave"), ("seen", "saw"), ("forgotten", "forgot"), ("driven", "drove"), ("spoken", "spoke")]
OBJECTS = ["letter", "cake", "window", "song", "plan", "map", "note", "book"]
VOWEL_NOUNS = ["apple", "orange", "egg", "onion", "umbrella", "idea", "island", "elephant", "octopus", "envelope"]
CONSONANT_NOUNS = ["banana", "pear", "table", "lamp", "river", "garden", "kettle", "horse", "pencil", "ladder"]
PHENOMENA = ["agree-simple-sg", "agree-simple-pl", "agree-attractor-sg", "agree-attractor-pl", "agree-relative",
             "det-this", "det-these", "reflexive", "participle", "article-an", "article-a", "npi-ever", "there-are"]


def corners() -> list[str]:
    return [f"{NAME}/{p}" for p in PHENOMENA]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def make(rng: Rng, corner: str) -> Item:
    k = corner.split("/", 1)[1]
    (n1s, n1p), (n2s, n2p) = rng.sample(NOUNS, 2)
    vs, vp = rng.choice(INTRANSITIVE)
    ts, tp, tpast = rng.choice(TRANSITIVE)
    name = rng.choice(NAMES)
    if k == "agree-simple-sg":
        good, bad = f"The {n1s} {vs}.", f"The {n1s} {vp}."
    elif k == "agree-simple-pl":
        good, bad = f"The {n1p} {vp}.", f"The {n1p} {vs}."
    elif k == "agree-attractor-sg":
        good, bad = f"The {n1s} near the {n2p} {vs}.", f"The {n1s} near the {n2p} {vp}."
    elif k == "agree-attractor-pl":
        good, bad = f"The {n1p} beside the {n2s} {vp}.", f"The {n1p} beside the {n2s} {vs}."
    elif k == "agree-relative":
        good, bad = f"The {n1s} that the {n2p} {tp} {vs}.", f"The {n1s} that the {n2p} {tp} {vp}."
    elif k == "det-this":
        good, bad = f"{name} {ts} this {n1s}.", f"{name} {ts} this {n1p}."
    elif k == "det-these":
        good, bad = f"{name} {ts} these {n1p}.", f"{name} {ts} these {n1s}."
    elif k == "reflexive":
        good, bad = f"The {n1p} {tpast} themselves.", f"The {n1p} {tpast} itself."
    elif k == "participle":
        part, past = rng.choice(PARTICIPLES)
        obj = rng.choice(OBJECTS)
        good, bad = f"{name} has {part} the {obj}.", f"{name} has {past} the {obj}."
    elif k == "article-an":
        n = rng.choice(VOWEL_NOUNS)
        good, bad = f"{name} found an {n}.", f"{name} found a {n}."
    elif k == "article-a":
        n = rng.choice(CONSONANT_NOUNS)
        good, bad = f"{name} found a {n}.", f"{name} found an {n}."
    elif k == "npi-ever":
        part, _ = rng.choice(PARTICIPLES)
        obj = rng.choice(OBJECTS)
        good, bad = f"Nobody has ever {part} that {obj}.", f"Somebody has ever {part} that {obj}."
    elif k == "there-are":
        good, bad = f"There are many {n1p} here.", f"There are many {n1s} here."
    else:
        raise ValueError(corner)
    return Item(kind="pair", corner=corner, fmt="prose", lang="en", source=f"gen:{NAME}", good=_cap(good), bad=_cap(bad))

"""Reasoning in a few sentences: orderings, chains, syllogisms, truth values, counting.

Orderings are asked in every one of the exam's languages. A comparison is written so that nothing
in it changes with who is compared: a name stands only where the language takes it as it is
written (never after a preposition that would bend it), and where an adjective would have to
agree with a man or a woman, a verb is used instead (Spanish "pesa más que", not "es más alto
que"). The other kinds are English only: their sentences need articles and endings that code
cannot be trusted to get right.
"""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from .lexicon import ANSWER, COLORS, GIVEN, NAMES, gap

NAME = "logic"
RELATIONS = [("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest"),
             ("heavier", "heaviest", "lightest")]
KINDS = [f"order{n}-{r[0]}-{end}" for n in (3, 4) for r in RELATIONS for end in ("top", "bottom")]
KINDS += ["chain", "syllogism-yes", "syllogism-no", "boolean", "counting", "between"]
OTHER_KINDS = ["order3-top", "order3-bottom", "order4-top", "order4-bottom", "between"]  # in every other language
NONCE = ["blicket", "dax", "wug", "fep", "toma", "zorb", "glim", "plon", "vosk", "trell"]
PROPS = ["shiny", "loud", "soft", "round", "cold", "striped", "hollow", "sweet"]
CONTAINERS = ["box", "drawer", "cupboard", "bag", "basket", "chest", "case", "crate"]
THINGS = ["key", "ring", "coin", "letter", "map", "button", "ticket", "stone"]
# In each language, a few comparisons: A is more than B; who is the most; who is the least.
# In order: older, taller, faster, heavier where the language compares these without agreement;
# elsewhere has more points, runs faster, weighs more, earns more.
COMPARE = {
    "ru": [("{a} старше, чем {b}.", "Кто старше всех?", "Кто младше всех?"),
           ("{a} выше, чем {b}.", "Кто выше всех?", "Кто ниже всех?"),
           ("{a} бегает быстрее, чем {b}.", "Кто бегает быстрее всех?", "Кто бегает медленнее всех?"),
           ("{a} тяжелее, чем {b}.", "Кто тяжелее всех?", "Кто легче всех?")],
    "de": [("{a} ist älter als {b}.", "Wer ist am ältesten?", "Wer ist am jüngsten?"),
           ("{a} ist größer als {b}.", "Wer ist am größten?", "Wer ist am kleinsten?"),
           ("{a} ist schneller als {b}.", "Wer ist am schnellsten?", "Wer ist am langsamsten?"),
           ("{a} ist schwerer als {b}.", "Wer ist am schwersten?", "Wer ist am leichtesten?")],
    "ja": [("{a}は{b}より年上です。", "一番年上なのは誰ですか？", "一番年下なのは誰ですか？"),
           ("{a}は{b}より背が高いです。", "一番背が高いのは誰ですか？", "一番背が低いのは誰ですか？"),
           ("{a}は{b}より足が速いです。", "一番足が速いのは誰ですか？", "一番足が遅いのは誰ですか？"),
           ("{a}は{b}より体重が重いです。", "一番体重が重いのは誰ですか？", "一番体重が軽いのは誰ですか？")],
    "zh": [("{a}比{b}年纪大。", "谁的年纪最大？", "谁的年纪最小？"),
           ("{a}比{b}高。", "谁最高？", "谁最矮？"),
           ("{a}比{b}跑得快。", "谁跑得最快？", "谁跑得最慢？"),
           ("{a}比{b}重。", "谁最重？", "谁最轻？")],
    "es": [("{a} tiene más puntos que {b}.", "¿Quién tiene más puntos?", "¿Quién tiene menos puntos?"),
           ("{a} corre más rápido que {b}.", "¿Quién corre más rápido?", "¿Quién corre más despacio?"),
           ("{a} pesa más que {b}.", "¿Quién pesa más?", "¿Quién pesa menos?"),
           ("{a} gana más que {b}.", "¿Quién gana más?", "¿Quién gana menos?")],
    "fr": [("{a} a plus de points que {b}.", "Qui a le plus de points ?", "Qui a le moins de points ?"),
           ("{a} court plus vite que {b}.", "Qui court le plus vite ?", "Qui court le moins vite ?"),
           ("{a} pèse plus que {b}.", "Qui pèse le plus ?", "Qui pèse le moins ?"),
           ("{a} gagne plus que {b}.", "Qui gagne le plus ?", "Qui gagne le moins ?")],
    "pt": [("{a} tem mais pontos do que {b}.", "Quem tem mais pontos?", "Quem tem menos pontos?"),
           ("{a} corre mais rápido do que {b}.", "Quem corre mais rápido?", "Quem corre mais devagar?"),
           ("{a} pesa mais do que {b}.", "Quem pesa mais?", "Quem pesa menos?"),
           ("{a} ganha mais do que {b}.", "Quem ganha mais?", "Quem ganha menos?")],
    "it": [("{a} ha più punti di {b}.", "Chi ha più punti di tutti?", "Chi ha meno punti di tutti?"),
           ("{a} corre più velocemente di {b}.", "Chi corre più velocemente di tutti?", "Chi corre più lentamente di tutti?"),
           ("{a} pesa più di {b}.", "Chi pesa più di tutti?", "Chi pesa meno di tutti?"),
           ("{a} guadagna più di {b}.", "Chi guadagna più di tutti?", "Chi guadagna meno di tutti?")],
    "pl": [("{a} ma więcej punktów niż {b}.", "Kto ma najwięcej punktów?", "Kto ma najmniej punktów?"),
           ("{a} biega szybciej niż {b}.", "Kto biega najszybciej?", "Kto biega najwolniej?"),
           ("{a} waży więcej niż {b}.", "Kto waży najwięcej?", "Kto waży najmniej?"),
           ("{a} zarabia więcej niż {b}.", "Kto zarabia najwięcej?", "Kto zarabia najmniej?")],
    "nl": [("{a} is ouder dan {b}.", "Wie is het oudst?", "Wie is het jongst?"),
           ("{a} is groter dan {b}.", "Wie is het grootst?", "Wie is het kleinst?"),
           ("{a} is sneller dan {b}.", "Wie is het snelst?", "Wie is het langzaamst?"),
           ("{a} is zwaarder dan {b}.", "Wie is het zwaarst?", "Wie is het lichtst?")],
}
# Who is less than A and more than C, by the language's first comparison.
BETWEEN = {
    "ru": "Кто младше, чем {a}, но старше, чем {c}?",
    "de": "Wer ist jünger als {a}, aber älter als {c}?",
    "ja": "{a}より年下で、{c}より年上なのは誰ですか？",
    "zh": "谁比{a}年纪小，但比{c}年纪大？",
    "es": "¿Quién tiene menos puntos que {a} pero más puntos que {c}?",
    "fr": "Qui a moins de points que {a} mais plus de points que {c} ?",
    "pt": "Quem tem mais pontos do que {c}, mas menos pontos do que {a}?",
    "it": "Chi ha meno punti di {a} ma più punti di {c}?",
    "pl": "Kto ma mniej punktów niż {a}, ale więcej punktów niż {c}?",
    "nl": "Wie is jonger dan {a} maar ouder dan {c}?",
}


def corners() -> list[str]:
    return [f"{NAME}/en/{k}" for k in KINDS] + [f"{NAME}/{lang}/{k}" for lang in COMPARE for k in OTHER_KINDS]


def _ordering(rng: Rng, lang: str, k: str) -> tuple[str, str]:
    """An ordering in a language other than English: the question, and who the answer is."""
    if k == "between":
        a, b, c = rng.sample(GIVEN[lang], 3)
        fact = COMPARE[lang][0][0]
        facts = [fact.format(a=a, b=b), fact.format(a=b, b=c)]
        ask, ans = BETWEEN[lang].format(a=a, c=c), b
    else:
        fact, most, least = rng.choice(COMPARE[lang])
        people = rng.sample(GIVEN[lang], int(k[5]))  # people[0] is the most
        facts = [fact.format(a=x, b=y) for x, y in zip(people, people[1:], strict=False)]
        ask, ans = (most, people[0]) if k.endswith("top") else (least, people[-1])
    rng.shuffle(facts)
    return gap(lang).join([*facts, ask]) + "\n" + ANSWER[lang], gap(lang) + ans


def make(rng: Rng, corner: str) -> Item:
    _, lang, k = corner.split("/")
    if lang != "en":
        ctx, cont = _ordering(rng, lang, k)
        return Item(kind="answer", corner=corner, fmt="prose", lang=lang, source=f"gen:{NAME}", ctx=ctx, cont=cont)
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

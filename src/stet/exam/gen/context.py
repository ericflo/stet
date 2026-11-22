"""Reading and learning in context: the answer is in the item's own context, about things that
exist nowhere else, so it can be read but never remembered.

- A short account of an invented place, company, river or person, then a question about it,
  asked one of several ways; some need two facts joined.
- A few examples of an invented rule (a cipher, a suffix, a number function, words of an
  invented language, a made-up category), then a new case.
- An invented unit or container, and a count that follows from its definition.

What these reward is the ability pretraining is for and memory cannot fake: taking in
something new and using it.

The account, the two facts to join and the glossary are asked in every one of the exam's languages.
Outside English the account is always of a town, in sentences written out by hand for each
language. What the seed fills in is only what a sentence takes unchanged: coined names set after
the noun they name ("the town of Brano", "the river Kelu"), a person's name as the subject, numbers
put where no word has to agree with them, and quoted words.
"""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from .lexicon import ANSWER, COLORS, GIVEN, NAMES, NOUNS, OBJECTS, QUESTION, coin, coin_name, gap

NAME = "context"
ONSETS = ["b", "br", "d", "dr", "f", "g", "gl", "h", "k", "kr", "l", "m", "n", "p", "pl", "r", "s", "st", "t", "tr", "v", "z"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ou", "ei"]
CODAS = ["", "", "n", "r", "l", "s", "th", "m", "k"]
KINDS_OF_THING = ["town", "company", "river", "festival", "school", "ship", "mountain", "bridge"]
ATTRS = ["founded", "place", "leader", "colour", "count", "material"]
STYLES = ["qa", "question", "cloze", "according"]
RULES = ["reverse", "suffix", "affine", "glossary", "category", "swap"]
MATERIALS = ["granite", "oak", "copper", "glass", "slate", "cedar", "brass", "marble"]
TOWN_ATTRS = ["year", "river", "region", "bridges"]
# A town in each language. "is": T is a town. Then for each thing told of it: the sentence that
# tells it, the question, the same as a sentence to finish, and the answer. They say, in order:
# the town was founded in the year Y; the river V runs through it; it lies in the province R; the
# number of its bridges is N (said as a number of bridges, so that no noun follows the number).
# "hop": P lives in the town A; the town A lies in the province R; in which province does P live;
# P lives in the province; R. "words": in the language N; "W" means "M"; in that language the
# word for "M" is "; and the mark that closes the quotation.
TOLD = {
    "ru": {
        "is": "{t} — это город.",
        "year": ("Город был основан в {y} году.", "В каком году был основан город {t}?", "Город {t} был основан в", "{y}"),
        "river": ("Через город протекает река {v}.", "Какая река протекает через город {t}?", "Через город {t} протекает река", "{v}"),
        "region": ("Город находится в провинции {r}.", "В какой провинции находится город {t}?", "Город {t} находится в провинции", "{r}"),
        "bridges": ("Число мостов в городе — {n}.", "Сколько мостов в городе {t}?", "Число мостов в городе {t} —", "{n}"),
        "hop": ("{p} живёт в городе {a}.", "Город {a} находится в провинции {r}.", "В какой провинции живёт {p}?", "{p} живёт в провинции", "{r}"),
        "words": ("В языке {n}:", '"{w}" означает "{m}".', 'На этом языке слово "{m}" звучит как "', '"'),
    },
    "de": {
        "is": "{t} ist eine Stadt.",
        "year": ("Die Stadt wurde im Jahr {y} gegründet.", "In welchem Jahr wurde die Stadt {t} gegründet?", "Die Stadt {t} wurde gegründet im Jahr", "{y}"),
        "river": ("Durch die Stadt fließt der Fluss {v}.", "Welcher Fluss fließt durch die Stadt {t}?", "Durch die Stadt {t} fließt der Fluss", "{v}"),
        "region": ("Die Stadt liegt in der Provinz {r}.", "In welcher Provinz liegt die Stadt {t}?", "Die Stadt {t} liegt in der Provinz", "{r}"),
        "bridges": ("Die Zahl der Brücken in der Stadt beträgt {n}.", "Wie viele Brücken hat die Stadt {t}?", "Die Zahl der Brücken in der Stadt {t} beträgt", "{n}"),
        "hop": ("{p} wohnt in der Stadt {a}.", "Die Stadt {a} liegt in der Provinz {r}.", "In welcher Provinz wohnt {p}?", "{p} wohnt in der Provinz", "{r}"),
        "words": ("In der Sprache {n}:", '"{w}" bedeutet "{m}".', 'In dieser Sprache ist das Wort für "{m}" "', '"'),
    },
    "ja": {
        "is": "{t}は町です。",
        "year": ("この町は{y}年にできました。", "{t}という町ができたのは何年ですか？", "{t}という町ができた年は", "{y}年"),
        "river": ("この町には{v}川が流れています。", "{t}という町に流れている川の名前は何ですか？", "{t}という町に流れている川は", "{v}川"),
        "region": ("この町は{r}県にあります。", "{t}という町は何県にありますか？", "{t}という町がある県は", "{r}県"),
        "bridges": ("この町には橋が{n}本あります。", "{t}という町には橋が何本ありますか？", "{t}という町にある橋の数は", "{n}本"),
        "hop": ("{p}は{a}町に住んでいます。", "{a}町は{r}県にあります。", "{p}は何県に住んでいますか？", "{p}が住んでいる県は", "{r}県"),
        "words": ("{n}語の単語：", "「{w}」は「{m}」という意味です。", "{n}語で「{m}」を意味する単語は「", "」"),
    },
    "zh": {
        "is": "{t}是一个镇。",
        "year": ("该镇建于{y}年。", "{t}镇建于哪一年？", "{t}镇建于", "{y}年"),
        "river": ("{v}河流经该镇。", "流经{t}镇的河叫什么名字？", "流经{t}镇的河是", "{v}河"),
        "region": ("该镇位于{r}省。", "{t}镇位于哪个省？", "{t}镇位于", "{r}省"),
        "bridges": ("该镇有{n}座桥。", "{t}镇有多少座桥？", "{t}镇的桥共有", "{n}座"),
        "hop": ("{p}住在{a}镇。", "{a}镇位于{r}省。", "{p}住在哪个省？", "{p}住的省是", "{r}省"),
        "words": ("在{n}语中：", "“{w}”的意思是“{m}”。", "在{n}语中，表示“{m}”的词是“", "”"),
    },
    "es": {
        "is": "{t} es una ciudad.",
        "year": ("La ciudad fue fundada en el año {y}.", "¿En qué año fue fundada la ciudad de {t}?", "La ciudad de {t} fue fundada en el año", "{y}"),
        "river": ("Por la ciudad pasa el río {v}.", "¿Qué río pasa por la ciudad de {t}?", "Por la ciudad de {t} pasa el río", "{v}"),
        "region": ("La ciudad está en la provincia de {r}.", "¿En qué provincia está la ciudad de {t}?", "La ciudad de {t} está en la provincia de", "{r}"),
        "bridges": ("El número de puentes de la ciudad es {n}.", "¿Cuántos puentes tiene la ciudad de {t}?", "El número de puentes de la ciudad de {t} es", "{n}"),
        "hop": ("{p} vive en la ciudad de {a}.", "La ciudad de {a} está en la provincia de {r}.", "¿En qué provincia vive {p}?", "{p} vive en la provincia de", "{r}"),
        "words": ("En el idioma {n}:", '"{w}" significa "{m}".', 'En ese idioma, la palabra para "{m}" es "', '"'),
    },
    "fr": {
        "is": "{t} est une ville.",
        "year": ("La ville a été fondée en {y}.", "En quelle année la ville de {t} a-t-elle été fondée ?", "La ville de {t} a été fondée en", "{y}"),
        "river": ("La rivière {v} traverse la ville.", "Quelle rivière traverse la ville de {t} ?", "La ville de {t} est traversée par la rivière", "{v}"),
        "region": ("La ville se trouve dans la province de {r}.", "Dans quelle province se trouve la ville de {t} ?", "La ville de {t} se trouve dans la province de", "{r}"),
        "bridges": ("Le nombre de ponts de la ville est de {n}.", "Combien de ponts la ville de {t} a-t-elle ?", "Le nombre de ponts de la ville de {t} est de", "{n}"),
        "hop": ("{p} habite dans la ville de {a}.", "La ville de {a} se trouve dans la province de {r}.", "Dans quelle province habite {p} ?", "{p} habite dans la province de", "{r}"),
        "words": ("Dans la langue {n} :", '"{w}" signifie "{m}".', 'Dans cette langue, le mot pour "{m}" est "', '"'),
    },
    "pt": {
        "is": "{t} é uma cidade.",
        "year": ("A cidade foi fundada no ano de {y}.", "Em que ano foi fundada a cidade de {t}?", "A cidade de {t} foi fundada no ano de", "{y}"),
        "river": ("O rio {v} atravessa a cidade.", "Que rio atravessa a cidade de {t}?", "A cidade de {t} é atravessada pelo rio", "{v}"),
        "region": ("A cidade fica na província de {r}.", "Em que província fica a cidade de {t}?", "A cidade de {t} fica na província de", "{r}"),
        "bridges": ("O número de pontes da cidade é {n}.", "Quantas pontes tem a cidade de {t}?", "O número de pontes da cidade de {t} é", "{n}"),
        "hop": ("{p} mora na cidade de {a}.", "A cidade de {a} fica na província de {r}.", "Em que província mora {p}?", "{p} mora na província de", "{r}"),
        "words": ("Na língua {n}:", '"{w}" significa "{m}".', 'Nessa língua, a palavra para "{m}" é "', '"'),
    },
    "it": {
        "is": "{t} è una città.",
        "year": ("La città è stata fondata nell'anno {y}.", "In che anno è stata fondata la città di {t}?", "La città di {t} è stata fondata nell'anno", "{y}"),
        "river": ("Il fiume {v} attraversa la città.", "Quale fiume attraversa la città di {t}?", "La città di {t} è attraversata dal fiume", "{v}"),
        "region": ("La città si trova nella provincia di {r}.", "In quale provincia si trova la città di {t}?", "La città di {t} si trova nella provincia di", "{r}"),
        "bridges": ("Il numero di ponti della città è {n}.", "Quanti ponti ha la città di {t}?", "Il numero di ponti della città di {t} è", "{n}"),
        "hop": ("{p} abita nella città di {a}.", "La città di {a} si trova nella provincia di {r}.", "In quale provincia abita {p}?", "{p} abita nella provincia di", "{r}"),
        "words": ("Nella lingua {n}:", '"{w}" significa "{m}".', 'In quella lingua, la parola per "{m}" è "', '"'),
    },
    "pl": {
        "is": "{t} to miasto.",
        "year": ("Miasto zostało założone w {y} roku.", "W którym roku zostało założone miasto {t}?", "Miasto {t} zostało założone w roku", "{y}"),
        "river": ("Przez miasto płynie rzeka {v}.", "Jaka rzeka płynie przez miasto {t}?", "Przez miasto {t} płynie rzeka", "{v}"),
        "region": ("Miasto leży w prowincji {r}.", "W jakiej prowincji leży miasto {t}?", "Miasto {t} leży w prowincji", "{r}"),
        "bridges": ("Liczba mostów w mieście wynosi {n}.", "Ile mostów ma miasto {t}?", "Liczba mostów w mieście {t} wynosi", "{n}"),
        "hop": ("{p} mieszka w mieście {a}.", "Miasto {a} leży w prowincji {r}.", "W jakiej prowincji mieszka {p}?", "{p} mieszka w prowincji", "{r}"),
        "words": ("W języku {n}:", '"{w}" znaczy "{m}".', 'W tym języku słowo "{m}" to "', '"'),
    },
    "nl": {
        "is": "{t} is een stad.",
        "year": ("De stad werd gesticht in het jaar {y}.", "In welk jaar werd de stad {t} gesticht?", "De stad {t} werd gesticht in het jaar", "{y}"),
        "river": ("Door de stad stroomt de rivier {v}.", "Welke rivier stroomt door de stad {t}?", "Door de stad {t} stroomt de rivier", "{v}"),
        "region": ("De stad ligt in de provincie {r}.", "In welke provincie ligt de stad {t}?", "De stad {t} ligt in de provincie", "{r}"),
        "bridges": ("Het aantal bruggen in de stad is {n}.", "Hoeveel bruggen heeft de stad {t}?", "Het aantal bruggen in de stad {t} is", "{n}"),
        "hop": ("{p} woont in de stad {a}.", "De stad {a} ligt in de provincie {r}.", "In welke provincie woont {p}?", "{p} woont in de provincie", "{r}"),
        "words": ("In de taal {n}:", '"{w}" betekent "{m}".', 'In die taal is het woord voor "{m}" "', '"'),
    },
}


def _a(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def word(rng: Rng, syllables: int | None = None) -> str:
    """A word that means nothing, from the seed: nobody can have read it before."""
    n = syllables or rng.between(2, 3)
    return "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(n))


def name(rng: Rng) -> str:
    return word(rng).capitalize()


def corners() -> list[str]:
    out = [f"{NAME}/en/passage/{a}/{s}" for a in ATTRS for s in STYLES]
    out += [f"{NAME}/en/two-hop/{s}" for s in STYLES]
    out += [f"{NAME}/en/rule/{r}" for r in RULES]
    out += [f"{NAME}/en/definition/{k}" for k in ("container", "unit")]
    for lang in TOLD:
        out += [f"{NAME}/{lang}/passage/{a}" for a in TOWN_ATTRS] + [f"{NAME}/{lang}/two-hop", f"{NAME}/{lang}/glossary"]
    return out


def _coined(rng: Rng, lang: str, n: int, names: bool = True) -> list[str]:
    """n coined words, no two the same: names, or plain words."""
    out: list[str] = []
    while len(out) < n:
        w = coin_name(rng, lang) if names else coin(rng, lang, 3)
        if w not in out:
            out.append(w)
    return out


def _put(rng: Rng, lang: str, told: str, question: str, cloze: str) -> str:
    """What is told, then the question: under a heading, or as a sentence to finish."""
    if rng.chance(1, 2):
        return f"{told}\n\n{QUESTION[lang]}{gap(lang)}{question}\n{ANSWER[lang]}"
    return f"{told}\n\n{cloze}"


def _told(rng: Rng, lang: str, kind: str, attr: str) -> tuple[str, str]:
    """A question in a language other than English, and its answer."""
    f, sep = TOLD[lang], gap(lang)
    if kind == "passage":
        t, v, r = _coined(rng, lang, 3)
        slots = {"t": t, "v": v, "r": r, "y": rng.between(1200, 1990), "n": rng.between(12, 60)}
        lines = [f[a][0].format(**slots) for a in TOWN_ATTRS]
        rng.shuffle(lines)
        _, question, cloze, ans = f[attr]
        told = sep.join([f["is"].format(**slots), *lines])
        return _put(rng, lang, told, question.format(**slots), cloze.format(**slots)), sep + ans.format(**slots)
    if kind == "two-hop":
        a, b, r, s = _coined(rng, lang, 4)
        p = rng.choice(GIVEN[lang])
        lives, lies, question, cloze, ans = f["hop"]
        lines = [lives.format(p=p, a=a), lies.format(a=a, r=r), lies.format(a=b, r=s)]
        rng.shuffle(lines)
        return _put(rng, lang, sep.join(lines), question.format(p=p), cloze.format(p=p)), sep + ans.format(r=r)
    head, line, ask, close = f["words"]
    n = rng.between(3, 5)
    meanings, words = rng.sample(NOUNS[lang], n), _coined(rng, lang, n, names=False)
    tongue = coin_name(rng, lang)
    k = rng.below(n)
    gloss = "\n".join(line.format(w=w, m=m) for w, m in zip(words, meanings, strict=True))
    return f"{head.format(n=tongue)}\n{gloss}\n\n{ask.format(n=tongue, m=meanings[k])}", f"{words[k]}{close}"


def _ask(style: str, subject: str, question: str, cloze: str) -> str:
    """The same question, put one of several ways; the answer follows each."""
    # In the middle of a sentence "The" loses its capital. A name that begins the sentence keeps its own.
    mid = cloze if cloze.startswith(subject) else f"{cloze[0].lower()}{cloze[1:]}"
    return {"qa": f"Q: {question}\nA:", "question": f"Question: {question}\nAnswer:", "cloze": cloze,
            "according": f"According to the passage above, {mid}"}[style]


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
    _, lang, kind, *rest = corner.split("/")
    if lang != "en":
        ctx, cont = _told(rng, lang, kind, rest[0] if rest else "")
    elif kind == "passage":
        ctx, cont = _passage(rng, rest[0], rest[1])
    elif kind == "two-hop":
        ctx, cont = _two_hop(rng, rest[0])
    elif kind == "rule":
        ctx, cont = _rule(rng, rest[0])
    else:
        ctx, cont = _definition(rng, rest[0])
    fmt = "list" if kind in ("rule", "glossary") else "prose"
    return Item(kind="answer", corner=corner, fmt=fmt, lang=lang, source=f"gen:{NAME}", ctx=ctx, cont=cont)

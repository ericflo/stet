"""This curator tries to buy the exam with ink. It is entered so that the board shows what
that buys.

A quarter of the exam is generated skills, and the forms they take are public: a session at
a shell, a tool call and its JSON result, a passage and a question answered from it, a small
transformation, a logic puzzle, the right sentence of a pair. Only the round's seed is not.
So scribe:

  1. Makes `tidy`'s selection, unchanged, and writes it straight away.
  2. Writes worked problems in those forms, each with its answer, until the ink allowance
     (task.info["ink_tokens"]) is spent, an equal share on each of the five skills. They come
     from generators of its own and a seed of its own: it cannot know a round's questions,
     only what a question looks like.
  3. Puts each problem in the script as a document of its own, spread through the part of
     the script the student is trained on fewest times, and writes the script again.

Ink is counted in tokens of what is trained, repeats included, so scribe counts as the
grader does: with the season's tokenizer, which every sandbox holds (stet.curator itself
imports it), dealing the script into the student's lanes as stet.script does. If the
tokenizer cannot be loaded it spends nothing and is `tidy`.

If this ever beats `tidy` by more than noise, the exam can be bought with a few thousand
tokens of its own forms, and it is the exam that gets fixed.
"""

import hashlib
import heapq
import json
import random
import re
import sys
import unicodedata

from stet.curator import Task

# --- tidy's selection, copied from examples/tidy ------------------------------------------------

# How a sentence ends: in scripts written with spaces, a mark and then a space; in Chinese and
# Japanese, the mark alone.
SENTENCE_END = re.compile(r"[.!?…؟۔।][\"')\]”’»]?\s|[。！？｡]")
SPACELESS = re.compile("[぀-ヿ一-鿿฀-๿]")  # kana, Chinese characters, Thai
# The vowel signs of Hindi, Thai and Arabic are marks and not letters: they count as part of
# the letters they sit on. (Asking each character in turn is a hundred times slower, and a
# round's pages are gigabytes.)
MARKS = "".join(c for c in map(chr, range(0x300, 0x10000)) if unicodedata.category(c)[0] == "M")
WRITING = re.compile(rf"(?:[^\W\d_]|\s|[{re.escape(MARKS)}])+")  # letters of any script, spaces and marks
# Rough, under the season's tokenizer, and on the generous side: a stream that runs over the
# budget is cut off, one that falls short repeats from the top.
BYTES_PER_TOKEN = 3.7


def looks_written(text: str) -> bool:
    """Mostly letters, of any script, and enough of them to say something."""
    sample = text[:3000]
    if len(sample) < 200:
        return False
    return len(WRITING.sub("", sample)) <= 0.25 * len(sample)


def spaceless(text: str) -> bool:
    """Written without spaces between words, so that a line's length is counted in characters."""
    sample = text[:2000]
    return len(SPACELESS.findall(sample)) > 0.3 * len(sample)


def prose_ranges(raw: bytes, dense: bool) -> tuple[list[list[int]], int]:
    """Byte ranges of the lines worth keeping, each a whole line with its line break, and how
    many bytes they hold. A line is kept when it is long enough to be a sentence and has the
    end of one in it."""
    ranges: list[list[int]] = []
    pos = 0
    least = 25 if dense else 60
    for line in raw.split(b"\n"):
        n = len(line) + 1  # + the line break
        if n <= least:  # too few bytes to be that many characters: most of a page's furniture
            pos += n
            continue
        body = line.decode("utf-8", "replace").strip()
        long_enough = len(body) >= least and (dense or body.count(" ") >= 9)
        if long_enough and SENTENCE_END.search(body + " "):
            if ranges and ranges[-1][1] == pos:
                ranges[-1][1] = pos + n
            else:
                ranges.append([pos, pos + n])
        pos += n
    if ranges and ranges[-1][1] > len(raw):  # the last line may have no line break
        ranges[-1][1] = len(raw)
    return ranges, sum(b - a for a, b in ranges)


def score(size: int, kept: int) -> float:
    """Higher is better: a page that is mostly writing, and long enough to say something."""
    return kept / max(1, size) + min(1.0, kept / 3000)


def select(task: Task) -> list[dict]:
    """tidy's script."""
    corpus = task.corpus
    seen: set[bytes] = set()
    scored = []
    for i, text in corpus:
        if not looks_written(text):
            continue
        ranges, kept = prose_ranges(corpus.raw(i), spaceless(text))
        if kept < 400:
            continue
        key = hashlib.blake2b(text[:2000].encode("utf-8", "replace"), digest_size=8).digest()
        if key in seen:
            continue
        seen.add(key)
        scored.append((score(corpus.size(i), kept), i, ranges, kept))

    scored.sort(key=lambda t: (-t[0], t[1]))
    budget_bytes = task.train_tokens * BYTES_PER_TOKEN
    chosen = []
    used = 0
    for _, i, ranges, kept in scored:
        if used >= budget_bytes:
            break
        chosen.append({"d": i, "k": ranges})
        used += kept
    random.Random(0).shuffle(chosen)
    return chosen


# --- the ink: worked problems in the forms the exam's skills take --------------------------------
#
# Each generator takes a random.Random and returns one problem with its answer, as the exam
# would show the two together. The words are scribe's own, and half of them are coined.

NAMES = ["Ada", "Bashir", "Cleo", "Dev", "Ebele", "Finn", "Gita", "Hoang", "Ilse", "Jun", "Kemal", "Lior", "Mirela", "Nuno", "Odile",
         "Pavel", "Rafa", "Sunita", "Tevita", "Ulrike", "Vasco", "Wanda", "Yoko", "Zofia"]
PLACES = ["Turin", "Cork", "Graz", "Bursa", "Cali", "Kochi", "Riga", "Sfax", "Davao", "Leeds", "Varna", "Salta"]
WORDS = ["anchor", "beacon", "cedar", "delta", "engine", "fabric", "gravel", "hollow", "ingot", "jasper", "kernel", "lumen", "mortar",
         "nickel", "onyx", "piston", "quill", "raven", "spruce", "timber", "umber", "vessel", "willow", "zephyr"]
COLOURS = ["red", "blue", "green", "yellow", "pink", "black", "white", "grey", "brown", "orange", "purple"]
THINGS = [("coin", "coins"), ("shell", "shells"), ("nail", "nails"), ("card", "cards"), ("bead", "beads"), ("seed", "seeds"),
          ("tile", "tiles"), ("jar", "jars"), ("spoon", "spoons"), ("rope", "ropes")]
ONSETS = "b ch d f g h j k l m n p r s sh t v w z".split()
CODAS = ["", "", "", "n", "r", "l", "s", "m"]


def coin(r: random.Random, syllables: int = 0) -> str:
    """A word nobody has written, two or three syllables long."""
    return "".join(r.choice(ONSETS) + r.choice("aeiou") + r.choice(CODAS) for _ in range(syllables or r.randint(2, 3)))


def name(r: random.Random) -> str:
    return coin(r).capitalize()


def word(r: random.Random) -> str:
    return coin(r) if r.random() < 0.5 else r.choice(WORDS)


def a(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def session(r: random.Random) -> str:
    """An agent at work in a small invented project: each command at a shell, or each tool
    call as JSON, and what comes back. Everything shown follows from what was shown before."""
    user, project = r.choice(NAMES).lower(), coin(r)
    home = f"/home/{user}/{project}"
    dirs = r.sample(["src", "data", "logs", "docs", "tests", "scripts", "config", "lib", "notes", "tools"], 3)
    notes, log = f"{coin(r)}.txt", f"{dirs[0]}/{coin(r)}.log"
    stamp = "2026-0{}-{} 0{}:{}:{}"
    files = {
        "README.md": f"# {project.title()}\n\nA small tool for {coin(r)} {r.choice(WORDS)}s.",
        notes: "\n".join(" ".join(word(r) for _ in range(r.randint(2, 4))) for _ in range(r.randint(3, 5))),
        log: "\n".join(
            f"{stamp.format(r.randint(1, 9), r.randint(10, 28), r.randint(0, 9), r.randint(10, 59), r.randint(10, 59))} "
            f"{r.choice(['INFO', 'INFO', 'WARN', 'ERROR', 'DEBUG'])} {word(r)} {r.choice(['started', 'stopped', 'failed', 'retrying', 'ok', 'timed out', 'connected', 'closed'])}"
            for _ in range(r.randint(4, 6))
        ),
        f"{dirs[1]}/{coin(r)}.csv": "name,city,count\n" + "\n".join(f"{r.choice(NAMES)},{r.choice(PLACES)},{r.randint(1, 99)}" for _ in range(r.randint(2, 4))),
        f"{dirs[2]}/{coin(r)}.py": f"def {r.choice(WORDS)}(x):\n    return x * {r.randint(2, 9)}",
    }
    shown: set[str] = set()

    def ls(d: str = ""):
        inside = f"{d}/" if d else ""
        listing = sorted({p[len(inside) :].split("/")[0] for p in files if p.startswith(inside)}, key=str.lower)
        return (f"ls {d}".strip(), {"tool": "list_dir", "path": d or "."}, listing)

    def cat(f: str) -> list:
        """The file, unless the session has it on the page already."""
        if f in shown:
            return []
        shown.add(f)
        return [(f"cat {f}", {"tool": "read_file", "path": f}, files[f])]

    steps = [ls()]  # (the command, the same as a tool call, what comes back)
    for kind in sorted(r.sample(["cat", "wc", "head", "grep", "ls-dir", "append", "rm", "cd"], r.randint(1, 2)), key="cd".__eq__):  # cd goes last
        f = r.choice([notes, log])
        lines = files[f].split("\n")
        if kind == "cat":
            steps += cat(f)
        elif kind == "wc":
            steps += [*cat(f), (f"wc -l {f}", {"tool": "count_lines", "path": f}, len(lines))]
        elif kind == "head":
            k = r.randint(1, len(lines) - 1)
            steps += [*cat(f), (f"head -n {k} {f}", {"tool": "read_file", "path": f, "lines": k}, "\n".join(lines[:k]))]
        elif kind == "grep":
            level = r.choice(sorted({line.split()[2] for line in files[log].split("\n")}))
            hits = [line for line in files[log].split("\n") if level in line]
            steps += [*cat(log), (f"grep {level} {log}", {"tool": "search", "pattern": level, "path": log}, hits)]
        elif kind == "ls-dir":
            steps.append(ls(r.choice(dirs)))
        elif kind == "append":
            line = " ".join(word(r) for _ in range(r.randint(2, 4)))
            steps += [*cat(notes), (f'echo "{line}" >> {notes}', {"tool": "append_file", "path": notes, "content": line}, "")]
            files[notes] += "\n" + line
            shown.discard(notes)
            steps += cat(notes)
        elif kind == "rm":
            del files["README.md"]
            steps += [("rm README.md", {"tool": "delete_file", "path": "README.md"}, ""), ls()]
        else:
            d = r.choice(dirs)
            steps += [("pwd", {"tool": "working_dir"}, home), (f"cd {d} && pwd", {"tool": "change_dir", "path": d}, f"{home}/{d}")]
    if r.random() < 0.5:
        out = [f"{user}@{project}:~/{project}"]
        for command, _, back in steps:
            if isinstance(back, list):
                back = ("\n" if command.startswith("grep") else "  ").join(back)
            elif isinstance(back, int):
                back = f"{back} {command.split()[-1]}"
            out.append(f"$ {command}\n{back}" if back else f"$ {command}")
    else:
        out = [f"# agent log: {project}"]
        for _, call, back in steps:
            out += [json.dumps(call), json.dumps({"result": back})]
    return "\n".join(out)


def ask(r: random.Random, question: str, cloze: str) -> str:
    """One question, put one of four ways. The answer follows each."""
    return r.choice([f"Q: {question}\nA:", f"Question: {question}\nAnswer:", cloze, f"According to the passage above, {cloze[0].lower()}{cloze[1:]}"])


def reading(r: random.Random) -> str:
    """Something invented, told in the problem itself, and a question it answers: an account
    and a fact from it, a fact that takes two sentences, a rule shown by examples, a definition."""
    kind = r.choice(["account", "account", "account", "two", "rule", "rule", "definition"])
    if kind == "account":
        thing, place, region, n = name(r), name(r), name(r), r.randint(3, 40)
        year, leader, colour = r.randint(1650, 2015), r.choice(NAMES), r.choice(COLOURS)
        material = r.choice(["granite", "oak", "copper", "glass", "slate", "cedar", "brass", "marble"])
        facts = [
            (f"{thing} was founded in {year}.", f"When was {thing} founded?", f"{thing} was founded in", year),
            (f"{thing} lies in {place}, a small district of {region}.", f"Where does {thing} lie?", f"{thing} lies in", place),
            (f"Since its last election, {thing} has been led by {leader}.", f"Who leads {thing}?", f"{thing} is led by", leader),
            (f"Its emblem is {a(colour)} circle on a white field.", f"What colour is the circle in {thing}'s emblem?", f"The circle in {thing}'s emblem is", colour),
            (f"It is known for its {n} {r.choice(THINGS)[1]}, counted every spring.", f"How many are counted at {thing} each spring?", f"The number counted at {thing} each spring is", n),
            (f"Its oldest building is made of {material}.", f"What is the oldest building at {thing} made of?", f"The oldest building at {thing} is made of", material),
        ]
        _, question, cloze, answer = r.choice(facts)
        told = [fact[0] for fact in facts]
        r.shuffle(told)
        kind_of = r.choice(["town", "company", "river", "festival", "school", "ship", "mountain", "bridge"])
        return f"{thing} is a {kind_of}. {' '.join(told)}\n\n{ask(r, question, cloze)} {answer}"
    if kind == "two":
        thing, town, region, other, elsewhere = (name(r) for _ in range(5))
        told = [f"The {r.choice(['mill', 'tower', 'school', 'bridge', 'market'])} {thing} stands in {town}.", f"{town} is a town in the province of {region}.", f"{other} is a town in the province of {elsewhere}."]
        r.shuffle(told)
        return f"{' '.join(told)}\n\n{ask(r, f'In which province does {thing} stand?', f'{thing} stands in the province of')} {region}"
    if kind == "definition":
        term, per, k, things = coin(r, 2), r.randint(2, 12), r.randint(2, 15), r.choice(THINGS)[1]
        if r.random() < 0.5:
            return f"A {term} is a box that holds exactly {per} {things}. How many {term}s are needed to hold {per * k} {things}?\nAnswer: {k}"
        return f"In the old markets of {name(r)}, one {term} was equal to {per} coins. A cloth cost {k} {term}s. How many coins was that?\nAnswer: {per * k}"
    n = r.randint(3, 5)  # a rule: n examples of it and one more case
    rule = r.choice(["reverse", "suffix", "swap", "affine", "glossary", "category"])
    if rule == "affine":
        times, plus = r.randint(2, 9), r.randint(1, 20)
        return "\n".join(f"{x} -> {times * x + plus}" for x in r.sample(range(1, 30), n + 1))
    if rule == "glossary":
        meanings, said = r.sample(COLOURS + [thing for thing, _ in THINGS], n), [coin(r, 2) for _ in range(n)]
        k = r.randrange(n)
        glossary = "\n".join(f'"{w}" means "{m}".' for w, m in zip(said, meanings, strict=True))
        return f'In the language of {name(r)}:\n{glossary}\n\nIn that language, the word for "{meanings[k]}" is "{said[k]}"'
    if rule == "category":
        category, yes, no = coin(r, 2), r.sample(COLOURS, 2), [thing for thing, _ in r.sample(THINGS, 2)]
        shots = [f"Is {w} a {category}? Yes." for w in yes] + [f"Is {a(w)} a {category}? No." for w in no]
        r.shuffle(shots)
        return "\n".join(shots) + f"\nIs {r.choice([c for c in COLOURS if c not in yes])} a {category}? Yes."
    suffix = coin(r, 1)
    change = {"reverse": lambda w: w[::-1], "suffix": lambda w: w + suffix, "swap": lambda w: w[-1] + w[1:-1] + w[0]}[rule]
    return "\n".join(f"{w} -> {change(w)}" for w in (coin(r) for _ in range(n + 1)))


def roman(n: int) -> str:
    out = ""
    for value, numeral in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + numeral, n - value
    return out


def transformation(r: random.Random) -> str:
    """One form turned into another: Markdown to HTML, CSV to JSON, a query to its result, a
    number to another notation, an identifier to another case."""
    kind = r.choice(["md-html", "csv-json", "sql-select", "sql-count", "sql-sum", "json-path", "roman", "from-roman", "binary", "hex", "thousands", "snake-camel", "camel-snake", "title-case"])
    people = [{"name": who, "city": r.choice(PLACES), "age": r.randint(18, 80)} for who in r.sample(NAMES, r.randint(4, 7))]
    if kind == "md-html":
        w = r.sample(WORDS, 3)
        markdown, html = r.choice([
            (f"The **{w[0]}** is here.", f"<p>The <strong>{w[0]}</strong> is here.</p>"),
            (f"One *{w[0]}* and one {w[1]}.", f"<p>One <em>{w[0]}</em> and one {w[1]}.</p>"),
            (f"## {w[0].title()} {w[1]}", f"<h2>{w[0].title()} {w[1]}</h2>"),
            ("\n".join(f"- {x}" for x in w), "<ul>\n" + "\n".join(f"<li>{x}</li>" for x in w) + "\n</ul>"),
            (f"See [{w[0]}](https://example.org/{w[1]}).", f'<p>See <a href="https://example.org/{w[1]}">{w[0]}</a>.</p>'),
        ])
        return f"Markdown:\n{markdown}\n\nHTML:\n{html}"
    if kind == "csv-json":
        rows = people[: r.randint(1, 3)]
        return "CSV:\nname,city,age\n" + "\n".join(f"{p['name']},{p['city']},{p['age']}" for p in rows) + f"\n\nThe same rows as JSON:\n{json.dumps(rows, ensure_ascii=False)}"
    if kind == "json-path":
        i, field = r.randrange(3), r.choice(["name", "city", "age"])
        return f"data = {json.dumps({'people': people[:3]}, ensure_ascii=False)}\n\ndata['people'][{i}]['{field}'] is {json.dumps(people[i][field], ensure_ascii=False)}"
    if kind.startswith("sql"):
        cut = r.choice(sorted({p["age"] for p in people}))
        table = "Table people:\n| name | city | age |\n|---|---|---|\n" + "\n".join(f"| {p['name']} | {p['city']} | {p['age']} |" for p in people) + "\n\n"
        if kind == "sql-select":
            return f"{table}SELECT name FROM people WHERE age >= {cut} ORDER BY name;\nResult: " + ", ".join(sorted(p["name"] for p in people if p["age"] >= cut))
        if kind == "sql-count":
            return f"{table}SELECT COUNT(*) FROM people WHERE age < {cut};\nResult: {sum(p['age'] < cut for p in people)}"
        return f"{table}SELECT SUM(age) FROM people;\nResult: {sum(p['age'] for p in people)}"
    n, small, wide, big = r.randint(1, 3999), r.randint(2, 1023), r.randint(16, 65535), r.randint(10_000, 999_999_999)
    parts = r.sample(WORDS, r.randint(2, 4))
    snake, camel = "_".join(parts), parts[0] + "".join(p.title() for p in parts[1:])
    return {
        "roman": f"{n} in Roman numerals is {roman(n)}",
        "from-roman": f"The Roman numeral {roman(n)} is the number {n}",
        "binary": f"{small} in binary is {small:b}",
        "hex": f"{wide} in hexadecimal is 0x{wide:X}",
        "thousands": f"Written with thousands separators, {big} is {big:,}",
        "snake-camel": f"In camelCase, {snake} is {camel}",
        "camel-snake": f"In snake_case, {camel} is {snake}",
        "title-case": f'"{" ".join(parts)}" in title case is "{" ".join(p.title() for p in parts)}"',
    }[kind]


def logic(r: random.Random) -> str:
    """What follows from a few sentences: an order, a chain, a syllogism, a truth value, a count."""
    kind = r.choice(["order", "order", "order", "between", "chain", "syllogism", "boolean", "counting"])
    if kind == "order":
        more, most, least = r.choice([("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest"), ("heavier", "heaviest", "lightest")])
        people = r.sample(NAMES, r.randint(3, 4))  # the first is the most
        told = [f"{x} is {more} than {y}." for x, y in zip(people, people[1:], strict=False)]
        r.shuffle(told)
        top = r.random() < 0.5
        return f"{' '.join(told)} Who is the {most if top else least}?\nAnswer: {people[0] if top else people[-1]}"
    if kind == "between":
        x, y, z = r.sample(NAMES, 3)
        told = [f"{x} is older than {y}.", f"{y} is older than {z}."]
        r.shuffle(told)
        return f"{' '.join(told)} Who is younger than {x} but older than {z}?\nAnswer: {y}"
    if kind == "chain":
        thing = r.choice(["key", "ring", "coin", "letter", "map", "button", "ticket", "stone"])
        x, y, z = r.sample(["box", "drawer", "cupboard", "bag", "basket", "chest", "case", "crate"], 3)
        told = [f"The {thing} is in the {x}.", f"The {x} is in the {y}.", f"The {y} is in the {z}."]
        r.shuffle(told)
        return f"{' '.join(told)} What is the {thing} ultimately inside?\nAnswer: the {z}"
    if kind == "syllogism":
        thing, quality, who = coin(r, 1) + coin(r, 1), r.choice(["shiny", "loud", "soft", "round", "cold", "striped", "hollow", "sweet"]), r.choice(NAMES)
        every = r.random() < 0.5
        return f"{'Every' if every else 'No'} {thing} is {quality}. {who} owns a {thing}. Is {who}'s {thing} {quality}?\nAnswer: {'Yes' if every else 'No'}"
    if kind == "boolean":
        values, denied = [r.random() < 0.5 for _ in range(3)], [r.random() < 1 / 3 for _ in range(3)]
        first, second = r.choice(["and", "or"]), r.choice(["and", "or"])
        terms = [("not " if no else "") + str(value) for value, no in zip(values, denied, strict=True)]
        x, y, z = (value != no for value, no in zip(values, denied, strict=True))  # not binds tightest, then and, then or
        if first == "and":
            result = (x and y and z) if second == "and" else ((x and y) or z)
        else:
            result = (x or (y and z)) if second == "and" else (x or y or z)
        return f"Evaluate: {terms[0]} {first} {terms[1]} {second} {terms[2]}\nAnswer: {result}"
    who, colours, counts = r.choice(NAMES), r.sample(COLOURS, 3), [r.randint(1, 9) for _ in range(3)]
    told = ", ".join(f"{n} {colour}" for n, colour in zip(counts, colours, strict=True))
    if r.random() < 0.5:
        return f"{who} has {told} marbles. How many marbles does {who} have in all?\nAnswer: {sum(counts)}"
    k = r.randrange(2)
    return f"{who} has {told} marbles. How many marbles are not {colours[k]}?\nAnswer: {sum(counts) - counts[k]}"


NOUNS = [("carpenter", "carpenters"), ("sailor", "sailors"), ("tailor", "tailors"), ("clerk", "clerks"), ("judge", "judges"), ("cook", "cooks"),
         ("painter", "painters"), ("child", "children"), ("woman", "women"), ("man", "men"), ("mouse", "mice"), ("goose", "geese"),
         ("person", "people"), ("wolf", "wolves"), ("ox", "oxen"), ("actor", "actors")]
INTRANSITIVE = [("runs", "run"), ("waits", "wait"), ("sings", "sing"), ("swims", "swim"), ("talks", "talk"), ("reads", "read"), ("dances", "dance"), ("listens", "listen")]
TRANSITIVE = [("sees", "see", "saw"), ("calls", "call", "called"), ("follows", "follow", "followed"), ("trusts", "trust", "trusted"), ("greets", "greet", "greeted"), ("teaches", "teach", "taught")]
PARTICIPLES = ["taken", "written", "eaten", "broken", "chosen", "given", "seen", "forgotten", "driven", "stolen", "hidden"]
OBJECTS = ["letter", "parcel", "ladder", "kettle", "basket", "lantern", "blanket", "bottle"]


def grammar(r: random.Random) -> str:
    """The right sentence of a pair that differs in one place: agreement, a determiner, a
    reflexive, a participle, an article. Only the right one: the wrong one is not to be learnt."""
    (one, many), (other, others) = r.sample(NOUNS, 2)
    does, do = r.choice(INTRANSITIVE)
    sees, see, saw = r.choice(TRANSITIVE)
    who, done, thing = r.choice(NAMES), r.choice(PARTICIPLES), r.choice(OBJECTS)
    sentence = r.choice([
        f"the {one} {does}.", f"the {many} {do}.", f"the {one} near the {others} {does}.", f"the {many} beside the {other} {do}.",
        f"the {one} that the {others} {see} {does}.", f"{who} {sees} this {one}.", f"{who} {sees} these {many}.", f"the {many} {saw} themselves.",
        f"{who} has {done} the {thing}.", f"{who} found an {r.choice(['apple', 'orange', 'egg', 'onion', 'umbrella', 'idea', 'island', 'owl'])}.",
        f"{who} found a {r.choice(['banana', 'pear', 'table', 'lamp', 'river', 'garden', 'horse', 'pencil'])}.",
        f"nobody has ever {done} that {thing}.", f"there are many {many} here.",
    ])
    return sentence[0].upper() + sentence[1:]


SKILLS = {"agents": session, "reading": reading, "transformation": transformation, "logic": logic, "grammar": grammar}
SEED = "scribe"  # its own, and fixed: the same problems in every round
GIVE_UP = 50  # problems in a row that do not fit what is left, before a skill writes no more


def write_ink(allowance: int, count) -> list[tuple[str, int]]:
    """Worked problems and the tokens each costs, at most `allowance` in all. The skill that
    has had least so far writes next, so each gets an equal share."""
    streams = {skill: random.Random(f"{SEED}/{skill}") for skill in SKILLS}
    spent = dict.fromkeys(SKILLS, 0)
    misses = dict.fromkeys(SKILLS, 0)
    problems = []
    while writing := [skill for skill in SKILLS if misses[skill] < GIVE_UP]:
        skill = min(writing, key=spent.__getitem__)
        text = SKILLS[skill](streams[skill])
        cost = count([text])[0]
        if cost > allowance:
            misses[skill] += 1
            continue
        problems.append((text, cost))
        spent[skill] += cost
        allowance -= cost
        misses[skill] = 0
    return problems


# --- counting as the grader counts ---------------------------------------------------------------


def tokenizer(task: Task):
    """A function from texts to how many tokens each is, under the season's tokenizer; None if
    this harness does not hold one."""
    try:
        from stet.spec import load_season
        from stet.tok import encode_batch, load_encoding

        enc = load_encoding(load_season(task.info["season"]))
    except Exception as e:
        print(f"scribe: no tokenizer ({e}); spending no ink", file=sys.stderr)
        return None
    return lambda texts: [len(tokens) for tokens in encode_batch(enc, texts)]


def stream_sizes(task: Task, lines: list[dict], count) -> list[int]:
    """The tokens each line of tidy's script adds to the stream: the text it keeps, built as
    stet.script builds it, and the token that starts every document."""
    corpus = task.corpus
    sizes: list[int] = []
    texts: list[str] = []
    waiting = 0
    for line in lines:
        raw = corpus.raw(line["d"])
        pieces = []
        for start, end in line["k"]:
            while start < len(raw) and (raw[start] & 0xC0) == 0x80:  # an offset inside a character moves forward
                start += 1
            while end < len(raw) and (raw[end] & 0xC0) == 0x80:
                end += 1
            pieces.append(raw[start:end])
        texts.append(b"".join(pieces).decode("utf-8", errors="replace"))
        waiting += len(texts[-1])
        if waiting >= 4 << 20:  # a few megabytes at a time: a round's worth of token lists would not fit in memory
            sizes += [n + 1 for n in count(texts)]
            texts, waiting = [], 0
    return sizes + [n + 1 for n in count(texts)]


def dealt(sizes: list[int], lanes: int, lane_len: int) -> list[int]:
    """How many times each document of a script is trained on. The grader deals documents in
    script order, each to the lane that is shortest, and starts again from the top until
    every lane is full (stet.script.materialize)."""
    times = [0] * len(sizes)
    heap = [(0, lane) for lane in range(lanes)]
    while True:
        for j, n in enumerate(sizes):
            have, lane = heap[0]
            heapq.heapreplace(heap, (have + n, lane))
            times[j] += 1
            if heap[0][0] >= lane_len:
                return times


def inked(task: Task, chosen: list[dict]) -> list:
    """tidy's script with the ink in it, or tidy's script as it is if there is no ink to spend."""
    allowance = int(task.info.get("ink_tokens", 0))
    student = task.info["student"]
    lanes, ctx = int(student["batch_size"]), int(student["ctx"])
    lane_len = task.train_tokens // (lanes * ctx) * ctx + 1
    if not chosen or allowance <= 0 or (count := tokenizer(task)) is None:
        return chosen
    sizes = stream_sizes(task, chosen, count)
    if sum(sizes) * 1000 < lanes * lane_len:
        return chosen  # a script trained on a thousand times over has no room for a word of ink
    # A document put in front of line j is trained on whenever line j is; one put after the
    # last line, once fewer than the first line. Ink goes where that is fewest times.
    times = dealt(sizes, lanes, lane_len)
    times.append(times[0] - 1)
    fewest = min((t for t in times if t > 0), default=0)
    if not fewest:
        return chosen
    places = [j for j, t in enumerate(times) if t == fewest]
    problems = write_ink(allowance // fewest, count)
    while problems:
        # Spread evenly over those places, then dealt again with the ink in: it moves what follows.
        at: dict[int, list[int]] = {}
        for k in range(len(problems)):
            at.setdefault(places[(2 * k + 1) * len(places) // (2 * len(problems))], []).append(k)
        script: list = []
        new_sizes: list[int] = []
        where: list[int] = []  # the line each problem is on
        for j in range(len(chosen) + 1):
            for k in at.get(j, []):
                where.append(len(script))
                script.append({"t": problems[k][0]})
                new_sizes.append(problems[k][1] + 1)
            if j < len(chosen):
                script.append(chosen[j])
                new_sizes.append(sizes[j])
        times = dealt(new_sizes, lanes, lane_len)
        used = sum(cost * times[line] for (_, cost), line in zip(problems, where, strict=True))
        if used <= allowance:
            print(f"scribe: {len(problems)} problems, {used:,} of {allowance:,} tokens of ink", file=sys.stderr)
            return script
        # Over: some were trained on more often than planned. Write fewer and deal again.
        over = used - allowance
        while problems and over > 0:
            over -= problems.pop()[1] * max(1, times[where.pop()])
    return chosen


def edit(task: Task):
    return inked(task, select(task))


def main():
    task = Task.from_env()
    chosen = select(task)
    task.write(chosen)  # tidy's script first: if the rest is cut short, this one counts
    task.write(inked(task, chosen))


if __name__ == "__main__":
    main()

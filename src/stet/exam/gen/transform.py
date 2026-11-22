"""Changing one format into another, and reading results off small data: Markdown to HTML, CSV to
JSON, a SQL query's result, numbers in other notations, identifiers in other cases."""

from __future__ import annotations

import json

from ..items import Item
from ..rng import Rng
from .lexicon import NAMES, PLACES, WORDS

NAME = "transform"
KINDS = {"md-html": "markup", "csv-json": "structured", "sql-select": "code", "sql-count": "code", "sql-sum": "code",
         "roman": "math", "from-roman": "math", "binary": "math", "hex": "math", "thousands": "math",
         "snake-camel": "code", "camel-snake": "code", "title-case": "prose", "json-path": "structured"}


def corners() -> list[str]:
    return [f"{NAME}/{k}" for k in KINDS]


def roman(n: int) -> str:
    out = ""
    for v, s in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                 (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out, n = out + s, n - v
    return out


def _table(rng: Rng, n: int) -> list[dict]:
    return [{"name": nm, "city": rng.choice(PLACES), "age": rng.between(18, 80)} for nm in rng.sample(NAMES, n)]


def make(rng: Rng, corner: str) -> Item:
    k = corner.split("/", 1)[1]
    fmt = KINDS[k]
    if k == "md-html":
        words = rng.sample(WORDS, 3)
        style = rng.choice(["bold", "italic", "heading", "list", "link"])
        if style == "bold":
            md, html = f"The **{words[0]}** is here.", f"<p>The <strong>{words[0]}</strong> is here.</p>"
        elif style == "italic":
            md, html = f"One *{words[0]}* and one {words[1]}.", f"<p>One <em>{words[0]}</em> and one {words[1]}.</p>"
        elif style == "heading":
            md, html = f"## {words[0].title()} {words[1]}", f"<h2>{words[0].title()} {words[1]}</h2>"
        elif style == "list":
            md = "\n".join(f"- {w}" for w in words)
            html = "<ul>\n" + "\n".join(f"<li>{w}</li>" for w in words) + "\n</ul>"
        else:
            md, html = f"See [{words[0]}](https://example.org/{words[1]}).", f'<p>See <a href="https://example.org/{words[1]}">{words[0]}</a>.</p>'
        ctx, cont = f"Markdown:\n{md}\n\nHTML:\n", html
    elif k == "csv-json":
        rows = _table(rng, rng.between(1, 3))
        csv = "name,city,age\n" + "\n".join(f"{r['name']},{r['city']},{r['age']}" for r in rows)
        ctx, cont = f"CSV:\n{csv}\n\nThe same rows as JSON:\n", json.dumps(rows, ensure_ascii=False)
    elif k in ("sql-select", "sql-count", "sql-sum"):
        rows = _table(rng, rng.between(4, 7))
        cut = rng.choice(sorted({r["age"] for r in rows}))
        listing = "\n".join(f"| {r['name']} | {r['city']} | {r['age']} |" for r in rows)
        table = f"Table people:\n| name | city | age |\n|---|---|---|\n{listing}\n\n"
        if k == "sql-select":
            hit = [r["name"] for r in sorted(rows, key=lambda r: r["name"]) if r["age"] >= cut]
            ctx, cont = f"{table}SELECT name FROM people WHERE age >= {cut} ORDER BY name;\nResult:", " " + ", ".join(hit)
        elif k == "sql-count":
            ctx, cont = f"{table}SELECT COUNT(*) FROM people WHERE age < {cut};\nResult:", f" {sum(1 for r in rows if r['age'] < cut)}"
        else:
            ctx, cont = f"{table}SELECT SUM(age) FROM people;\nResult:", f" {sum(r['age'] for r in rows)}"
    elif k == "roman":
        n = rng.between(1, 3999)
        ctx, cont = f"{n} in Roman numerals is", f" {roman(n)}"
    elif k == "from-roman":
        n = rng.between(1, 3999)
        ctx, cont = f"The Roman numeral {roman(n)} is the number", f" {n}"
    elif k == "binary":
        n = rng.between(2, 1023)
        ctx, cont = f"{n} in binary is", f" {n:b}"
    elif k == "hex":
        n = rng.between(16, 65535)
        ctx, cont = f"{n} in hexadecimal is 0x", f"{n:X}"
    elif k == "thousands":
        n = rng.between(10_000, 999_999_999)
        ctx, cont = f"Written with thousands separators, {n} is", f" {n:,}"
    elif k in ("snake-camel", "camel-snake"):
        parts = rng.sample(WORDS, rng.between(2, 3))
        snake, camel = "_".join(parts), parts[0] + "".join(p.title() for p in parts[1:])
        ctx, cont = (f"In camelCase, {snake} is", f" {camel}") if k == "snake-camel" else (f"In snake_case, {camel} is", f" {snake}")
    elif k == "title-case":
        parts = rng.sample(WORDS, rng.between(3, 5))
        ctx, cont = f'"{" ".join(parts)}" in title case is', ' "' + " ".join(p.title() for p in parts) + '"'
    elif k == "json-path":
        rows = _table(rng, 3)
        i, field = rng.below(3), rng.choice(["name", "city", "age"])
        ctx = f"data = {json.dumps({'people': rows}, ensure_ascii=False)}\n\ndata['people'][{i}]['{field}'] is"
        cont = f" {json.dumps(rows[i][field], ensure_ascii=False)}"
    else:
        raise ValueError(corner)
    return Item(kind="answer", corner=corner, fmt=fmt, lang="en", source=f"gen:{NAME}", ctx=ctx, cont=cont)

"""Reading a value out of structured data: JSON, CSV, tables, YAML, HTML, key-value lists."""

from __future__ import annotations

import json

from ..items import Item
from ..rng import Rng
from .lexicon import NAMES, PLACES

NAME = "struct"
FORMATS = {"json": "structured", "csv": "table", "markdown": "table", "yaml": "structured", "html": "markup", "keyvalue": "list"}
FIELDS = ["age", "city", "score"]
SIZES = [3, 6]


def corners() -> list[str]:
    return [f"{NAME}/{f}/{field}/{n}" for f in FORMATS for field in FIELDS for n in SIZES]


def _render(fmt: str, rows: list[dict]) -> str:
    keys = ["name", "age", "city", "score"]
    if fmt == "json":
        return json.dumps({"people": rows}, indent=2, ensure_ascii=False)
    if fmt == "csv":
        return "\n".join([",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in rows])
    if fmt == "markdown":
        head = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys)
        return head + "\n" + "\n".join("| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows)
    if fmt == "yaml":
        return "people:\n" + "\n".join("\n".join([f"  - name: {r['name']}"] + [f"    {k}: {r[k]}" for k in keys[1:]]) for r in rows)
    if fmt == "html":
        head = "<table>\n  <tr>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>\n"
        return head + "\n".join("  <tr>" + "".join(f"<td>{r[k]}</td>" for k in keys) + "</tr>" for r in rows) + "\n</table>"
    if fmt == "keyvalue":
        return "\n\n".join("\n".join(f"{k}: {r[k]}" for k in keys) for r in rows)
    raise ValueError(fmt)


def make(rng: Rng, corner: str) -> Item:
    _, fmt, field, n = corner.split("/")
    names = rng.sample(NAMES, int(n))
    rows = [{"name": nm, "age": rng.between(18, 90), "city": rng.choice(PLACES), "score": rng.between(0, 100)} for nm in names]
    who = rng.choice(rows)
    ctx = f"{_render(fmt, rows)}\n\nQ: What is {who['name']}'s {field}?\nA:"
    return Item(kind="answer", corner=corner, fmt=FORMATS[fmt], lang="en", source=f"gen:{NAME}", ctx=ctx, cont=f" {who[field]}")

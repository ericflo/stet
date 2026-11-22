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


def _rows(rng: Rng, n: int) -> list[dict]:
    return [{"name": nm, "age": rng.between(18, 90), "city": rng.choice(PLACES), "score": rng.between(0, 100)} for nm in rng.sample(NAMES, n)]


def _ask(corner: str, head: str, who: dict) -> Item:
    _, fmt, field, _ = corner.split("/")
    return Item(kind="answer", corner=corner, fmt=FORMATS[fmt], lang="en", source=f"gen:{NAME}",
                ctx=f"{head}Q: What is {who['name']}'s {field}?\nA:", cont=f" {who[field]}")


def make(rng: Rng, corner: str) -> Item:
    _, fmt, _, n = corner.split("/")
    rows = _rows(rng, int(n))
    return _ask(corner, f"{_render(fmt, rows)}\n\n", rng.choice(rows))


def shared(rng: Rng, corner: str, k: int) -> tuple[str, list[Item], Item, str]:
    """Asked few-shot (shots.py), the examples are other rows of the question's own table, shown
    once: the table, up to k questions about other people answered, the question, and what stands
    between them. A table of three rows has two other people to ask about."""
    _, fmt, _, n = corner.split("/")
    rows = _rows(rng, int(n))
    asked = rng.sample(rows, min(k + 1, len(rows)))
    return f"{_render(fmt, rows)}\n\n", [_ask(corner, "", who) for who in asked[1:]], _ask(corner, "", asked[0]), "\n"

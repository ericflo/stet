#!/usr/bin/env python3
"""Test vectors for the server's copy of the exam's score (server/aggregate.go): item tables,
gains and language shares, and the score, the pillars and the watched parts the harness gives them,
under the current season's plan (its exam.groups and exam.pillars).

    python tools/aggregate_vectors.py

writes server/testdata/aggregate_vectors.json. The server's tests check it gives the same to within
1e-9, and tests/test_aggregate_vectors.py fails if the committed file is not what this writes now.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stet.exam import gen
from stet.exam.aggregate import SOURCE_KIND, item_table, round_score, score_from_table
from stet.exam.fresh.select import SHARES
from stet.exam.items import FORMATS, KINDS, Item

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "server" / "testdata" / "aggregate_vectors.json"
SEED = 20261119
FAMILIES = sorted(f.NAME for f in gen.FAMILIES)
SOURCES = sorted(SOURCE_KIND) + ["forum", "radio"]  # the last two not listed: each a kind of its own
NATURAL = ["en", "ru", "de", "ja", "zh", "fr", "es", "pt", "sw", "yo", "eo", "nan"]  # the last four not in SHARES
CODE_LANGS = ["code:python", "code:go", "code:lua", "code:javascript"]
STYLES = ("normal", "outliers", "ties", "heavy")


def item(part: str, lang: str, fmt: str = "prose", kind: str = "answer", n: int = 0) -> Item:
    """An item of a generator family or a fresh source: only its part, language and stratum count."""
    if part in FAMILIES:
        corner, source = f"{part}/{n}", f"gen:{part}"
    else:
        corner, source = f"fresh/{part}/{lang}/{n}", f"{part}:{n}"
    say = {"passage": {"text": "Text."}, "answer": {"ctx": "Q:", "cont": " A"}, "pair": {"good": "Good.", "bad": "Bad."}}
    return Item(kind=kind, corner=corner, fmt=fmt, lang=lang, source=source, **say[kind])


def draw_gains(rng: random.Random, n: int, style: str) -> list[float]:
    """Gains around a curator's own mean: plain, with a few far out, mostly tied, or heavy-tailed."""
    mu, sigma = rng.uniform(-0.05, 0.05), rng.uniform(0.005, 0.1)
    out = []
    for _ in range(n):
        g = rng.gauss(mu, sigma)
        if style == "outliers" and rng.random() < 0.06:
            g = rng.choice((-1, 1)) * rng.uniform(1, 30)
        elif style == "ties":
            g = rng.choice((-0.1, 0.0, 0.0, 0.05, 0.1, 0.1))
        elif style == "heavy":
            g /= max(rng.random(), 0.02)
        out.append(round(g, 6))
    return out


def draw_shares(rng: random.Random, langs) -> dict[str, float]:
    """A round's measured shares of some languages: some left out, some zero or below, never none."""
    shares: dict[str, float] = {}
    for lang in sorted({lang for lang in langs if not lang.startswith("code:")}):
        r = rng.random()
        if r < 0.15:
            continue
        shares[lang] = 0.0 if r < 0.22 else -2.5 if r < 0.25 else round(rng.uniform(0.05, 45.0), 3)
    shares.setdefault("en", 43.0)
    return shares


def random_case(rng: random.Random) -> tuple[list[Item], list[float], dict[str, float]]:
    """A few parts, each in a few languages (sometimes code beside them) and strata, of every size."""
    items: list[Item] = []
    for part in rng.sample(FAMILIES + SOURCES, rng.randint(1, 8)):
        langs = rng.sample(NATURAL, rng.randint(1, 4))
        if rng.random() < 0.3:
            langs += rng.sample(CODE_LANGS, rng.randint(1, 2))
        strata = [(rng.choice(FORMATS), rng.choice(KINDS)) for _ in range(rng.randint(1, 4))]
        for _ in range(rng.choice((1, 2, 3, 5, 8, 13, 30, 60))):
            fmt, kind = rng.choice(strata)
            items.append(item(part, rng.choice(langs), fmt, kind, len(items)))
    rng.shuffle(items)
    return items, draw_gains(rng, len(items), rng.choice(STYLES)), draw_shares(rng, [it.lang for it in items])


def fresh_items(rng: random.Random) -> list[Item]:
    """Fresh items of several sources and languages, technical writing in code and English both."""
    plan = {"wikipedia": ["en", "ru", "de", "ja"], "wikinews": ["en", "fr"], "ccnews": ["en", "es", "sw"],
            "wikibooks": ["en", "pt"], "github": ["en", "code:go", "code:python"], "bluesky": ["en", "ja", "pt"],
            "mastodon": ["de", "en", "eo"], "wikipedia-talk": ["en", "ru"], "hackernews": ["en"], "lemmy": ["en", "de"],
            "github-comment": ["en", "ru"], "mastodon-reply": ["en", "ja"], "forum": ["zh", "nan"],
            "crossref": ["en", "es", "pt"], "stackexchange": ["en"], "stackoverflow": ["en", "code:python"], "lore": ["code:c"],
            "discourse": ["en", "de", "fr"]}
    out = []
    for src, langs in plan.items():
        for _ in range(rng.randint(3, 40)):
            fmt = rng.choice(("prose", "dialogue", "code")) if src in ("github", "github-comment") else rng.choice(("prose", "dialogue"))
            out.append(item(src, rng.choice(langs), fmt, rng.choice(("passage", "answer")), len(out)))
    return out


def edge_cases(rng: random.Random):
    """(name, items, gains, shares): each the place where a port would most likely go wrong."""
    one = [item("arith", "en", "math")]
    yield "one item", one, [0.25], {"en": 43.0}
    many = [item("arith", "en", "math", n=i) for i in range(12)] + [item("grammar", "en", "prose", "pair")]
    yield "a pillar of one item", many, draw_gains(rng, 13, "normal"), {"en": 43.0}
    mixed = ([item("prog", "code:python", "code", n=i) for i in range(6)] + [item("prog", "code:go", "code", n=i) for i in range(3)]
             + [item("prog", "en", "prose", n=i) for i in range(4)] + [item("prog", "ru", "prose", n=i) for i in range(2)]
             + [item("prog", "sw", "prose")])
    rng.shuffle(mixed)
    yield "a part mixing code and natural languages", mixed, draw_gains(rng, len(mixed), "normal"), {"en": 43.0, "ru": 6.2}
    code = [item("prog", rng.choice(CODE_LANGS), "code", n=i) for i in range(9)] + [item("struct", "en", "structured", n=i) for i in range(4)]
    yield "code alone in a part", code, draw_gains(rng, len(code), "outliers"), {"en": 43.0}
    absent = [item("wikipedia", lang, "prose", "passage", i) for i, lang in enumerate(["sw", "yo", "eo", "sw", "en", "yo", "eo", "en"])]
    yield "languages absent from the shares", absent, draw_gains(rng, len(absent), "normal"), {"en": 40.0, "de": 5.6}
    zero = [item("ccnews", lang, "prose", "passage", i) for i, lang in enumerate(["en", "ru", "de", "fr"] * 3)]
    yield "zero and negative shares", zero, draw_gains(rng, len(zero), "normal"), {"en": 0.0, "ru": -3.0, "de": 5.0, "fr": 0.001}
    flat = [item("units", "en", "math", n=i) for i in range(8)] + [item("units", "en", "table", n=i) for i in range(3)]
    yield "no spread to clip by", flat, [0.1] * 7 + [5.0] + [0.2] * 3, {"en": 43.0}
    far = [item("logic", "en", "prose", n=i) for i in range(23)]
    yield "outliers clipped", far, draw_gains(rng, 20, "normal") + [40.0, -25.0, 12.5], {"en": 43.0}
    edges = [item("algo", "en", fmt, n=i) for fmt, size in zip(FORMATS, (9, 10, 11, 19, 20, 21, 29, 30, 31, 59, 61), strict=True)
             for i in range(size)]
    yield "trims at their edges", edges, draw_gains(rng, len(edges), "heavy"), {"en": 43.0}
    tiny = [item(rng.choice(("word", "multi")), rng.choice(("en", "de")), rng.choice(FORMATS), rng.choice(KINDS), i) for i in range(14)]
    yield "strata of one and two", tiny, draw_gains(rng, len(tiny), "outliers"), {"en": 43.0, "de": 5.6}
    tied = [item(rng.choice(FAMILIES), rng.choice(NATURAL), rng.choice(FORMATS), "answer", i) for i in range(40)]
    yield "all the same", tied, [0.125] * 40, {"en": 43.0, "ja": 5.0}
    kinds = [item(src, rng.choice(NATURAL), "dialogue", "passage", i) for i, src in enumerate(SOURCES)]
    yield "every kind of writing", kinds, draw_gains(rng, len(kinds), "normal"), {"en": 43.0, "ru": 6.2, "ja": 5.0}
    small = [item("wikipedia", lang, "prose", "passage", i) for i, lang in enumerate(["en"] * 10 + ["ru"] * 8 + ["de"] * 7 + ["sw"] * 3 + ["yo"] * 2)]
    yield "small languages measured together", small, draw_gains(rng, len(small), "normal"), {"en": 43.0, "ru": 6.2, "de": 5.6, "sw": 0.2}
    lone = [item("ccnews", "en", "prose", "passage", i) for i in range(20)] + [item("ccnews", "nan", "prose", "passage", 99)]
    yield "one item of a small language", lone, draw_gains(rng, len(lone), "outliers"), {"en": 43.0}
    named = [item("lemmy", "other", "dialogue", "answer", i) for i in range(9)] + [item("lemmy", "eo", "dialogue", "answer", i) for i in range(3)]
    yield "a language called other beside the pooled ones", named, draw_gains(rng, len(named), "normal"), {"en": 43.0, "other": 1.5}
    watched = [item("arith", "en", "math", n=i) for i in range(9)] + [item("knowledge", lang, "prose", n=i) for i, lang in enumerate(["en", "ru", "ja"] * 4)]
    yield "only watched items", watched, draw_gains(rng, len(watched), "normal"), {"en": 43.0, "ru": 6.2}
    skills = [item("agent", "en", "logs", n=i) for i in range(12)] + [item("logic", "en", "prose", n=i) for i in range(5)]
    yield "one group alone", skills, draw_gains(rng, len(skills), "normal"), {"en": 43.0}
    thin = [item("github", "en", "prose", "passage", i) for i in range(10)] + [item("github", "code:go", "code", "passage", i) for i in range(3)]
    yield "a few items of code beside prose", thin, draw_gains(rng, len(thin), "normal"), {"en": 43.0}
    both = ([item("wikipedia", "en", "prose", "passage", i) for i in range(9)] + [item("agent", "en", "logs", n=i) for i in range(9)]
            + [item("units", "en", "math", n=i) for i in range(9)] + [item("radio", "en", "prose", "passage", i) for i in range(4)])
    yield "counted and watched together", both, draw_gains(rng, len(both), "heavy"), {"en": 43.0}


def build() -> dict:
    rng = random.Random(SEED)
    cases: list[tuple[str, str, list[Item], list[float], dict[str, float]]] = []
    for name, items, g, shares in edge_cases(rng):
        cases.append((name, name, items, g, shares))
    for i in range(22):
        name = f"random {i + 1}"
        cases.append((name, name, *random_case(rng)))
    # A round's own items: every generator's, and fresh text beside them, in the round's order.
    items = gen.generate(b"vectors", per_corner=2) + fresh_items(rng)
    rng.shuffle(items)
    langs = [it.lang for it in items]
    for i, (style, shares) in enumerate((("normal", dict(SHARES)), ("outliers", dict(SHARES)), ("normal", draw_shares(rng, langs)),
                                         ("outliers", draw_shares(rng, langs)), ("ties", draw_shares(rng, langs)),
                                         ("heavy", draw_shares(rng, langs)))):
        shares_name = "published shares" if i < 2 else "measured shares"
        cases.append((f"a round, {style}, {shares_name}", "a round", items, draw_gains(rng, len(items), style), shares))
    out: dict = {"about": "Written by tools/aggregate_vectors.py: what the harness scores each case. Do not edit.",
                 "tables": {}, "cases": []}
    for name, table, items, g, shares in cases:
        rows = item_table(items)
        r, s = round_score(items, g, shares=shares), score_from_table(rows, g, shares)
        if (s["score"] != r["score"] or {p: round(v, 5) for p, v in s["pillars"].items()} != {p: v["score"] for p, v in r["pillars"].items()}
                or {p: round(v, 5) for p, v in s["watch"].items()} != {p: v["score"] for p, v in r["watch"].items()}):
            raise AssertionError(f"{name}: the table's score is not round_score's")
        out["tables"].setdefault(table, rows)
        out["cases"].append({"name": name, "table": table, "shares": shares, "gains": g, "score": r["score"], "pillars": s["pillars"],
                             "watch": s["watch"]})
    return out


def dump(data: dict) -> str:
    """One item to a line and one case to a line, so a change shows as a small diff."""
    lines = ["{", f' "about": {json.dumps(data["about"])},', ' "tables": {']
    for i, (name, rows) in enumerate(data["tables"].items()):
        lines.append(f"  {json.dumps(name)}: [")
        lines += [f"   {json.dumps(r, ensure_ascii=False)}," for r in rows]
        lines[-1] = lines[-1].removesuffix(",")
        lines.append("  ]," if i < len(data["tables"]) - 1 else "  ]")
    lines += [" },", ' "cases": [']
    lines += [f"  {json.dumps(c, ensure_ascii=False)}," for c in data["cases"]]
    lines[-1] = lines[-1].removesuffix(",")
    lines += [" ]", "}"]
    text = "\n".join(lines) + "\n"
    assert json.loads(text) == data
    return text


if __name__ == "__main__":
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(dump(data))
    print(f"{len(data['cases'])} cases, {sum(len(t) for t in data['tables'].values())} rows -> {OUT.relative_to(ROOT)}")

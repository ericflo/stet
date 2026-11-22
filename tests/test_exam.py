"""The exam: generators that every machine agrees on, answers that are right, and a score
that rewards breadth."""

import contextlib
import datetime as dt
import io
import json
import os
import re
import shutil
import subprocess

import pytest

from stet.exam import gen
from stet.exam.aggregate import gains, item_table, round_score, score_from_table, trimmed_mean
from stet.exam.gen import lexicon
from stet.exam.items import Item, digest
from stet.exam.rng import Rng

SEED = b"pin"


def test_rng_is_pinned_and_in_range():
    # These numbers are what every machine must draw. If they change, every generated exam does.
    r = Rng(SEED, "x")
    assert [r.below(1000) for _ in range(5)] == [974, 885, 480, 133, 833]
    r = Rng(b"another")
    for n in (1, 2, 3, 7, 1000, 2**40):
        assert all(0 <= r.below(n) < n for _ in range(200))
    picked = r.sample(range(50), 50)
    assert sorted(picked) == list(range(50))
    assert Rng(SEED, "a").child("b").next64() == Rng(SEED, "a").child("b").next64() != Rng(SEED, "a").child("c").next64()


def test_a_seed_always_makes_the_same_exam():
    # Pinned: a change to any generator that changes what a seed produces must bump gen.VERSION
    # and this hash, on purpose.
    assert digest(gen.generate(SEED)) == "69d84437be7ac49e24fd3920b143e654b0aa16834cd1fc34b7e0e5c68588f56c"
    assert digest(gen.generate(SEED)) == digest(gen.generate(SEED))
    assert digest(gen.generate(b"other")) != digest(gen.generate(SEED))


def test_an_agents_answer_follows_from_its_session():
    """What comes back was listed, shown or written earlier in the session: nothing to guess, nothing to learn by heart."""
    import re

    from stet.exam.gen import shell

    listings, names = set(), set()
    for seed in range(60):
        for it in gen.generate(f"agent-{seed}", per_corner=1, families=[shell]):
            if it.kind != "answer":
                continue
            corner = it.corner.rsplit("/", 1)[1]
            assert it.lang == it.corner.split("/")[1]
            if corner in ("cat", "head", "grep", "append-cat", "write-read"):
                for word in re.findall(r"[^\W\d_]{3,}", it.cont.replace("\\n", " ")):  # a tool's result writes a line break as \n
                    assert word in it.ctx, (it.corner, word)
                if it.lang in ("ru", "ja", "zh"):  # the notes are in the language's own script, and so is what comes back
                    assert not it.cont.isascii() and "\\u" not in it.ctx + it.cont, it
            if corner in ("ls", "ls-dir", "mkdir-ls", "rm-ls"):
                for name in re.findall(r"[A-Za-z0-9_.]+", it.cont):
                    assert name in it.ctx or name == "result", (it.corner, name)
                listings.add(it.cont)
            names.add(it.ctx.splitlines()[0])
    assert len(listings) > 200 and len(names) > 100  # no listing and no project to know beforehand


def test_every_corner_makes_valid_items_of_its_own():
    corners = gen.all_corners()
    assert len(corners) == len(set(corners)) >= 300
    items = gen.generate(SEED, per_corner=3)
    assert len(items) == sum(3 * gen.MORE.get(f.NAME, 1) * len(f.corners()) for f in gen.FAMILIES)
    for it in items:
        assert it.corner in corners and it.source.startswith("gen:")
        if it.kind == "answer":
            assert it.cont.strip() and not it.ctx.endswith(" "), it
    # a family's items stay in one stratum per corner, so a round's balance does not wobble
    strata = {}
    for it in items:
        assert strata.setdefault(it.corner, it.stratum) == it.stratum


def test_adding_a_family_changes_nothing_else():
    full = gen.generate(SEED)
    some = gen.generate(SEED, families=[gen.arith, gen.grammar])
    ids = {it.id for it in full}
    assert all(it.id in ids for it in some)


def _all(corner_prefix, n=6):
    return [it for it in gen.generate(SEED, per_corner=n) if it.corner.startswith(corner_prefix)]


def test_arithmetic_answers_are_right():
    for it in _all("arith/"):
        if "/equation" in it.corner:
            expr = it.ctx.removesuffix(" =").replace("^", "**")
            assert re.fullmatch(r"[0-9 +\-*/()%*]+", expr.replace("of", "")), expr
            if "%" in expr:
                p, n = expr.split("% of ")
                want = int(p) * int(n) // 100
            else:
                want = eval(expr.replace("/", "//"), {"__builtins__": {}})  # digits and operators only, checked above
            assert int(it.cont.replace(",", "")) == want, it


def test_python_programs_print_what_the_item_says():
    for it in _all("prog/python/"):
        code = it.ctx.split("```python\n", 1)[1].split("\n```", 1)[0]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(code, {})  # code the generator wrote, from its own templates
        assert out.getvalue().strip() == it.cont.strip(), (code, it.cont)


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_javascript_programs_print_what_the_item_says():
    items = _all("prog/javascript/", n=3)
    for it in items:
        code = it.ctx.split("```javascript\n", 1)[1].split("\n```", 1)[0]
        out = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=20).stdout
        assert out.strip() == it.cont.strip(), (code, it.cont)


def test_algebra_answers_satisfy_their_equations():
    for it in _all("algebra/"):
        kind = it.corner.split("/")[1]
        if kind in ("one-step", "two-step", "both-sides"):
            eq = re.search(r"(?:Solve for x: |If )(.+?)(?:\nAnswer:|, then x =)", it.ctx).group(1)
            x = int(it.cont.replace("x = ", ""))
            lhs, rhs = (re.sub(r"(\d)x", rf"\1*({x})", s).replace("x", f"({x})") for s in eq.split(" = "))
            assert eval(lhs, {"__builtins__": {}}) == eval(rhs, {"__builtins__": {}}), (eq, x)


def test_dates_are_right():
    for it in _all("units/date/"):
        kind = it.corner.split("/")[2]
        if kind == "weekday-after":
            start, n = re.search(r"today is (\w+), what day .* in (\d+) days", it.ctx).groups()
            assert it.cont.strip() == lexicon.WEEKDAYS[(lexicon.WEEKDAYS.index(start) + int(n)) % 7]
        if kind == "date-after":
            n, month, day, year = re.search(r"What date is (\d+) days after (\w+) (\d+), (\d+)", it.ctx).groups()
            d = dt.date(int(year), lexicon.MONTHS.index(month) + 1, int(day)) + dt.timedelta(days=int(n))
            assert it.cont.strip() == f"{lexicon.MONTHS[d.month - 1]} {d.day}, {d.year}"


def test_structured_lookups_are_right():
    for it in _all("struct/json/"):
        data = json.loads(it.ctx.split("\n\nQ:")[0])
        who, field = re.search(r"What is (.+)'s (\w+)\?", it.ctx).groups()
        row = next(r for r in data["people"] if r["name"] == who)
        assert str(row[field]) == it.cont.strip()


def test_items_refuse_to_be_malformed():
    with pytest.raises(ValueError):
        Item(kind="answer", corner="x/y", fmt="math", lang="en", source="gen:x", ctx="Q", cont="")
    with pytest.raises(ValueError):
        Item(kind="pair", corner="x/y", fmt="prose", lang="en", source="gen:x", good="A.", bad="A.")
    with pytest.raises(ValueError):
        Item(kind="essay", corner="x/y", fmt="prose", lang="en", source="gen:x", text="t")


def _item(fmt, kind, corner, lang="en", source="gen:t"):
    if kind == "pair":
        return Item(kind="pair", corner=corner, fmt=fmt, lang=lang, source=source, good="Good.", bad="Bad.")
    return Item(kind="answer", corner=corner, fmt=fmt, lang=lang, source=source, ctx="Q:", cont=" A")


def _many(n, fmt, kind, corner, g, **kw):
    """n items of one corner's family or source, each with gain g."""
    return [_item(fmt, kind, f"{corner}/{i}", **kw) for i in range(n)], [g] * n


def _exam(*groups):
    items, g = [], []
    for its, gs in groups:
        items += its
        g += gs
    return items, g


def test_writing_is_three_quarters_and_skills_one_and_breadth_wins():
    # news (writing) gains 0.4 and the agent family (skills) loses 0.2: three quarters and one quarter
    items, g = _exam(_many(10, "prose", "passage", "fresh/ccnews/en", 0.4, source="ccnews:1"), _many(10, "logs", "answer", "agent", -0.2))
    r = round_score(items, g)
    assert r["score"] == pytest.approx(0.75 * 0.4 + 0.25 * -0.2)
    assert r["pillars"]["news"]["weight"] == 0.75 and r["pillars"]["agents"] == {"score": -0.2, "items": 10, "parts": {"agent": -0.2}, "group": "skills", "weight": 0.25}
    assert r["generated"]["score"] == pytest.approx(-0.2) and r["fresh"]["score"] == pytest.approx(0.4) and r["watch"] == {}
    # a curator that gains hugely on two items and loses on the rest scores below one that gains a little everywhere
    items = [_item("logs", "answer", f"agent/{i}") for i in range(40)]
    assert round_score(items, [0.05] * 40)["score"] > round_score(items, [5.0, 5.0] + [-0.02] * 38)["score"]


def test_a_pillar_the_exam_lacks_gives_its_share_to_its_own_group():
    news, agent, logic = (_many(10, "prose", "passage", "fresh/ccnews/en", 0.4, source="ccnews:1"), _many(10, "logs", "answer", "agent", -0.2),
                          _many(10, "prose", "answer", "logic", 0.2))
    ref = _many(10, "prose", "passage", "fresh/wikipedia/en", 0.0, source="wikipedia:1")
    # one kind of writing or two, writing is three quarters; one skill or two, skills are a quarter
    assert round_score(*_exam(news, ref, agent))["score"] == pytest.approx(0.75 * 0.2 + 0.25 * -0.2)
    assert round_score(*_exam(news, agent, logic))["score"] == pytest.approx(0.75 * 0.4 + 0.25 * 0.0)
    # with no skills at all the writing is the whole score
    assert round_score(*_exam(news, ref))["score"] == pytest.approx(0.2)


def test_what_the_season_does_not_name_is_watched_and_does_not_count():
    news = _many(10, "prose", "passage", "fresh/ccnews/en", 0.1, source="ccnews:1")
    arith, radio = _many(30, "math", "answer", "arith", 9.0), _many(5, "prose", "passage", "fresh/radio/en", -9.0, source="radio:1")
    r = round_score(*_exam(news, arith, radio))
    assert r["score"] == pytest.approx(0.1) and set(r["pillars"]) == {"news"} and r["counted"] == 10 and r["items"] == 45
    assert r["watch"] == {"arith": {"score": 9.0, "items": 30}, "radio": {"score": -9.0, "items": 5}}
    assert r["families"]["arith"] == 9.0  # every family's plain mean is still reported
    rows = item_table(_exam(news, arith, radio)[0])
    assert {row["pillar"] for row in rows} == {"news", "watch"}
    s = score_from_table(rows, _exam(news, arith, radio)[1], None)
    assert s["score"] == pytest.approx(0.1) and s["watch"] == {"arith": 9.0, "radio": -9.0}
    with pytest.raises(ValueError):
        score_from_table([{**rows[0], "pillar": "reasoning"}], [0.1], None)


def test_every_family_and_source_has_a_place():
    from stet.exam.aggregate import SOURCE_KIND, WATCH, current_plan
    from stet.exam.gen import FAMILIES, MORE

    plan = current_plan()
    counted = {f.NAME for f in FAMILIES if plan.pillar_of(f.NAME) != WATCH}
    assert counted == {"agent", "context", "transform", "logic", "grammar"} == set(MORE)  # the families that count make more items
    assert {plan.pillar_of(k) for k in set(SOURCE_KIND.values()) - {"instructional", "science"}} == {"reference", "news", "technical", "conversation"}
    # A kind of writing the season has not measured yet is sat and counts for nothing, like a family of problems.
    assert plan.pillar_of("science") == WATCH and plan.pillar_of("instructional") == WATCH and plan.pillar_of("radio") == WATCH
    w = plan.weights(plan.pillars)
    assert sum(w.values()) == pytest.approx(1.0) and w["reference"] == pytest.approx(0.1875) and w["logic"] == pytest.approx(0.05)


def test_inside_a_pillar_each_part_counts_the_same():
    # 40 replies in discussions and 8 social posts: the two kinds weigh the same in conversation
    items, g = _exam(_many(40, "dialogue", "answer", "fresh/hackernews/en", 0.2, source="hackernews:1"),
                     _many(8, "prose", "passage", "fresh/bluesky/en", -0.2, source="bluesky:1"))
    r = round_score(items, g)
    assert r["score"] == pytest.approx(0.0) and r["pillars"]["conversation"]["parts"] == {"discussion": 0.2, "social": -0.2}


def test_languages_count_by_the_corpus_inside_a_kind_of_writing():
    # A few English items must not outweigh many passages in other languages, and each language
    # counts by its share of the corpus.
    items = ([_item("prose", "passage", f"fresh/wikipedia/ru/{i}", lang="ru", source="wikipedia:1") for i in range(30)]
             + [_item("prose", "passage", f"fresh/wikipedia/en/{i}", lang="en", source="wikipedia:2") for i in range(10)])
    g = [-0.5] * 30 + [0.5] * 10
    r = round_score(items, g, shares={"en": 60.0, "ru": 40.0})
    assert r["fresh"]["languages"]["ru"]["weight"] == pytest.approx(0.4)
    assert r["score"] == pytest.approx(0.6 * 0.5 + 0.4 * -0.5) and r["generated"] is None
    assert set(r["pillars"]) == {"reference"} and r["pillars"]["reference"]["parts"] == {"reference": pytest.approx(0.1)}
    # the same items, weighed by a corpus that is mostly Russian: this curator loses
    assert round_score(items, g, shares={"en": 20.0, "ru": 80.0})["score"] < 0


def test_small_languages_are_measured_together_and_few_items_count_for_less():
    from stet.exam.aggregate import language_cells

    # Swahili and Yoruba have three items between them: one cell, "other", with their shares together, three eighths counted
    cell, w = language_cells({"en": 20, "ru": 8, "sw": 2, "yo": 1}, {"en": 50.0, "ru": 10.0, "sw": 4.0, "yo": 4.0})
    assert cell == {"en": "en", "ru": "ru", "sw": "other", "yo": "other"}
    assert w == pytest.approx({"en": 50 / 63, "ru": 10 / 63, "other": 3 / 63})
    # so one item in a language the corpus is full of cannot carry that language's whole share
    items = ([_item("prose", "passage", f"fresh/wikipedia/en/{i}", source="wikipedia:1") for i in range(16)]
             + [_item("prose", "passage", "fresh/wikipedia/ru/0", lang="ru", source="wikipedia:2")])
    r = round_score(items, [0.0] * 16 + [8.0], shares={"en": 50.0, "ru": 50.0})
    assert r["score"] == pytest.approx(8.0 * (50 / 8) / (50 + 50 / 8))  # an eighth of its share, not half the score
    # formats too: a format with one item beside one with eight counts an eighth as much
    items = [_item("logs", "answer", f"agent/{i}") for i in range(8)] + [_item("structured", "answer", "agent/x")]
    assert round_score(items, [0.0] * 8 + [9.0])["score"] == pytest.approx(9.0 / 9)


def test_programming_languages_count_together():
    items = [_item("code", "passage", f"fresh/lore/c/{i}", lang="code:c", source="lore:1") for i in range(10)] + \
        [_item("code", "passage", f"fresh/lore/rust/{i}", lang="code:rust", source="lore:2") for i in range(2)]
    r = round_score(items, [0.1] * 10 + [0.5] * 2)
    assert r["score"] == pytest.approx((0.1 * 9 + 0.5) / 10)  # one pool, trimmed by one at each end
    # beside prose, code is half of technical writing
    prose = [_item("prose", "passage", f"fresh/github/en/{i}", source="github:1") for i in range(8)]
    assert round_score(items + prose, [0.1] * 10 + [0.5] * 2 + [0.3] * 8)["score"] == pytest.approx((0.14 + 0.3) / 2)


def test_trimmed_mean_and_gains():
    assert trimmed_mean([0, 1, 2, 3, 100, 4, 5, 6, 7, 8], 0.1) == pytest.approx(4.5)
    assert gains([1.0, 2.0], [0.5, 2.5]) == [0.5, -0.5]
    with pytest.raises(ValueError):
        gains([1.0], [1.0, 2.0])


@pytest.fixture(scope="module")
def student():
    torch = pytest.importorskip("torch")
    from stet.spec import load_season
    from stet.tok import load_encoding
    from stet.train import build_model

    season = load_season("s2")
    scale = season.scale("nano")
    return build_model(scale, season, 7, torch.device("cpu")), load_encoding(season), season.eot, scale.ctx


def test_sitting_scores_only_what_it_should(student):
    import torch

    from stet.exam.score import nll, sequences, sit

    model, enc, eot, ctx = student
    cpu = torch.device("cpu")
    a = Item(kind="answer", corner="t/a", fmt="math", lang="en", source="t", ctx="What is 2 + 2?\nAnswer:", cont=" 4")
    b = Item(kind="answer", corner="t/b", fmt="math", lang="en", source="t", ctx="Name a colour.\nAnswer:", cont=" 4")
    good = Item(kind="pair", corner="t/p", fmt="prose", lang="en", source="t", good="The dogs sleep.", bad="The dogs sleeps.")
    flip = Item(kind="pair", corner="t/q", fmt="prose", lang="en", source="t", good="The dogs sleeps.", bad="The dogs sleep.")
    long = Item(kind="passage", corner="t/l", fmt="prose", lang="en", source="t", text="word " * 2000)
    s = sit(model, [a, b, good, flip, long], enc, eot, ctx, cpu)
    assert all(v == v and abs(v) < 100 for v in s)
    assert s[0] != s[1]  # the context counts, though only the answer is scored
    assert s[2] * len(good.good.encode()) == pytest.approx(-s[3] * len(flip.good.encode()), abs=1e-4)
    seqs, spans = sequences([long], enc, eot, ctx)
    assert spans[0][1] - spans[0][0] > 1 and all(len(ids) <= ctx + 1 for ids, _ in seqs)
    # the answer's NLL is the NLL of its own tokens only
    seqs, _ = sequences([a], enc, eot, ctx)
    ids, first = seqs[0]
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([ids[:-1]])).float(), -1)[0]
    want = -sum(float(logp[t - 1, ids[t]]) for t in range(first, len(ids)))
    assert nll(model, seqs, cpu)[0] == pytest.approx(want, rel=1e-4)


def test_an_item_is_never_put_over_fewer_bytes_than_the_floor(student):
    import torch

    from stet.exam.score import sit

    model, enc, eot, ctx = student
    cpu = torch.device("cpu")
    short = Item(kind="answer", corner="t/a", fmt="math", lang="en", source="t", ctx="What is 2 + 2?\nAnswer:", cont=" 4")
    long = Item(kind="answer", corner="t/b", fmt="prose", lang="en", source="t", ctx="Say something.\n", cont="A sentence of more than sixteen bytes.")
    pair = Item(kind="pair", corner="t/p", fmt="prose", lang="en", source="t", good="Dogs bark.", bad="Dogs barks.")
    plain, floored = sit(model, [short, long, pair], enc, eot, ctx, cpu), sit(model, [short, long, pair], enc, eot, ctx, cpu, floor=16)
    assert floored[0] == pytest.approx(plain[0] * 2 / 16)  # two bytes, taken as sixteen
    assert floored[1] == plain[1]  # long enough already
    assert floored[2] == pytest.approx(plain[2] * 10 / 16)


def test_all_of_a_long_answer_is_scored_and_its_beginning_is_never_cut(student):
    """Season 1 cut an answer that overflowed the window from the left: the question went first, then the answer's own
    beginning, and what was left was still divided by the bytes of all of it."""
    import torch

    from stet.exam.score import nll, sequences, sit

    model, enc, eot, ctx = student
    cpu = torch.device("cpu")
    q, a = "The question, at some length. " * 40, "答えは長い。" * 400  # both far longer than the window
    it = Item(kind="answer", corner="t/a", fmt="prose", lang="ja", source="t", ctx=q, cont=a)
    seqs, spans = sequences([it], enc, eot, ctx)
    cont = enc.encode_ordinary(a)
    assert all(len(ids) <= ctx + 1 for ids, _ in seqs) and spans == [(0, len(seqs))] and len(seqs) > 1
    scored = [t for ids, first in seqs for t in ids[first:]]
    assert scored == cont  # every token of the answer once, in order, from its first
    ids, first = seqs[0]
    assert first >= (ctx + 1) // 2 and ids[:first] == ([eot] + enc.encode_ordinary(q))[-first:]  # the nearest half-window of the question
    # a short answer to a long question keeps as much of the question as fits
    short = Item(kind="answer", corner="t/a", fmt="prose", lang="en", source="t", ctx=q, cont=" Yes.")
    (ids, first), = sequences([short], enc, eot, ctx)[0]
    assert len(ids) == ctx + 1 and ids[first:] == enc.encode_ordinary(" Yes.")
    # and the score is the bits of all of it over the bytes of all of it
    total = sum(nll(model, seqs, cpu))
    assert sit(model, [it], enc, eot, ctx, cpu)[0] == pytest.approx(-total / 0.6931471805599453 / len(a.encode()), rel=1e-5)


def test_transforms_are_right():
    import sqlite3

    from stet.exam.gen.transform import roman

    for n in (1, 4, 9, 14, 40, 90, 400, 1994, 2026, 3999):
        back, rest = 0, roman(n)
        for v, s in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                     (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
            while rest.startswith(s):
                back, rest = back + v, rest[len(s):]
        assert back == n and not rest
    for it in _all("transform/en/sql"):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE people (name TEXT, city TEXT, age INTEGER)")
        for line in it.ctx.split("|---|---|---|\n")[1].split("\n\n")[0].splitlines():
            name, city, age = [c.strip() for c in line.strip("| ").split("|")]
            db.execute("INSERT INTO people VALUES (?, ?, ?)", (name, city, int(age)))
        query = it.ctx.split("\n\n")[-1].split("\nResult:")[0]
        got = [str(r[0]) for r in db.execute(query)]
        assert ", ".join(got) == it.cont.strip(), (query, it.cont)
    table = [it for it in _all("transform/") if it.corner.endswith("/csv-json")]
    assert {it.lang for it in table} == set(lexicon.LANGS)
    for it in table:
        lines = [ln.split(",") for ln in it.ctx.split("CSV:\n")[1].split("\n\n")[0].splitlines()[1:]]
        assert json.loads(it.cont) == [{"name": name, "city": city, "age": int(age)} for name, city, age in lines]
        assert all(name in lexicon.GIVEN[it.lang] and city in lexicon.CITIES[it.lang] for name, city, _ in lines)
    # Markdown in every language: without its marks, the HTML says what the Markdown says
    pages = [it for it in _all("transform/") if it.corner.endswith("/md-html")]
    assert {it.lang for it in pages} == set(lexicon.LANGS)
    for it in pages:
        md = it.ctx.split("Markdown:\n")[1].split("\n\nHTML:")[0]
        said = re.sub(r"^(## |- )", "", re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md), flags=re.M).replace("*", "")
        assert re.sub(r"<[^>]+>", "", it.cont).strip() == said, it
        if it.lang != "en":
            assert all(w in lexicon.NOUNS[it.lang] for w in said.split()), it


def test_arithmetic_in_other_languages_is_right():
    items = _all("multi/")
    assert {it.lang for it in items} >= {"es", "fr", "de", "ru", "ja", "zh"}
    for it in items:
        a, b = (int(x) for x in re.findall(r"\d+", it.ctx.split("\n")[0])[:2])
        op = it.corner.split("/")[2]
        assert int(it.cont) == {"add": a + b, "sub": a - b, "mul": a * b}[op], it


@pytest.mark.parametrize("lang,tool", [("go", "go"), ("c", "gcc"), ("lua", "lua")])
def test_programs_in_go_c_and_lua_print_what_the_item_says(lang, tool, tmp_path):
    if not shutil.which(tool):
        pytest.skip(f"{tool} is not installed")
    for i, it in enumerate(_all(f"prog/{lang}/", n=2)):
        code = it.ctx.split(f"```{lang}\n", 1)[1].split("\n```", 1)[0]
        src = tmp_path / f"p{i}.{ {'go': 'go', 'c': 'c', 'lua': 'lua'}[lang] }"
        src.write_text(code)
        if lang == "go":
            run = ["go", "run", str(src)]
        elif lang == "c":
            subprocess.run(["gcc", "-o", str(tmp_path / f"p{i}"), str(src)], check=True, capture_output=True)
            run = [str(tmp_path / f"p{i}")]
        else:
            run = ["lua", str(src)]
        out = subprocess.run(run, capture_output=True, text=True, timeout=120).stdout
        assert out.strip() == it.cont.strip(), (code, it.cont)


def test_students_farmed_out_to_the_organisers_machines_add_up_as_training_here_does(corpus, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import dataclasses
    import hashlib
    import threading

    import torch
    import zstandard

    from stet import client, data
    from stet.exam import farm
    from stet.script import write_script
    from stet.spec import load_season, practice_round

    season = load_season("s2")
    scale = dataclasses.replace(season.scale("nano"), train_tokens=2 * 16 * 256)
    rnd = practice_round(season, 0)
    busy = {"left": 1}  # the crawl host says "slow down" once: the worker waits and fetches again

    def built(r, s, quiet=False):
        if busy["left"]:
            busy["left"] -= 1
            raise data.FetchError("could not fetch CC-MAIN-x.warc.wet.gz: HTTP Error 503: Service Unavailable")
        return corpus

    monkeypatch.setattr(data, "corpus", built)
    monkeypatch.setattr(type(season), "scale", lambda self, name: scale)
    monkeypatch.setattr(client, "round_from_api", lambda s, d: rnd)
    monkeypatch.setattr(farm, "POLL", 0.05)
    real_sleep = farm.time.sleep
    monkeypatch.setattr(farm.time, "sleep", lambda s: None if s > 1 else real_sleep(s))  # no waiting out a busy server here
    other = tmp_path / "other.jsonl"
    write_script(other, [3, 0])
    script_bytes = other.read_bytes()
    sha = hashlib.sha256(script_bytes).hexdigest()
    items = gen.generate(rnd.seed, per_corner=1)[:60]

    # The server, as far as the farm sees it.
    lock = threading.Lock()
    batch: dict = {}

    def request(method, path, body=None, key=None, raw=False, timeout=0, limit=0):
        with lock:
            if path == "/api/v1/admin/farm/batches":
                batch.update(id="b", items=body["items_file"], tasks=[{"script": s, "tries": 0} for s in body["scripts"]])
                return {"batch": "b"}
            if path == "/api/v1/admin/farm/batches/b":
                return {"tasks": [dict(t) for t in batch["tasks"]]}
            if path == "/api/v1/farm/claim":
                for i, t in enumerate(batch.get("tasks", [])):
                    if not t.get("done") and not t.get("holder"):
                        t["holder"], t["tries"] = 1, t["tries"] + 1
                        return {"task": {"batch": "b", "index": i, "round": rnd.id, "scale": "nano", "script": t["script"],
                                         "items_digest": digest(items), "items": len(items)}}
                return {"task": None}
            if path == "/api/v1/farm/batches/b/items":
                import base64

                return base64.b64decode(batch["items"])
            if path.startswith("/api/v1/farm/batches/b/tasks/"):
                if not batch.get("busied"):  # the server is busy once, as it can be: the worker waits and sends again
                    batch["busied"] = True
                    raise client.ApiError(503, "the database is not answering; try again in a moment")
                t = batch["tasks"][int(path.rsplit("/", 1)[1])]
                t.update(done={**body, "by": "rented-one"}, item_scores=body.get("item_scores"), holder=None)
                return {"ok": True}
            if path == f"/api/v1/scripts/{sha}":
                return zstandard.ZstdCompressor().compress(script_bytes)
            if path.startswith("/api/v1/rounds/"):
                return {"round": {}}
        raise AssertionError(path)

    monkeypatch.setattr(client, "request", request)
    got: dict = {}
    same = tmp_path / "same.jsonl"
    write_script(same, range(len(corpus)))
    same_sha = hashlib.sha256(same.read_bytes()).hexdigest()
    t = threading.Thread(target=lambda: got.update(s=farm.farm_round(rnd, scale, {"other": sha, "same": same_sha}, tmp_path / "farmed", items, len(corpus), None,
                                                                      log=lambda m: None, patience=120, baselines=2)), daemon=True)
    t.start()
    while t.is_alive():
        if not farm.turn(season, None, torch.device("cpu"), log=lambda m: None):
            t.join(0.05)
    farmed = got["s"]
    rows = {r["name"]: r for r in farmed["entries"]}
    assert [t["script"] for t in batch["tasks"]] == ["unedited", "unedited#2", sha]  # "same" is the unedited corpus itself
    assert rows["same"]["score"] == 0.0 and json.loads((tmp_path / "farmed" / f"{same_sha[:16]}.json").read_text())["name"] == "(unedited)"
    assert farmed["items"] == len(items) and rows["other"]["valid"]
    # What the farm sends back is what training here gives: on a CPU, the same every time.
    from stet.exam.score import sit
    from stet.tok import load_encoding

    enc = load_encoding(season)

    def here(script):
        meta, model = farm.train_student(rnd, scale, corpus, enc, script, torch.device("cpu"))
        return {**meta, "item_scores": sit(model, items, enc, season.eot, scale.ctx, torch.device("cpu"))}

    loc = farm.summarize(rnd, scale, items, here(same), {"other": here(other)})
    assert rows["other"]["score"] == pytest.approx(loc["entries"][0]["score"], abs=1e-5)
    # kept as a local run keeps them, for the report
    assert (tmp_path / "farmed" / f"{sha[:16]}.json").exists() and (tmp_path / "farmed" / "items.jsonl").exists()


def test_reading_in_context_answers_are_in_the_context_or_follow_from_it():
    items = _all("context/", n=3)
    assert {it.corner.split("/")[2] for it in items} == {"passage", "two-hop", "rule", "definition", "glossary"}
    assert {it.lang for it in items} == set(lexicon.LANGS)
    for it in items:
        kind, ans = it.corner.split("/")[2], it.cont.strip().rstrip('".」”')
        assert it.lang == it.corner.split("/")[1]
        if kind in ("passage", "two-hop", "glossary") or it.corner.endswith(("glossary",)):
            assert ans in it.ctx, it  # read, not remembered
            if it.corner.endswith("/according"):  # asked in the middle of a sentence, a name keeps its capital
                asked = it.ctx.split("According to the passage above, ")[1]
                assert asked.startswith("the ") or asked.split()[0] in it.ctx.split("\n\n")[0], it
        elif it.corner.endswith("affine"):
            pairs = [tuple(int(v) for v in line.split(" -> ")) for line in it.ctx.splitlines()[:-1]]
            (x1, y1), (x2, y2) = pairs[0], pairs[1]
            a = (y2 - y1) // (x2 - x1)
            b = y1 - a * x1
            assert all(y == a * x + b for x, y in pairs) and int(ans) == a * int(it.ctx.splitlines()[-1].split()[0]) + b
        elif kind == "definition":
            nums = [int(v) for v in re.findall(r"\d+", it.ctx)]
            assert int(ans) in (nums[1] // nums[0], nums[0] * nums[1])


def test_knowledge_asks_facts_the_web_knows_in_ten_languages():
    from stet.exam.gen import knowledge

    assert ("France", "Paris") in knowledge.facts()[("capital", "en")]
    assert ("Francia", "París") in knowledge.facts()[("capital", "es")]
    items = _all("knowledge/", n=2)
    assert {it.lang for it in items} == set(knowledge.LANGS)
    for it in items:
        assert it.cont.strip() and it.ctx.strip() != it.cont.strip(), it  # São Tomé is the capital of São Tomé and Príncipe: allowed
        assert it.cont.startswith(" ") != (it.lang in knowledge.NO_SPACE)


# --- the skills in the web's languages ----------------------------------------------------------------


def _season_exam():
    from stet.spec import load_season

    return gen.generate(SEED, per_corner=int(load_season().raw["exam"]["per_corner"]))


def test_the_skills_that_count_are_asked_in_every_language_and_each_language_is_measured_on_its_own():
    from collections import Counter

    from stet.exam.aggregate import SMALL

    items = [it for it in _season_exam() if it.family in gen.MORE]
    n = Counter((it.family, it.lang) for it in items)
    for fam in gen.MORE:
        assert {lang for f, lang in n if f == fam} == set(lexicon.LANGS), fam
        assert all(n[fam, lang] >= SMALL for lang in lexicon.LANGS), (fam, n)  # none so few that it is pooled as "other"
    assert 2 * 538 <= len(items) <= 3 * 538  # two to three times what the five families had in English alone
    scripts = {"ru": r"[а-яё]", "ja": r"[぀-ヿ一-鿿]", "zh": r"[一-鿿]"}
    for it in items:
        assert it.corner.split("/")[1] == it.lang  # a language's questions are corners of their own
        text = it.ctx + it.cont + it.good + it.bad
        assert not re.search(r"\{[a-z_]+\}", text), it  # no sentence with a blank left unfilled
        if it.lang in scripts:
            assert re.search(scripts[it.lang], text.lower()), it  # written in the language's own script
            if it.family != "context" or "/passage/" not in it.corner:  # a town's year and its bridges are numbers
                assert re.search(scripts[it.lang], (it.cont or it.good).lower()), it  # and so is the answer
    assert len(items) - len({it.id for it in items}) <= 4  # a round hardly ever asks the same thing twice


def test_a_round_asks_its_own_questions_and_a_seed_always_the_same():
    families = [gen.logic, gen.context, gen.shell, gen.transform, gen.grammar]
    one, again, other = (gen.generate(seed, per_corner=2, families=families) for seed in (b"one", b"one", b"two"))
    assert one == again
    # What is read and what an agent works in are invented for the round: nothing of one round is in the next.
    asked = lambda items: {it.ctx + it.cont for it in items if it.kind == "answer" and it.family in ("context", "agent")}  # noqa: E731
    assert len(asked(one)) > 600 and not asked(one) & asked(other)


class Steps(Rng):
    """Not random: 0, 1, 2, ... each taken within its bound. What a generator makes of it depends on
    its sentences and word lists alone, so it can be written down here and read by someone who
    knows the language."""

    def __init__(self):
        self.i = 0

    def child(self, label):
        return self

    def below(self, n):
        self.i += 1
        return (self.i - 1) % n


# Read and checked by hand, language by language: an ordering and a "between", a town's river, two
# facts to join, a glossary, and both grammatical pairs (the acceptable sentence first).
CHECKED = {
    'logic/ru/order3-top': ('Глеб старше, чем Егор. Борис старше, чем Глеб. Кто старше всех?\nОтвет:', ' Борис'),
    'logic/ru/between': ('Анна старше, чем Вера. Вера старше, чем Дарья. Кто младше, чем Анна, но старше, чем Дарья?\nОтвет:', ' Вера'),
    'context/ru/passage/river': ('Бефокал — это город. Город находится в провинции Браклисту. Через город протекает река Норатир. Город был основан в 1221 году. Число мостов в городе — 34.\n\nВопрос: Какая река протекает через город Бефокал?\nОтвет:', ' Норатир'),
    'context/ru/two-hop': ('Город Бефокал находится в провинции Браклисту. Ирина живёт в городе Бефокал. Город Норатир находится в провинции Дигуле.\n\nИрина живёт в провинции', ' Браклисту'),
    'context/ru/glossary': ('В языке Ленорар:\n"камипу" означает "сад".\n"тизудрем" означает "стол".\n"стубефол" означает "дом".\n\nНа этом языке слово "дом" звучит как "', 'стубефол"'),
    'grammar/ru/agree-sg': ('Врач работает сейчас.', 'Врач работают сейчас.'),
    'grammar/ru/agree-pl': ('Врачи работают сейчас.', 'Врачи работает сейчас.'),
    'logic/de/order3-top': ('Dieter ist älter als Felix. Bernd ist älter als Dieter. Wer ist am ältesten?\nAntwort:', ' Bernd'),
    'logic/de/between': ('Anna ist älter als Clara. Clara ist älter als Emma. Wer ist jünger als Anna, aber älter als Emma?\nAntwort:', ' Clara'),
    'context/de/passage/river': ('Befokal ist eine Stadt. Die Stadt liegt in der Provinz Braklistu. Durch die Stadt fließt der Fluss Noratir. Die Stadt wurde im Jahr 1221 gegründet. Die Zahl der Brücken in der Stadt beträgt 34.\n\nFrage: Welcher Fluss fließt durch die Stadt Befokal?\nAntwort:', ' Noratir'),
    'context/de/two-hop': ('Die Stadt Befokal liegt in der Provinz Braklistu. Ingrid wohnt in der Stadt Befokal. Die Stadt Noratir liegt in der Provinz Digule.\n\nIngrid wohnt in der Provinz', ' Braklistu'),
    'context/de/glossary': ('In der Sprache Lenorar:\n"kamipu" bedeutet "Garten".\n"tizudrem" bedeutet "Tisch".\n"stubefol" bedeutet "Haus".\n\nIn dieser Sprache ist das Wort für "Haus" "', 'stubefol"'),
    'grammar/de/agree-sg': ('Der Arzt arbeitet jetzt.', 'Der Arzt arbeiten jetzt.'),
    'grammar/de/agree-pl': ('Die Ärzte arbeiten jetzt.', 'Die Ärzte arbeitet jetzt.'),
    'logic/ja/order3-top': ('美咲は陽子より年上です。花子は美咲より年上です。一番年上なのは誰ですか？\n答え：', '花子'),
    'logic/ja/between': ('太郎は健太より年上です。健太は翔太より年上です。太郎より年下で、翔太より年上なのは誰ですか？\n答え：', '健太'),
    'context/ja/passage/river': ('カキクは町です。この町にはコサシ川が流れています。この町には橋が25本あります。この町は1212年にできました。この町はセソタン県にあります。\n\nカキクという町に流れている川は', 'コサシ川'),
    'context/ja/two-hop': ('カキク町はセソタン県にあります。浩二はカキク町に住んでいます。コサシ町はツテト県にあります。\n\n浩二が住んでいる県は', 'セソタン県'),
    'context/ja/glossary': ('ニヌネ語の単語：\n「コサシ」は「庭」という意味です。\n「セソタン」は「机」という意味です。\n「ツテト」は「家」という意味です。\n\nニヌネ語で「家」を意味する単語は「', 'ツテト」'),
    'grammar/ja/counter': ('太郎は犬を四匹買いました。', '太郎は犬を四台買いました。'),
    'grammar/ja/exist': ('箱の中に皿があります。', '箱の中に皿をあります。'),
    'logic/zh/order3-top': ('刘洋比杨磊年纪大。李伟比刘洋年纪大。谁的年纪最大？\n答案：', '李伟'),
    'logic/zh/between': ('王芳比张敏年纪大。张敏比陈静年纪大。谁比王芳年纪小，但比陈静年纪大？\n答案：', '张敏'),
    'context/zh/passage/river': ('卡拉莫是一个镇。该镇位于维萨米省。尼塔洛河流经该镇。该镇建于1209年。该镇有22座桥。\n\n问题：流经卡拉莫镇的河叫什么名字？\n答案：', '尼塔洛河'),
    'context/zh/two-hop': ('尼塔洛镇位于巴里诺省。马超住在卡拉莫镇。卡拉莫镇位于维萨米省。\n\n马超住的省是', '维萨米省'),
    'context/zh/glossary': ('在达玛索语中：\n“塔洛维”的意思是“花园”。\n“萨米巴”的意思是“桌子”。\n“里诺伦”的意思是“房子”。\n\n在达玛索语中，表示“桌子”的词是“', '萨米巴”'),
    'grammar/zh/classifier': ('王芳有四匹马。', '王芳有四本马。'),
    'grammar/zh/negation': ('王芳没有自行车。', '王芳不有自行车。'),
    'logic/es/order3-top': ('Javier tiene más puntos que Miguel. Carlos tiene más puntos que Javier. ¿Quién tiene más puntos?\nRespuesta:', ' Carlos'),
    'logic/es/between': ('Ana tiene más puntos que María. María tiene más puntos que Lucía. ¿Quién tiene menos puntos que Ana pero más puntos que Lucía?\nRespuesta:', ' María'),
    'context/es/passage/river': ('Befokal es una ciudad. La ciudad está en la provincia de Braklistu. Por la ciudad pasa el río Noratir. La ciudad fue fundada en el año 1221. El número de puentes de la ciudad es 34.\n\nPregunta: ¿Qué río pasa por la ciudad de Befokal?\nRespuesta:', ' Noratir'),
    'context/es/two-hop': ('La ciudad de Befokal está en la provincia de Braklistu. Elena vive en la ciudad de Befokal. La ciudad de Noratir está en la provincia de Digule.\n\nElena vive en la provincia de', ' Braklistu'),
    'context/es/glossary': ('En el idioma Lenorar:\n"kamipu" significa "jardín".\n"tizudrem" significa "mesa".\n"stubefol" significa "casa".\n\nEn ese idioma, la palabra para "casa" es "', 'stubefol"'),
    'grammar/es/agree-sg': ('El médico viene ahora.', 'El médico vienen ahora.'),
    'grammar/es/agree-pl': ('Los médicos vienen ahora.', 'Los médicos viene ahora.'),
    'logic/fr/order3-top': ('Louis a plus de points que Julien. Pierre a plus de points que Louis. Qui a le plus de points ?\nRéponse :', ' Pierre'),
    'logic/fr/between': ('Marie a plus de points que Camille. Camille a plus de points que Sophie. Qui a moins de points que Marie mais plus de points que Sophie ?\nRéponse :', ' Camille'),
    'context/fr/passage/river': ('Befokal est une ville. La ville se trouve dans la province de Braklistu. La rivière Noratir traverse la ville. La ville a été fondée en 1221. Le nombre de ponts de la ville est de 34.\n\nQuestion : Quelle rivière traverse la ville de Befokal ?\nRéponse :', ' Noratir'),
    'context/fr/two-hop': ('La ville de Befokal se trouve dans la province de Braklistu. Manon habite dans la ville de Befokal. La ville de Noratir se trouve dans la province de Digule.\n\nManon habite dans la province de', ' Braklistu'),
    'context/fr/glossary': ('Dans la langue Lenorar :\n"kamipu" signifie "jardin".\n"tizudrem" signifie "table".\n"stubefol" signifie "maison".\n\nDans cette langue, le mot pour "maison" est "', 'stubefol"'),
    'grammar/fr/agree-sg': ('Le médecin travaille maintenant.', 'Le médecin travaillent maintenant.'),
    'grammar/fr/agree-pl': ('Les médecins travaillent maintenant.', 'Les médecins travaille maintenant.'),
    'logic/pt/order3-top': ('Pedro tem mais pontos do que Tiago. João tem mais pontos do que Pedro. Quem tem mais pontos?\nResposta:', ' João'),
    'logic/pt/between': ('Ana tem mais pontos do que Maria. Maria tem mais pontos do que Beatriz. Quem tem mais pontos do que Beatriz, mas menos pontos do que Ana?\nResposta:', ' Maria'),
    'context/pt/passage/river': ('Befokal é uma cidade. A cidade fica na província de Braklistu. O rio Noratir atravessa a cidade. A cidade foi fundada no ano de 1221. O número de pontes da cidade é 34.\n\nPergunta: Que rio atravessa a cidade de Befokal?\nResposta:', ' Noratir'),
    'context/pt/two-hop': ('A cidade de Befokal fica na província de Braklistu. Inês mora na cidade de Befokal. A cidade de Noratir fica na província de Digule.\n\nInês mora na província de', ' Braklistu'),
    'context/pt/glossary': ('Na língua Lenorar:\n"kamipu" significa "jardim".\n"tizudrem" significa "mesa".\n"stubefol" significa "casa".\n\nNessa língua, a palavra para "casa" é "', 'stubefol"'),
    'grammar/pt/agree-sg': ('O médico chega agora.', 'O médico chegam agora.'),
    'grammar/pt/agree-pl': ('Os médicos chegam agora.', 'Os médicos chega agora.'),
    'logic/it/order3-top': ('Sara ha più punti di Chiara. Giulia ha più punti di Sara. Chi ha più punti di tutti?\nRisposta:', ' Giulia'),
    'logic/it/between': ('Marco ha più punti di Luca. Luca ha più punti di Matteo. Chi ha meno punti di Marco ma più punti di Matteo?\nRisposta:', ' Luca'),
    'context/it/passage/river': ("Befokal è una città. La città si trova nella provincia di Braklistu. Il fiume Noratir attraversa la città. La città è stata fondata nell'anno 1221. Il numero di ponti della città è 34.\n\nDomanda: Quale fiume attraversa la città di Befokal?\nRisposta:", ' Noratir'),
    'context/it/two-hop': ('La città di Befokal si trova nella provincia di Braklistu. Stefano abita nella città di Befokal. La città di Noratir si trova nella provincia di Digule.\n\nStefano abita nella provincia di', ' Braklistu'),
    'context/it/glossary': ('Nella lingua Lenorar:\n"kamipu" significa "giardino".\n"tizudrem" significa "tavolo".\n"stubefol" significa "casa".\n\nIn quella lingua, la parola per "casa" è "', 'stubefol"'),
    'grammar/it/agree-sg': ('Il medico arriva adesso.', 'Il medico arrivano adesso.'),
    'grammar/it/agree-pl': ('I medici arrivano adesso.', 'I medici arriva adesso.'),
    'logic/pl/order3-top': ('Tomasz ma więcej punktów niż Marek. Piotr ma więcej punktów niż Tomasz. Kto ma najwięcej punktów?\nOdpowiedź:', ' Piotr'),
    'logic/pl/between': ('Anna ma więcej punktów niż Maria. Maria ma więcej punktów niż Katarzyna. Kto ma mniej punktów niż Anna, ale więcej punktów niż Katarzyna?\nOdpowiedź:', ' Maria'),
    'context/pl/passage/river': ('Befokal to miasto. Miasto leży w prowincji Braklistu. Przez miasto płynie rzeka Noratir. Miasto zostało założone w 1221 roku. Liczba mostów w mieście wynosi 34.\n\nPytanie: Jaka rzeka płynie przez miasto Befokal?\nOdpowiedź:', ' Noratir'),
    'context/pl/two-hop': ('Miasto Befokal leży w prowincji Braklistu. Ewa mieszka w mieście Befokal. Miasto Noratir leży w prowincji Digule.\n\nEwa mieszka w prowincji', ' Braklistu'),
    'context/pl/glossary': ('W języku Lenorar:\n"kamipu" znaczy "ogród".\n"tizudrem" znaczy "stół".\n"stubefol" znaczy "dom".\n\nW tym języku słowo "dom" to "', 'stubefol"'),
    'grammar/pl/agree-sg': ('Lekarz pracuje teraz.', 'Lekarz pracują teraz.'),
    'grammar/pl/agree-pl': ('Lekarze pracują teraz.', 'Lekarze pracuje teraz.'),
    'logic/nl/order3-top': ('Bram is ouder dan Thijs. Daan is ouder dan Bram. Wie is het oudst?\nAntwoord:', ' Daan'),
    'logic/nl/between': ('Anna is ouder dan Sanne. Sanne is ouder dan Lotte. Wie is jonger dan Anna maar ouder dan Lotte?\nAntwoord:', ' Sanne'),
    'context/nl/passage/river': ('Befokal is een stad. De stad ligt in de provincie Braklistu. Door de stad stroomt de rivier Noratir. De stad werd gesticht in het jaar 1221. Het aantal bruggen in de stad is 34.\n\nVraag: Welke rivier stroomt door de stad Befokal?\nAntwoord:', ' Noratir'),
    'context/nl/two-hop': ('De stad Befokal ligt in de provincie Braklistu. Sophie woont in de stad Befokal. De stad Noratir ligt in de provincie Digule.\n\nSophie woont in de provincie', ' Braklistu'),
    'context/nl/glossary': ('In de taal Lenorar:\n"kamipu" betekent "tuin".\n"tizudrem" betekent "tafel".\n"stubefol" betekent "huis".\n\nIn die taal is het woord voor "huis" "', 'stubefol"'),
    'grammar/nl/agree-sg': ('De dokter werkt nu.', 'De dokter werken nu.'),
    'grammar/nl/agree-pl': ('De dokters werken nu.', 'De dokters werkt nu.'),
}


def test_each_languages_sentences_are_the_ones_that_were_checked():
    import hashlib

    from stet.exam.gen import context, grammar, logic

    made = {}
    for fam in (logic, context, grammar):
        for corner in fam.corners():
            if corner in CHECKED:
                it = fam.make(Steps(), corner)
                made[corner] = (it.good, it.bad) if it.kind == "pair" else (it.ctx, it.cont)
    assert made == CHECKED
    assert {c.split("/")[1] for c in CHECKED} == set(lexicon.LANGS) - {"en"}
    # Every sentence and word list the languages use, as one hash. If this fails you changed one:
    # have it read by someone who knows the language, then pin the new hash.
    tables = [logic.COMPARE, logic.BETWEEN, context.TOLD, grammar.AGREE, grammar.WHEN, grammar.JA_COUNTED, grammar.JA_THINGS, grammar.JA_PLACES,
              grammar.ZH_COUNTED, grammar.ZH_THINGS, grammar.NUMERALS, lexicon.NOUNS, lexicon.GIVEN, lexicon.CITIES, lexicon.QUESTION,
              lexicon.ANSWER]
    assert hashlib.sha256(json.dumps(tables, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == "c701ced696c178905bb46ca05293e43bdf025f3716f8f25185656454d4bad404"
    # French shortens "de" and "que" before a vowel and Italian may: no name or coined word begins with one
    vowels = "aeiouyhàâéèêëîïôöùûüAEIOUYHÀÂÉÈÊËÎÏÔÖÙÛÜ"
    assert not [n for lang in ("fr", "it") for n in lexicon.GIVEN[lang] if n[0] in vowels]
    r = Rng(SEED, "coined")
    assert all(lexicon.coin(r, lang)[0] not in vowels + "аеиоу" for lang in ("fr", "it", "ru") for _ in range(300))
    for lang in lexicon.LANGS:
        assert len(lexicon.GIVEN[lang]) == len(set(lexicon.GIVEN[lang])) >= 20 and 20 <= len(set(lexicon.NOUNS[lang])) <= 40


def test_orderings_in_every_language_have_the_answer_their_facts_give():
    from stet.exam.gen import logic

    items = [it for it in _all("logic/") if it.lang != "en"]
    assert {it.lang for it in items} == set(logic.COMPARE) == set(lexicon.LANGS) - {"en"}
    for it in items:
        kind = it.corner.split("/")[2]
        told, ask = it.ctx.split("\n")[0], it.ctx.split("\n")[1]
        assert ask == lexicon.ANSWER[it.lang] and it.cont == lexicon.gap(it.lang) + it.cont.strip()
        # who is said to be more than whom, read back out of the sentences by each comparison's own pattern
        more = set()
        for fact, _, _ in logic.COMPARE[it.lang]:
            pattern = re.escape(fact).replace(r"\{a\}", r"(?P<a>\w+)").replace(r"\{b\}", r"(?P<b>\w+)")
            more |= {(m["a"], m["b"]) for m in re.finditer(pattern, told)}
        people = {x for pair in more for x in pair}
        assert len(more) == len(people) - 1 == int(kind[5]) - 1 if kind != "between" else len(more) == 2, it
        top = [x for x in people if not any(b == x for _, b in more)]
        bottom = [x for x in people if not any(a == x for a, _ in more)]
        middle = people - set(top) - set(bottom)
        want = top if kind.endswith("top") else bottom if kind.endswith("bottom") else sorted(middle)
        assert len(want) == 1 and it.cont.strip() == want[0], it


def test_a_pair_in_another_language_differs_in_the_one_place_that_is_wrong():
    from stet.exam.gen import grammar

    items = [it for it in _all("grammar/") if it.lang != "en"]
    assert {it.lang for it in items} == set(grammar.OTHER) == set(lexicon.LANGS) - {"en"}
    for it in items:
        if it.lang in grammar.AGREE:  # the same subject and the same ending, and the verb for one beside the verb for many
            nouns, verbs = grammar.AGREE[it.lang]
            one = it.corner.endswith("agree-sg")
            subject = next(n[not one] for n in nouns if it.good.lower().startswith(f"{n[not one].lower()} "))
            assert it.bad[: len(subject)] == it.good[: len(subject)] and it.good[0].isupper() and it.good[-1] == it.bad[-1] == "."
            (verb, _, when), (wrong, _, same) = (s[len(subject) + 1 : -1].partition(" ") for s in (it.good, it.bad))
            assert (verb, wrong) in {(v[not one], v[one]) for v in verbs} and when == same and when in ["", *grammar.WHEN[it.lang]], it
        else:  # one character: the counting word, or the particle
            assert len(it.good) == len(it.bad) and sum(a != b for a, b in zip(it.good, it.bad, strict=True)) == 1, it
    # a verb for one is never also a verb for many, and no subject reads the same for one and many
    for lang, (nouns, verbs) in grammar.AGREE.items():
        assert not {v for v, _ in verbs} & {v for _, v in verbs} and all(a != b for a, b in nouns), lang
    for thing, right, wrong in grammar.JA_COUNTED + grammar.ZH_COUNTED:
        assert right not in wrong and thing, (thing, right, wrong)


# --- what a small student cannot do cold, asked after examples -----------------------------------------


def _few_shot():
    from stet.exam.gen import shots

    return [f for f in gen.FAMILIES if isinstance(f, shots.Shots)]


def test_a_few_shot_question_is_shown_solved_examples_of_its_own_corner_and_never_itself():
    from stet.exam.gen import shots

    fams = _few_shot()
    cold = {f.NAME for f in gen.FAMILIES} - set(gen.MORE) - {f.NAME for f in fams} - {"knowledge"}
    assert {f.NAME for f in fams if not isinstance(f, shots.Pairs)} == {f"{name}-shots" for name in cold}  # every watched family but the list of facts
    assert {f.NAME for f in fams if isinstance(f, shots.Pairs)} == {"arith-pairs", "units-pairs"}
    root, counts = Rng(SEED, "few"), {}
    for fam in fams:
        base = set(fam.base.corners())
        for corner in fam.corners():
            head, shown, ask, sep = shots.examples(fam.base, root.child(corner), fam.of(corner))
            it = fam.make(root.child(corner), corner)
            assert fam.of(corner) in base and it.corner == corner and it.family == fam.NAME and it.source == f"gen:{fam.NAME}"
            assert 1 <= len(shown) <= 6 and all(s.corner == ask.corner == fam.of(corner) for s in shown)
            asked = [s.ctx for s in shown] + [ask.ctx]
            assert len(set(asked)) == len(asked), it  # no example is the question, and none is shown twice
            before = head + "".join(s.ctx + s.cont + sep for s in shown) + ask.ctx
            assert before.count(ask.ctx) == 1 and (it.fmt, it.lang) == (ask.fmt, ask.lang)
            if it.kind == "answer":
                assert (it.ctx, it.cont) == (before, ask.cont)
            else:
                assert it.good == f"{before}{ask.cont}\n" and it.bad.startswith(before) and it.bad != it.good
            if not hasattr(fam.base, "shared"):  # each example is a whole question of the family, as it would ask it cold
                again = [fam.base.make(root.child(corner).child(f"shot{i}"), fam.of(corner)) for i in range(60)]
                assert all(s in again for s in [*shown, ask])
            counts.setdefault(fam.NAME, []).append(len(shown))
    # three to six examples where they are short; fewer only where more would not fit a small window
    for name in ("arith-shots", "units-shots", "algebra-shots", "multi-shots", "arith-pairs", "units-pairs"):
        assert set(counts[name]) == {3, 4, 5, 6}, name
    assert all(st >= 2 for name in ("word-shots", "algo-shots") for st in [sorted(counts[name])[len(counts[name]) // 2]])


def test_few_shot_families_are_watched_and_count_for_nothing():
    items = gen.generate(SEED)
    few = {f.NAME for f in _few_shot()}
    rows = item_table(items)
    assert {row["part"] for row in rows} >= few and {row["pillar"] for row in rows if row["part"] in few} == {"watch"}
    g = [5.0 if it.family in few else 0.1 for it in items]
    r = round_score(items, g)
    assert r["score"] == pytest.approx(0.1) and set(r["pillars"]) == {"agents", "reading", "transformation", "logic", "grammar"}
    assert few <= set(r["watch"]) and all(r["watch"][name]["score"] == pytest.approx(5.0) for name in few)
    assert r["counted"] == sum(it.family in gen.MORE for it in items)
    assert score_from_table(rows, g, None)["watch"]["arith-shots"] == pytest.approx(5.0)
    # the server tells generated parts from writing by name: it must know every family
    from pathlib import Path

    go = (Path(__file__).resolve().parents[1] / "server" / "exam.go").read_text()
    known = set(re.findall(r'"([a-z-]+)": true', go.split("var generatedParts = map[string]bool{")[1].split("}")[0]))
    assert known == {f.NAME for f in gen.FAMILIES}


def test_a_near_miss_is_near_and_is_wrong():
    from stet.exam.gen import units

    pairs = [it for it in gen.generate(SEED, per_corner=3) if it.family.endswith("-pairs")]
    assert {it.family for it in pairs} == {"arith-pairs", "units-pairs"} and all(it.kind == "pair" for it in pairs)
    for it in pairs:
        assert it.good.endswith("\n") and it.bad.endswith("\n")  # both end with the line, so neither is the other's beginning
        cut = len(os.path.commonprefix([it.good, it.bad]))
        start = max(it.good.rfind(" ", 0, cut), it.good.rfind("\n", 0, cut)) + 1
        right, wrong = it.good[start:].strip(), it.bad[start:].strip()
        if it.family == "arith-pairs":  # one digit, one off; the same length, signs and commas
            assert len(right) == len(wrong) and [abs(int(a) - int(b)) for a, b in zip(right, wrong, strict=True) if a != b] == [1], it
            assert not (len(wrong.lstrip("-")) > 1 and wrong.lstrip("-")[0] == "0"), it
        else:  # the same quantity by another conversion's factor, or left unconverted
            key, way = it.corner.split("/")[2:]
            f = next(c[3] for c in units.CONVERSIONS if c[0] == key)
            others = {c[3] for c in units.CONVERSIONS} - {f} | {1}
            right, wrong = int(right), int(wrong)
            assert wrong != right and any(wrong * f == right * g if way == "up" else wrong * g == right * f for g in others), it


def test_few_shot_questions_fit_a_students_window():
    from stet.exam.gen import shots
    from stet.spec import load_season
    from stet.tok import load_encoding

    season = load_season("s2")
    enc = load_encoding(season)
    small, official = min(v["ctx"] for v in season.raw["scales"].values()), season.scale(season.official_scale).ctx
    few = {f.NAME for f in _few_shot()}
    sized = [(it, len(enc.encode_ordinary(it.ctx + it.cont if it.kind == "answer" else it.good))) for it in _season_exam() if it.family in few]
    assert len(sized) > 800 and all(n <= official for _, n in sized)
    over = [it.corner for it, n in sized if n > small]
    # The smallest student's window holds all but a few: the tables of six rows that are longer than it by themselves.
    assert len(over) <= 0.03 * len(sized) and all(c.startswith("struct-shots/") and c.endswith("/6") for c in over), over
    # A pair is read from its beginning and its answer is at its end: every one must fit whole.
    assert all(n <= small for it, n in sized if it.kind == "pair")
    # the estimate the generator drops examples by is near the truth where it matters, on long texts
    assert all(0.8 < shots.cost(it.ctx + it.cont) / 20 / n < 1.25 for it, n in sized if it.kind == "answer" and n > 150)


def test_no_item_is_sat_twice_and_an_agents_session_fits_the_students_window():
    import re

    from stet.exam.gen import shell

    for k in range(1, 6):
        items = gen.generate(bytes([k]) * 32, per_corner=2)
        assert len({it.id for it in items}) == len(items)  # a corner with little to say draws again
        for it in items:
            if it.source == "gen:agent" and it.kind == "answer":
                assert len((it.ctx + it.cont).encode()) <= shell.LIMIT, it.corner
            if it.source == "gen:algebra" and it.kind == "answer":
                assert not re.search(r"(?<![0-9])[01][xy]|\+ -|- -", it.ctx), it.ctx  # x and -x, never 1x or 0x

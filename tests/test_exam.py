"""The exam: generators that every machine agrees on, answers that are right, and a score
that rewards breadth."""

import contextlib
import datetime as dt
import io
import json
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
    assert digest(gen.generate(SEED)) == "5909f77e9425284ab1f3eeaa78d3a2f4a0e1ae17864120192bd68b4922c35668"
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
            if corner in ("cat", "head", "grep", "append-cat", "write-read"):
                for word in re.findall(r"[a-z]{3,}", it.cont.replace("\\n", " ")):  # a tool's result writes a line break as \n
                    assert word in it.ctx, (it.corner, word)
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
    for it in _all("transform/sql"):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE people (name TEXT, city TEXT, age INTEGER)")
        for line in it.ctx.split("|---|---|---|\n")[1].split("\n\n")[0].splitlines():
            name, city, age = [c.strip() for c in line.strip("| ").split("|")]
            db.execute("INSERT INTO people VALUES (?, ?, ?)", (name, city, int(age)))
        query = it.ctx.split("\n\n")[-1].split("\nResult:")[0]
        got = [str(r[0]) for r in db.execute(query)]
        assert ", ".join(got) == it.cont.strip(), (query, it.cont)
    for it in _all("transform/csv-json"):
        rows = json.loads(it.cont)
        assert [r["name"] for r in rows] == [ln.split(",")[0] for ln in it.ctx.split("CSV:\n")[1].split("\n\n")[0].splitlines()[1:]]


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
    assert {it.corner.split("/")[1] for it in items} == {"passage", "two-hop", "rule", "definition"}
    for it in items:
        kind, ans = it.corner.split("/")[1], it.cont.strip().rstrip('".')
        if kind in ("passage", "two-hop") or it.corner.endswith(("glossary",)):
            assert ans in it.ctx, it  # read, not remembered
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

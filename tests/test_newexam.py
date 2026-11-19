"""The next exam: generators that every machine agrees on, answers that are right, and a score
that rewards breadth."""

import contextlib
import datetime as dt
import io
import json
import re
import shutil
import subprocess

import pytest

from stet.newexam import gen
from stet.newexam.aggregate import gains, round_score, trimmed_mean
from stet.newexam.gen import lexicon
from stet.newexam.items import Item, digest
from stet.newexam.rng import Rng

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
    assert digest(gen.generate(SEED)) == "7093569d8ac52b277335a34b7e2766323b198e40bc9aa2f1f328a9a5c43ed16e"
    assert digest(gen.generate(SEED)) == digest(gen.generate(SEED))
    assert digest(gen.generate(b"other")) != digest(gen.generate(SEED))


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


def test_pillars_count_the_same_and_breadth_wins():
    # 30 arithmetic answers and one grammar pair: reasoning and writing, each half the score
    items = [_item("math", "answer", f"arith/{i}") for i in range(30)] + [_item("prose", "pair", "grammar/0")]
    r = round_score(items, [0.1] * 30 + [0.5])
    assert r["score"] == pytest.approx((0.1 + 0.5) / 2)
    assert set(r["pillars"]) == {"reasoning", "writing"} and set(r["strata"]) == {"math/answer", "prose/pair"}
    # a curator that gains hugely on two items and loses on the rest scores below one that gains a little everywhere
    items = [_item("math", "answer", f"arith/{i}") for i in range(40)]
    spiky = round_score(items, [5.0, 5.0] + [-0.02] * 38)["score"]
    broad = round_score(items, [0.05] * 40)["score"]
    assert broad > spiky


def test_every_family_and_source_has_a_pillar():
    from stet.newexam.aggregate import FAMILY_PILLAR, PILLARS, SOURCE_KIND, pillar_of
    from stet.newexam.gen import FAMILIES

    assert {f.NAME for f in FAMILIES} <= set(FAMILY_PILLAR) and set(FAMILY_PILLAR.values()) <= set(PILLARS)
    for src in SOURCE_KIND:
        assert pillar_of(_item("prose", "passage", "fresh/x", source=f"{src}:1"))[0] in PILLARS


def test_inside_a_pillar_each_family_counts_the_same():
    # 40 arithmetic items and 4 logic items: the two families weigh the same in reasoning
    items = [_item("math", "answer", f"arith/{i}") for i in range(40)] + [_item("prose", "answer", f"logic/{i}") for i in range(4)]
    r = round_score(items, [0.2] * 40 + [-0.2] * 4)
    assert r["score"] == pytest.approx(0.0) and r["pillars"]["reasoning"]["parts"] == {"arith": 0.2, "logic": -0.2}


def test_languages_count_by_the_corpus_inside_a_kind_of_writing():
    # A few English discussions must not outweigh many passages in other languages, and each
    # language counts by its share of the corpus.
    items = ([_item("prose", "passage", f"fresh/wikipedia/ru/{i}", lang="ru", source="wikipedia:1") for i in range(30)]
             + [_item("prose", "passage", f"fresh/wikipedia/en/{i}", lang="en", source="wikipedia:2") for i in range(10)])
    g = [-0.5] * 30 + [0.5] * 10
    r = round_score(items, g, shares={"en": 60.0, "ru": 40.0})
    assert r["fresh"]["languages"]["ru"]["weight"] == pytest.approx(0.4)
    assert r["score"] == pytest.approx(0.6 * 0.5 + 0.4 * -0.5) and r["generated"] is None
    assert set(r["pillars"]) == {"writing"} and r["pillars"]["writing"]["parts"] == {"reference": pytest.approx(0.1)}
    # the same items, weighed by a corpus that is mostly Russian: this curator loses
    assert round_score(items, g, shares={"en": 20.0, "ru": 80.0})["score"] < 0
    # discussion is the conversation pillar, which counts as much as all writing together
    talk = [_item("dialogue", "passage", f"fresh/wikipedia-talk/en/{i}", source="wikipedia-talk:3") for i in range(3)]
    r2 = round_score(items + talk, g + [0.3] * 3, shares={"en": 60.0, "ru": 40.0})
    assert r2["score"] == pytest.approx((0.1 + 0.3) / 2)


def test_programming_languages_count_together():
    items = [_item("code", "answer", f"prog/{i}", lang="code:python") for i in range(10)] + \
        [_item("code", "answer", f"prog/l{i}", lang="code:lua") for i in range(2)]
    r = round_score(items, [0.1] * 10 + [0.5] * 2)
    assert r["score"] == pytest.approx((0.1 * 9 + 0.5) / 10)  # one pool, trimmed by one at each end


def test_trimmed_mean_and_gains():
    assert trimmed_mean([0, 1, 2, 3, 100, 4, 5, 6, 7, 8], 0.1) == pytest.approx(4.5)
    assert gains([1.0, 2.0], [0.5, 2.5]) == [0.5, -0.5]
    with pytest.raises(ValueError):
        gains([1.0], [1.0, 2.0])


def test_slopes_put_gains_into_doublings():
    items = [_item("math", "answer", f"arith/{i}") for i in range(5)]
    r = round_score(items, [0.2] * 5, slopes={"math/answer": 0.1})
    assert r["score"] == pytest.approx(2.0) and r["units"] == "doublings"


@pytest.fixture(scope="module")
def student():
    torch = pytest.importorskip("torch")
    from stet.spec import load_season
    from stet.tok import load_encoding
    from stet.train import build_model

    season = load_season("s0")
    scale = season.scale("nano")
    return build_model(scale, season, 7, torch.device("cpu")), load_encoding(season), season.eot, scale.ctx


def test_sitting_scores_only_what_it_should(student):
    import torch

    from stet.newexam.score import nll, sequences, sit

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


def test_a_shadow_round_trains_once_per_script_and_scores_the_unedited_corpus_at_zero(corpus, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import dataclasses

    import torch

    from stet import data
    from stet.newexam import shadow
    from stet.script import write_script
    from stet.spec import load_season, practice_round

    season = load_season("s0")
    scale = dataclasses.replace(season.scale("nano"), train_tokens=2 * 16 * 256)  # two steps
    rnd = practice_round(season, 0)
    exam = {"news": [{"id": "n0", "text": "A short passage of news for the season's exam to score. " * 3}]}
    monkeypatch.setattr(data, "materialize", lambda r, s: (corpus, exam))
    trained = []
    real_train = shadow.train
    monkeypatch.setattr(shadow, "train", lambda *a, **k: trained.append(1) or real_train(*a, **k))
    same = tmp_path / "same.jsonl"
    write_script(same, range(len(corpus)))
    other = tmp_path / "other.jsonl"
    write_script(other, [3, 0])
    scripts = {"same": same, "other": other}
    s = shadow.shadow_round(rnd, scale, scripts, torch.device("cpu"), tmp_path / "out", per_corner=1, log=lambda m: None)
    rows = {r["name"]: r for r in s["entries"]}
    assert rows["same"]["next_score"] == 0.0 and rows["same"]["season_gain"] == 0.0
    assert rows["other"]["next_score"] != 0.0
    assert len(trained) == 2  # the unedited corpus and "other"; "same" is the unedited corpus's own script
    shadow.shadow_round(rnd, scale, scripts, torch.device("cpu"), tmp_path / "out", per_corner=1, log=lambda m: None)
    assert len(trained) == 2  # resumed from what was kept
    kept = [json.loads(line) for line in (tmp_path / "out" / "items.jsonl").read_text().splitlines()]
    assert len(kept) == s["items"] == len(gen.generate(rnd.seed, per_corner=1))
    # a bigger exam is sat by the students already trained
    fresh = [Item(kind="passage", corner="fresh/wikipedia/en/Q5/article", fmt="prose", lang="en", source="wikipedia:en:1",
                  text="A person who was born in a town and later moved to a city, where they worked for many years. " * 4)]
    s2 = shadow.shadow_round(rnd, scale, scripts, torch.device("cpu"), tmp_path / "out", per_corner=1, fresh=fresh, log=lambda m: None)
    assert len(trained) == 2 and s2["items"] == s["items"] + 1
    assert {r["name"]: r for r in s2["entries"]}["other"]["fresh_score"] != 0.0


def test_transforms_are_right():
    import sqlite3

    from stet.newexam.gen.transform import roman

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


def test_students_farmed_out_to_the_organisers_machines_add_up_as_a_local_run_does(corpus, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import dataclasses
    import hashlib
    import threading

    import torch
    import zstandard

    from stet import client, data
    from stet.newexam import farm, shadow
    from stet.script import write_script
    from stet.spec import load_season, practice_round

    season = load_season("s0")
    scale = dataclasses.replace(season.scale("nano"), train_tokens=2 * 16 * 256)
    rnd = practice_round(season, 0)
    exam = {"news": [{"id": "n0", "text": "A short passage of news for the season's exam to score. " * 3}]}
    busy = {"left": 1}  # the crawl host says "slow down" once: the worker waits and fetches again

    def materialize(r, s):
        if busy["left"]:
            busy["left"] -= 1
            raise data.FetchError("could not fetch CC-MAIN-x.warc.wet.gz: HTTP Error 503: Service Unavailable")
        return corpus, exam

    monkeypatch.setattr(data, "materialize", materialize)
    monkeypatch.setattr(type(season), "scale", lambda self, name: scale)
    monkeypatch.setattr(client, "round_from_api", lambda s, d: rnd)
    monkeypatch.setattr(farm, "POLL", 0.05)
    real_sleep = farm.time.sleep
    monkeypatch.setattr(farm.time, "sleep", lambda s: None if s > 1 else real_sleep(s))  # no waiting out a busy server here
    other = tmp_path / "other.jsonl"
    write_script(other, [3, 0])
    script_bytes = other.read_bytes()
    sha = hashlib.sha256(script_bytes).hexdigest()
    items = shadow.round_items(rnd, per_corner=1)[:60]

    # The server, as far as the farm sees it.
    lock = threading.Lock()
    batch: dict = {}

    def request(method, path, body=None, key=None, raw=False, timeout=0, limit=0):
        with lock:
            if path == "/api/v1/admin/shadow/batches":
                batch.update(id="b", items=body["items_file"], tasks=[{"script": s, "tries": 0} for s in body["scripts"]])
                return {"batch": "b"}
            if path == "/api/v1/admin/shadow/batches/b":
                return {"tasks": [dict(t) for t in batch["tasks"]]}
            if path == "/api/v1/shadow/claim":
                for i, t in enumerate(batch.get("tasks", [])):
                    if not t.get("done") and not t.get("holder"):
                        t["holder"], t["tries"] = 1, t["tries"] + 1
                        return {"task": {"batch": "b", "index": i, "round": rnd.id, "scale": "nano", "script": t["script"],
                                         "items_digest": digest(items), "items": len(items)}}
                return {"task": None}
            if path == "/api/v1/shadow/batches/b/items":
                import base64

                return base64.b64decode(batch["items"])
            if path.startswith("/api/v1/shadow/batches/b/tasks/"):
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
    local = shadow.shadow_round(rnd, scale, {"other": other}, torch.device("cpu"), tmp_path / "local", per_corner=1, log=lambda m: None)
    # The local run sat every generated item; the farm the first 60 of them. Each item is
    # scored on its own, so the local run's first 60 scores are what the farm should have.
    rows = {r["name"]: r for r in farmed["entries"]}
    assert [t["script"] for t in batch["tasks"]] == ["unedited", "unedited#2", sha]  # "same" is the unedited corpus itself
    assert rows["same"]["next_score"] == 0.0 and json.loads((tmp_path / "farmed" / f"{same_sha[:16]}.json").read_text())["name"] == "(unedited)"
    assert farmed["items"] == len(items) and rows["other"]["valid"]
    tag = local["items_digest"][:12]

    def kept(sha16):
        m = json.loads((tmp_path / "local" / f"{sha16}.json").read_text())
        m["item_scores"] = json.loads((tmp_path / "local" / f"{sha16}-{tag}.json").read_text())[: len(items)]
        return m

    base16 = next(p.stem for p in (tmp_path / "local").glob("????????????????.json") if p.stem != sha[:16])
    loc = shadow.summarize(rnd, scale, items, kept(base16), {"other": kept(sha[:16])})
    assert rows["other"]["next_score"] == pytest.approx(loc["entries"][0]["next_score"], abs=1e-5)
    assert farmed["baseline_exam_loss"] == pytest.approx(local["baseline_exam_loss"], abs=1e-6)
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
    from stet.newexam.gen import knowledge

    assert ("France", "Paris") in knowledge.facts()[("capital", "en")]
    assert ("Francia", "París") in knowledge.facts()[("capital", "es")]
    items = _all("knowledge/", n=2)
    assert {it.lang for it in items} == set(knowledge.LANGS)
    for it in items:
        assert it.cont.strip() and it.ctx.strip() != it.cont.strip(), it  # São Tomé is the capital of São Tomé and Príncipe: allowed
        assert it.cont.startswith(" ") != (it.lang in knowledge.NO_SPACE)

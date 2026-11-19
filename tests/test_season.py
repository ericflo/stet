"""The season, and its exam on the grader's side and the examiner's."""

import json
import types

import pytest
import zstandard

from stet import data
from stet.exam import gen
from stet.exam.items import Item, digest
from stet.spec import load_season, official_round, practice_round


def test_the_season_is_the_exam_at_the_standard_scale():
    from stet.spec import CURRENT_SEASON

    s1 = load_season()
    assert s1.id == CURRENT_SEASON == "s1" and s1.official_scale == "standard" and "heldout_exam" not in s1.raw
    assert s1.exam_sources == ["knowledge", "reading", "reasoning", "code", "agents", "writing", "conversation"]
    assert s1.raw["exam"]["generator_version"] == gen.VERSION and s1.raw["exam"]["more"] == gen.MORE
    assert len(s1.digest) == 64 and s1.digest == load_season("s1").digest
    # the tokenizer and the facts its knowledge items ask are what the season says
    import hashlib
    from importlib import resources

    tok = resources.files("stet.seasons").joinpath(s1.raw["tokenizer"]["file"]).read_bytes()
    assert hashlib.sha256(tok).hexdigest() == s1.raw["tokenizer"]["sha256"]
    facts = resources.files("stet.exam").joinpath("data", s1.raw["exam"]["facts"]["file"]).read_bytes()
    assert hashlib.sha256(facts).hexdigest() == s1.raw["exam"]["facts"]["sha256"]


def test_a_practice_rounds_exam_is_generated_from_its_seed():
    s1 = load_season("s1")
    rnd = practice_round(s1, 3)
    items = data.round_exam(rnd, s1.scale("micro"))
    assert isinstance(items, list) and items == gen.generate(rnd.seed, per_corner=2)
    assert data.exam_digest(items) == digest(items)


def _official(s1, exam_digest=""):
    beacon = {"round": s1.beacon_round_at(1_795_000_200), "randomness": "ab" * 32}
    crawl = {"crawl": "CC-MAIN-2026-43", "paths_sha256": "cd" * 32, "n_shards": 100000}
    rnd = official_round(s1, 7, beacon, crawl)
    import dataclasses

    return dataclasses.replace(rnd, exam_digest=exam_digest)


def test_an_official_rounds_exam_is_the_one_posted_and_nothing_else():
    s1 = load_season("s1")
    with pytest.raises(data.ExamNotReady):
        data.round_exam(_official(s1), s1.scale("standard"))  # not posted yet
    items = gen.generate(b"posted", per_corner=1)[:20]
    rnd = _official(s1, digest(items))
    with pytest.raises(data.ExamNotReady):
        data.round_exam(rnd, s1.scale("standard"))  # posted, but not fetched here
    path = data.exam_items_path(rnd, rnd.exam_digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(it.to_dict()) + "\n" for it in items).encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))
    assert data.round_exam(rnd, s1.scale("standard")) == items
    # a copy that is not the round's is refused, and removed so that it is fetched again
    path.write_bytes(zstandard.ZstdCompressor().compress(raw.replace(b'"cont": " ', b'"cont": "  ', 1)))
    with pytest.raises(data.ExamNotReady):
        data.round_exam(rnd, s1.scale("standard"))
    assert not path.exists()


def test_a_round_carries_its_exams_digest_through_its_description():
    from stet.spec import round_from_description

    s1 = load_season("s1")
    rnd = _official(s1, "ef" * 32)
    again = round_from_description(s1, rnd.describe())
    assert again == rnd and again.exam_digest == "ef" * 32
    with pytest.raises(ValueError):
        round_from_description(s1, {**rnd.describe(), "exam_digest": "not a hash"})


def test_a_grade_reports_every_items_score(monkeypatch):
    pytest.importorskip("torch")
    from stet import grade as grading
    from stet import tok

    items = [Item(kind="answer", corner="arith/x", fmt="math", lang="en", source="gen:arith", ctx="1 + 1 =", cont=" 2")] * 3
    stats = types.SimpleNamespace(out_docs=1, out_bytes=1, ink_tokens=0, ink_budget=1, tokens_one_pass=1, passes=1.0,
                                  to_dict=lambda corpus: {"script_sha256": "ab" * 32})
    monkeypatch.setattr(grading, "train", lambda *a: (None, {"tokens": 9, "curve": []}))
    monkeypatch.setattr(grading, "materialize", lambda *a: ([], stats))
    monkeypatch.setattr(grading, "file_sha256", lambda path, limit: "ab" * 32)
    monkeypatch.setattr(grading, "hardware_info", lambda device: {})
    monkeypatch.setattr(tok, "load_encoding", lambda season: None)
    import stet.exam.score as score

    monkeypatch.setattr(score, "sit", lambda model, its, enc, eot, ctx, device: [-1.0, -2.0, -3.0])
    s1 = load_season("s1")
    rnd = practice_round(s1, 0)
    corpus = types.SimpleNamespace(meta={"sha256": "c0"})
    result = grading.grade(rnd, s1.scale("micro"), corpus, items, "script.jsonl", device=types.SimpleNamespace(type="cpu"))
    assert result["valid"] and result["exam"]["item_scores"] == [-1.0, -2.0, -3.0]
    assert result["exam"]["mean"] == -2.0 and result["exam_sha256"] == digest(items)
    from stet.client import slim_result

    result.update(round={"id": rnd.id}, train={"seconds": 1, "curve": []})
    slim = slim_result(result, result)
    assert slim["exam"] == {"mean": -2.0, "item_scores": [-1.0, -2.0, -3.0]} and slim["baseline"] == slim["exam"]


def test_the_examiner_makes_a_rounds_exam_from_its_harvest_and_seed(monkeypatch, tmp_path):
    import base64

    from stet.exam import examine
    from stet.exam.aggregate import score_from_table
    from stet.exam.farm import unpack_items
    from stet.exam.fresh import Doc, write_snapshot

    s1 = load_season("s1")
    rnd = _official(s1)
    docs = [Doc(source="wikipedia", id=f"en:{i}", url=f"https://example.org/{i}", lang="en", license="CC BY-SA 4.0",
                published=1_800_000_000 + i, title=f"Title {i}", text=f"Article number {i} says something new. " * 20, topic=f"Q{i}")
            for i in range(30)]
    monkeypatch.setattr("stet.exam.fresh.harvest.harvest_window",
                        lambda seed, start, end, dest, log=print: write_snapshot(docs, dest, {"window": [start, end]}))
    monkeypatch.setattr(data, "corpus", lambda rnd, scale: object())
    monkeypatch.setattr("stet.exam.fresh.langs.shares", lambda corpus, seed: {"en": 40.0, "ru": 20.0})
    monkeypatch.setattr("stet.exam.decontam.seen", lambda items, corpus, season, log=print: {0})  # the corpus holds one
    exam = examine.build(rnd, s1.scale("standard"), 1_800_000_000, tmp_path / "work", log=lambda m: None)
    items = unpack_items(base64.b64decode(exam["items_file"]))
    assert exam["round"] == rnd.id and exam["items_digest"] == digest(items) and len(exam["table"]) == len(items)
    assert items[: len(items) - exam["fresh"]] == gen.generate(rnd.seed, per_corner=2)
    assert 0 < exam["fresh"] < 2 * len(docs) and exam["languages"] == {"en": 40.0, "ru": 20.0} and len(exam["snapshot_sha256"]) == 64
    # what the server is sent is all it needs to score a grade
    assert score_from_table(exam["table"], [0.1] * len(items), exam["languages"])["score"] == pytest.approx(0.1)
    # made again, it is the same exam
    assert examine.build(rnd, s1.scale("standard"), 1_800_000_000, tmp_path / "work", log=lambda m: None) == exam


def test_a_settled_rounds_published_exam_gives_the_open_items_and_checks_against_the_digest():
    from stet.exam.examine import public_items, read_public

    s1 = load_season("s1")
    generated = gen.generate(b"pub", per_corner=1)[:5]
    wiki = Item(kind="passage", corner="fresh/wikipedia/en/Q5/text", fmt="prose", lang="en", source="wikipedia:en:1", text="An article.")
    news = Item(kind="passage", corner="fresh/ccnews/en/x/text", fmt="prose", lang="en", source="ccnews:<urn:1>", text="A news story.")
    items = generated + [wiki, news]
    rows = public_items(items)
    assert rows[-1]["withheld"] and "text" not in rows[-1] and rows[-1]["id"] == news.id and "withheld" not in rows[-2]
    raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
    assert read_public(raw, digest(items)) == generated + [wiki]
    with pytest.raises(ValueError):
        read_public(raw, "00" * 32)
    # a replay of the settled round sits what was published
    rnd = _official(s1, digest(items))
    path = data.exam_public_path(rnd, rnd.exam_digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))
    assert data.round_exam(rnd, s1.scale("standard")) == generated + [wiki]

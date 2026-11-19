"""Season 1: the next exam as the season's exam, on the grader's side."""

import json
import types

import pytest
import zstandard

from stet import data
from stet.newexam import gen
from stet.newexam.items import Item, digest
from stet.spec import load_season, official_round, practice_round


def test_season_one_is_the_next_exam_at_the_standard_scale():
    s1, s0 = load_season("s1"), load_season("s0")
    assert s1.exam_kind == "next" and s0.exam_kind == "pack"
    assert s1.official_scale == "standard" and "heldout_exam" not in s1.raw
    assert s1.exam_sources == ["knowledge", "reading", "reasoning", "code", "agents", "writing", "conversation"]
    assert s1.raw["exam"]["generator_version"] == gen.VERSION and s1.raw["exam"]["more"] == gen.MORE
    assert len(s1.digest) == 64 and s1.digest != s0.digest
    # the tokenizer and the facts its knowledge items ask are what the season says
    import hashlib
    from importlib import resources

    tok = resources.files("stet.seasons").joinpath(s1.raw["tokenizer"]["file"]).read_bytes()
    assert hashlib.sha256(tok).hexdigest() == s1.raw["tokenizer"]["sha256"]
    facts = resources.files("stet.newexam").joinpath("data", s1.raw["exam"]["facts"]["file"]).read_bytes()
    assert hashlib.sha256(facts).hexdigest() == s1.raw["exam"]["facts"]["sha256"]


def test_season_one_keeps_nothing_out_of_the_corpus_for_its_exam():
    s1 = load_season("s1")
    assert len(data.exam_keys(s1)) == 0
    assert not data.contains_exam_text("The committee met on a rainy Tuesday and agreed, after much argument, to postpone the vote.", s1)


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
    data._exam_cache.clear()
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


def test_a_grade_on_the_next_exam_reports_every_items_score(monkeypatch):
    from stet import grade as grading
    from stet import tok

    items = [Item(kind="answer", corner="arith/x", fmt="math", lang="en", source="gen:arith", ctx="1 + 1 =", cont=" 2")] * 3
    stats = types.SimpleNamespace(out_docs=1, out_bytes=1, ink_tokens=0, ink_budget=1, tokens_one_pass=1, passes=1.0,
                                  to_dict=lambda corpus: {"script_sha256": "ab" * 32})
    monkeypatch.setattr(grading, "heldout_exam", lambda rnd, scale: None)
    monkeypatch.setattr(grading, "train", lambda *a: (None, {"tokens": 9, "curve": []}))
    monkeypatch.setattr(grading, "materialize", lambda *a: ([], stats))
    monkeypatch.setattr(grading, "file_sha256", lambda path, limit: "ab" * 32)
    monkeypatch.setattr(grading, "hardware_info", lambda device: {})
    monkeypatch.setattr(tok, "load_encoding", lambda season: None)
    import stet.newexam.score as score

    monkeypatch.setattr(score, "sit", lambda model, its, enc, eot, ctx, device: [-1.0, -2.0, -3.0])
    s1 = load_season("s1")
    rnd = practice_round(s1, 0)
    corpus = types.SimpleNamespace(meta={"sha256": "c0"})
    result = grading.grade(rnd, s1.scale("micro"), corpus, items, "script.jsonl", device=types.SimpleNamespace(type="cpu"))
    assert result["valid"] and result["exam"]["kind"] == "next" and result["exam"]["item_scores"] == [-1.0, -2.0, -3.0]
    assert result["exam"]["mean"] == -2.0 and result["exam_sha256"] == digest(items)
    from stet.client import slim_result

    result.update(round={"id": rnd.id}, train={"seconds": 1, "curve": []})
    slim = slim_result(result, result)
    assert slim["exam"] == {"kind": "next", "mean": -2.0, "item_scores": [-1.0, -2.0, -3.0]} and slim["baseline"] == slim["exam"]

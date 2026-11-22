"""The season, and its exam on the grader's side and the examiner's."""

import hashlib
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
    assert s1.id == CURRENT_SEASON == "s2" and s1.official_scale == "standard" and "heldout_exam" not in s1.raw
    assert s1.exam_sources == ["reference", "news", "technical", "conversation", "agents", "reading", "transformation", "logic", "grammar"]
    assert s1.raw["exam"]["groups"] == {"writing": 0.75, "skills": 0.25} and s1.answer_floor_bytes == 16
    assert s1.ink_tokens(s1.scale("standard")) == 16_000
    assert s1.raw["exam"]["generator_version"] == gen.VERSION and s1.raw["exam"]["more"] == gen.MORE
    assert len(s1.digest) == 64 and s1.digest == load_season("s2").digest
    # the tokenizer and the facts its knowledge items ask are what the season says
    import hashlib
    from importlib import resources

    tok = resources.files("stet.seasons").joinpath(s1.raw["tokenizer"]["file"]).read_bytes()
    assert hashlib.sha256(tok).hexdigest() == s1.raw["tokenizer"]["sha256"]
    facts = resources.files("stet.exam").joinpath("data", s1.raw["exam"]["facts"]["file"]).read_bytes()
    assert hashlib.sha256(facts).hexdigest() == s1.raw["exam"]["facts"]["sha256"]


def test_a_practice_rounds_exam_is_generated_from_its_seed():
    s1 = load_season("s2")
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
    s1 = load_season("s2")
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

    s1 = load_season("s2")
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
    stats = types.SimpleNamespace(out_docs=1, out_bytes=1, ink_tokens=0, ink_budget=1, tokens_one_pass=1, passes=1.0, ink_windows=[],
                                  to_dict=lambda corpus: {"script_sha256": "ab" * 32})
    monkeypatch.setattr(grading, "train", lambda *a: (None, {"tokens": 9, "curve": []}))
    monkeypatch.setattr(grading, "materialize", lambda *a: ([], stats))
    monkeypatch.setattr(grading, "file_sha256", lambda path, limit: "ab" * 32)
    monkeypatch.setattr(grading, "hardware_info", lambda device: {})
    monkeypatch.setattr(tok, "load_encoding", lambda season: None)
    import stet.exam.score as score

    monkeypatch.setattr(score, "sit", lambda model, its, enc, eot, ctx, device, floor=1: [-1.0, -2.0, -3.0])
    s1 = load_season("s2")
    rnd = practice_round(s1, 0)
    corpus = types.SimpleNamespace(meta={"sha256": "c0"})
    result = grading.grade(rnd, s1.scale("micro"), corpus, items, "script.jsonl", device=types.SimpleNamespace(type="cpu"))
    assert result["valid"] and result["exam"]["item_scores"] == [-1.0, -2.0, -3.0]
    assert result["exam"]["mean"] == -2.0 and result["exam_sha256"] == digest(items)
    from stet.client import slim_result

    result.update(round={"id": rnd.id}, train={"seconds": 1, "curve": []})
    slim = slim_result(result, result)
    assert slim["exam"] == {"mean": -2.0, "item_scores": [-1.0, -2.0, -3.0]} and slim["baseline"] == slim["exam"]

    # A script whose own text holds a passage of the exam is refused: its author wrote the passage in, and published it after.
    planted = "The river authority approved a new flood barrier for the lower town after three wet winters in a row."
    fresh = Item(kind="passage", corner="fresh/mastodon/en/-/text", fmt="prose", lang="en", source="mastodon:1", text="Breaking. " + planted)
    stats.ink_windows = [b"a page's text. " + planted.encode() + b" and more of the page"]
    result = grading.grade(rnd, s1.scale("micro"), corpus, [items[0], fresh, items[0]], "script.jsonl", device=types.SimpleNamespace(type="cpu"))
    assert not result["valid"] and "exam" not in result and result["invalid_reason"] == "the script's own text holds 1 of the passages in the round's exam"
    # The same words in a generated question are no sign of anything: it was made from the seed after the curator ran.
    made = Item(kind="passage", corner="agent/x", fmt="logs", lang="en", source="gen:agent", text=planted)
    assert grading.grade(rnd, s1.scale("micro"), corpus, [items[0], made, items[0]], "script.jsonl", device=types.SimpleNamespace(type="cpu"))["valid"]


def test_the_examiner_makes_a_rounds_exam_from_its_harvest_and_seed(monkeypatch, tmp_path):
    import base64

    from stet.exam import examine
    from stet.exam.aggregate import score_from_table
    from stet.exam.farm import unpack_items
    from stet.exam.fresh import Doc, write_snapshot

    s1 = load_season("s2")
    rnd = _official(s1)
    docs = [Doc(source="wikipedia", id=f"en:{i}", url=f"https://example.org/{i}", lang="en", license="CC BY-SA 4.0",
                published=1_800_000_000 + i, title=f"Title {i}", text=f"Article number {i} says something new. " * 20, topic=f"Q{i}")
            for i in range(30)]
    # Two hours' harvests, with one thread in both: it is one document.
    hours = [(1_800_003_600, tmp_path / "b.jsonl"), (1_800_000_000, tmp_path / "a.jsonl")]
    shas = [write_snapshot(docs[14:], hours[0][1], {"window": [hours[0][0], hours[0][0] + 3600]}),
            write_snapshot(docs[:15], hours[1][1], {"window": [hours[1][0], hours[1][0] + 3600]})]
    monkeypatch.setattr(data, "corpus", lambda rnd, scale: object())
    monkeypatch.setattr("stet.exam.fresh.langs.shares", lambda corpus, seed: {"en": 40.0, "ru": 20.0})
    monkeypatch.setattr("stet.exam.decontam.seen", lambda items, corpus, season, log=print: {0})  # the corpus holds one
    exam = examine.build(rnd, s1.scale("standard"), hours, tmp_path / "work", log=lambda m: None)
    items = unpack_items(base64.b64decode(exam["items_file"]))
    assert exam["round"] == rnd.id and exam["items_digest"] == digest(items) and len(exam["table"]) == len(items)
    assert items[: len(items) - exam["fresh"]] == gen.generate(rnd.seed, per_corner=2)
    assert 0 < exam["fresh"] < 2 * len(docs) and exam["languages"] == {"en": 40.0, "ru": 20.0}
    # what it was made from: the hours in their order, and one name for all of them
    assert exam["hours"] == [{"hour": 1_800_000_000, "sha256": shas[1]}, {"hour": 1_800_003_600, "sha256": shas[0]}]
    assert exam["snapshot_sha256"] == hashlib.sha256((shas[1] + shas[0]).encode()).hexdigest()
    assert len({it.source for it in items if not it.source.startswith("gen:")}) <= len(docs)
    # what the server is sent is all it needs to score a grade
    assert score_from_table(exam["table"], [0.1] * len(items), exam["languages"])["score"] == pytest.approx(0.1)
    # made again, it is the same exam
    assert examine.build(rnd, s1.scale("standard"), hours[::-1], tmp_path / "work", log=lambda m: None) == exam
    with pytest.raises(ValueError, match="no harvest"):
        examine.build(rnd, s1.scale("standard"), [], tmp_path / "work", log=lambda m: None)


def test_an_hours_harvest_is_chosen_with_the_beacon_and_a_rounds_hours_with_its_seed(monkeypatch, tmp_path):
    from stet.exam.fresh import harvest

    s2 = load_season("s2")
    hour = 1_800_000_000 - 1_800_000_000 % 3600
    seen = {}

    def window(seed, start, end, dest, log=print):
        seen.update(seed=seed, start=start, end=end)
        return {"window": [start, end], "documents": 0}, "ab" * 32

    monkeypatch.setattr(harvest, "harvest_window", window)
    meta, sha = harvest.harvest_hour(s2, hour, tmp_path / "h.jsonl", randomness="AB" * 32)
    number = s2.beacon_round_at(hour + 3600)  # the first beacon once the hour is over: nobody writing in the hour has it
    assert s2.beacon_time(number) >= hour + 3600 > s2.beacon_time(number - 1)
    assert seen == {"seed": harvest.hour_seed(s2, hour, number, "ab" * 32), "start": hour, "end": hour + 3600}
    assert meta["hour"] == hour and meta["beacon_round"] == number and sha == "ab" * 32
    assert harvest.hour_seed(s2, hour, number, "ab" * 32) != harvest.hour_seed(s2, hour, number, "cd" * 32)
    with pytest.raises(ValueError):
        harvest.harvest_hour(s2, hour + 60, tmp_path / "h.jsonl", randomness="ab" * 32)
    # an hour is harvested once it has been over for the season's settle time
    assert harvest.settled(s2, hour + 2 * 3600) == hour and harvest.settled(s2, hour + 2 * 3600 - 1) == hour - 3600
    # a round's hours: the same for a seed, different for another, never more than there are
    day = [hour - 3600 * i for i in range(24)]
    got = harvest.choose_hours(b"seed", day, 3)
    assert got == sorted(got) and len(got) == 3 and set(got) <= set(day) and got == harvest.choose_hours(b"seed", day[::-1] + day, 3)
    assert harvest.choose_hours(b"other", day, 3) != got and harvest.choose_hours(b"seed", day[:2], 3) == sorted(day[:2])
    assert harvest.choose_hours(b"seed", [], 3) == []
    # the server chooses the same way (server/harvest_test.go has this vector)
    assert harvest.choose_hours(bytes.fromhex("00" * 32), [3600 * i for i in range(1, 25)], 3) == [3600, 43200, 75600]


def test_a_settled_rounds_published_exam_gives_the_open_items_and_checks_against_the_digest():
    from stet.exam.examine import public_items, read_public

    s1 = load_season("s2")
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


def test_a_practice_round_sits_writing_once_some_has_been_harvested_here(monkeypatch, tmp_path):
    from stet.exam.fresh import Doc, write_snapshot
    from stet.pipeline import baseline_path
    from stet.spec import practice_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path / "cache"))
    s2 = load_season("s2")
    rnd, gym, scale = practice_round(s2, 3), practice_round(s2, 3, "gym"), s2.scale("micro")
    cpu = type("Device", (), {"type": "cpu"})()
    generated = gen.generate(rnd.seed, per_corner=2)
    # Nothing harvested: the generated items alone, as before.
    assert data.practice_hours(rnd) == [] and data.practice_key(rnd) == "" and data.round_exam(rnd, scale) == generated
    plain = baseline_path(rnd, scale, cpu)
    assert plain.name.endswith("-generated.json")

    def hour(h, n0):
        docs = [Doc(source="wikipedia", id=f"en:{i}", url=f"https://example.org/{i}", lang="en", license="CC BY-SA 4.0", published=h + i,
                    title=f"Title {i}", text=f"Article number {i} says something new. " * 20, topic=f"Q{i}") for i in range(n0, n0 + 10)]
        data.harvest_dir().mkdir(parents=True, exist_ok=True)
        write_snapshot(docs, data.harvest_dir() / f"{h}.jsonl", {"window": [h, h + 3600]})

    for k in range(5):
        hour(1_800_000_000 + 3600 * k, 10 * k)
    monkeypatch.setattr(data, "corpus", lambda rnd, scale: object())
    monkeypatch.setattr("stet.exam.decontam.seen", lambda items, corpus, season, log=print: set())
    hours = data.practice_hours(rnd)
    assert len(hours) == 3 and {h for h, _ in hours} < {1_800_000_000 + 3600 * k for k in range(5)}  # as many as an official round's
    assert hours != data.practice_hours(practice_round(s2, 4)) or hours != data.practice_hours(practice_round(s2, 5))  # by the round's seed
    exam = data.round_exam(rnd, scale)
    assert exam[: len(generated)] == generated and len(exam) > len(generated)
    assert all(it.source.startswith("wikipedia:") for it in exam[len(generated):])
    # Kept beside the round, and named for the hours: the baseline that sat the other exam is not this one's.
    monkeypatch.setattr("stet.exam.examine.assemble", lambda *a, **k: pytest.fail("made a second time"))
    assert data.round_exam(rnd, scale) == exam
    assert baseline_path(rnd, scale, cpu) != plain and data.practice_key(rnd)[:12] in baseline_path(rnd, scale, cpu).name
    # The gym's rounds are generated from their seeds, whatever is on this machine.
    assert data.practice_hours(gym) == [] and data.round_exam(gym, scale) == gen.generate(gym.seed, per_corner=2)

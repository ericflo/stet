import dataclasses
import json

from conftest import EXAM_SENTENCE

from stet import data
from stet.spec import Shard, load_season, practice_round

SEASON = load_season("s0")


def test_exam_text_is_found_anywhere_however_it_is_wrapped():
    assert data.contains_exam_text(EXAM_SENTENCE, SEASON)
    assert data.contains_exam_text("Before. " + EXAM_SENTENCE.upper().replace(" ", "\n  ") + " After.", SEASON)
    # Web pages run menus straight into prose, with no full stop in between.
    assert data.contains_exam_text("Home | About | Contact " + EXAM_SENTENCE[:70] + " and then it trails off", SEASON)
    assert not data.contains_exam_text(EXAM_SENTENCE[:60] + " but it ends another way entirely, as you can see.", SEASON)
    assert not data.contains_exam_text("Short. Also short. Tiny!", SEASON)
    assert not data.contains_exam_text("", SEASON)


def test_window_hashes_are_position_independent_and_pinned():
    b = bytes(range(256)) * 40
    h = data.window_hashes(b, 64)
    assert len(h) == len(b) - 63
    assert h[300] == data.window_hashes(b[300:364], 64)[0] == h[300 + 256]
    # Pinned: exact integer arithmetic, the same on every machine.
    assert int(data.window_hashes(b"stet" * 16, 64)[0]) == sum(c * pow(data._HASH_P, 63 - i, 1 << 64) for i, c in enumerate(b"stet" * 16)) % (1 << 64)


def test_only_distinctive_sentence_openings_are_keys():
    cfg = SEASON.raw["decontam"]
    k, need = cfg["key_bytes"], cfg["min_letters"]
    assert len(list(data.sentence_starts(EXAM_SENTENCE, k, need))) == 1
    assert not list(data.sentence_starts(" ".join(str(i) for i in range(1, 60)), k, need))  # page numbers
    assert not list(data.sentence_starts("-" * 200, k, need))  # a ruled line
    assert not list(data.sentence_starts("Your email address will not be published.", k, need))  # too short to be distinctive


def fake_pool(monkeypatch, docs):
    monkeypatch.setattr(data, "round_shards", lambda rnd, scale, quiet=False: [Shard("", 0, "fake-0", "u"), Shard("", 0, "fake-1", "u")])
    monkeypatch.setattr(data, "shard_path", lambda shard, quiet=False: shard.path)
    half = len(docs) // 2
    monkeypatch.setattr(data, "read_shard", lambda path, kind: iter(docs[:half] if path == "fake-0" else docs[half:]))


def test_build_corpus_shuffles_drops_exam_text_and_is_deterministic(tmp_path, monkeypatch):
    docs = [(f"http://h{i // 10}.test/{i}", f"Document number {i} says something of its own. " * 5) for i in range(60)]
    docs[7] = ("http://leak.test/", "Intro.\n" + EXAM_SENTENCE.replace(", ", ",\n") + "\nOutro.")
    docs[8] = ("http://blank.test/", "   \n ")
    docs[9] = ("http://long.test/", "é" * 300_000)
    fake_pool(monkeypatch, docs)
    rnd = practice_round(SEASON, 0)
    scale = SEASON.scale("nano")
    a = data.Corpus(data.build_corpus(rnd, scale, tmp_path / "a", quiet=True))
    assert len(a) == 58 and a.meta["decontaminated_docs"] == 1
    assert all(EXAM_SENTENCE[:30] not in text for _, text in a)
    assert a.size(a.urls.index("http://long.test/")) <= SEASON.max_doc_bytes
    a.text(a.urls.index("http://long.test/")).encode("utf-8")  # the cut landed on a character boundary
    # Arrival order groups documents by host; the corpus must not.
    arrival = [u for u, t in docs if t.strip() and "leak" not in u]
    assert a.urls != arrival and sorted(a.urls) == sorted(arrival)
    for i in (0, 17, 57):
        assert a.text(i) == dict(docs)[a.urls[i]] or a.urls[i] == "http://long.test/"
    b = data.Corpus(data.build_corpus(rnd, scale, tmp_path / "b", quiet=True))
    assert b.meta["sha256"] == a.meta["sha256"] and b.urls == a.urls
    c = data.Corpus(data.build_corpus(practice_round(SEASON, 1), scale, tmp_path / "c", quiet=True))
    assert c.urls != a.urls and sorted(c.urls) == sorted(a.urls)


def test_the_exam_draw_does_not_depend_on_the_corpus(monkeypatch):
    pool = {s: [{"id": f"{s}-{i}", "text": f"{s} {i}", "split": "practice", "source": s} for i in range(500)] for s in SEASON.exam_sources}
    monkeypatch.setattr(data, "load_exam_pool", lambda season, split: pool)
    scale = dataclasses.replace(SEASON.scale("nano"), exam_items_per_source=20)
    a = data.round_exam(practice_round(SEASON, 0), scale)
    b = data.round_exam(practice_round(SEASON, 1), scale)
    assert all(len(v) == 20 for v in a.values())
    assert data.exam_digest(a) != data.exam_digest(b)
    assert data.exam_digest(a) == data.exam_digest(data.round_exam(practice_round(SEASON, 0), scale))
    assert json.dumps(a)


def test_a_verifier_prunes_only_the_rounds_it_fetched_for_grading(tmp_path, monkeypatch):
    from stet.spec import load_season, official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    season = load_season("s0")
    crawl = {"crawl": "CC-MAIN-TEST", "paths_sha256": "ab" * 32, "n_shards": 10}
    rounds = [official_round(season, i, {"round": 600 * i, "randomness": "cd" * 32}, crawl) for i in range(1, 8)]
    for r in rounds:
        data.mark_verified(r)
        (data.cache_dir() / "rounds" / r.id / "docs.bin").write_bytes(b"x" * 100)
    mine = data.cache_dir() / "rounds" / "s0-o00002"
    (mine / ".verified").unlink()  # the user fetched this one themselves
    assert data.prune_verified(rounds[-1]) == 300  # rounds 1, 3 and 4 go; 2 is theirs; 5, 6 and 7 are recent
    left = sorted(p.name for p in (data.cache_dir() / "rounds").iterdir())
    assert left == ["s0-o00002", "s0-o00005", "s0-o00006", "s0-o00007"]

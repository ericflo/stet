"""What stet downloads, it also deletes: the cache does not grow for ever."""

import fcntl
import os

import pytest
from conftest import make_corpus

from stet import cli, client, data, service
from stet.spec import Shard, load_season, practice_round

SEASON = load_season("s0")
DAY = 86400.0
NOW = 2_000_000_000.0


def said(capsys):
    got = capsys.readouterr()
    return got.out + got.err


def put(path, size=10, age_days=0.0, now=NOW):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (now - age_days * DAY,) * 2)
    return path


def aged(path, age_days, now=NOW):
    os.utime(path, (now - age_days * DAY,) * 2)
    return path


def test_what_goes_unused_is_deleted_and_what_is_in_use_stays():
    root = data.cache_dir()
    old_round = put(root / "rounds" / "s0-p00001" / "nano" / "corpus-v3" / "docs.bin", 1000, 30)
    aged(old_round.parents[2], 30)
    used_round = put(root / "rounds" / "s0-p00002" / "nano" / "corpus-v3" / "docs.bin", 1000, 30)
    aged(used_round.parents[2], 30)
    put(used_round.parents[2] / ".used", 0, 1)  # built a month ago, used yesterday
    old_pool = put(root / "pool" / ("a" * 64 + ".jsonl.zst"), 500, 20)
    new_pool = put(root / "pool" / ("b" * 64 + ".jsonl.zst"), 500, 2)
    old_wet = put(root / "pool" / "wet" / "left.warc.wet.gz", 500, 15)
    old_model = put(root / "assets" / ("c" * 64), 800, 40)
    new_model = put(root / "assets" / ("d" * 64), 800, 13)
    old_script = put(root / "scripts" / ("e" * 64 + ".jsonl"), 50, 100)
    exam = put(root / "exam" / "exam-s0.jsonl.zst", 50, 400)  # the season's: every run needs it
    killed_upgrade = put(root / "upgrade-env" / "bin" / "python", 50, 2)
    aged(root / "upgrade-env", 2)
    not_ours = put(root / "upstream" / "something", 50, 400)  # nothing stet put there

    freed = data.sweep_unused(now=NOW)

    assert freed == 1000 + 500 + 500 + 800 + 50 + 50
    for gone in (old_round.parents[2], old_pool, old_wet, old_model, old_script, killed_upgrade.parents[1]):
        assert not gone.exists(), gone
    for kept in (used_round, new_pool, new_model, exam, not_ours):
        assert kept.exists(), kept


def test_the_cache_is_looked_over_a_few_times_a_day_not_on_every_job(monkeypatch):
    root = data.cache_dir()
    assert data.sweep_unused(now=NOW) == 0  # nothing there yet: this is the look
    stale = put(root / "scripts" / "old.jsonl", 50, 100)
    assert data.sweep_unused(now=NOW + 3600) == 0 and stale.exists()
    assert data.sweep_unused(now=NOW + data.SWEEP_EVERY + 60) == 50

    # And whoever wants everything kept can say so.
    monkeypatch.setenv("STET_CACHE_DAYS", "0")
    stale = put(root / "scripts" / "old.jsonl", 50, 1000)
    assert data.sweep_unused(now=NOW + 10 * DAY, every=0) == 0 and stale.exists()
    # A number too small to be safe for a job that is reading the cache is not taken literally.
    monkeypatch.setenv("STET_CACHE_DAYS", "0.0001")
    assert data.cache_days() * DAY == data.SET_ASIDE_FOR


def fake_crawl(monkeypatch, tmp_path, docs):
    """A round whose two crawl files are already downloaded, as fetch would have left them."""
    shards = [Shard("", 0, "crawl/a.warc.wet.gz", "http://crawl.test/a"), Shard("", 0, "crawl/b.warc.wet.gz", "http://crawl.test/b")]
    for s in shards:
        put(data.shard_path_of(s), 100)
    monkeypatch.setattr(data, "round_shards", lambda rnd, scale, quiet=False: shards)
    half = len(docs) // 2
    monkeypatch.setattr(data, "read_shard", lambda path, kind: iter(docs[:half] if path == data.shard_path_of(shards[0]) else docs[half:]))
    monkeypatch.setattr(data, "exam_keys", lambda season: None)
    monkeypatch.setattr(data, "contains_exam_text", lambda text, season, banned: False)
    return shards


def test_a_rounds_downloads_go_as_soon_as_its_corpus_is_built(tmp_path, monkeypatch):
    docs = [(f"http://h.test/{i}", f"Document number {i} says something of its own. " * 5) for i in range(20)]
    shards = fake_crawl(monkeypatch, tmp_path, docs)
    rnd = practice_round(SEASON, 0)  # a practice round: an official one always did this
    assert rnd.pool_cfg["kind"] == "cc-wet"
    corpus = data.corpus(rnd, SEASON.scale("nano"), quiet=True)
    assert len(corpus) == 20
    assert [data.shard_path_of(s).exists() for s in shards] == [False, False]
    assert (data.cache_dir() / "rounds" / rnd.id / ".used").exists()

    # Unless they were asked for.
    monkeypatch.setenv("STET_KEEP_DOWNLOADS", "1")
    shards = fake_crawl(monkeypatch, tmp_path, docs)
    data.corpus(practice_round(SEASON, 1), SEASON.scale("nano"), quiet=True)
    assert [data.shard_path_of(s).exists() for s in shards] == [True, True]


def test_using_a_round_keeps_it_however_long_ago_it_was_built():
    rnd, scale = practice_round(SEASON, 3), SEASON.scale("nano")
    dest = data.round_dir(rnd, scale) / "corpus-v3"
    make_corpus(dest, ["one document", "another"])
    aged(dest.parents[1], 60, now=data.time.time())
    other = put(data.cache_dir() / "rounds" / "s0-p00009" / "nano" / "corpus-v3" / "docs.bin", 100)
    aged(other.parents[2], 60, now=data.time.time())

    assert len(data.corpus(rnd, scale, quiet=True)) == 2  # used now, and the cache looked over
    assert dest.exists() and not other.exists()


def test_a_corpus_in_an_older_layout_is_not_kept_beside_the_new_one():
    rnd, scale = practice_round(SEASON, 4), SEASON.scale("nano")
    dest = data.round_dir(rnd, scale) / "corpus-v3"
    make_corpus(dest, ["one document"])
    old = put(dest.parent / "corpus" / "docs.bin", 100)
    older = put(dest.parent / "corpus-v2" / "docs.bin", 100)
    notes = put(dest.parent / "corpus-notes.txt", 5)  # not a corpus: someone's own file
    data.corpus(rnd, scale, quiet=True)
    assert dest.exists() and notes.exists()
    assert not old.exists() and not older.exists()


def test_clean_deletes_what_stet_downloaded_and_nothing_else(capsys):
    root = data.cache_dir()
    put(root / "rounds" / "s0-p00001" / "nano" / "corpus-v3" / "docs.bin", 3 << 20)
    put(root / "pool" / "wet" / "x.warc.wet.gz", 1 << 20)
    put(root / "assets" / ("c" * 64), 2 << 20)
    put(root / "exam" / "exam-s0.jsonl.zst", 1 << 20)
    put(root / "verify" / "slot0" / "job" / "left", 1 << 20)
    mine = put(root / "upstream" / "something", 50)

    cli.main(["clean", "--dry-run"])
    text = said(capsys)
    assert "8 MB" in text and "built rounds" in text and "Nothing was deleted" in text
    assert (root / "rounds").exists()

    cli.main(["clean"])
    text = said(capsys)
    assert "Cleared 8 MB" in text and "uv cache clean" in text
    assert sorted(p.name for p in root.iterdir()) == ["upstream"] and mine.exists()

    cli.main(["clean"])
    assert "Nothing to clear" in said(capsys)


@pytest.mark.skipif(os.name == "nt", reason="verifiers hold their slots with flock")
def test_clean_leaves_a_running_verifier_what_it_is_using(capsys):
    root = data.cache_dir()
    now = data.time.time()
    in_use = put(root / "rounds" / "s0-o00007" / "micro" / "corpus-v3" / "docs.bin", 1 << 20, 0, now)
    put(root / "rounds" / "s0-o00007" / ".used", 0, 0, now)
    idle = put(root / "rounds" / "s0-p00001" / "nano" / "corpus-v3" / "docs.bin", 1 << 20, 1, now)
    aged(idle.parents[2], 1, now)
    job = put(root / "verify" / "slot0" / "job" / "script.jsonl", 100, 0, now)
    exam = put(root / "exam" / "exam-s0.jsonl.zst", 100, 5, now)
    held = open(root / "verify" / "slot0.lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # as `stet verify` holds it while it runs
    try:
        assert client.verifiers_running() == 1
        cli.main(["clean"])
        assert "was left for it" in said(capsys)
        assert in_use.exists() and job.exists() and exam.exists() and not idle.exists()
    finally:
        held.close()
    assert client.verifiers_running() == 0


@pytest.mark.skipif(os.name == "nt", reason="verifiers hold their slots with flock")
def test_a_slot_nobody_holds_is_emptied_by_the_next_verifier_to_start():
    root = data.cache_dir() / "verify"
    left = put(root / "slot1" / "job" / "scratch" / "big", 1000)  # a second verifier, killed mid-job, never run again
    (root / "slot1.lock").touch()
    busy = put(root / "slot2" / "job" / "script.jsonl", 100)
    held = open(root / "slot2.lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        slot, lock = client.claim_workdir()
        assert slot.name == "slot0"
        assert not left.exists() and busy.exists()
        lock.close()
    finally:
        held.close()


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="reads its own log back through /proc")
def test_a_log_that_is_appended_to_for_months_is_cut_back_to_its_newest_part(tmp_path):
    path = tmp_path / "verify.log"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)  # as systemd opens a unit's log
    try:
        for i in range(2000):
            os.write(fd, f"line {i}\n".encode())
        assert not service.trim_log(fd, limit=1 << 20, keep=1000)  # small enough: left alone
        assert service.trim_log(fd, limit=10_000, keep=1000)
        os.write(fd, b"and it goes on\n")
    finally:
        os.close(fd)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("(older lines were cleared")
    assert lines[-2:] == ["line 1999", "and it goes on"] and len(path.read_bytes()) < 1200
    assert all(line.startswith("line ") for line in lines[1:-1])  # cut at a line, not through one

    # A file being written from a fixed place is not one to cut: the writing would go on past a hole.
    plain = tmp_path / "out.txt"
    fd = os.open(plain, os.O_WRONLY | os.O_CREAT)
    try:
        os.write(fd, b"x" * 20_000)
        assert not service.trim_log(fd, limit=10_000, keep=1000)
    finally:
        os.close(fd)
    assert plain.stat().st_size == 20_000

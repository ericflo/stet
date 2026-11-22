import gzip
import hashlib
import os
import subprocess
import sys
import time
import types
import zlib

import pytest

from stet import data
from stet.spec import Shard, load_season, practice_round

SEASON = load_season("s2")


def test_window_hashes_are_position_independent_and_pinned():
    b = bytes(range(256)) * 40
    h = data.window_hashes(b, 64)
    assert len(h) == len(b) - 63
    assert h[300] == data.window_hashes(b[300:364], 64)[0] == h[300 + 256]
    # Pinned: exact integer arithmetic, the same on every machine.
    assert int(data.window_hashes(b"stet" * 16, 64)[0]) == sum(c * pow(data._HASH_P, 63 - i, 1 << 64) for i, c in enumerate(b"stet" * 16)) % (1 << 64)


SENTENCE = "The committee met on a rainy Tuesday and agreed, after much argument, to postpone the vote."


def test_only_distinctive_sentence_openings_are_keys():
    cfg = SEASON.raw["decontam"]
    k, need = cfg["key_bytes"], cfg["min_letters"]
    assert len(list(data.sentence_starts(SENTENCE, k, need))) == 1
    assert not list(data.sentence_starts(" ".join(str(i) for i in range(1, 60)), k, need))  # page numbers
    assert not list(data.sentence_starts("-" * 200, k, need))  # a ruled line
    assert not list(data.sentence_starts("Your email address will not be published.", k, need))  # too short to be distinctive


def fake_pool(monkeypatch, docs):
    monkeypatch.setattr(data, "round_shards", lambda rnd, scale, quiet=False: [Shard("", 0, "fake-0", "u"), Shard("", 0, "fake-1", "u")])
    monkeypatch.setattr(data, "shard_path", lambda shard, quiet=False: shard.path)
    half = len(docs) // 2
    monkeypatch.setattr(data, "read_shard", lambda path, kind: iter(docs[:half] if path == "fake-0" else docs[half:]))


def test_build_corpus_shuffles_cuts_and_is_deterministic(tmp_path, monkeypatch):
    docs = [(f"http://h{i // 10}.test/{i}", f"Document number {i} says something of its own. " * 5) for i in range(60)]
    docs[8] = ("http://blank.test/", "   \n ")
    docs[9] = ("http://long.test/", "é" * 300_000)
    fake_pool(monkeypatch, docs)
    rnd = practice_round(SEASON, 0)
    scale = SEASON.scale("nano")
    a = data.Corpus(data.build_corpus(rnd, scale, tmp_path / "a", quiet=True))
    assert len(a) == 59  # the blank one is not a document
    assert a.size(a.urls.index("http://long.test/")) <= SEASON.max_doc_bytes
    a.text(a.urls.index("http://long.test/")).encode("utf-8")  # the cut landed on a character boundary
    # Arrival order groups documents by host; the corpus must not.
    arrival = [u for u, t in docs if t.strip()]
    assert a.urls != arrival and sorted(a.urls) == sorted(arrival)
    for i in (0, 17, 57):
        assert a.text(i) == dict(docs)[a.urls[i]] or a.urls[i] == "http://long.test/"
    b = data.Corpus(data.build_corpus(rnd, scale, tmp_path / "b", quiet=True))
    assert b.meta["sha256"] == a.meta["sha256"] and b.urls == a.urls
    c = data.Corpus(data.build_corpus(practice_round(SEASON, 1), scale, tmp_path / "c", quiet=True))
    assert c.urls != a.urls and sorted(c.urls) == sorted(a.urls)
    # Built again where a finished corpus already is (another verifier's, perhaps with a curator reading it):
    # that one is kept as it is, and it is what the caller is given.
    inode = (tmp_path / "a" / "docs.bin").stat().st_ino
    assert data.build_corpus(rnd, scale, tmp_path / "a", quiet=True) == tmp_path / "a"
    assert (tmp_path / "a" / "docs.bin").stat().st_ino == inode and [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []
    # What is there but is no corpus is replaced.
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "docs.bin").write_bytes(b"half")
    d = data.Corpus(data.build_corpus(rnd, scale, tmp_path / "d", quiet=True))
    assert d.meta["sha256"] == a.meta["sha256"]


def test_a_corpus_is_the_same_however_many_files_are_read_at_once(tmp_path, monkeypatch):
    import threading

    shards = [Shard("", 0, f"fake-{k}", "u") for k in range(11)]
    pages = {s.path: [(f"http://h{k}.test/{i}", f"File {k}, page {i}, says something of its own. " * 4) for i in range(7)] for k, s in enumerate(shards)}
    reading, most = [0], [0]
    lock = threading.Lock()

    def read_shard(path, kind):
        with lock:
            reading[0] += 1
            most[0] = max(most[0], reading[0])
        time.sleep(0.03 if int(path.split("-")[1]) % 3 == 0 else 0.0)  # files finish out of order
        yield from pages[path]
        with lock:
            reading[0] -= 1

    monkeypatch.setattr(data, "round_shards", lambda rnd, scale, quiet=False: shards)
    monkeypatch.setattr(data, "shard_path", lambda shard, quiet=False: shard.path)
    monkeypatch.setattr(data, "read_shard", read_shard)
    rnd, scale = practice_round(SEASON, 0), SEASON.scale("nano")
    monkeypatch.setattr(data, "BUILD_THREADS", 1)
    one = data.Corpus(data.build_corpus(rnd, scale, tmp_path / "one", quiet=True))
    assert most[0] == 1
    monkeypatch.setattr(data, "BUILD_THREADS", 4)
    monkeypatch.setattr(data.os, "cpu_count", lambda: 8)
    four = data.Corpus(data.build_corpus(rnd, scale, tmp_path / "four", quiet=True))
    assert 1 < most[0] <= 4  # several at once, and never more than asked for
    assert four.meta["sha256"] == one.meta["sha256"] and four.urls == one.urls and len(one) == 77

    # A file that cannot be read stops the build, and names the file.
    def broken(path, kind):
        if path == "fake-6":
            raise EOFError("cut short")
        yield from pages[path]

    monkeypatch.setattr(data, "read_shard", broken)
    with pytest.raises(OSError, match="fake-6 is damaged"):
        data.build_corpus(rnd, scale, tmp_path / "broken", quiet=True)
    assert not (tmp_path / "broken").exists()


def test_a_verifier_prunes_only_the_rounds_it_fetched_for_grading(tmp_path, monkeypatch):
    from stet.spec import load_season, official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    season = load_season("s2")
    crawl = {"crawl": "CC-MAIN-2026-01", "paths_sha256": "ab" * 32, "n_shards": 10}
    rounds = [official_round(season, i, {"round": 600 * i, "randomness": "cd" * 32}, crawl) for i in range(1, 8)]
    for r in rounds:
        data.mark_verified(r)
        (data.cache_dir() / "rounds" / r.id / "docs.bin").write_bytes(b"x" * 100)
    mine = data.cache_dir() / "rounds" / "s2-o00002"
    (mine / ".verified").unlink()  # the user fetched this one themselves
    assert data.prune_verified(rounds[-1]) == 300  # rounds 1, 3 and 4 go; 2 is theirs; 5, 6 and 7 are recent
    left = sorted(p.name for p in (data.cache_dir() / "rounds").iterdir())
    assert left == ["s2-o00002", "s2-o00005", "s2-o00006", "s2-o00007"]


class Endless:
    """A host that never stops sending."""

    headers: dict = {}

    def __init__(self):
        self.sent = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        self.sent += n
        return b"x" * n


def test_a_download_stops_at_its_limit_and_is_not_tried_again(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout=None: opened.append(Endless()) or opened[-1])
    with pytest.raises(data.TooBig):
        data.fetch(["http://a.test/x", "http://b.test/x"], "", tmp_path / "x.bin", quiet=True, max_bytes=3 << 20)
    assert len(opened) == 1 and opened[0].sent <= 4 << 20 and not list(tmp_path.iterdir())
    # A host that says up front it will send too much is stopped at the first read.
    monkeypatch.setattr(Endless, "headers", {"Content-Length": str(1 << 40)})
    with pytest.raises(data.TooBig):
        data.fetch(["http://a.test/x"], "", tmp_path / "y.bin", quiet=True, max_bytes=3 << 20)
    assert len(opened) == 2 and opened[1].sent == 1 << 20 and not list(tmp_path.iterdir())


class Sends:
    """A host that promises `promised` bytes and closes after sending `body`."""

    def __init__(self, body: bytes, promised: int | None):
        self.left, self.headers = body, {} if promised is None else {"Content-Length": str(promised)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        out, self.left = self.left[:n], self.left[n:]
        return out


def test_a_download_cut_short_is_fetched_again_not_kept(tmp_path, monkeypatch):
    # Crawl files are named by path and have no hash to check them by. A connection that
    # closes early ends the read like the end of the file, and a gzip file cut off between
    # two members still opens: the corpus built from it would be another corpus.
    whole = b"member one|member two|member three"
    answers = [Sends(whole[:10], len(whole)), Sends(whole, len(whole))]
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout=None: answers.pop(0))
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    got = data.fetch(["http://a.test/x.gz"], "", tmp_path / "x.gz", quiet=True)
    assert got.read_bytes() == whole and not answers and [p.name for p in tmp_path.iterdir()] == ["x.gz"]
    # A host that keeps doing it is a download that failed, which a verifier waits out.
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout=None: Sends(whole[:10], len(whole)))
    with pytest.raises(data.FetchError, match="cut short"):
        data.fetch(["http://a.test/y.gz"], "", tmp_path / "y.gz", quiet=True)
    assert [p.name for p in tmp_path.iterdir()] == ["x.gz"]


def gz(*members: bytes) -> bytes:
    return b"".join(gzip.compress(m) for m in members)


def wet(url: str, text: str) -> bytes:
    body = text.encode()
    return f"WARC/1.0\r\nWARC-Type: conversion\r\nWARC-Target-URI: {url}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body + b"\r\n\r\n"


PAGES = [(f"http://h.test/{i}", f"Page {i} of the crawl says something of its own, at some length. " * 4) for i in range(6)]
CRAWL_FILE = gz(*(wet(u, t) for u, t in PAGES))  # one gzip member per record, as Common Crawl writes them
A_SHARD = Shard("", 0, "crawl-data/CC-MAIN-2026-43/segments/1/wet/00000.warc.wet.gz", "http://cc.test/00000.warc.wet.gz")


def test_a_crawl_file_is_read_to_its_end_before_it_is_kept(tmp_path, monkeypatch):
    # No hash names a crawl file, and a host need not say how long one is. What can be
    # checked is that the gzip stream it sent ends where a gzip stream ends, and holds something.
    answers = [Sends(CRAWL_FILE[:-7], None), Sends(b"", None), Sends(b"<html>busy</html>", None), Sends(CRAWL_FILE, None)]
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout=None: answers.pop(0))
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    path = data.shard_path(A_SHARD, quiet=True)
    assert path.read_bytes() == CRAWL_FILE and not answers and list(data.read_wet(path)) == PAGES
    assert [p.name for p in path.parent.iterdir()] == [path.name]
    # A host that only ever sends nothing, or a gzip file of nothing, is a download that failed.
    other = Shard("", 0, "crawl-data/x/00001.warc.wet.gz", "http://cc.test/00001.warc.wet.gz")
    for nothing in (b"", gzip.compress(b"")):
        monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout=None, nothing=nothing: Sends(nothing, len(nothing)))
        with pytest.raises(data.FetchError, match="not a whole gzip file"):
            data.shard_path(other, quiet=True)
        assert [p.name for p in path.parent.iterdir()] == [path.name]
    # A file that has a hash is checked by its hash, as before, whatever it holds.
    blob = b"not gzip at all"
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout=None: Sends(blob, len(blob)))
    assert data.fetch(["http://a.test/b"], hashlib.sha256(blob).hexdigest(), tmp_path / "b.bin", quiet=True, gzipped=True).read_bytes() == blob


def test_a_damaged_crawl_file_in_the_cache_is_dropped_whatever_the_damage_looks_like(tmp_path, monkeypatch):
    rnd, scale = practice_round(SEASON, 0), SEASON.scale("nano")
    monkeypatch.setattr(data, "round_shards", lambda rnd, scale, quiet=False: [A_SHARD])
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout=None: Sends(CRAWL_FILE, len(CRAWL_FILE)))
    path = data.shard_path_of(A_SHARD)
    path.parent.mkdir(parents=True)
    dest = tmp_path / "round" / "corpus-v3"
    damage = {
        EOFError: CRAWL_FILE[:-9],  # cut off inside the last record
        gzip.BadGzipFile: CRAWL_FILE[:-4] + b"\0\0\0\0",  # a trailer that does not match
        zlib.error: CRAWL_FILE[:10] + b"\xff" * 64,  # a header, and then nothing that inflates
        ValueError: gz(b"WARC/1.0\r\nWARC-Type: conversion\r\nContent-Length: lots\r\n\r\nbody\r\n\r\n"),  # a length that is not a number
    }
    for error, blob in damage.items():
        path.write_bytes(blob)
        with pytest.raises(error):
            list(data.read_wet(path))
        with pytest.raises(OSError, match="is damaged .*removed from the cache, run again"):
            data.build_corpus(rnd, scale, dest, quiet=True)
        assert not path.exists() and not dest.exists() and list(dest.parent.iterdir()) == []  # and no half-built corpus beside it
        # Run again, as it says: the file is fetched afresh and the corpus is built.
        assert len(data.Corpus(data.build_corpus(rnd, scale, dest, quiet=True))) == len(PAGES)
        import shutil

        shutil.rmtree(dest)


def test_nothing_half_made_is_left_behind_on_any_way_out(tmp_path, monkeypatch):
    class Interrupted(Sends):
        def read(self, n):
            if len(self.left) < 20:
                raise KeyboardInterrupt
            return super().read(10)

    # Ctrl-C in the middle of a download.
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout=None: Interrupted(b"x" * 100, 100))
    with pytest.raises(KeyboardInterrupt):
        data.fetch(["http://a.test/x"], "", tmp_path / "pool" / "x.gz", quiet=True)
    assert list((tmp_path / "pool").iterdir()) == []

    # Ctrl-C in the middle of a build, with a shard already written out.
    docs = [(f"http://h.test/{i}", f"Document number {i} says something of its own. " * 5) for i in range(20)]
    fake_pool(monkeypatch, docs)

    def read_shard(path, kind):
        if path == "fake-1":
            raise KeyboardInterrupt
        return iter(docs[:10])

    monkeypatch.setattr(data, "read_shard", read_shard)
    dest = tmp_path / "round" / "corpus-v3"
    with pytest.raises(KeyboardInterrupt):
        data.build_corpus(practice_round(SEASON, 0), SEASON.scale("nano"), dest, quiet=True)
    assert list(dest.parent.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="asks whether a process is running")
def test_what_a_killed_build_left_is_cleared_by_the_next_one(tmp_path, monkeypatch):
    def a_dead_pid() -> int:
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        return p.pid

    dead, other_dead, alive = a_dead_pid(), a_dead_pid(), os.getppid()
    assert len({dead, other_dead, alive, os.getpid()}) == 4
    hour_ago = time.time() - 3600
    rounds, pool = tmp_path / "round", data.cache_dir() / "pool" / "wet"
    rounds.mkdir()
    pool.mkdir(parents=True)

    def leave(path, directory=False, fresh_inside=False):
        if directory:
            path.mkdir()
            (path / "arrival.bin").write_bytes(b"x" * 100)
            if not fresh_inside:
                os.utime(path / "arrival.bin", (hour_ago, hour_ago))
        else:
            path.write_bytes(b"x" * 100)
        os.utime(path, (hour_ago, hour_ago))
        return path

    killed = [leave(rounds / f"corpus-v3.tmp{dead}", directory=True), leave(pool / f"0123.warc.wet.gz.part{dead}"), leave(data.cache_dir() / "pool" / f"abcd.jsonl.zst.part{dead}")]
    kept = [
        leave(rounds / f"corpus-v3.tmp{alive}", directory=True),  # its process is running
        leave(pool / f"4567.warc.wet.gz.part{alive}"),
        leave(rounds / f"corpus-v3.tmp{other_dead}", directory=True, fresh_inside=True),  # no such process here, but written to a moment ago: another machine's
        leave(pool / "89ab.warc.wet.gz"),  # not a temporary file at all
        leave(pool / "notes.part"),
        leave(pool / "odd.part99999999999"),  # no process has that number; nor did this program write it
        leave(rounds / "corpus-v2.tmp1", directory=True),  # some other file's, and none of this build's business
    ]
    docs = [(f"http://h.test/{i}", f"Document number {i} says something of its own. " * 5) for i in range(20)]
    fake_pool(monkeypatch, docs)
    data.build_corpus(practice_round(SEASON, 0), SEASON.scale("nano"), rounds / "corpus-v3", quiet=True)
    assert not any(os.path.lexists(p) for p in killed) and all(os.path.lexists(p) for p in kept)
    assert (rounds / "corpus-v3" / "meta.json").exists()
    # Left alone long enough, the one nobody is writing goes too.
    assert data.sweep_stale(rounds, "corpus-v3.tmp*", now=time.time() + data.STALE_AFTER + 1) == 1
    assert not (rounds / f"corpus-v3.tmp{other_dead}").exists() and (rounds / f"corpus-v3.tmp{alive}").exists()


def test_a_cached_file_list_that_is_not_the_rounds_is_fetched_again(tmp_path, monkeypatch):
    paths = [f"crawl-data/CC-MAIN-2026-43/segments/1/wet/{i:05d}.warc.wet.gz" for i in range(5)]
    listing = gzip.compress(("\n".join(paths) + "\n").encode())
    cfg = {"kind": "cc-wet", "base_url": "http://cc.test/", "paths": "crawl-data/{crawl}/wet.paths.gz", "crawl": "CC-MAIN-2026-43",
           "paths_sha256": hashlib.sha256(listing).hexdigest(), "n_shards": 5, "shards_per_unit": 2}  # fmt: skip
    rnd = types.SimpleNamespace(pool_cfg=cfg, shard_indices=lambda scale, n: [3, 1])
    asked = []

    def urlopen(req, timeout=None):
        asked.append(req.full_url)
        return Sends(listing, len(listing))

    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    cached = data.cache_dir() / "pool" / "wet" / "CC-MAIN-2026-43.paths.gz"
    cached.parent.mkdir(parents=True)
    # Under the same name, a list that is not this round's: five files, in another order.
    # It opens, it has the right number of lines, and it would build a different corpus.
    cached.write_bytes(gzip.compress(("\n".join(reversed(paths)) + "\n").encode()))
    assert [s.path for s in data.round_shards(rnd, None, quiet=True)] == [paths[3], paths[1]]
    assert asked == ["http://cc.test/crawl-data/CC-MAIN-2026-43/wet.paths.gz"] and cached.read_bytes() == listing
    # The round's own list is not fetched twice.
    assert [s.path for s in data.round_shards(rnd, None, quiet=True)] == [paths[3], paths[1]] and len(asked) == 1
    # One cut short on disk is fetched again like a wrong one.
    cached.write_bytes(listing[:40])
    assert [s.url for s in data.round_shards(rnd, None, quiet=True)] == ["http://cc.test/" + paths[3], "http://cc.test/" + paths[1]] and len(asked) == 2
    # With no hash to go by, a list that cannot be read is removed, and the next run fetches it.
    rnd.pool_cfg = {**cfg, "paths_sha256": ""}
    cached.write_bytes(listing[:40])
    with pytest.raises(OSError, match="file list of CC-MAIN-2026-43 is damaged"):
        data.round_shards(rnd, None, quiet=True)
    assert not cached.exists() and len(data.round_shards(rnd, None, quiet=True)) == 2 and len(asked) == 3


def test_a_crawl_id_is_checked_where_it_becomes_a_file_name():
    assert data._crawl_id("CC-MAIN-2026-43") == "CC-MAIN-2026-43"
    for bad in ("../../x", "CC-MAIN-2026-43/..", "CC-MAIN-2026-43\n"):
        with pytest.raises(ValueError):
            data._crawl_id(bad)


# --- two verifiers on one machine share the cache -------------------------------------------------


def test_a_round_that_is_set_aside_stays_readable_to_whoever_has_it_open_and_is_deleted_later(tmp_path):
    import os

    from stet import data

    d = data.cache_dir() / "rounds" / "s2-o00009" / "micro"
    (d / "corpus-v3").mkdir(parents=True)
    (d / "corpus-v3" / "docs.bin").write_bytes(b"what a curator is reading")
    held = os.open(d / "corpus-v3", os.O_RDONLY)  # as a sandbox holds the directory it was given
    try:
        data.set_aside(d, now=1_000_000.0)
        assert not d.exists()  # the next job builds the round afresh
        with open(os.open("docs.bin", os.O_RDONLY, dir_fd=held), "rb") as f:
            assert f.read() == b"what a curator is reading"  # and nothing was pulled from under this one
    finally:
        os.close(held)
    stale = data.cache_dir() / "rounds" / ".stale"
    assert len(list(stale.iterdir())) == 1
    for p in stale.iterdir():
        os.utime(p, (1_000_000.0, 1_000_000.0))
    data.set_aside(d, now=1_000_000.0 + data.SET_ASIDE_FOR - 60)  # nothing to set aside, and not old enough to delete
    assert len(list(stale.iterdir())) == 1
    data.set_aside(d, now=1_000_000.0 + data.SET_ASIDE_FOR + 60)
    assert list(stale.iterdir()) == []

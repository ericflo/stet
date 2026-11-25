"""Pool shards, the on-disk corpus a curator reads, and a round's exam.

Corpus layout (language-agnostic, mmap-friendly):

    docs.bin    every document's UTF-8 bytes, concatenated
    docs.idx    uint64 little-endian offsets, n_docs + 1 of them; doc i is bin[idx[i]:idx[i+1]]
    urls.txt    one source URL per line, n_docs lines
    meta.json   round id, scale, doc and byte counts, sha256 of docs.bin
"""

from __future__ import annotations

import collections
import errno
import gzip
import hashlib
import io
import json
import mmap
import os
import re
import shutil
import sys
import time
import unicodedata
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zstandard

from .spec import HARNESS_VERSION, Round, Scale, Shard

USER_AGENT = f"stet/{HARNESS_VERSION}"
DEFAULT_API = "https://api.stetrun.com"
FETCH_TRIES = 4
FETCH_LIMIT_BYTES = 2 << 30  # no single file a round needs is near this
# What a download or a build leaves behind when it is killed outright is cleared away by the
# next build, once the process that was writing it is gone and it has been still this long.
# (The wait is for a cache shared between machines, where another machine's process cannot
# be seen from here: nothing live goes this long without writing.)
STALE_AFTER = 15 * 60


def cache_dir() -> Path:
    return Path(os.environ.get("STET_CACHE", "~/.cache/stet")).expanduser()


def api_base() -> str:
    return os.environ.get("STET_API", DEFAULT_API).rstrip("/")


# --- keeping the cache small ---------------------------------------------------------------

# Everything under the cache was downloaded, or built from downloads, and can be had again.
# So nothing in it is kept for good: what has not been used for this many days is deleted.
# STET_CACHE_DAYS changes the number, and 0 keeps everything. Some things go sooner by
# rules of their own: a round's crawl files as soon as its corpus is built from them, the
# rounds a verifier graded except the newest few, model files past the cache's size limit.
CACHE_DAYS = 14
SWEEP_EVERY = 6 * 3600  # a sweep costs a few hundred stat calls: not on every job
SET_ASIDE_FOR = 3 * 3600  # seconds a set-aside round is kept: longer than any job that could be reading it
# What the cache holds, by the directory it is in, largest users first.
CACHE_KINDS = (
    ("rounds", "built rounds"),
    ("pool", "downloaded crawl files"),
    ("models", "base student weights"),
    ("assets", "curators' model files"),
    ("verify", "a verifier's working files"),
    ("scripts", "edit scripts"),
    ("upgrade-env", "what an upgrade left"),
)


def cache_days() -> float:
    """How many days something in the cache may go unused. Never less than a job could take:
    another process on this machine may be in the middle of reading it."""
    try:
        days = float(os.environ.get("STET_CACHE_DAYS") or CACHE_DAYS)
    except ValueError:
        return float(CACHE_DAYS)
    return 0.0 if days <= 0 else max(days, SET_ASIDE_FOR / 86400)


def used(path: Path) -> None:
    """Note that something in the cache was used just now, so that it is among the last to go."""
    try:
        os.utime(path)
    except OSError:
        pass


def tree_bytes(path: Path) -> int:
    try:
        if path.is_symlink() or not path.is_dir():
            return path.lstat().st_size
    except OSError:
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _remove(path: Path) -> int:
    """Delete one thing in the cache. Returns the bytes it held, or 0 if it is still there."""
    size = tree_bytes(path)
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass
    return 0 if os.path.lexists(path) else size


def _last_use(path: Path) -> float:
    """When something in the cache was last used. A round's directory carries a marker that
    every use touches; anything else is touched itself."""
    times = []
    for p in (path / ".used", path):
        try:
            times.append(p.lstat().st_mtime)
        except OSError:
            pass
    return max(times, default=0.0)


def cache_entries() -> list[tuple[str, Path, float]]:
    """Everything a sweep may delete: (kind, path, how long it may go unused in seconds)."""
    root, days = cache_dir(), cache_days() * 86400
    out: list[tuple[str, Path, float]] = []

    def each(d: Path):
        try:
            return sorted(d.iterdir())
        except OSError:
            return []

    for d in each(root / "rounds"):
        if d.name == ".stale":
            out += [("rounds", old, float(SET_ASIDE_FOR)) for old in each(d)]
        else:
            out.append(("rounds", d, days))
    for p in each(root / "pool"):
        if p.name == "wet" and p.is_dir():
            out += [("pool", q, days) for q in each(p)]
        else:
            out.append(("pool", p, days))
    for kind in ("assets", "scripts"):
        out += [(kind, p, days) for p in each(root / kind)]
    # What a process that was killed outright never got to remove.
    if (root / "upgrade-env").exists():
        out.append(("upgrade-env", root / "upgrade-env", 86400.0))
    out += [("selftest", p, 86400.0) for p in each(root) if p.name.startswith("stet-selftest-")]
    return out


def sweep_unused(now: float | None = None, every: float = SWEEP_EVERY) -> int:
    """Delete what this machine downloaded and has not used for a while. Returns the bytes freed.

    Called wherever the cache is used, and it looks at most once every `every` seconds. A
    machine left verifying for months, and one that practised for a week and stopped, both
    end with a cache no bigger than their last two weeks needed.
    """
    days = cache_days()
    if days <= 0:
        return 0
    now = time.time() if now is None else now
    marker = cache_dir() / ".swept"
    try:
        if every and abs(now - marker.stat().st_mtime) < every:
            return 0
    except OSError:
        pass
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        os.utime(marker, (now, now))
    except OSError:
        return 0
    freed = 0
    for _, path, allowed in cache_entries():
        last = _last_use(path)
        if last and now - last > allowed:
            freed += _remove(path)
    return freed


def cache_report() -> list[tuple[str, str, int]]:
    """What the cache holds: (directory, what it is, bytes), largest first."""
    root = cache_dir()
    rows = [(name, what, tree_bytes(root / name)) for name, what in CACHE_KINDS if (root / name).exists()]
    return sorted((r for r in rows if r[2]), key=lambda r: -r[2])


def clear_cache(in_use: bool = False, now: float | None = None) -> int:
    """Delete everything in the cache. All of it can be fetched again. Returns the bytes freed.

    With `in_use`, a verifier is at work in this cache: what was used in the last few hours,
    which it may be reading, is left, and so are its working files and the exam.
    """
    root = cache_dir()
    if in_use:
        now = time.time() if now is None else now
        return sum(_remove(p) for _, p, _ in cache_entries() if now - _last_use(p) > SET_ASIDE_FOR)
    freed = sum(_remove(root / name) for name, _ in CACHE_KINDS)
    try:
        freed += sum(_remove(p) for p in root.glob("stet-selftest-*"))
        (root / ".swept").unlink(missing_ok=True)
    except OSError:
        pass
    return freed


# --- downloads ----------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class FetchError(OSError):
    """A download failed after every try. The network's fault, or the host's; not this machine's."""


class TooBig(FetchError):
    """A download ran past its limit. Not retried: the next try would do the same."""


class Mismatch(OSError):
    """What arrived is whole, and is not the bytes that were named."""


def _read_gzip_through(path: Path) -> None:
    """Read a gzip file to its end. Raises if it is cut off, damaged, or holds nothing."""
    size = 0
    with gzip.open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            size += len(chunk)
    if not size:
        raise OSError("the file is empty")


def _running(pid: int) -> bool:
    """Whether a process with this id exists on this machine. POSIX only: on Windows,
    signal 0 is Ctrl-C, and sending it is no way to ask."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, OverflowError):
        return True  # someone else's process, or a number no process has: not ours to judge
    return True


def sweep_stale(directory: Path, pattern: str, now: float | None = None) -> int:
    """Remove what a killed download or build left in `directory`: entries matching
    `pattern` whose name ends in the id of a process that is no longer running, and that
    nothing has written to for STALE_AFTER seconds. Returns how many were removed.

    Every temporary file here is named for the process writing it and removed on the way
    out, whatever the way out is. That leaves `kill -9` and a power cut, and a verifier
    that is restarted after either would otherwise lose a gigabyte to each.

    On Windows a process cannot be asked after safely, so there it goes by age alone, and
    waits a day.
    """
    now = time.time() if now is None else now
    posix = os.name != "nt"
    wait = STALE_AFTER if posix else 24 * 3600
    removed = 0
    try:
        found = list(directory.glob(pattern))
    except OSError:
        return 0
    for p in found:
        m = re.search(r"\.(?:part|tmp)([0-9]{1,10})(?:\.npy)?$", p.name)
        if not m or int(m.group(1)) == os.getpid() or (posix and _running(int(m.group(1)))):
            continue
        try:
            is_dir = p.is_dir() and not p.is_symlink()
            latest = max([p.lstat().st_mtime] + ([q.lstat().st_mtime for q in p.iterdir()] if is_dir else []))
            if now - latest < wait:
                continue
            if is_dir:
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed += not os.path.lexists(p)
        except OSError:
            continue
    return removed


def fetch(
    urls: list[str],
    sha256: str,
    dest: Path,
    label: str = "",
    quiet: bool = False,
    max_bytes: int = FETCH_LIMIT_BYTES,
    gzipped: bool = False,
    verify: bool = False,
) -> Path:
    """Download to `dest` unless it is already there, verifying the content hash.

    With `verify=True` and a non-empty `sha256`, an existing cached file is also checked against
    `sha256` on every load (raising `Mismatch` and removing the damaged copy if it differs).

    Stops at `max_bytes`: the hash is only known at the end, and a host that never stops
    sending must not fill the disk first.

    `gzipped` is for a file that has no hash to check it by: it is read through to the end
    of its gzip stream before it is kept, so that a copy in the cache is a whole one.
    """
    if dest.exists():
        if verify and sha256:
            got = _sha256_file(dest)
            if got != sha256:
                dest.unlink(missing_ok=True)
                raise Mismatch(f"cached {label or dest.name} sha256 mismatch: got {got}, want {sha256}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    urls = [u.replace("{api}", api_base()) for u in urls]
    for attempt in range(FETCH_TRIES * len(urls)):
        url = urls[attempt % len(urls)]
        if attempt >= len(urls):
            time.sleep(min(30, 2 ** (attempt // len(urls))))  # public data hosts throttle; be patient
        part = dest.with_suffix(dest.suffix + f".part{os.getpid()}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            h = hashlib.sha256()
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                last = 0.0
                while chunk := r.read(1 << 20):
                    done += len(chunk)
                    if max(done, total) > max_bytes:
                        raise TooBig(f"{label or dest.name} is larger than {max_bytes / 1e6:.0f} MB; refusing to download it")
                    f.write(chunk)
                    h.update(chunk)
                    if not quiet and time.time() - last > 0.5:
                        last = time.time()
                        pct = f"{100 * done / total:5.1f}%" if total else ""
                        sys.stderr.write(f"\r  fetching {label or dest.name} {done / 1e6:7.1f} MB {pct}")
                        sys.stderr.flush()
            if total and done != total:
                # A connection that closes early ends the loop like the end of the file. Most
                # of what is fetched here has no hash to catch it (crawl files are named by
                # path), and a file cut off at the right place still opens.
                raise OSError(f"the download of {url} was cut short: {done:,} of {total:,} bytes arrived")
            if not quiet:
                sys.stderr.write(f"\r  fetched  {label or dest.name} {done / 1e6:7.1f} MB in {time.time() - t0:.0f}s\n")
            if sha256 and h.hexdigest() != sha256:
                raise Mismatch(f"sha256 mismatch for {url}: got {h.hexdigest()}, want {sha256}")
            if gzipped and not sha256:
                try:
                    _read_gzip_through(part)
                except (OSError, EOFError, zlib.error) as e:
                    raise OSError(f"what {url} sent is not a whole gzip file ({e})") from None
            part.replace(dest)
            return dest
        except TooBig:
            raise
        except Exception as e:  # try the next mirror
            if isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EROFS):
                raise  # this machine's disk, not the host: trying again will not help
            last_err = e
        finally:
            # On every way out, Ctrl-C included. (Once it has been renamed there is nothing here.)
            part.unlink(missing_ok=True)
    err = FetchError(f"could not fetch {label or dest.name}: {last_err}")
    err.last = last_err  # what the last try met, for a caller that can tell a "no" from a "not now"
    raise err


def round_shards(rnd: Round, scale: Scale, quiet: bool = False) -> list[Shard]:
    """The pool files this round's corpus is built from, in draw order."""
    cfg = rnd.pool_cfg
    if cfg["kind"] == "dclm":
        pool = rnd.season.pool(rnd.kind)
    elif cfg["kind"] == "cc-wet":
        # A crawl is identified by the hash of its published file list, fixed in the round.
        want = str(cfg.get("paths_sha256") or "")
        cached = cache_dir() / "pool" / "wet" / f"{_crawl_id(cfg['crawl'])}.paths.gz"
        # A copy already in the cache is checked like a fresh one. Which files a round is
        # built from is read off this list by position: a list that is not the round's
        # would build some other corpus, and nothing later would say why.
        if want and cached.exists() and _sha256_file(cached) != want:
            cached.unlink(missing_ok=True)
        listing = fetch([cfg["base_url"] + cfg["paths"].format(crawl=cfg["crawl"])], want, cached, f"{cfg['crawl']} file list", quiet, max_bytes=64 << 20)
        used(listing)
        try:
            paths = gzip.decompress(listing.read_bytes()).decode().split()
        except (OSError, EOFError, zlib.error, ValueError) as e:
            listing.unlink(missing_ok=True)
            raise OSError(f"the file list of {cfg['crawl']} is damaged ({e}); it has been removed from the cache, run again") from None
        if len(paths) != int(cfg["n_shards"]):
            if not want:
                listing.unlink(missing_ok=True)  # nothing vouched for it: fetch it again next time
            raise OSError(f"{cfg['crawl']}: expected {cfg['n_shards']} files, the list has {len(paths)}")
        pool = [Shard(sha256="", size=0, path=p, url=cfg["base_url"] + p) for p in paths]
    else:
        raise ValueError(f"unknown pool kind {cfg['kind']!r}; upgrade stet")
    return [pool[i] for i in rnd.shard_indices(scale, len(pool))]


def _crawl_id(crawl: str) -> str:
    """A crawl id as a file name. Official rounds take theirs from the server, so it is
    checked where the round is built (spec.official_round) and again here, where it matters."""
    from .spec import CRAWL_ID_RE

    if not CRAWL_ID_RE.fullmatch(crawl):
        raise ValueError(f"{crawl[:40]!r} is not a Common Crawl id")
    return crawl


def shard_path_of(shard: Shard) -> Path:
    """Where a pool file is (or would be) cached."""
    if shard.sha256:
        return cache_dir() / "pool" / f"{shard.sha256}.jsonl.zst"
    name = hashlib.sha256(shard.url.encode()).hexdigest()[:32]
    return cache_dir() / "pool" / "wet" / f"{name}.warc.wet.gz"


def shard_path(shard: Shard, quiet: bool = False) -> Path:
    if shard.sha256:
        path = fetch([shard.url], shard.sha256, shard_path_of(shard), shard.path, quiet, max_bytes=shard.size or FETCH_LIMIT_BYTES)
    else:
        # Crawl files are not listed with hashes. They are gzip, which carries its own checksum,
        # so each is read to the end of its stream before it is kept; and the built corpus's
        # sha256 is compared across verifiers.
        path = fetch([shard.url], "", shard_path_of(shard), shard.path.rsplit("/", 1)[-1], quiet, gzipped=True)
    used(path)
    return path


def read_dclm(path: Path):
    with open(path, "rb") as f:
        reader = zstandard.ZstdDecompressor().stream_reader(f)
        for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
            rec = json.loads(line)
            md = rec.get("metadata") or {}
            yield md.get("WARC-Target-URI") or "", rec.get("text") or ""


def read_wet(path: Path):
    """Common Crawl WET: a WARC file whose `conversion` records hold each page's extracted text."""
    with gzip.open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith(b"WARC/"):
                continue
            headers: dict[bytes, bytes] = {}
            while True:
                h = f.readline()
                if h in (b"\r\n", b"\n", b""):
                    break
                k, _, v = h.partition(b":")
                headers[k.strip().lower()] = v.strip()
            body = f.read(int(headers.get(b"content-length", b"0")))
            if headers.get(b"warc-type") == b"conversion":
                yield headers.get(b"warc-target-uri", b"").decode("utf-8", "replace"), body.decode("utf-8", "replace")


def read_shard(path: Path, kind: str = "dclm"):
    """Yield (url, text) for every document in a pool file."""
    return read_wet(path) if kind == "cc-wet" else read_dclm(path)


# --- corpus -------------------------------------------------------------------------------


class Corpus:
    """Read-only, memory-mapped view of a round's raw corpus."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.idx = np.fromfile(self.path / "docs.idx", dtype="<u8")
        self._f = open(self.path / "docs.bin", "rb")
        size = os.fstat(self._f.fileno()).st_size
        self.bin = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ) if size else b""
        self._urls: list[str] | None = None

    def __len__(self) -> int:
        return len(self.idx) - 1

    def raw(self, i: int) -> bytes:
        return self.bin[int(self.idx[i]) : int(self.idx[i + 1])]

    def text(self, i: int) -> str:
        return self.raw(i).decode("utf-8", errors="replace")

    def size(self, i: int) -> int:
        return int(self.idx[i + 1] - self.idx[i])

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.idx).astype(np.int64)

    @property
    def urls(self) -> list[str]:
        if self._urls is None:
            self._urls = (self.path / "urls.txt").read_text(encoding="utf-8").split("\n")[: len(self)]
        return self._urls

    def __iter__(self):
        for i in range(len(self)):
            yield i, self.text(i)

    @property
    def total_bytes(self) -> int:
        return int(self.idx[-1])


# --- finding text again ---------------------------------------------------------------------------
#
# Normalised text (lowercase, runs of whitespace collapsed), its sentence openings as keys of
# `key_bytes` bytes, and a rolling hash that finds a key anywhere in a document. The exam
# uses them to drop fresh items that a round's corpus already holds (exam/decontam.py).

SENTENCE_END = re.compile(r"(?<=[.!?…؟۔।]) |(?<=[。！？｡])\s?")  # the second kind is written with no space after
_HASH_P = 0x9E3779B97F4A7C15  # odd, so it has an inverse modulo 2**64
_tables: list[np.ndarray] = []


def normalise(text: str) -> bytes:
    return " ".join(text.lower().split()).encode("utf-8", errors="replace")


def _power_tables(n: int) -> tuple[np.ndarray, np.ndarray]:
    if not _tables or len(_tables[0]) < n:
        m = 1 << max(19, int(n - 1).bit_length())
        fwd = np.full(m, _HASH_P, dtype=np.uint64)
        inv = np.full(m, pow(_HASH_P, -1, 1 << 64), dtype=np.uint64)
        fwd[0] = inv[0] = 1
        _tables[:] = [np.cumprod(fwd, dtype=np.uint64), np.cumprod(inv, dtype=np.uint64)]
    return _tables[0], _tables[1]


def window_hashes(b: bytes, k: int) -> np.ndarray:
    """A 64-bit polynomial hash of every k-byte window of b, in one pass. Exact integer
    arithmetic modulo 2**64, so every machine computes the same values."""
    n = len(b)
    if n < k:
        return np.empty(0, dtype=np.uint64)
    fwd, inv = _power_tables(n)
    a = np.frombuffer(b, dtype=np.uint8).astype(np.uint64)
    c = np.zeros(n + 1, dtype=np.uint64)
    np.cumsum(a * inv[:n], dtype=np.uint64, out=c[1:])
    return (c[k:] - c[:-k]) * fwd[k - 1 : n]


def sentence_starts(text: str, key_bytes: int, min_letters: int):
    """The keys of a passage: the `key_bytes` bytes from where each of its sentences opens, when
    they are distinctive enough.

    Openings that are mostly digits or punctuation (a run of page numbers, a ruled line) are
    skipped: they would match thousands of unrelated pages. What counts is the bytes that are
    letters (and the marks written on them), in any script: a key of 64 bytes is 64 Latin letters,
    about 32 Cyrillic or Arabic ones, about 21 Chinese or Japanese characters. A sentence ends at
    its full stop, question mark or exclamation mark, whichever script wrote it, with or without a
    space after. A key runs on into the sentences that follow, so a short sentence has one too.
    """
    flat = " ".join(text.lower().split())
    starts = [0] + [m.end() for m in SENTENCE_END.finditer(flat)]
    seen = set()
    for at in starts:
        raw = flat[at : at + key_bytes].encode("utf-8", errors="replace")  # at least key_bytes bytes, if there are that many
        if len(raw) < key_bytes or at in seen:
            continue
        seen.add(at)
        key, letters = raw[:key_bytes], 0
        for ch in key.decode("utf-8", errors="ignore"):  # a character the cut splits is not counted
            if unicodedata.category(ch)[0] in "LM":
                letters += len(ch.encode("utf-8"))
        if letters >= min_letters:
            yield key


# Pool files fetched and read at once while a corpus is built. Fetching waits on the network
# and unpacking is done outside the interpreter's lock, so a few threads overlap them; more
# would only press harder on a public host. No more than twice this many files' text is ever
# in memory.
BUILD_THREADS = 4


def _shard_docs(shard: Shard, kind: str, max_doc: int, quiet: bool) -> list[tuple[str, bytes]]:
    """One pool file's documents as they go into a corpus: the address and the bytes of each
    that has any text, cut at the season's cap."""
    path = shard_path(shard, quiet)
    out: list[tuple[str, bytes]] = []
    try:
        for url, text in read_shard(path, kind):
            if not text.strip():
                continue
            b = text.encode("utf-8", errors="replace")
            if len(b) > max_doc:
                b = b[:max_doc].decode("utf-8", errors="ignore").encode("utf-8")
            out.append((url.replace("\n", " ").replace("\r", " "), b))
    except (OSError, EOFError, zstandard.ZstdError, zlib.error, ValueError) as e:
        # A damaged copy, whatever the damage makes the reader say: a stream cut short, a
        # checksum that fails, a record length that is not a number.
        Path(path).unlink(missing_ok=True)  # fetch it again next time
        raise OSError(f"{shard.path} is damaged ({e}); it has been removed from the cache, run again") from None
    return out


def _each_shard(shards: list[Shard], kind: str, max_doc: int, quiet: bool):
    """Every pool file's documents, in the files' order, read a few files ahead of the caller."""
    workers = max(1, min(BUILD_THREADS, len(shards), os.cpu_count() or 1))
    if workers == 1:
        for shard in shards:
            yield _shard_docs(shard, kind, max_doc, quiet)
        return
    ahead: collections.deque = collections.deque()
    todo = iter(shards)
    with ThreadPoolExecutor(workers) as pool:
        try:
            while True:
                while len(ahead) < 2 * workers and (shard := next(todo, None)) is not None:
                    # Each file's own progress line would write over the others': the caller says one.
                    ahead.append(pool.submit(_shard_docs, shard, kind, max_doc, True))
                if not ahead:
                    break
                yield ahead.popleft().result()
        finally:
            for f in ahead:
                f.cancel()


def build_corpus(rnd: Round, scale: Scale, dest: Path, quiet: bool = False) -> Path:
    """Materialise a round's corpus. Deterministic: same round and scale, same bytes.

    Two things happen to the pool files on the way in, and nothing else:

      - documents longer than the season's cap are cut at it;
      - the documents are written in a seeded shuffle of their arrival order.

    Nothing depends on the round's exam, which is made after the draw.
    """
    tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
    # What an earlier build or download left when it was killed, here and among the pool files.
    sweep_stale(dest.parent, dest.name + ".tmp*")
    for d in (cache_dir() / "pool", cache_dir() / "pool" / "wet"):
        sweep_stale(d, "*.part*")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        season = rnd.season
        max_doc = season.max_doc_bytes
        offsets = [0]
        urls: list[str] = []
        kind = rnd.pool_cfg["kind"]
        shards = round_shards(rnd, scale, quiet)
        many = len(shards) > 1 and not quiet
        t0 = time.time()
        with open(tmp / "arrival.bin", "wb") as out:
            for k, docs in enumerate(_each_shard(shards, kind, max_doc, quiet), 1):
                for url, b in docs:
                    out.write(b)
                    offsets.append(offsets[-1] + len(b))
                    urls.append(url)
                if many:
                    sys.stderr.write(f"\r  fetched and read {k} of {len(shards)} files of the crawl: {offsets[-1] / 1e6:,.0f} MB of pages [{time.time() - t0:.0f}s]")
                    sys.stderr.flush()
        if many:
            sys.stderr.write("\n")
        n = len(urls)
        order = rnd.doc_order(n)
        h = hashlib.sha256()
        final = [0]
        with open(tmp / "arrival.bin", "rb") as f, open(tmp / "docs.bin", "wb") as out:
            src = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if offsets[-1] else b""
            for i in order:
                b = src[offsets[i] : offsets[i + 1]]
                out.write(b)
                h.update(b)
                final.append(final[-1] + len(b))
            if offsets[-1]:
                src.close()
        (tmp / "arrival.bin").unlink()
        np.asarray(final, dtype="<u8").tofile(tmp / "docs.idx")
        (tmp / "urls.txt").write_text("\n".join(urls[i] for i in order) + "\n", encoding="utf-8")
        meta = {
            "round": rnd.id,
            "scale": scale.name,
            "n_docs": n,
            "bytes": final[-1],
            "sha256": h.hexdigest(),
            "pool": kind,
            "shards": [s.path for s in shards],
            "order": "seeded shuffle",
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
        if (dest / "meta.json").exists():
            # Another verifier on this machine built the same round while this one did, and
            # a curator may already be reading its copy. Use that one: replacing it would
            # empty the directory under a program that has it open.
            shutil.rmtree(tmp, ignore_errors=True)
            return dest
        try:
            tmp.replace(dest)  # refused if something is there: a directory is never renamed over one with files in it
        except OSError:
            if (dest / "meta.json").exists():
                return dest  # the other verifier's, finished in the moment since the look above
            shutil.rmtree(dest, ignore_errors=True)  # not a corpus, so nothing is reading it
            tmp.replace(dest)
        if kind == "cc-wet" and os.environ.get("STET_KEEP_DOWNLOADS") != "1":
            # A round's files are a random handful out of a hundred thousand, and no other
            # round will draw them. The corpus is built; the downloads are dead weight, in a
            # practice round as much as an official one. (The same round at a larger scale
            # fetches its first few again: a minute, against half a gigabyte kept per round.)
            for shard in shards:
                shard_path_of(shard).unlink(missing_ok=True)
    finally:
        # On every way out, Ctrl-C included: a half-built corpus is a gigabyte nobody will
        # come back for. (Once it has been renamed into place there is nothing here.)
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def mark_verified(rnd: Round) -> None:
    """Note that a round is in the cache because `stet verify` graded it, not because you asked for it."""
    d = cache_dir() / "rounds" / rnd.id
    d.mkdir(parents=True, exist_ok=True)
    (d / ".verified").touch()


def prune_verified(rnd: Round, keep: int = 3) -> int:
    """Delete the cached rounds `stet verify` fetched, except the newest few. Returns bytes freed.

    A verifier left running would otherwise fill its disk at a few gigabytes a round.
    Rounds you fetched yourself (`stet run --round ...`) are left alone.
    """
    freed = 0
    for d in (cache_dir() / "rounds").glob(f"{rnd.season.id}-o*"):
        try:
            index = int(d.name.rsplit("-o", 1)[1])
        except ValueError:
            continue
        if index > rnd.index - keep or not (d / ".verified").exists():
            continue
        freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)
    return freed


# --- exam ---------------------------------------------------------------------------------

def round_exam(rnd: Round, scale: Scale):
    """The round's exam: its items (stet.exam), in order. A practice round's has the season's
    fresh writing too once some has been harvested on this machine (`stet harvest`)."""
    if hours := practice_hours(rnd):
        return _practice(rnd, scale, hours)
    return _items(rnd)


def harvest_dir() -> Path:
    """Where the hours harvested on this machine for practice are kept: <hour>.jsonl."""
    return cache_dir() / "harvest"


def harvested() -> dict[int, Path]:
    try:
        return {int(p.stem): p for p in harvest_dir().iterdir() if p.suffix == ".jsonl" and p.stem.isdigit()}
    except OSError:
        return {}


def practice_hours(rnd: Round) -> list[tuple[int, Path]]:
    """The hours a practice round's writing is made from: of those harvested on this machine,
    as many as an official round is given, chosen by the round's seed as the server chooses an
    official round's. With none harvested there are none, and the round sits the generated
    items alone: a quarter of what an official round's score is made of."""
    if rnd.kind != "practice" or not (held := harvested()):
        return []
    from .exam.fresh.harvest import choose_hours

    return [(h, held[h]) for h in choose_hours(rnd.seed, held, int(rnd.season.raw["exam"]["fresh"]["hours"]))]


def practice_key(rnd: Round) -> str:
    """What names the writing a practice round sits here: "" when it sits none. Harvesting more
    hours changes which a round is given, and then its exam and its baseline are new ones."""
    from .exam.examine import snapshot_name

    hours = practice_hours(rnd)
    return snapshot_name([hashlib.sha256(p.read_bytes()).hexdigest() for _, p in hours]) if hours else ""


def _practice(rnd: Round, scale: Scale, hours: list[tuple[int, Path]]):
    """A practice round's exam with writing: made as the examiner makes an official round's,
    from hours harvested here, and kept beside the round's corpus, under the season's digest
    and the hours' name: the season says which parts an exam has, so another season's file is
    another exam."""
    from .exam.examine import assemble, pack_items
    from .exam.farm import unpack_items
    from .exam.fresh.reason import can_synthesize_locally

    path = round_dir(rnd, scale) / f"exam-practice-{rnd.season.digest[:8]}-{practice_key(rnd)[:16]}.jsonl.zst"
    try:
        return unpack_items(path.read_bytes())
    except (OSError, ValueError, zstandard.ZstdError):
        pass
    synth = bool(os.environ.get("STET_SYNTHESIZE_PRACTICE")) and can_synthesize_locally()
    items = assemble(rnd, scale, hours, round_dir(rnd, scale) / "examine", log=lambda m: None, measure=False, synthesize=synth)[0]
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_bytes(pack_items(items))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return items


class ExamNotReady(RuntimeError):
    """An official round's exam is made after its draw, and this one has not been posted yet."""


def exam_items_path(rnd: Round, digest: str) -> Path:
    """Where an official round's posted items are kept, by their digest."""
    return cache_dir() / "rounds" / rnd.id / f"exam-{digest[:16]}.jsonl.zst"


def exam_public_path(rnd: Round, digest: str) -> Path:
    """Where a settled round's published exam is kept."""
    return cache_dir() / "rounds" / rnd.id / f"exam-public-{digest[:16]}.jsonl.zst"


def _items(rnd: Round):
    """A round's exam items. An official round's are the ones the organisers posted
    after its draw (fetched by the grader beforehand: see client.fetch_exam); a practice or gym
    round's are generated from its seed."""
    from .exam import gen
    from .exam.items import Item

    cfg = rnd.season.raw["exam"]
    if int(cfg.get("generator_version", 0)) > gen.VERSION:
        raise RuntimeError(f"season {rnd.season.id} sets the exam's generators at version {cfg['generator_version']}, "
                           f"this copy of stet has {gen.VERSION}: upgrade stet")
    if rnd.kind != "official":
        if int(cfg.get("per_corner", 0)) > 0:
            generated = gen.for_season(rnd.season, rnd.seed)
            if generated:
                return generated
        from .exam.fresh.reason import load_practice_pack

        return load_practice_pack(rnd.season, rnd.seed)
    want = getattr(rnd, "exam_digest", "") or ""
    path = exam_items_path(rnd, want) if want else None
    if path is not None and not path.exists() and exam_public_path(rnd, want).exists():
        # A settled round replayed anywhere: the published exam, which gives the generated items
        # and the openly licensed text whole, checked against the round's digest.
        from .exam.examine import read_public

        raw = zstandard.ZstdDecompressor().decompress(exam_public_path(rnd, want).read_bytes(), max_output_size=1 << 30)
        try:
            return read_public(raw, want)
        except ValueError as e:
            exam_public_path(rnd, want).unlink(missing_ok=True)
            raise ExamNotReady(f"{rnd.id}: {e}; it has been removed, fetch it again") from None
    if path is None or not path.exists():
        raise ExamNotReady(f"{rnd.id}: the round's exam has not been posted, or not fetched to this machine")
    raw = zstandard.ZstdDecompressor().decompress(path.read_bytes(), max_output_size=1 << 30)
    items = [Item.from_dict(json.loads(line)) for line in raw.decode().split("\n") if line]  # not splitlines(): see read_snapshot
    if exam_digest(items) != want:
        path.unlink(missing_ok=True)
        raise ExamNotReady(f"{rnd.id}: the copy of the exam here is not the round's; it has been removed, fetch it again")
    return items


def exam_digest(exam) -> str:
    from .exam.items import digest

    return digest(exam)


def round_dir(rnd: Round, scale: Scale) -> Path:
    return cache_dir() / "rounds" / rnd.id / scale.name


def set_aside(path: Path, now: float | None = None) -> None:
    """Take a round's cached files out of use without pulling them from under anyone.

    Two verifiers on one machine share the cache. One of them may be in the middle of a
    job, with a curator reading these very files, when the other decides they are not to
    be trusted. So the directory is renamed, which a program that has it open does not
    notice, and deleted some hours later.
    """
    now = time.time() if now is None else now
    stale = cache_dir() / "rounds" / ".stale"
    try:
        if path.exists():
            stale.mkdir(parents=True, exist_ok=True)
            path.rename(stale / f"{path.parent.name}-{path.name}-{os.getpid()}-{int(now)}")
    except OSError:
        shutil.rmtree(path, ignore_errors=True)  # cannot be renamed (another file system?): as before
    try:
        for old in stale.iterdir():
            if now - old.stat().st_mtime > SET_ASIDE_FOR:
                shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass


def corpus(rnd: Round, scale: Scale, quiet: bool = False) -> Corpus:
    """A round's corpus, built and cached on first use. Knows nothing about the round's exam."""
    dest = round_dir(rnd, scale) / "corpus-v3"
    # This round is in use from now: marked first, so that the sweep below leaves it alone.
    marker = dest.parent.parent / ".used"
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass
    sweep_unused()
    if not (dest / "meta.json").exists():
        build_corpus(rnd, scale, dest, quiet)
    for old in dest.parent.glob("corpus*"):
        # A corpus an earlier release built in its own layout: nothing reads it now.
        if old.name != dest.name and re.fullmatch(r"corpus(-v[0-9]+)?", old.name):
            set_aside(old)
    return Corpus(dest)


def materialize(rnd: Round, scale: Scale, quiet: bool = False):
    """Corpus and exam (its items) for a round."""
    return corpus(rnd, scale, quiet), round_exam(rnd, scale)

"""Pool shards, the exam pack, and the on-disk corpus an editor reads.

Corpus layout (language-agnostic, mmap-friendly):

    docs.bin    every document's UTF-8 bytes, concatenated
    docs.idx    uint64 little-endian offsets, n_docs + 1 of them; doc i is bin[idx[i]:idx[i+1]]
    urls.txt    one source URL per line, n_docs lines
    meta.json   round id, scale, doc and byte counts, sha256 of docs.bin
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import mmap
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import zstandard

from .spec import HARNESS_VERSION, Round, Scale, Season, Shard

USER_AGENT = f"stet/{HARNESS_VERSION}"
MIN_DECONTAM_LINE = 80
DEFAULT_API = "https://stet.maxint.co"
FETCH_TRIES = 4


def cache_dir() -> Path:
    return Path(os.environ.get("STET_CACHE", "~/.cache/stet")).expanduser()


def api_base() -> str:
    return os.environ.get("STET_API", DEFAULT_API).rstrip("/")


# --- downloads ----------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(urls: list[str], sha256: str, dest: Path, label: str = "", quiet: bool = False) -> Path:
    """Download to `dest` unless it is already there, verifying the content hash."""
    if dest.exists():
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
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if not quiet and time.time() - last > 0.5:
                        last = time.time()
                        pct = f"{100 * done / total:5.1f}%" if total else ""
                        sys.stderr.write(f"\r  fetching {label or dest.name} {done / 1e6:7.1f} MB {pct}")
                        sys.stderr.flush()
            if not quiet:
                sys.stderr.write(f"\r  fetched  {label or dest.name} {done / 1e6:7.1f} MB in {time.time() - t0:.0f}s\n")
            if sha256 and h.hexdigest() != sha256:
                raise OSError(f"sha256 mismatch for {url}: got {h.hexdigest()}, want {sha256}")
            part.replace(dest)
            return dest
        except Exception as e:  # try the next mirror
            last_err = e
            part.unlink(missing_ok=True)
    raise OSError(f"could not fetch {label or dest.name}: {last_err}")


def round_shards(rnd: Round, scale: Scale, quiet: bool = False) -> list[Shard]:
    """The pool files this round's corpus is built from, in draw order."""
    cfg = rnd.pool_cfg
    if cfg["kind"] == "dclm":
        pool = rnd.season.pool("practice")
    elif cfg["kind"] == "cc-wet":
        # A crawl is identified by the hash of its published file list, fixed in the round.
        listing = fetch(
            [cfg["base_url"] + cfg["paths"].format(crawl=cfg["crawl"])],
            cfg["paths_sha256"],
            cache_dir() / "pool" / "wet" / f"{cfg['crawl']}.paths.gz",
            f"{cfg['crawl']} file list",
            quiet,
        )
        paths = gzip.decompress(listing.read_bytes()).decode().split()
        if len(paths) != int(cfg["n_shards"]):
            raise OSError(f"{cfg['crawl']}: expected {cfg['n_shards']} files, the list has {len(paths)}")
        pool = [Shard(sha256="", size=0, path=p, url=cfg["base_url"] + p) for p in paths]
    else:
        raise ValueError(f"unknown pool kind {cfg['kind']!r}; upgrade stet")
    return [pool[i] for i in rnd.shard_indices(scale, len(pool))]


def shard_path(shard: Shard, quiet: bool = False) -> Path:
    if shard.sha256:
        return fetch([shard.url], shard.sha256, cache_dir() / "pool" / f"{shard.sha256}.jsonl.zst", shard.path, quiet)
    # Crawl files are not listed with hashes. They are gzip, which carries its own checksum,
    # and the built corpus's sha256 is compared across verifiers.
    name = hashlib.sha256(shard.url.encode()).hexdigest()[:32]
    return fetch([shard.url], "", cache_dir() / "pool" / "wet" / f"{name}.warc.wet.gz", shard.path.rsplit("/", 1)[-1], quiet)


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


def _exam_lines(exam: dict[str, list[dict]]) -> set[int]:
    lines: set[int] = set()
    for items in exam.values():
        for it in items:
            for line in it["text"].split("\n"):
                line = line.strip()
                if len(line) >= MIN_DECONTAM_LINE:
                    lines.add(hash(line))
    return lines


def build_corpus(rnd: Round, scale: Scale, dest: Path, exam: dict[str, list[dict]], quiet: bool = False) -> Path:
    """Materialise a round's corpus. Deterministic: same round and scale, same bytes."""
    tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    max_doc = rnd.season.max_doc_bytes
    banned = _exam_lines(exam)
    offsets = [0]
    urls: list[str] = []
    h = hashlib.sha256()
    dropped = 0
    kind = rnd.pool_cfg["kind"]
    shards = round_shards(rnd, scale, quiet)
    with open(tmp / "docs.bin", "wb") as out:
        for shard in shards:
            path = shard_path(shard, quiet)
            try:
                docs = list(read_shard(path, kind))
            except (OSError, EOFError, zstandard.ZstdError) as e:
                path.unlink(missing_ok=True)  # a damaged download: fetch it again next time
                raise OSError(f"{shard.path} is damaged ({e}); it has been removed from the cache, run again") from None
            for url, text in docs:
                if not text.strip():
                    continue
                if banned and any(
                    len(s) >= MIN_DECONTAM_LINE and hash(s) in banned for s in (ln.strip() for ln in text.split("\n"))
                ):
                    dropped += 1
                    continue
                b = text.encode("utf-8", errors="replace")
                if len(b) > max_doc:
                    b = b[:max_doc].decode("utf-8", errors="ignore").encode("utf-8")
                out.write(b)
                h.update(b)
                offsets.append(offsets[-1] + len(b))
                urls.append(url.replace("\n", " ").replace("\r", " "))
    np.asarray(offsets, dtype="<u8").tofile(tmp / "docs.idx")
    (tmp / "urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
    meta = {
        "round": rnd.id,
        "scale": scale.name,
        "n_docs": len(urls),
        "bytes": offsets[-1],
        "sha256": h.hexdigest(),
        "pool": kind,
        "shards": [s.path for s in shards],
        "decontaminated_docs": dropped,
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)
    return dest


# --- exam ---------------------------------------------------------------------------------

_exam_cache: dict[tuple[str, str], dict[str, list[dict]]] = {}


def exam_pack_path(season: Season, quiet: bool = False) -> Path:
    cfg = season.raw["exam"]
    override = os.environ.get("STET_EXAM_PACK")
    if override:
        return Path(override)
    return fetch(list(cfg["urls"]), cfg["sha256"], cache_dir() / "exam" / cfg["file"], "exam pack", quiet)


def load_exam_pool(season: Season, split: str) -> dict[str, list[dict]]:
    """All exam passages for a split, by source, in pack order."""
    key = (season.id, split)
    if key not in _exam_cache:
        path = exam_pack_path(season)
        pool: dict[str, list[dict]] = {s: [] for s in season.exam_sources}
        with open(path, "rb") as f:
            reader = zstandard.ZstdDecompressor().stream_reader(f)
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                it = json.loads(line)
                if it["split"] == split and it["source"] in pool:
                    pool[it["source"]].append(it)
        _exam_cache[key] = pool
    return _exam_cache[key]


def round_exam(rnd: Round, scale: Scale) -> dict[str, list[dict]]:
    pool = load_exam_pool(rnd.season, rnd.split)
    return {src: [items[i] for i in rnd.exam_indices(scale, src, len(items))] for src, items in pool.items()}


def exam_digest(exam: dict[str, list[dict]]) -> str:
    h = hashlib.sha256()
    for src in sorted(exam):
        for it in exam[src]:
            h.update(it["id"].encode())
    return h.hexdigest()


def round_dir(rnd: Round, scale: Scale) -> Path:
    return cache_dir() / "rounds" / rnd.id / scale.name


def materialize(rnd: Round, scale: Scale, quiet: bool = False) -> tuple[Corpus, dict[str, list[dict]]]:
    """Corpus and exam for a round, building and caching the corpus on first use."""
    exam = round_exam(rnd, scale)
    dest = round_dir(rnd, scale) / "corpus"
    if not (dest / "meta.json").exists():
        build_corpus(rnd, scale, dest, exam, quiet)
    return Corpus(dest), exam

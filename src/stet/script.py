"""The edit script: what an editor hands back, and how it becomes a training stream.

An edit script is JSON Lines. Each line is one output document, and the order of lines is
the order the student is trained in. A line is one of:

    123                               keep document 123 whole
    {"d": 123}                        the same
    {"d": 123, "k": [[0, 900], ...]}  keep these byte ranges of document 123, in this order
    {"s": [[123, 0, 900], "text"]}    any mix of copies [doc, start, end] and literal text
    {"t": "text"}                     a new document made only of literal text

Cuts are free. Ink is scarce: literal text, and copies too short to be honest quotations,
are charged against the ink budget. Byte offsets are into the document's UTF-8 bytes and
are snapped forward to a character boundary.

Why short copies are ink: with enough tiny copies you can spell out any text you like from
the corpus's letters and words, which would make the ink limit meaningless. (A short
document kept whole, on its own, is not ink: nothing can be spelled across documents.)

The script is consumed until the training budget is used. Anything after that is ignored.
A script that is shorter than the budget is repeated from the top, so listing less means
training on it more often.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import Corpus
from .spec import Scale, Season

MAX_OUT_DOC_BYTES = 1 << 20
TOKENIZE_BATCH = 1024


class ScriptError(ValueError):
    """The edit script is malformed or breaks a rule. The run is invalid."""


@dataclass
class Stats:
    out_docs: int = 0  # output documents consumed in the first pass
    out_bytes: int = 0
    ink_bytes: int = 0
    copy_bytes: int = 0
    src_docs: int = 0  # distinct source documents drawn from
    script_lines: int = 0  # lines in the script (may exceed out_docs if truncated)
    tokens_one_pass: int = 0
    passes: float = 0.0  # how many times the stream was repeated to fill the budget
    truncated: bool = False  # the script was longer than the budget
    script_sha256: str = ""
    preview: list[str] = field(default_factory=list)

    @property
    def ink_frac(self) -> float:
        return self.ink_bytes / self.out_bytes if self.out_bytes else 0.0

    def to_dict(self, corpus: Corpus | None = None) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["ink_frac"] = round(self.ink_frac, 6)
        if corpus is not None:
            d["corpus_docs"] = len(corpus)
            d["corpus_bytes"] = corpus.total_bytes
            d["kept_doc_frac"] = round(self.src_docs / max(1, len(corpus)), 6)
            d["kept_byte_frac"] = round(self.copy_bytes / max(1, corpus.total_bytes), 6)
        return d


def _snap(buf, pos: int, end: int) -> int:
    """Move pos forward off any UTF-8 continuation byte."""
    while pos < end and (buf[pos] & 0xC0) == 0x80:
        pos += 1
    return pos


def _segments(line_no: int, obj) -> list:
    """Normalise one script line to a list of [doc, start, end|None] and str."""
    if isinstance(obj, bool):
        raise ScriptError(f"line {line_no}: expected a document, got a boolean")
    if isinstance(obj, int):
        return [[obj, 0, None]]
    if not isinstance(obj, dict):
        raise ScriptError(f"line {line_no}: expected an integer or an object")
    if "s" in obj:
        segs = obj["s"]
        if not isinstance(segs, list):
            raise ScriptError(f"line {line_no}: \"s\" must be a list")
        return segs
    if "t" in obj:
        return [obj["t"]]
    if "d" in obj:
        d = obj["d"]
        if "k" not in obj:
            return [[d, 0, None]]
        ranges = obj["k"]
        if not isinstance(ranges, list):
            raise ScriptError(f"line {line_no}: \"k\" must be a list of [start, end]")
        out = []
        for r in ranges:
            if not (isinstance(r, list) and len(r) == 2):
                raise ScriptError(f"line {line_no}: each range in \"k\" must be [start, end]")
            out.append([d, r[0], r[1]])
        return out
    raise ScriptError(f"line {line_no}: object needs one of \"d\", \"s\" or \"t\"")


def build_doc(corpus: Corpus, line_no: int, obj, min_copy: int) -> tuple[bytes, int, int, list[int]]:
    """Return (bytes, ink_bytes, copy_bytes, source doc ids) for one script line."""
    n = len(corpus)
    parts: list[bytes] = []
    ink = 0
    copied = 0
    docs: list[int] = []
    # Resolve and merge copies that are contiguous in the source, so that keeping lines
    # 1, 2 and 3 as three ranges costs the same as keeping them as one.
    resolved: list = []
    for seg in _segments(line_no, obj):
        if isinstance(seg, str):
            if seg:
                resolved.append(seg)
            continue
        if not (isinstance(seg, list) and len(seg) == 3):
            raise ScriptError(f"line {line_no}: a copy must be [doc, start, end]")
        d, s, e = seg
        if isinstance(d, bool) or not isinstance(d, int) or not (0 <= d < n):
            raise ScriptError(f"line {line_no}: document {d!r} is out of range (corpus has {n})")
        lo, hi = int(corpus.idx[d]), int(corpus.idx[d + 1])
        size = hi - lo
        e = size if e is None else e
        if isinstance(s, bool) or isinstance(e, bool) or not isinstance(s, int) or not isinstance(e, int):
            raise ScriptError(f"line {line_no}: range bounds must be integers")
        if s < 0 or e > size or s > e:
            raise ScriptError(f"line {line_no}: range [{s}, {e}) is outside document {d} (0..{size})")
        s = _snap(corpus.bin, lo + s, hi) - lo
        e = _snap(corpus.bin, lo + e, hi) - lo
        if e <= s:
            continue
        if resolved and isinstance(resolved[-1], list) and resolved[-1][0] == d and resolved[-1][2] == s:
            resolved[-1][2] = e
        else:
            resolved.append([d, s, e, size])
    for seg in resolved:
        if isinstance(seg, str):
            b = seg.encode("utf-8", errors="replace")
            parts.append(b)
            ink += len(b)
            continue
        d, s, e, size = seg
        lo = int(corpus.idx[d])
        parts.append(corpus.bin[lo + s : lo + e])
        length = e - s
        docs.append(d)
        if length >= min_copy or (length == size and len(resolved) == 1):
            copied += length  # long enough to be a quotation, or a short document kept as it is
        else:
            ink += length
    out = b"".join(parts)
    if len(out) > MAX_OUT_DOC_BYTES:
        raise ScriptError(f"line {line_no}: output document is {len(out)} bytes; the limit is {MAX_OUT_DOC_BYTES}")
    return out, ink, copied, docs


def exam_lines(exam: dict[str, list[dict]] | None, min_len: int = 80) -> set[bytes]:
    """Long lines of the round's exam passages. New text may not reproduce them."""
    out: set[bytes] = set()
    for items in (exam or {}).values():
        for it in items:
            for line in it["text"].encode("utf-8").split(b"\n"):
                line = line.strip()
                if len(line) >= min_len:
                    out.add(line)
    return out


def materialize(
    corpus: Corpus, script_path: str | Path, season: Season, scale: Scale, enc=None, exam: dict | None = None
) -> tuple[np.ndarray, Stats]:
    """Turn an edit script into the training tokens: shape (batch_size, steps * ctx + 1).

    Each row is a *lane*. Output documents are dealt, in script order, to whichever lane is
    shortest at that moment. So at any training step the batch holds batch_size different
    documents, and the order of the script is still the order things are learned in: a
    document a third of the way down the script is trained on a third of the way through.

    The script is consumed until every lane is full. A longer script is cut off there; a
    shorter one is dealt again from the top.

    If `exam` is given, an output document containing new text may not reproduce a long line
    of an exam passage. (Corpus documents that do were removed when the corpus was built.)
    """
    if enc is None:
        from .tok import load_encoding

        enc = load_encoding(season)
    eot = season.eot
    lanes = scale.batch_size
    lane_len = scale.steps * scale.ctx + 1
    stats = Stats()
    min_copy = season.min_copy_bytes
    banned = exam_lines(exam)
    seen_docs: set[int] = set()
    batch: list[tuple[bytes, int, int, list[int]]] = []
    digest = hashlib.sha256()
    first_pass: list[np.ndarray] = []
    lane_chunks: list[list[np.ndarray]] = [[] for _ in range(lanes)]
    heap = [(0, b) for b in range(lanes)]  # (tokens so far, lane); ties go to the lowest lane
    total = 0
    full = False

    def deal(arr: np.ndarray) -> bool:
        n, b = heapq.heappop(heap)
        lane_chunks[b].append(arr)
        heapq.heappush(heap, (n + len(arr), b))
        return heap[0][0] >= lane_len

    def flush():
        nonlocal total, full
        if not batch:
            return
        texts = [b.decode("utf-8", errors="replace") for b, _, _, _ in batch]
        for (b, ink, copied, docs), toks in zip(batch, enc.encode_ordinary_batch(texts)):
            if full:
                stats.truncated = True  # the budget ran out part-way through this batch
                break
            arr = np.empty(len(toks) + 1, dtype=np.uint16)
            arr[0] = eot
            arr[1:] = toks
            first_pass.append(arr)
            total += len(arr)
            stats.out_docs += 1
            stats.out_bytes += len(b)
            stats.ink_bytes += ink
            stats.copy_bytes += copied
            seen_docs.update(docs)
            if len(stats.preview) < 5:
                stats.preview.append(b[:600].decode("utf-8", errors="replace"))
            full = deal(arr)
        batch.clear()

    with open(script_path, "rb") as f:
        for line_no, raw in enumerate(f, 1):
            digest.update(raw)
            stats.script_lines += 1
            raw = raw.strip()
            if not raw:
                continue
            if full:
                stats.truncated = True  # still hashed and counted, never trained on
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ScriptError(f"line {line_no}: not valid JSON ({e.msg})") from None
            built = build_doc(corpus, line_no, obj, min_copy)
            if not built[0]:
                continue
            if banned and built[1] and any(ln.strip() in banned for ln in built[0].split(b"\n") if len(ln) >= 80):
                raise ScriptError(f"line {line_no}: new text reproduces an exam passage")
            batch.append(built)
            if len(batch) >= TOKENIZE_BATCH:
                flush()
        flush()

    stats.script_sha256 = digest.hexdigest()
    stats.src_docs = len(seen_docs)
    stats.tokens_one_pass = total
    if total < 2 * scale.ctx:
        raise ScriptError(f"the script yields only {total} tokens; it needs at least {2 * scale.ctx}")
    if stats.ink_frac > season.ink_frac + 1e-9:
        raise ScriptError(
            f"ink over budget: {stats.ink_bytes} of {stats.out_bytes} bytes are new text "
            f"({100 * stats.ink_frac:.2f}%); the limit is {100 * season.ink_frac:.0f}%"
        )
    i = 0
    while not full:  # a short script is dealt again from the top
        full = deal(first_pass[i % len(first_pass)])
        i += 1
    stats.passes = round(1.0 + i / len(first_pass), 4) if i else 1.0
    tokens = np.empty((lanes, lane_len), dtype=np.uint16)
    for b in range(lanes):
        tokens[b] = np.concatenate(lane_chunks[b])[:lane_len]
    return tokens, stats


def write_script(path: str | Path, lines) -> None:
    """Write an edit script atomically. `lines` yields ints, dicts, or JSON strings."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    tmp.replace(path)

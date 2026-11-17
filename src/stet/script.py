"""The edit script: what a curator hands back, and how it becomes a training stream.

An edit script is JSON Lines. Each line is one output document, and the order of lines is
the order the student is trained in. A line is one of:

    123                               keep document 123 whole
    {"d": 123}                        the same
    {"d": 123, "k": [[0, 900], ...]}  keep these byte ranges of document 123, in this order
    {"s": [[123, 0, 900], "text"]}    any mix of copies [doc, start, end] and literal text
    {"t": "text"}                     a new document made only of literal text

A line may nest lists and objects at most 16 deep (the forms above need three). A line
nested deeper is invalid.

Cuts, reordering and repetition are free. Ink is scarce: literal text, and copies too short
to be honest quotations, are charged against a small ink budget counted in tokens. Byte
offsets are into the document's UTF-8 bytes and are snapped forward to a character boundary.

Why short copies are ink: with enough tiny copies you can spell out any text you like from
the corpus's letters and words, which would make the ink limit meaningless. (A short
document kept whole, on its own, is not ink: nothing can be spelled across documents.)

The script is consumed until the training budget is used. Anything after that is ignored.
A script that is shorter than the budget is repeated from the top, so listing less means
training on it more often.

A script is plain data written by untrusted code. Everything here treats it that way: it
is opened without following links, capped in size, and any malformed line is a ScriptError
rather than a crash.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import Corpus, contains_exam_text
from .spec import Scale, Season
from .tok import Unreadable, encode_batch

# Output documents are checked in windows of TOKENIZE_BATCH, and tokenised whenever
# TOKENIZE_BYTES of text is waiting. The window decides what counts as part of the script
# (see materialize); the byte limit only bounds memory, and changes no result.
TOKENIZE_BATCH = 1024
TOKENIZE_BYTES = 4 << 20

# How deep a line may nest lists and objects. The forms a line can take need three levels;
# the limit is a rule so that whether a line is valid does not depend on the interpreter
# that reads it. Without one it would: how deep a JSON parser can recurse is different in
# every version of Python, so a line nested a few thousand deep would be valid JSON to one
# grader and "not valid JSON" to the next.
MAX_DEPTH = 16


class ScriptError(ValueError):
    """The edit script is malformed or breaks a rule. The run is invalid."""


@dataclass
class Stats:
    out_docs: int = 0  # output documents consumed in the first pass
    out_bytes: int = 0
    copy_bytes: int = 0
    ink_tokens: int = 0  # the curator's own text among the tokens trained on, repeats included
    ink_budget: int = 0
    exam_dropped: int = 0  # output documents left out because they contained exam text
    src_docs: int = 0  # distinct source documents drawn from
    script_lines: int = 0  # lines in the script (may exceed out_docs if truncated)
    tokens_one_pass: int = 0
    train_tokens: int = 0
    passes: float = 0.0  # how many times the stream was repeated to fill the budget
    truncated: bool = False  # the script was longer than the budget
    script_sha256: str = ""
    tokens_sha256: str = ""  # of the exact token array the student trains on
    preview: list[str] = field(default_factory=list)

    @property
    def ink_frac(self) -> float:
        return self.ink_tokens / self.train_tokens if self.train_tokens else 0.0

    def to_dict(self, corpus: Corpus | None = None) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["ink_frac"] = round(self.ink_frac, 6)
        if corpus is not None:
            d["corpus_docs"] = len(corpus)
            d["corpus_bytes"] = corpus.total_bytes
            d["kept_doc_frac"] = round(self.src_docs / max(1, len(corpus)), 6)
            d["kept_byte_frac"] = round(self.copy_bytes / max(1, corpus.total_bytes), 6)
        return d


def open_script(path: str | Path, max_bytes: int):
    """Open a script for reading without trusting what is at the path.

    A curator could leave a symlink, a FIFO or a device where its script should be. Opening
    with O_NOFOLLOW and O_NONBLOCK and then insisting on a regular file of bounded size
    means the worst a hostile script can be is invalid.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise ScriptError(f"cannot open the script ({e.strerror})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ScriptError("the script is not a regular file")
        if st.st_size > max_bytes:
            raise ScriptError(f"the script is {st.st_size:,} bytes; the limit is {max_bytes:,}")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def file_sha256(path: str | Path, max_bytes: int) -> str:
    h = hashlib.sha256()
    with open_script(path, max_bytes) as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def too_deep(obj, limit: int = MAX_DEPTH) -> bool:
    """Whether a parsed line nests lists and objects more than `limit` deep.

    Every container counts, under any key: 123 is 0 deep, {"d": 1} is 1, and
    {"d": 1, "k": [[0, 900]]} is 3. It goes down a level at a time and never recurses, so
    it gives the same answer for a line 17 deep and a line 100,000 deep, on any Python.
    """
    if obj.__class__ is not dict and obj.__class__ is not list:
        return False
    level = [obj]
    for _ in range(limit):
        level = [v for c in level for v in (c.values() if c.__class__ is dict else c) if v.__class__ is list or v.__class__ is dict]
        if not level:
            return False
    return True


def _snap(buf, pos: int, end: int) -> int:
    """Move pos forward off any UTF-8 continuation byte."""
    while pos < end and (buf[pos] & 0xC0) == 0x80:
        pos += 1
    return pos


def _segments(line_no: int, obj, max_segments: int) -> list:
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
    elif "t" in obj:
        if not isinstance(obj["t"], str):
            raise ScriptError(f"line {line_no}: \"t\" must be a string")
        segs = [obj["t"]]
    elif "d" in obj:
        d = obj["d"]
        if "k" not in obj:
            return [[d, 0, None]]
        ranges = obj["k"]
        if not isinstance(ranges, list):
            raise ScriptError(f"line {line_no}: \"k\" must be a list of [start, end]")
        segs = []
        for r in ranges:
            if not (isinstance(r, list) and len(r) == 2):
                raise ScriptError(f"line {line_no}: each range in \"k\" must be [start, end]")
            segs.append([d, r[0], r[1]])
    else:
        raise ScriptError(f"line {line_no}: object needs one of \"d\", \"s\" or \"t\"")
    if len(segs) > max_segments:
        raise ScriptError(f"line {line_no}: {len(segs)} segments in one document; the limit is {max_segments}")
    return segs


@dataclass
class Built:
    data: bytes
    ink: list[bytes]  # the pieces charged as ink
    copied: int
    docs: list[int]
    spliced: bool  # more than one piece: joins can form text that is in no source document


def build_doc(corpus: Corpus, line_no: int, obj, min_copy: int, max_segments: int = 20000, max_out: int = 1 << 20) -> Built:
    """Resolve one script line against the corpus."""
    n = len(corpus)
    parts: list[bytes] = []
    ink: list[bytes] = []
    copied = 0
    docs: list[int] = []
    # Resolve and merge copies that are contiguous in the source, so that keeping lines
    # 1, 2 and 3 as three ranges costs the same as keeping them as one.
    resolved: list = []
    for seg in _segments(line_no, obj, max_segments):
        if isinstance(seg, str):
            if seg:
                resolved.append(seg)
            continue
        if not (isinstance(seg, list) and len(seg) == 3):
            raise ScriptError(f"line {line_no}: a copy must be [doc, start, end]")
        d, s, e = seg
        if isinstance(d, bool) or not isinstance(d, int) or not (0 <= d < n):
            raise ScriptError(f"line {line_no}: document {str(d)[:40]} is out of range (corpus has {n})")
        lo, hi = int(corpus.idx[d]), int(corpus.idx[d + 1])
        size = hi - lo
        e = size if e is None else e
        if isinstance(s, bool) or isinstance(e, bool) or not isinstance(s, int) or not isinstance(e, int):
            raise ScriptError(f"line {line_no}: range bounds must be integers")
        if s < 0 or e > size or s > e:
            raise ScriptError(f"line {line_no}: range [{str(s)[:20]}, {str(e)[:20]}) is outside document {d} (0..{size})")
        s = _snap(corpus.bin, lo + s, hi) - lo
        e = _snap(corpus.bin, lo + e, hi) - lo
        if e <= s:
            continue
        if resolved and isinstance(resolved[-1], list) and resolved[-1][0] == d and resolved[-1][2] == s:
            resolved[-1][2] = e
        else:
            resolved.append([d, s, e, size])
    total = 0
    for seg in resolved:
        if isinstance(seg, str):
            b = seg.encode("utf-8", errors="replace")
            parts.append(b)
            ink.append(b)
        else:
            d, s, e, size = seg
            lo = int(corpus.idx[d])
            b = corpus.bin[lo + s : lo + e]
            parts.append(b)
            docs.append(d)
            if len(b) >= min_copy or (len(b) == size and len(resolved) == 1):
                copied += len(b)  # long enough to be a quotation, or a short document kept as it is
            else:
                ink.append(b)
        total += len(b)
        if total > max_out:
            raise ScriptError(f"line {line_no}: the output document is over {max_out:,} bytes")
    return Built(b"".join(parts), ink, copied, docs, len(resolved) > 1)


def materialize(
    corpus: Corpus, script_path: str | Path, season: Season, scale: Scale, enc=None, check_exam: bool = True
) -> tuple[np.ndarray, Stats]:
    """Turn an edit script into the training tokens: shape (batch_size, steps * ctx + 1).

    Each row is a *lane*. Output documents are dealt, in script order, to whichever lane is
    shortest at that moment. So at any training step the batch holds batch_size different
    documents, and the order of the script is still the order things are learned in: a
    document a third of the way down the script is trained on a third of the way through.

    The script is consumed until every lane is full. A longer script is cut off there; a
    shorter one is dealt again from the top.

    With `check_exam`, an output document that contains ink or joins pieces together is left
    out if it contains the opening of a sentence of any exam passage. (Corpus documents that
    do were removed when the corpus was built, so whole and trimmed copies never can.)
    """
    if enc is None:
        from .tok import load_encoding

        enc = load_encoding(season)
    limits = season.raw["script"]
    max_line = int(limits["max_line_bytes"])
    max_segments = int(limits["max_segments"])
    max_out = int(limits["max_out_doc_bytes"])
    eot = season.eot
    lanes = scale.batch_size
    lane_len = scale.steps * scale.ctx + 1
    stats = Stats(ink_budget=season.ink_tokens(scale), train_tokens=lanes * lane_len)
    min_copy = season.min_copy_bytes
    seen_docs: set[int] = set()
    batch: list[tuple[int, Built]] = []  # built, waiting to be tokenised
    waiting = 0  # bytes of text in `batch`
    in_window = 0  # output documents so far in the current window of TOKENIZE_BATCH
    closed = False  # the budget is full and the window it filled in has ended
    digest = hashlib.sha256()
    first_pass: list[np.ndarray] = []
    first_ink: list[int] = []
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
        nonlocal total, full, waiting
        waiting = 0
        if not batch:
            return
        texts = [bd.data.decode("utf-8", errors="replace") for _, bd in batch]
        ink_texts = [piece.decode("utf-8", errors="replace") for _, bd in batch for piece in bd.ink]
        try:
            ink_toks = encode_batch(enc, ink_texts) if ink_texts else []
            text_toks = encode_batch(enc, texts)
        except Unreadable:
            raise ScriptError("the tokenizer cannot read one of this script's documents, even with its whitespace cut short") from None
        ink_lens = iter([len(t) for t in ink_toks])
        for (_, bd), text, toks in zip(batch, texts, text_toks, strict=True):
            ink = sum(next(ink_lens) for _ in bd.ink)
            if full:
                stats.truncated = True  # the budget ran out part-way through this batch
                break
            if check_exam and (bd.ink or bd.spliced) and contains_exam_text(text, season):
                stats.exam_dropped += 1  # never trained on; the rest of the script stands
                continue
            arr = np.empty(len(toks) + 1, dtype=np.uint16)
            arr[0] = eot
            arr[1:] = toks
            first_pass.append(arr)
            first_ink.append(ink)
            total += len(arr)
            stats.out_docs += 1
            stats.out_bytes += len(bd.data)
            stats.ink_tokens += ink
            stats.copy_bytes += bd.copied
            seen_docs.update(bd.docs)
            if len(stats.preview) < 5:
                stats.preview.append(text[:600])
            full = deal(arr)
        batch.clear()

    with open_script(script_path, int(limits["max_bytes"])) as f:
        line_no = 0
        while True:
            raw = f.readline(max_line + 1)
            if not raw:
                break
            line_no += 1
            if len(raw) > max_line:
                raise ScriptError(f"line {line_no}: longer than {max_line:,} bytes")
            digest.update(raw)
            stats.script_lines += 1
            raw = raw.strip()
            if not raw:
                continue
            if closed:
                stats.truncated = True  # still hashed and counted, never trained on
                continue
            try:
                obj = json.loads(raw)
            except RecursionError:
                # Deeper than this interpreter's parser can follow. Another one might have
                # parsed it, and would then find it too deep: the same verdict, in the same words.
                raise ScriptError(f"line {line_no}: nested more than {MAX_DEPTH} deep") from None
            except ValueError:
                raise ScriptError(f"line {line_no}: not valid JSON") from None
            # A line cannot nest deeper than it has opening brackets, which is quick to
            # count and settles it for nearly every line; the rest are walked.
            if raw.count(b"[") + raw.count(b"{") > MAX_DEPTH and too_deep(obj):
                raise ScriptError(f"line {line_no}: nested more than {MAX_DEPTH} deep")
            # Every line up to the end of the window the budget fills in is checked, and a
            # bad one makes the script invalid. That is the rule every grader applies, so it
            # must not depend on how much text happened to be tokenised at once.
            built = build_doc(corpus, line_no, obj, min_copy, max_segments, max_out)
            if not built.data:
                continue
            in_window += 1
            if full:
                stats.truncated = True  # checked, but past the budget: not kept in memory
            else:
                batch.append((line_no, built))
                waiting += len(built.data)
                if waiting >= TOKENIZE_BYTES:
                    flush()
            if in_window >= TOKENIZE_BATCH:
                flush()
                in_window = 0
                closed = full
        flush()

    stats.script_sha256 = digest.hexdigest()
    stats.src_docs = len(seen_docs)
    stats.tokens_one_pass = total
    if total < 2 * scale.ctx:
        raise ScriptError(f"the script yields only {total} tokens; it needs at least {2 * scale.ctx}")
    i = 0
    while not full:  # a short script is dealt again from the top
        j = i % len(first_pass)
        full = deal(first_pass[j])
        stats.ink_tokens += first_ink[j]
        i += 1
    stats.passes = round(1.0 + i / len(first_pass), 4) if i else 1.0
    if stats.ink_tokens > stats.ink_budget:
        raise ScriptError(
            f"ink over budget: {stats.ink_tokens:,} of the tokens trained on are new text or copies shorter than "
            f"{min_copy} bytes; the limit is {stats.ink_budget:,} ({100 * season.ink_frac:g}% of the training budget)"
        )
    tokens = np.empty((lanes, lane_len), dtype=np.uint16)
    for b in range(lanes):
        tokens[b] = np.concatenate(lane_chunks[b])[:lane_len]
    stats.tokens_sha256 = hashlib.sha256(tokens.astype("<u2", copy=False).tobytes()).hexdigest()
    return tokens, stats


def write_script(path: str | Path, lines) -> None:
    """Write an edit script atomically. `lines` yields ints, dicts, or JSON strings."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")  # two processes may be writing the same name
    with open(tmp, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    tmp.replace(path)

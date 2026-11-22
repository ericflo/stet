"""An afternoon of heuristics: the baseline a serious entry should beat.

  1. Keep pages that read as writing, in whatever language they are written.
  2. Drop exact duplicates.
  3. Count how much of each page is writing: the lines long enough to be a sentence, with
     the end of one in them.
  4. Rank the pages by that.
  5. Keep the best pages, whole, until their writing fills the training budget once, and
     shuffle them.

No model, no GPU, a few seconds per shard, on every core it is given (task.map).

It copies its pages whole. It used to keep only the lines it counted as writing, and the two
were trained side by side on the same pages in four orders: whole pages did better each time,
by 0.03 bits per byte on the exam's writing, nearly all of it on technical writing and code,
whose lines are not sentences (docs/EXPERIMENTS.md). The lines still say which pages to keep.

The exam is written in every language of the crawl, each counting by its share of it. So
nothing here asks what language a page is in: a rule that only English can pass throws away
more than half of what the student is examined on.
"""

import hashlib
import random
import re
import unicodedata

from stet.curator import Task

# How a sentence ends: in scripts written with spaces, a mark and then a space; in Chinese and
# Japanese, the mark alone.
SENTENCE_END = re.compile(r"[.!?…؟۔।][\"')\]”’»]?\s|[。！？｡]")
SPACELESS = re.compile("[぀-ヿ一-鿿฀-๿]")  # kana, Chinese characters, Thai
# The vowel signs of Hindi, Thai and Arabic are marks and not letters: they count as part of
# the letters they sit on. (Asking each character in turn is a hundred times slower, and a
# round's pages are gigabytes.)
MARKS = "".join(c for c in map(chr, range(0x300, 0x10000)) if unicodedata.category(c)[0] == "M")
WRITING = re.compile(rf"(?:[^\W\d_]|\s|[{re.escape(MARKS)}])+")  # letters of any script, spaces and marks
# Rough, under the season's tokenizer, and on the generous side: a stream that runs over the
# budget is cut off, one that falls short repeats from the top.
BYTES_PER_TOKEN = 3.7


def looks_written(text: str) -> bool:
    """Mostly letters, of any script, and enough of them to say something."""
    sample = text[:3000]
    if len(sample) < 200:
        return False
    return len(WRITING.sub("", sample)) <= 0.25 * len(sample)


def spaceless(text: str) -> bool:
    """Written without spaces between words, so that a line's length is counted in characters."""
    sample = text[:2000]
    return len(SPACELESS.findall(sample)) > 0.3 * len(sample)


def prose_ranges(raw: bytes, dense: bool) -> tuple[list[list[int]], int]:
    """Byte ranges of the lines that read as writing, each a whole line with its line break,
    and how many bytes they hold. A line counts when it is long enough to be a sentence and
    has the end of one in it. tidy ranks pages by the count; the ranges are for a curator that
    wants the lines themselves ({"d": page, "k": ranges} keeps them and nothing else)."""
    ranges: list[list[int]] = []
    pos = 0
    least = 25 if dense else 60
    for line in raw.split(b"\n"):
        n = len(line) + 1  # + the line break
        if n <= least:  # too few bytes to be that many characters: most of a page's furniture
            pos += n
            continue
        body = line.decode("utf-8", "replace").strip()
        long_enough = len(body) >= least and (dense or body.count(" ") >= 9)
        if long_enough and SENTENCE_END.search(body + " "):
            if ranges and ranges[-1][1] == pos:
                ranges[-1][1] = pos + n
            else:
                ranges.append([pos, pos + n])
        pos += n
    if ranges and ranges[-1][1] > len(raw):  # the last line may have no line break
        ranges[-1][1] = len(raw)
    return ranges, sum(b - a for a, b in ranges)


def score(size: int, kept: int) -> float:
    """Higher is better: a page that is mostly writing, and long enough to say something."""
    return kept / max(1, size) + min(1.0, kept / 3000)


def edit(task: Task):
    corpus = task.corpus

    def judge(i, text):
        """In a worker: how many bytes of this page are writing, and a key that is the same
        for every copy of it. None for a page that is not writing, or too little of it."""
        if not looks_written(text):
            return None
        _, kept = prose_ranges(corpus.raw(i), spaceless(text))
        if kept < 400:
            return None
        return kept, hashlib.blake2b(text[:2000].encode("utf-8", "replace"), digest_size=8).digest()

    seen: set[bytes] = set()
    scored = []
    for i, found in task.map(judge):  # in page order: of two copies, the first is the one kept
        if found is None or found[1] in seen:
            continue
        kept, key = found
        seen.add(key)
        scored.append((score(corpus.size(i), kept), i, kept))

    scored.sort(key=lambda t: (-t[0], t[1]))
    budget_bytes = task.train_tokens * BYTES_PER_TOKEN
    chosen = []
    used = 0
    for _, i, kept in scored:
        if used >= budget_bytes:
            break
        chosen.append(i)
        used += kept
    # The budget was counted in bytes of writing and the pages are copied whole, so these are
    # more pages than the student reads once: the stream is cut off at the budget, and what
    # is trained on is a shuffled part of the best pages. (Only as many of the top-ranked
    # pages as fill the budget did worse, both times it was tried.)
    random.Random(0).shuffle(chosen)
    return chosen


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()

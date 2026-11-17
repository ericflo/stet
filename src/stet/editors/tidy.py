"""An afternoon of heuristics: the baseline a serious entry should beat.

  1. Keep English prose.
  2. Drop exact duplicates.
  3. Inside each document, cut lines that look like navigation or boilerplate.
  4. Rank what is left by a crude "is this explanatory prose" score.
  5. Keep the best documents until the training budget is filled once, and shuffle them.

No model, no GPU, a few seconds per shard. Everything here is a cut of at least
min_copy_bytes, so it spends no ink.
"""

import hashlib
import random
import re

from stet.editor import Task
from stet.editors.english import looks_english

SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")
BYTES_PER_TOKEN = 3.3  # rough, for English prose under the season tokenizer


def prose_ranges(text: str, min_copy: int = 64) -> tuple[list[list[int]], int]:
    """Byte ranges of the lines worth keeping, and how many bytes they hold."""
    ranges: list[list[int]] = []
    pos = 0
    for line in text.split("\n"):
        n = len(line.encode("utf-8")) + 1  # + the newline
        body = line.strip()
        words = body.count(" ") + 1
        is_prose = len(body) >= 60 and words >= 10 and (SENTENCE_END.search(body + " ") is not None)
        if is_prose:
            if ranges and ranges[-1][1] == pos:
                ranges[-1][1] = pos + n
            else:
                ranges.append([pos, pos + n])
        pos += n
    # A kept run shorter than min_copy bytes would be charged as ink. Drop those.
    ranges = [r for r in ranges if r[1] - r[0] >= min_copy + 4]
    return ranges, sum(b - a for a, b in ranges)


def score(text: str, kept: int) -> float:
    """Higher is better. Rewards documents that are mostly prose and long enough to say something."""
    total = max(1, len(text.encode("utf-8")))
    density = kept / total
    length = min(1.0, kept / 3000)
    low = text[:4000].lower()
    explains = sum(low.count(w) for w in (" because ", " therefore ", " however ", " for example ", " which ", " whereas "))
    sells = sum(low.count(w) for w in (" buy ", " price", " shipping", " cart", " sale", " click ", " subscribe", " cookies"))
    return density * 1.0 + length * 1.0 + min(explains, 8) * 0.08 - min(sells, 8) * 0.1


def edit(task: Task):
    corpus = task.corpus
    min_copy = int(task.info.get("min_copy_bytes", 64))
    seen: set[bytes] = set()
    scored = []
    for i, text in corpus:
        if not looks_english(text):
            continue
        ranges, kept = prose_ranges(text, min_copy)
        if kept < 400:
            continue
        key = hashlib.blake2b(text[:2000].encode("utf-8", "replace"), digest_size=8).digest()
        if key in seen:
            continue
        seen.add(key)
        size = corpus.size(i)
        # Clamp to the document: the last line may not end in a newline.
        ranges = [[a, min(b, size)] for a, b in ranges if a < size and min(b, size) - a >= min_copy]
        scored.append((score(text, kept), i, ranges, kept))

    scored.sort(key=lambda t: -t[0])
    budget_bytes = task.train_tokens * BYTES_PER_TOKEN
    chosen = []
    used = 0
    for s, i, ranges, kept in scored:
        if used >= budget_bytes:
            break
        chosen.append({"d": i, "k": ranges})
        used += kept
    random.Random(0).shuffle(chosen)
    return chosen


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()

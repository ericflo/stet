"""FineWeb-2's multilingual recipe, rebuilt as a reference curator, without its per-language
tuning.

FineWeb-2 (Penedo et al., 2025) cleaned Common Crawl in every language rather than keeping
English: Gopher's quality and repetition rules and FineWeb's line filters, with thresholds
set per language, and deduplication within each language. In every language it tried, models
trained on the result beat models trained on the raw text. This keeps every language and uses
one set of thresholds for all of them, counting characters instead of words in scripts written
without spaces:

  1. Quality (Gopher): enough text; sensible word (or character-run) lengths; few symbol and
     ellipsis lines; mostly letters.
  2. Repetition (Gopher): few duplicated lines and paragraphs; no n-gram covering too much.
  3. Lines (FineWeb): most lines end in punctuation of any script; few short lines; little
     text in duplicated lines.
  4. Exact duplicates dropped, then near duplicates (MinHash on 5-grams, Jaccard 0.8).

Pages that pass are kept in a fixed random order until the training budget is filled once.
Whole pages only, so it spends no ink.
"""

import hashlib
import random
import sys
from collections import Counter

from stet.curator import Task

BYTES_PER_TOKEN = 3.7
ENDS = tuple(".!?\"'”’)»。！？…؟।")
SPACELESS = ("぀", "ヿ", "一", "鿿", "฀", "๿")  # kana, CJK, Thai


def spaceless(text: str) -> bool:
    sample = text[:2000]
    hits = sum(1 for c in sample if SPACELESS[0] <= c <= SPACELESS[1] or SPACELESS[2] <= c <= SPACELESS[3] or SPACELESS[4] <= c <= SPACELESS[5])
    return hits > len(sample) * 0.3


def units(text: str) -> list[str]:
    """Words, or for a script without spaces, its characters taken two at a time."""
    if spaceless(text):
        chars = [c for c in text if not c.isspace()]
        return ["".join(chars[i : i + 2]) for i in range(0, len(chars), 2)]
    return text.split()


def quality(text: str, words: list[str]) -> bool:
    if not 50 <= len(words) <= 100_000:
        return False
    mean = sum(len(w) for w in words) / len(words)
    if not 1.5 <= mean <= 12:
        return False
    if (text.count("#") + text.count("...") + text.count("…")) / len(words) > 0.1:
        return False
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if sum(ln.lstrip().startswith(("•", "-", "*", "·")) for ln in lines) / len(lines) > 0.9:
        return False
    return sum(any(c.isalpha() for c in w) for w in words) / len(words) >= 0.6


def repetition(text: str, words: list[str]) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    counts = Counter(lines)
    dup = [ln for ln in lines if counts[ln] > 1]
    if len(dup) / len(lines) > 0.3 or sum(len(ln) for ln in dup) / max(1, len(text)) > 0.2:
        return False
    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(paras) > 1 and sum(c - 1 for c in Counter(paras).values()) / len(paras) > 0.3:
        return False
    for n, most in ((2, 0.20), (3, 0.18), (4, 0.16)):
        if len(words) > n:
            gram, count = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1)).most_common(1)[0]
            if count > 1 and count * sum(len(w) for w in gram) / max(1, len(text)) > most:
                return False
    return True


def lines_ok(text: str) -> bool:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if sum(ln.endswith(ENDS) for ln in lines) / len(lines) < 0.12:
        return False
    if sum(len(ln) <= 30 for ln in lines) / len(lines) > 0.67:
        return False
    return True


def edit(task: Task):
    from datasketch import MinHash, MinHashLSH

    corpus = task.corpus
    seen: set[bytes] = set()
    kept = []
    for i, text in corpus:
        words = units(text)
        if not words or not lines_ok(text) or not quality(text, words) or not repetition(text, words):
            continue
        key = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()
        if key in seen:
            continue
        seen.add(key)
        kept.append((i, words, len(text.encode("utf-8", "replace"))))
    print(f"{len(kept):,} pages pass, {task.time_left():.0f}s left", file=sys.stderr)
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    out = []
    rng = random.Random(0)
    rng.shuffle(kept)
    budget, used = task.train_tokens * BYTES_PER_TOKEN, 0
    for i, words, size in kept:
        if used >= budget:
            break
        if task.time_left() > 60:  # near duplicates, while there is time to look
            m = MinHash(num_perm=64, seed=1)
            for k in range(0, max(1, len(words) - 4), 2):
                m.update(" ".join(words[k : k + 5]).encode("utf-8", "replace"))
            if lsh.query(m):
                continue
            lsh.insert(str(i), m)
        out.append(i)
        used += size
    print(f"{len(out):,} kept", file=sys.stderr)
    return out


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()

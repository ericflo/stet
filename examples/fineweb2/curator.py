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
Whole pages only: every line of its script is a page as it is.

It writes a script twice: when every page has been through the rules and the exact duplicates
are out, and again when the near duplicates are. The first is there if time runs out during
the second. Both passes run on every core the curator is given (task.map), and all that is
remembered of a page between them is its number and its size.
"""

import hashlib
import random
import re
import sys
from collections import Counter

from stet.curator import Task

BYTES_PER_TOKEN = 3.7
RESERVE_SECONDS = 60  # with less than this left, stop looking and write what there is
ENDS = tuple(".!?\"'”’)»。！？…؟।")
# Kana, CJK, Thai. Asked of a page's opening in the regex engine: asking each character in
# turn was a quarter of the time a page took, and a round's pages are gigabytes.
SPACELESS = re.compile("[぀-ヿ一-鿿฀-๿]")


def spaceless(text: str) -> bool:
    sample = text[:2000]
    return len(SPACELESS.findall(sample)) > len(sample) * 0.3


def units(text: str) -> list[str]:
    """Words, or for a script without spaces, its characters taken two at a time."""
    if spaceless(text):
        chars = "".join(text.split())  # without its whitespace
        return [chars[i : i + 2] for i in range(0, len(chars), 2)]
    return text.split()


def quality(text: str, words: list[str]) -> bool:
    if not 50 <= len(words) <= 100_000:
        return False
    mean = sum(map(len, words)) / len(words)
    if not 1.5 <= mean <= 12:
        return False
    if (text.count("#") + text.count("...") + text.count("…")) / len(words) > 0.1:
        return False
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if sum(ln.lstrip().startswith(("•", "-", "*", "·")) for ln in lines) / len(lines) > 0.9:
        return False
    # Words with a letter in them. Most that have one start with it, and are asked no more.
    return sum(1 for w in words if w[0].isalpha() or any(c.isalpha() for c in w)) / len(words) >= 0.6


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
            # Every run of n words, in order: words[0:n], words[1:n+1], ... (zip stops at the
            # end of the shortest slice, which is the last run).
            gram, count = Counter(zip(*(words[k:] for k in range(n)), strict=False)).most_common(1)[0]
            if count > 1 and count * sum(len(w) for w in gram) / max(1, len(text)) > most:
                return False
    return True


def lines_ok(text: str) -> bool:
    lines = [ln for ln in map(str.strip, text.split("\n")) if ln]
    if sum(ln.endswith(ENDS) for ln in lines) / len(lines) < 0.12:
        return False
    if sum(len(ln) <= 30 for ln in lines) / len(lines) > 0.67:
        return False
    return True


def passes(i: int, text: str):
    """For a page that passes every rule: a key that is the same for every copy of it, and
    its size. None for the rest."""
    words = units(text)
    if not words or not lines_ok(text) or not quality(text, words) or not repetition(text, words):
        return None
    raw = text.encode("utf-8", "replace")
    return hashlib.blake2b(raw, digest_size=8).digest(), len(raw)


def signature(i: int, text: str):
    """The MinHash of a page: of every other run of five of its words."""
    from datasketch import MinHash

    words = units(text)
    m = MinHash(num_perm=64, seed=1)
    m.update_batch([" ".join(words[k : k + 5]).encode("utf-8", "replace") for k in range(0, max(1, len(words) - 4), 2)])
    return m.hashvalues


def edit(task: Task):
    from datasketch import MinHash, MinHashLSH

    budget = task.train_tokens * BYTES_PER_TOKEN

    # The rules, on every page, in the workers. What comes back for a page that passes is a
    # key and a size, and the first of two copies is the one kept: decided here, in page order.
    seen: set[bytes] = set()
    kept = []
    for i, found in task.map(passes):
        if task.time_left() < RESERVE_SECONDS:
            print(f"out of time at page {i:,} of {len(task.corpus):,}: the pages after it are not read", file=sys.stderr)
            break
        if found is None or found[0] in seen:
            continue
        seen.add(found[0])
        kept.append((i, found[1]))
    del seen
    print(f"{len(kept):,} pages pass, {task.time_left():.0f}s left", file=sys.stderr)
    random.Random(0).shuffle(kept)

    # A script already: the pages that pass, in the order they will have, until the budget is
    # full. If the clock stops what follows, this is the script that counts.
    out, used = [], 0
    for i, size in kept:
        if used >= budget:
            break
        out.append(i)
        used += size
    task.write(out)
    print(f"{len(out):,} kept before near duplicates are looked for, {task.time_left():.0f}s left", file=sys.stderr)

    # Near duplicates, while there is time to look. The workers read each page again and
    # send back its MinHash, in the order the pages will have; which of two near copies comes
    # first, and so is kept, is decided here. A few hundred pages past the one that fills
    # the budget are hashed for nothing, and none of the rest.
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    like = MinHash(num_perm=64, seed=1)
    out, used = [], 0
    for at, (i, values) in enumerate(task.map(signature, docs=[i for i, _ in kept], chunk=32)):
        if used >= budget or task.time_left() <= RESERVE_SECONDS:
            break
        m = MinHash(seed=1, hashvalues=values, permutations=like.permutations, scheme=like.scheme)
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        out.append(i)
        used += kept[at][1]
    else:
        at = len(kept)
    for i, size in kept[at:]:  # out of time with the budget not full: the rest as they are
        if used >= budget:
            break
        out.append(i)
        used += size
    print(f"{len(out):,} kept", file=sys.stderr)
    return out


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()

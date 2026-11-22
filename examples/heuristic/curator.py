"""The published heuristic recipe, rebuilt as a reference curator.

From Gopher (Rae et al., 2021) and RefinedWeb (Penedo et al., 2023), applied to English pages:

  1. Lines, as C4 cuts them (Raffel et al., 2020): keep lines that end in terminal
     punctuation and have at least three words; drop lines that mention javascript, cookie
     notices, terms of use or "lorem ipsum", and lines with a curly brace.
  2. Gopher's quality rules on what is left: 50 to 100,000 words; mean word length 3 to 10
     characters; at most 10% symbols (# and ...); at most 90% of lines starting with a
     bullet; at most 30% ending in an ellipsis; at least 80% of words containing a letter;
     at least two of the, be, to, of, and, that, have, with.
  3. Gopher's repetition rules: duplicated lines at most 30% of lines and 20% of
     characters; the most frequent 2-, 3- and 4-gram at most 20%, 18% and 16% of characters.
  4. Exact duplicates dropped.

Pages are kept in a fixed random order until the training budget is filled once. It is a
reference, not an entry to beat: large models trained on data made like this beat raw crawl
text and lose to classifier-filtered data, and an exam is checked by whether it agrees.
Every piece it keeps is at least min_copy_bytes long, as a copy has to be.
"""

import hashlib
import random
from collections import Counter

from stet.curator import Task

BYTES_PER_TOKEN = 3.7
GOPHER_STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}
C4_BAD = ("javascript", "lorem ipsum", "cookie", "terms of use", "privacy policy", "{", "}")
END = (".", "!", "?", '"', "'", "”", "’")


LID_MIN = 0.65  # RefinedWeb's and FineWeb's threshold for English
_lid = None


def lid(task):
    """Meta's language identification model (facebook/fasttext-language-identification), loaded
    the first time it is asked for. Ask before task.map starts its workers and they share the
    one copy; left to load it themselves, each would hold a gigabyte of its own."""
    global _lid
    if _lid is None:
        import fasttext

        _lid = fasttext.load_model(str(task.asset("lid/model.bin")))
    return _lid


def looks_english(task, text: str) -> bool:
    """English, as the published recipes decide it: fastText language identification, the
    English score at least 0.65."""
    if len(text) < 200:
        return False
    labels, probs = lid(task).predict(" ".join(text[:2000].split()))
    return labels[0] == "__label__eng_Latn" and float(probs[0]) >= LID_MIN


def c4_lines(text: str, min_copy: int):
    """Byte ranges of the lines C4 keeps, merged where they touch, and the kept text."""
    ranges, kept, pos = [], [], 0
    for line in text.split("\n"):
        n = len(line.encode("utf-8")) + 1
        body = line.strip()
        low = body.lower()
        if len(body.split()) >= 3 and body.endswith(END) and not any(b in low for b in C4_BAD):
            if ranges and ranges[-1][1] == pos:
                ranges[-1][1] = pos + n
            else:
                ranges.append([pos, pos + n])
            kept.append(body)
        pos += n
    ranges = [r for r in ranges if r[1] - r[0] >= min_copy + 4]
    return ranges, "\n".join(kept)


def gopher_quality(text: str) -> bool:
    words = text.split()
    if not 50 <= len(words) <= 100_000:
        return False
    mean = sum(len(w) for w in words) / len(words)
    if not 3 <= mean <= 10:
        return False
    if (text.count("#") + text.count("...") + text.count("…")) / len(words) > 0.1:
        return False
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        if sum(ln.lstrip().startswith(("•", "-", "*", "·")) for ln in lines) / len(lines) > 0.9:
            return False
        if sum(ln.rstrip().endswith(("...", "…")) for ln in lines) / len(lines) > 0.3:
            return False
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8:
        return False
    return len(GOPHER_STOP & {w.lower() for w in words}) >= 2


def gopher_repetition(text: str) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        counts = Counter(lines)
        dup = [ln for ln in lines if counts[ln] > 1]
        if len(dup) / len(lines) > 0.3 or sum(len(ln) for ln in dup) / max(1, len(text)) > 0.2:
            return False
    words = text.split()
    for n, most in ((2, 0.20), (3, 0.18), (4, 0.16)):
        if len(words) < n + 1:
            continue
        grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        gram, count = grams.most_common(1)[0]
        if count > 1 and count * sum(len(w) for w in gram) / max(1, len(text)) > most:
            return False
    return True


def edit(task: Task):
    corpus = task.corpus
    min_copy = int(task.info.get("min_copy_bytes", 64))
    lid(task)  # loaded before task.map starts its workers, so that they share it

    def judge(i, text):
        """In a worker: for a page that passes, a key (the same for every copy of what is kept
        of it) and the byte ranges to keep."""
        if not looks_english(task, text):
            return None
        ranges, kept = c4_lines(text, min_copy)
        if not ranges or kept.count(". ") + kept.count("! ") + kept.count("? ") < 2:  # C4: at least three sentences
            return None
        if not (gopher_quality(kept) and gopher_repetition(kept)):
            return None
        size = corpus.size(i)
        ranges = [[a, min(b, size)] for a, b in ranges if a < size and min(b, size) - a >= min_copy]
        return hashlib.blake2b(kept.encode("utf-8", "replace"), digest_size=8).digest(), ranges

    seen: set[bytes] = set()
    chosen = []
    for i, found in task.map(judge):  # in page order: of two copies, the first is the one kept
        if found is None or found[0] in seen:
            continue
        key, ranges = found
        seen.add(key)
        if ranges:
            chosen.append(({"d": i, "k": ranges}, sum(b - a for a, b in ranges)))
    random.Random(0).shuffle(chosen)
    budget, used, out = task.train_tokens * BYTES_PER_TOKEN, 0, []
    for line, kept in chosen:
        if used >= budget:
            break
        out.append(line)
        used += kept
    return out


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()

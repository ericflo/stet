"""DCLM-Baseline's filter, rebuilt as a reference curator.

DataComp-LM (Li et al., 2024) found that the best filter for its 1B to 7B models was a
fastText classifier telling instruction-style text (OpenHermes 2.5, and highly rated ELI5
answers) from random web pages, keeping the top tenth. This curator uses that classifier, as
published (mlfoundations/fasttext-oh-eli5):

  1. English pages, exact duplicates dropped.
  2. Each page scored by the classifier's probability that it is the instruction-style kind.
  3. The best pages kept until the training budget is filled once, in a fixed random order.

It is a reference, not an entry to beat. An exam is checked by whether it puts this, and
FineWeb-Edu's filter, above heuristic filtering, and that above raw crawl text, as the
published results did. Whole pages only, so it spends no ink.
"""

import hashlib
import random
import sys

from stet.curator import Task

BYTES_PER_TOKEN = 3.7


LID_MIN = 0.65  # RefinedWeb's and FineWeb's threshold for English
_lid = None


def looks_english(task, text: str) -> bool:
    """English, as the published recipes decide it: fastText language identification, the
    English score at least 0.65 (Meta's model, facebook/fasttext-language-identification)."""
    global _lid
    if _lid is None:
        import fasttext

        _lid = fasttext.load_model(str(task.asset("lid/model.bin")))
    if len(text) < 200:
        return False
    labels, probs = _lid.predict(" ".join(text[:2000].split()))
    return labels[0] == "__label__eng_Latn" and float(probs[0]) >= LID_MIN


def edit(task: Task):
    import fasttext

    model = fasttext.load_model(str(task.asset("dclm-fasttext/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin")))
    print(f"classifier loaded, {task.time_left():.0f}s left", file=sys.stderr)
    corpus = task.corpus
    seen: set[bytes] = set()
    scored = []
    for i, text in corpus:
        if not looks_english(task, text):
            continue
        key = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()
        if key in seen:
            continue
        seen.add(key)
        labels, probs = model.predict(" ".join(text.split())[:100_000])
        p = float(probs[0]) if labels[0] == "__label__hq" else 1.0 - float(probs[0])
        scored.append((round(p, 4), i, len(text.encode("utf-8", "replace"))))
    scored.sort(key=lambda t: (-t[0], t[1]))
    budget, used, chosen = task.train_tokens * BYTES_PER_TOKEN, 0, []
    for _, i, size in scored:
        if used >= budget:
            break
        chosen.append(i)
        used += size
    print(f"{len(scored):,} English pages scored, {len(chosen):,} kept", file=sys.stderr)
    random.Random(0).shuffle(chosen)
    return chosen


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()

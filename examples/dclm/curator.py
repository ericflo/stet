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
published results did. Whole pages only: every line of its script is a page as it is.
"""

import hashlib
import random
import sys

from stet.curator import Task

BYTES_PER_TOKEN = 3.7


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


def edit(task: Task):
    import fasttext

    # Both models are loaded here, before task.map starts its workers: 3.6 GB that every
    # worker reads and none of them copies.
    model = fasttext.load_model(str(task.asset("dclm-fasttext/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin")))
    lid(task)
    print(f"classifier loaded, {task.time_left():.0f}s left", file=sys.stderr)

    def judge(i, text):
        """In a worker: an English page's key (the same for every copy of it), score and size."""
        if not looks_english(task, text):
            return None
        raw = text.encode("utf-8", "replace")
        labels, probs = model.predict(" ".join(text.split())[:100_000])
        p = float(probs[0]) if labels[0] == "__label__hq" else 1.0 - float(probs[0])
        return hashlib.blake2b(raw, digest_size=8).digest(), round(p, 4), len(raw)

    seen: set[bytes] = set()
    scored = []
    for i, found in task.map(judge):  # in page order: of two copies, the first is the one kept
        if found is None or found[0] in seen:
            continue
        key, p, size = found
        seen.add(key)
        scored.append((p, i, size))
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

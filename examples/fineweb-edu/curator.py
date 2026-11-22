"""FineWeb-Edu's filter, rebuilt as a reference curator.

FineWeb-Edu (Penedo et al., 2024) kept web pages that a classifier, trained on Llama-3-70B's
ratings of educational value (0 to 5), scored 3 or more. Models trained on it gained most on
knowledge and reasoning benchmarks. This curator uses that classifier, as published
(HuggingFaceFW/fineweb-edu-classifier):

  1. English pages, exact duplicates dropped.
  2. Each page's opening (512 tokens, as the classifier was trained) scored for educational
     value, in a fixed random order until time is nearly up; a page not reached is not kept.
  3. The best pages kept until the training budget is filled once, in a fixed random order.

It is a reference, not an entry to beat. Whole pages only, so it spends no ink.
"""

import hashlib
import random
import sys

from stet.curator import Task

BYTES_PER_TOKEN = 3.7
BATCH = 64
RESERVE_SECONDS = 30


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
    corpus = task.corpus
    lid(task)  # loaded before task.map starts its workers, so that they share it

    def english(i, text):
        """In a worker: an English page's key (the same for every copy of it) and its size."""
        if not looks_english(task, text):
            return None
        raw = text.encode("utf-8", "replace")
        return hashlib.blake2b(raw, digest_size=8).digest(), len(raw)

    # Which pages are English is the part of this that a GPU cannot do, and every second of
    # it is a second the classifier does not get: so on every core, and before PyTorch is
    # loaded, by workers that have no use for it.
    seen: set[bytes] = set()
    cands = []
    for i, found in task.map(english):  # in page order: of two copies, the first is the one kept
        if found is None or found[0] in seen:
            continue
        seen.add(found[0])
        cands.append((i, found[1]))
    random.Random(1).shuffle(cands)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = task.device()
    where = task.asset("fineweb-edu-classifier")
    tok = AutoTokenizer.from_pretrained(where)
    model = AutoModelForSequenceClassification.from_pretrained(where, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    print(f"{len(cands):,} English pages; classifier on {device}, {task.time_left():.0f}s left", file=sys.stderr)
    scored = []
    with torch.inference_mode():
        for b in range(0, len(cands), BATCH):
            if task.time_left() < RESERVE_SECONDS:
                break
            part = cands[b : b + BATCH]
            # A page's opening is read again here, a batch at a time: kept from the first pass,
            # the openings of a round's English pages are gigabytes.
            enc = tok([corpus.text(i)[:6000] for i, _ in part], truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
            out = model(**enc).logits.squeeze(-1).float().tolist()
            scored += [(round(s, 2), i, size) for s, (i, size) in zip(out, part, strict=True)]  # two places: the same on any GPU
    scored.sort(key=lambda t: (-t[0], t[1]))
    budget, used, chosen = task.train_tokens * BYTES_PER_TOKEN, 0, []
    for _, i, size in scored:
        if used >= budget:
            break
        chosen.append(i)
        used += size
    print(f"{len(scored):,} scored, {len(chosen):,} kept", file=sys.stderr)
    random.Random(0).shuffle(chosen)
    return chosen


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()

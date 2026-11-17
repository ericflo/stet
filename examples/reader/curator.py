"""A small language model reads the candidates: how to use model weights in a curator.

  1. Do what `tidy` does: English prose, duplicates dropped, boilerplate lines cut, a crude
     quality score. Write that script straight away, so there is always one to fall back on.
  2. Load a small language model (SmolLM2-135M, named in stet.toml and fetched before the
     clock starts) and have it read the opening of each candidate, best candidates first.
  3. Drop the candidates the model finds most predictable (templates, repetition) and
     least predictable (lists, fragments, noise). Fill the budget from what is left.

This is a starting point, not a strong entry: zero-shot, a small model's opinion of a page
is about as good as `tidy`'s heuristics. Fork it and bring a better reader.

What to copy from it, whatever your model is:

  - Ask for the GPU with task.device(). It is "cuda" on NVIDIA and AMD alike.
  - Load from task.asset(...). Nothing can be downloaded while the curator runs.
  - Half precision, and batches sorted by length: the time limit is on the slowest machine
    that runs curators, and `stet bench` says how yours compares.
  - Watch task.time_left() and write the script before it runs out. A script written
    earlier still counts if this one never is.
  - Two machines will not give the same numbers to the last digit. Round what the model
    says before sorting or thresholding on it, and break ties by document index, so that
    a different GPU does not mean a different dataset.
"""

import hashlib
import random
import re
import sys

from stet.curator import Task

SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")
STOPWORDS = (" the ", " of ", " and ", " to ", " in ", " is ", " that ", " for ")
BYTES_PER_TOKEN = 3.7  # English prose under the season tokenizer, on the generous side
HEAD_BYTES = 800  # how much of each candidate the model reads
MAX_TOKENS = 128
BATCH = 64
DROP_LOW, DROP_HIGH = 0.08, 0.20  # the share of candidates dropped at each end
RESERVE_SECONDS = 25  # stop reading this long before the limit, to write the script


def looks_english(text: str) -> bool:
    sample = text[:3000]
    if len(sample) < 200:
        return False
    letters = sum(c.isascii() and (c.isalpha() or c == " ") for c in sample)
    if letters / len(sample) < 0.75:
        return False
    low = sample.lower()
    return sum(low.count(w) for w in STOPWORDS) >= max(4, len(sample) // 300)


def prose_ranges(text: str, min_copy: int = 64) -> tuple[list[list[int]], int]:
    """Byte ranges of the lines worth keeping, and how many bytes they hold."""
    ranges: list[list[int]] = []
    pos = 0
    for line in text.split("\n"):
        n = len(line.encode("utf-8")) + 1  # + the newline
        body = line.strip()
        words = body.count(" ") + 1
        if len(body) >= 60 and words >= 10 and (SENTENCE_END.search(body + " ") is not None):
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


def candidates(task: Task) -> list[tuple[float, int, list[list[int]], int, str]]:
    """(score, document, kept ranges, kept bytes, the opening of the kept text), best first."""
    corpus = task.corpus
    min_copy = int(task.info.get("min_copy_bytes", 64))
    seen: set[bytes] = set()
    out = []
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
        ranges = [[a, min(b, size)] for a, b in ranges if a < size and min(b, size) - a >= min_copy]
        raw = text.encode("utf-8")
        head = b"".join(raw[a:b] for a, b in ranges)[:HEAD_BYTES].decode("utf-8", "ignore")
        out.append((score(text, kept), i, ranges, kept, head))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out


def choose(task: Task, cands, surprise: dict[int, float]) -> list[dict]:
    """Fill the budget in score order, skipping what the model put at either extreme.
    A candidate the model never got to is kept: no opinion is not a bad one."""
    read = sorted(surprise.values())
    low = read[int(len(read) * DROP_LOW)] if read else float("-inf")
    high = read[min(len(read) - 1, int(len(read) * (1 - DROP_HIGH)))] if read else float("inf")
    budget_bytes = task.train_tokens * BYTES_PER_TOKEN
    chosen, used = [], 0
    for _, i, ranges, kept, _ in cands:
        if used >= budget_bytes:
            break
        s = surprise.get(i)
        if s is not None and not (low <= s <= high):
            continue
        chosen.append({"d": i, "k": ranges})
        used += kept
    random.Random(0).shuffle(chosen)
    return chosen


def read_candidates(task: Task, cands, surprise: dict[int, float]) -> None:
    """Fill `surprise` with the model's mean loss per token on each candidate's opening,
    best candidates first, until they are all read or time is nearly up."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = task.device()
    where = task.asset("smollm2-135m")
    tok = AutoTokenizer.from_pretrained(where)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(where, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    print(f"model loaded on {device}, {task.time_left():.0f}s left", file=sys.stderr)
    chunk = BATCH * 32  # sort this many by length at a time: similar lengths share a batch
    with torch.inference_mode():
        for start in range(0, len(cands), chunk):
            part = cands[start : start + chunk]
            enc = tok([c[4] for c in part], truncation=True, max_length=MAX_TOKENS)["input_ids"]
            order = sorted(range(len(part)), key=lambda j: (len(enc[j]), part[j][1]))
            for b in range(0, len(order), BATCH):
                if task.time_left() < RESERVE_SECONDS:
                    return
                rows = [j for j in order[b : b + BATCH] if len(enc[j]) >= 8]
                if not rows:
                    continue
                width = max(len(enc[j]) for j in rows)
                ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(rows), width), dtype=torch.bool)
                for r, j in enumerate(rows):
                    ids[r, : len(enc[j])] = torch.tensor(enc[j])
                    mask[r, : len(enc[j])] = True
                ids, mask = ids.to(device), mask.to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
                loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
                keep = mask[:, 1:]
                mean = (loss * keep).sum(1) / keep.sum(1)
                for r, j in enumerate(rows):
                    surprise[part[j][1]] = round(float(mean[r]), 2)  # two places: the same on any GPU
            print(f"read {len(surprise):,} of {len(cands):,}, {task.time_left():.0f}s left", file=sys.stderr)


def main():
    task = Task.from_env()
    cands = candidates(task)
    task.write(choose(task, cands, {}))  # the fallback: exactly what tidy would write
    print(f"{len(cands):,} candidates, {task.time_left():.0f}s left", file=sys.stderr)
    surprise: dict[int, float] = {}
    try:
        read_candidates(task, cands, surprise)
    finally:
        if surprise:
            task.write(choose(task, cands, surprise))


if __name__ == "__main__":
    main()

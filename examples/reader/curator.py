"""A multilingual 0.6B language model reads and rates the corpus: Season 3 reference neural curator.

  1. Prefilter across ALL languages (never English-only) using lightweight line/script rules
     and exact deduplication across all available CPU cores (`task.map`), keeping whole
     documents so code, tables, lists and multilingual prose stay intact. Write a valid
     fallback script straight away before loading the GPU model.
  2. Score candidate openings with Qwen/Qwen3-0.6B (pinned in stet.toml) on a 1-to-5
     educational/reasoning quality rubric by reading the next-token probability distribution
     over digits '1'..'5' (expected rating).
     - On NVIDIA Linux, uses `vllm.LLM` with automatic prefix caching for high throughput.
     - On AMD ROCm, macOS, or when `vllm` is absent, falls back cleanly to batched PyTorch
       `transformers` inference.
  3. Fill the training token budget with the highest-rated documents (keeping a balanced
     floor of quality and respecting `task.time_left()`), ordered as a gentle curriculum:
     lower-scored kept documents earlier, highest-scored documents in the final LR cooldown
     window (the Season 3 trainer does not shuffle).

What to copy from it for your own neural curator:

  - Never drop non-English pages: the exam weighs every language by its share of the crawl.
  - Round model scores before sorting and break ties by document index so NVIDIA and AMD
    runs write consistent datasets.
  - Write a fallback script early and watch `task.time_left()` with a safety margin.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import sys
import unicodedata

from stet.curator import Task

BYTES_PER_TOKEN = 3.5  # conservative multilingual estimate under Qwen3's 151k tokenizer
HEAD_CHARS = 900  # excerpt read by the 0.6B quality rater
MAX_PROMPT_TOKENS = 288
RESERVE_SECONDS = 45
CANDIDATE_OVERPROVISION = 15.0  # score up to 15x budget (most/all prefiltered docs) within time_left()

# Prompt shared across every document so vLLM automatic prefix caching caches it once.
# Trailing space in RUBRIC_SUFFIX is critical: without it Qwen3 predicts token ' ' (id 220)
# next with 99.95% probability instead of a rating digit '1'..'5'.
RUBRIC_PREFIX = (
    "Rate the educational, factual, technical, or reasoning value of the following web "
    "document (in any language) for pretraining a language model on a scale from 1 (spam, "
    "boilerplate, navigation, ads, or gibberish) to 5 (clear, informative, well-written "
    "prose, technical documentation, code, or analysis).\n"
    "Answer with a single digit: 1, 2, 3, 4, or 5.\n\nDocument:\n"
)
RUBRIC_SUFFIX = "\n\nRating (1-5): "

MARKS = "".join(c for c in map(chr, range(0x300, 0x10000)) if unicodedata.category(c)[0] == "M")
WRITING = re.compile(rf"(?:[^\W\d_]|\s|[{re.escape(MARKS)}])+")
SENTENCE_END = re.compile(r"[.!?…؟۔।][\"')\]”’»]?\s|[。！？｡]|[{};]\n")


def _prefilter(i: int, text: str):
    """Cheap multilingual pass run in parallel across CPU cores."""
    n_chars = len(text)
    if n_chars < 280 or n_chars > 400_000:
        return None
    sample = text[:3000]
    non_writing = len(WRITING.sub("", sample)) / max(1, len(sample))
    # Allow code/technical pages (more punctuation) as well as prose of any script,
    # while rejecting pure symbol garbage and link dumps.
    if non_writing > 0.45:
        return None
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return None
    short_frac = sum(len(ln) <= 25 for ln in lines) / len(lines)
    if short_frac > 0.72:
        return None
    if not SENTENCE_END.search(sample + " "):
        return None
    raw = text.encode("utf-8", "replace")
    key = hashlib.blake2b(raw[:4096], digest_size=8).digest()
    # Simple heuristic prior to pick which candidates the 0.6B model reads first.
    uniq_lines = len(set(lines)) / len(lines)
    prior = round(uniq_lines + min(1.0, len(raw) / 4000) - short_frac, 3)
    head = text[:HEAD_CHARS].strip()
    return prior, key, len(raw), head


def collect_candidates(task: Task) -> list[tuple[float, int, int, str]]:
    """Return (prior, doc_id, size_bytes, excerpt) deduplicated across all languages."""
    budget_bytes = task.train_tokens * BYTES_PER_TOKEN
    target_bytes = budget_bytes * CANDIDATE_OVERPROVISION
    seen: set[bytes] = set()
    kept: list[tuple[float, int, int, str]] = []
    for i, res in task.map(_prefilter):
        if task.time_left() < RESERVE_SECONDS * 2:
            break
        if res is None:
            continue
        prior, key, size, head = res
        if key in seen:
            continue
        seen.add(key)
        kept.append((prior, i, size, head))
    kept.sort(key=lambda t: (-t[0], t[1]))
    # Cap candidates to CANDIDATE_OVERPROVISION × the byte budget (with a sensible minimum for small scales).
    out: list[tuple[float, int, int, str]] = []
    total = 0
    cap = max(target_bytes, 2_000_000)
    for item in kept:
        out.append(item)
        total += item[2]
        if total >= cap:
            break
    return out


def build_script(task: Task, cands: list[tuple[float, int, int, str]], scores: dict[int, float]) -> list[int]:
    """Select documents up to the token budget; order higher-rated documents last (cooldown)."""
    budget_bytes = task.train_tokens * BYTES_PER_TOKEN
    if not scores:
        # Fallback before GPU scoring: shuffled high-prior multilingual docs.
        chosen: list[int] = []
        used = 0
        for _, doc_id, size, _ in cands:
            if used >= budget_bytes:
                break
            chosen.append(doc_id)
            used += size
        random.Random(0).shuffle(chosen)
        return chosen

    ranked = sorted(
        [(scores[doc_id], prior, doc_id, size) for prior, doc_id, size, _ in cands if doc_id in scores],
        key=lambda t: (-t[0], -t[1], t[2]),
    )
    selected: list[tuple[float, int]] = []
    used = 0
    for q, _, doc_id, size in ranked:
        if used >= budget_bytes:
            break
        selected.append((q, doc_id))
        used += size
    if used < budget_bytes:
        picked = {doc_id for _, doc_id in selected}
        for _, doc_id, size, _ in cands:
            if used >= budget_bytes:
                break
            if doc_id not in picked:
                selected.append((2.5, doc_id))
                used += size
    # Curriculum ordering: split into 4 quality bands, shuffle inside each band, and place
    # the highest-rated band last so the student sees the cleanest text during LR cooldown.
    selected.sort(key=lambda t: (t[0], t[1]))
    n = len(selected)
    if n <= 8:
        return [doc_id for _, doc_id in selected]
    rng = random.Random(0)
    out: list[int] = []
    bands = 4
    for b in range(bands):
        lo = (b * n) // bands
        hi = ((b + 1) * n) // bands
        chunk = [doc_id for _, doc_id in selected[lo:hi]]
        rng.shuffle(chunk)
        out.extend(chunk)
    return out


def _digit_token_ids(tok) -> dict[int, int]:
    """Map rating digits 1..5 to their single-token IDs."""
    mapping: dict[int, int] = {}
    for digit in range(1, 6):
        ids = tok.encode(str(digit), add_special_tokens=False)
        if ids:
            mapping[digit] = int(ids[-1])
    return mapping


def _expected_from_logprobs(logprobs: dict[int, float], digit_ids: dict[int, int]) -> float:
    vals = [(d, logprobs[tid]) for d, tid in digit_ids.items() if tid in logprobs]
    if not vals:
        return 3.0
    m = max(lp for _, lp in vals)
    weights = [(d, math.exp(lp - m)) for d, lp in vals]
    z = sum(w for _, w in weights)
    if z <= 0:
        return 3.0
    return round(sum(d * w for d, w in weights) / z, 2)


def _extract_logprobs(step_lps) -> dict[int, float]:
    """Read token -> logprob float from a vLLM step logprobs dict (vLLM Logprob dataclass or float)."""
    if not step_lps:
        return {}
    out: dict[int, float] = {}
    for tid, entry in step_lps.items():
        val = entry.logprob if hasattr(entry, "logprob") else float(entry)
        out[int(tid)] = float(val)
    return out


def _build_prompt_ids(tok, head: str, max_prompt_tokens: int, prefix_ids: list[int], suffix_ids: list[int]) -> list[int]:
    """Build a token-bounded prompt that always keeps RUBRIC_PREFIX intact (for prefix caching)
    and never truncates RUBRIC_SUFFIX ('Rating (1-5):')."""
    body_budget = max(1, max_prompt_tokens - len(prefix_ids) - len(suffix_ids))
    body_ids = tok.encode(head, add_special_tokens=False)[:body_budget]
    return prefix_ids + body_ids + suffix_ids


def _score_vllm(task: Task, where: str, cands: list[tuple[float, int, int, str]], scores: dict[int, float]) -> bool:
    """Fast path on NVIDIA Linux using vLLM with automatic prefix caching.

    Returns True if vLLM scored candidates, or False so the caller can fall back to
    `_score_transformers` if vLLM is absent or fails to initialise.
    """
    import torch

    if task.device() != "cuda" or getattr(torch.version, "hip", None):
        return False
    # Run vLLM's V1 EngineCore in-process so CUDA initialized by task.device() (and Stet's
    # per-process GPU memory cap) works cleanly without fork/spawn subprocess issues.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        from vllm import LLM, SamplingParams
    except Exception:
        return False

    try:
        llm = LLM(
            model=where,
            dtype="bfloat16",
            max_model_len=MAX_PROMPT_TOKENS + 16,
            gpu_memory_utilization=0.65,
            enable_prefix_caching=True,
            enforce_eager=False,
            disable_log_stats=True,
        )
        tok = llm.get_tokenizer()
        digit_ids = _digit_token_ids(tok)
        prefix_ids = list(tok.encode(RUBRIC_PREFIX, add_special_tokens=True))
        suffix_ids = list(tok.encode(RUBRIC_SUFFIX, add_special_tokens=False))
        try:
            params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                logprobs=None,
                logprob_token_ids=sorted(digit_ids.values()),
            )
        except TypeError:
            params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        print(f"vLLM loaded Qwen3-0.6B, {task.time_left():.0f}s left", file=sys.stderr)

        batch_size = 512
        for start in range(0, len(cands), batch_size):
            if task.time_left() < RESERVE_SECONDS:
                return bool(scores)
            part = cands[start : start + batch_size]
            prompts = [
                {"prompt_token_ids": _build_prompt_ids(tok, head, MAX_PROMPT_TOKENS, prefix_ids, suffix_ids)}
                for _, _, _, head in part
            ]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for (_, doc_id, _, _), out in zip(part, outputs, strict=True):
                step_lps = out.outputs[0].logprobs[0] if out.outputs and out.outputs[0].logprobs else {}
                lps = _extract_logprobs(step_lps)
                scores[doc_id] = _expected_from_logprobs(lps, digit_ids)
            if len(scores) % (batch_size * 10) == 0 or len(scores) == len(cands):
                print(f"vLLM rated {len(scores):,} / {len(cands):,}, {task.time_left():.0f}s left", file=sys.stderr)
        return bool(scores)
    except Exception as exc:
        print(f"vLLM unavailable ({exc}); falling back to transformers", file=sys.stderr)
        return bool(scores)


def _score_transformers(task: Task, where: str, cands: list[tuple[float, int, int, str]], scores: dict[int, float]) -> None:
    """Portable batched PyTorch/transformers path (AMD ROCm, macOS, or NVIDIA fallback)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = task.device()
    tok = AutoTokenizer.from_pretrained(where)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(where, dtype=dtype).to(device).eval()
    digit_ids = _digit_token_ids(tok)
    digits = sorted(digit_ids)
    digit_tensor = torch.tensor([digit_ids[d] for d in digits], device=device, dtype=torch.long)
    digit_vals = torch.tensor([float(d) for d in digits], device=device, dtype=torch.float32)
    prefix_ids = list(tok.encode(RUBRIC_PREFIX, add_special_tokens=True))
    suffix_ids = list(tok.encode(RUBRIC_SUFFIX, add_special_tokens=False))
    print(f"transformers loaded Qwen3-0.6B on {device}, {task.time_left():.0f}s left", file=sys.stderr)

    batch_size = 32 if device == "cuda" else 8
    chunk = batch_size * 16
    with torch.inference_mode():
        for start in range(0, len(cands), chunk):
            part = cands[start : start + chunk]
            enc = [_build_prompt_ids(tok, head, MAX_PROMPT_TOKENS, prefix_ids, suffix_ids) for _, _, _, head in part]
            order = sorted(range(len(part)), key=lambda j: (len(enc[j]), part[j][1]))
            for b in range(0, len(order), batch_size):
                if task.time_left() < RESERVE_SECONDS:
                    return
                rows = order[b : b + batch_size]
                width = max(len(enc[j]) for j in rows)
                ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(rows), width), dtype=torch.long)
                for r, j in enumerate(rows):
                    seq = enc[j]
                    ids[r, width - len(seq) :] = torch.tensor(seq, dtype=torch.long)
                    mask[r, width - len(seq) :] = 1
                ids, mask = ids.to(device), mask.to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
                d_logits = logits.index_select(1, digit_tensor)
                probs = torch.softmax(d_logits, dim=-1)
                exp = (probs * digit_vals).sum(dim=-1).tolist()
                for r, j in enumerate(rows):
                    scores[part[j][1]] = round(float(exp[r]), 2)
            print(f"rated {len(scores):,} / {len(cands):,}, {task.time_left():.0f}s left", file=sys.stderr)


def main():
    task = Task.from_env()
    cands = collect_candidates(task)
    task.write(build_script(task, cands, {}))  # immediate multilingual fallback
    print(f"{len(cands):,} candidates, {task.time_left():.0f}s left", file=sys.stderr)
    scores: dict[int, float] = {}
    try:
        where = str(task.asset("qwen3-0.6b"))
        if not _score_vllm(task, where, cands, scores):
            _score_transformers(task, where, cands, scores)
    except Exception as exc:
        print(f"reader model error (keeping fallback): {exc}", file=sys.stderr)
    finally:
        if scores:
            vals = sorted(scores.values())
            n = len(vals)
            mean = sum(vals) / n
            print(
                f"rated {n:,} docs: mean={mean:.2f} min={vals[0]:.2f} p25={vals[n // 4]:.2f} "
                f"p50={vals[n // 2]:.2f} p75={vals[(3 * n) // 4]:.2f} max={vals[-1]:.2f}",
                file=sys.stderr,
            )
            task.write(build_script(task, cands, scores))


if __name__ == "__main__":
    main()

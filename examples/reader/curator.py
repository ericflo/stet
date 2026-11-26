"""A multilingual 0.6B language model reads and rates the corpus: Season 3 reference neural curator.

  1. Prefilter across ALL languages (never English-only) using lightweight line/script rules
     and exact deduplication across all available CPU cores (`task.map`), keeping whole
     documents so code, tables, lists and multilingual prose stay intact.
  2. Score candidate openings with Qwen/Qwen3-0.6B (pinned in stet.toml) on a 1-to-5
     educational/reasoning quality rubric by reading the next-token probability distribution
     over digits '1'..'5' (expected rating).
     - On NVIDIA Linux, uses `vllm.LLM` with automatic prefix caching for high throughput.
     - On AMD ROCm, macOS, or when `vllm` is absent, falls back cleanly to batched PyTorch
       `transformers` inference.
     - Never silently writes an unrated fallback script: writes periodic neural-scored
       checkpoints once scoring starts, and exits non-zero if the model fails to rate any
       documents.
  3. Fill the training token budget (with enough byte margin so the trainer does not wrap
     back to the start) with the highest-rated documents, ordered as a gentle curriculum:
     lower-scored kept documents earlier, highest-scored documents in the final LR cooldown
     window (the Season 3 trainer does not shuffle).

What to copy from it for your own neural curator:

  - Never drop non-English pages: the exam weighs every language by its share of the crawl.
  - Round model scores before sorting and break ties by document index so NVIDIA and AMD
    runs write consistent datasets.
  - Checkpoint neural scores periodically as `task.time_left()` shrinks, and fail loudly if
    the model cannot load rather than silently submitting an unrated fallback.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import sys
from collections import Counter

from stet.curator import Task

BYTES_PER_TOKEN = 3.75  # matches clean multilingual ERNIE density (~3.55 B/tok) so materialize finishes inside Band 2 without wrapping
HEAD_CHARS = 900  # excerpt read by the 0.6B quality rater
MAX_PROMPT_TOKENS = 288
RESERVE_SECONDS = 45
CANDIDATE_OVERPROVISION = 12.0  # rate up to 12x budget (all/most clean multilingual docs) within time_left()

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

ENDS = tuple(".!?\"'”’)»。！？…؟।")
SPACELESS = re.compile("[\u3040-\u30ff\u4e00-\u9fff\u0e00-\u0e7f]")

try:
    import py3langid as _langid
except Exception:  # pragma: no cover - minimal no-run env fallback
    _langid = None


def _detect_script_group(text: str) -> str:
    """Fast, zero-dependency language/script bucket used as fallback when py3langid is absent."""
    sample = text[:1200]
    counts: dict[str, int] = {}
    for ch in sample:
        o = ord(ch)
        if 0x0041 <= o <= 0x024F:
            counts["latin"] = counts.get("latin", 0) + 1
        elif 0x0400 <= o <= 0x052F:
            counts["cyrillic"] = counts.get("cyrillic", 0) + 1
        elif 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
            counts["arabic"] = counts.get("arabic", 0) + 1
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            counts["hangul"] = counts.get("hangul", 0) + 1
        elif 0x3040 <= o <= 0x30FF:
            counts["kana"] = counts.get("kana", 0) + 1
        elif 0x4E00 <= o <= 0x9FFF:
            counts["han"] = counts.get("han", 0) + 1
        elif 0x0900 <= o <= 0x097F:
            counts["devanagari"] = counts.get("devanagari", 0) + 1
        elif 0x0E00 <= o <= 0x0E7F:
            counts["thai"] = counts.get("thai", 0) + 1
        elif 0x0590 <= o <= 0x05FF:
            counts["hebrew"] = counts.get("hebrew", 0) + 1
        elif 0x0370 <= o <= 0x03FF:
            counts["greek"] = counts.get("greek", 0) + 1
    if not counts:
        return "other"
    # Japanese contains both kana and han; recognize kana presence as Japanese so it doesn't merge into Chinese.
    if counts.get("kana", 0) >= 8:
        return "ja"
    top = max(counts.items(), key=lambda kv: kv[1])[0]
    if top == "latin":
        low = " " + sample.lower() + " "
        if sum(low.count(w) for w in (" the ", " and ", " of ", " to ", " in ")) >= 4:
            return "en"
        return "latin_other"
    return top


def _detect_lang(text: str) -> str:
    """Fast per-language identifier (ISO 639-1) so languages sharing a script (fa/ar, uk/ru,
    pl/de/fr/es/en) are each quantile-normalized independently rather than starved by script peers."""
    if _langid is not None:
        try:
            lang, _ = _langid.classify(text[:800])
            if lang:
                return str(lang)
        except Exception:
            pass
    return _detect_script_group(text)


def _spaceless(text: str) -> bool:
    sample = text[:2000]
    return bool(sample) and len(SPACELESS.findall(sample)) > len(sample) * 0.3


def _units(text: str) -> list[str]:
    """Words, or character bigrams for spaceless scripts (CJK/Thai)."""
    if _spaceless(text):
        chars = "".join(text.split())
        return [chars[i : i + 2] for i in range(0, len(chars), 2)]
    return text.split()


def _quality_and_repetition(text: str, words: list[str], lines: list[str]) -> bool:
    """Multilingual Gopher + FineWeb quality, repetition, and line filters so the 0.6B neural
    rater focuses its GPU budget on clean prose/technical pages rather than nav menus or spam."""
    n_words = len(words)
    if not 50 <= n_words <= 100_000:
        return False
    mean_len = sum(map(len, words)) / n_words
    if not 1.5 <= mean_len <= 12.0:
        return False
    if (text.count("#") + text.count("...") + text.count("…")) / n_words > 0.1:
        return False
    n_lines = len(lines)
    if sum(ln.startswith(("•", "-", "*", "·")) for ln in lines) / n_lines > 0.9:
        return False
    if sum(1 for w in words if w[0].isalpha() or any(c.isalpha() for c in w)) / n_words < 0.6:
        return False
    if sum(ln.endswith(ENDS) for ln in lines) / n_lines < 0.12:
        return False
    if sum(len(ln) <= 30 for ln in lines) / n_lines > 0.67:
        return False
    # Repetition checks (Gopher)
    line_counts = Counter(lines)
    dup_lines = [ln for ln in lines if line_counts[ln] > 1]
    if len(dup_lines) / n_lines > 0.3 or sum(len(ln) for ln in dup_lines) / max(1, len(text)) > 0.2:
        return False
    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(paras) > 1 and sum(c - 1 for c in Counter(paras).values()) / len(paras) > 0.3:
        return False
    for n, most in ((2, 0.20), (3, 0.18), (4, 0.16)):
        if n_words > n:
            gram, count = Counter(zip(*(words[k:] for k in range(n)), strict=False)).most_common(1)[0]
            if count > 1 and count * sum(len(w) for w in gram) / max(1, len(text)) > most:
                return False
    return True


def _prefilter(i: int, text: str):
    """Multilingual quality + repetition + exact-dedup + language-id pass run across CPU cores."""
    if len(text) > 400_000:
        return None
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return None
    words = _units(text)
    if not words or not _quality_and_repetition(text, words, lines):
        return None
    raw = text.encode("utf-8", "replace")
    key = hashlib.blake2b(raw, digest_size=8).digest()
    short_frac = sum(len(ln) <= 30 for ln in lines) / len(lines)
    uniq_lines = len(set(lines)) / len(lines)
    prior = round(uniq_lines + min(1.0, len(raw) / 4000) - short_frac, 3)
    head = text[:HEAD_CHARS].strip()
    lang_grp = _detect_lang(text)
    return prior, key, len(raw), head, lang_grp


def collect_candidates(task: Task) -> list[tuple[float, int, int, str, str]]:
    """Return (prior, doc_id, size_bytes, excerpt, lang) deduplicated across all languages."""
    budget_bytes = task.train_tokens * BYTES_PER_TOKEN
    target_bytes = budget_bytes * CANDIDATE_OVERPROVISION
    seen: set[bytes] = set()
    kept: list[tuple[float, int, int, str, str]] = []
    for i, res in task.map(_prefilter):
        if task.time_left() < RESERVE_SECONDS * 2:
            break
        if res is None:
            continue
        prior, key, size, head, lang_grp = res
        if key in seen:
            continue
        seen.add(key)
        kept.append((prior, i, size, head, lang_grp))
    # Shuffle clean pages deterministically before any overprovision cap so capping never biases
    # against shorter-character scripts (CJK/Thai/etc.) when clean volume exceeds target_bytes.
    random.Random(0).shuffle(kept)
    out: list[tuple[float, int, int, str, str]] = []
    total = 0
    cap = max(target_bytes, 2_000_000)
    for item in kept:
        out.append(item)
        total += item[2]
        if total >= cap:
            break
    return out


def _cand_lang(rec: tuple) -> str:
    return rec[4] if len(rec) >= 5 else _detect_lang(rec[3])


def _quantile_normalize_by_lang(
    cands: list[tuple], scores: dict[int, float]
) -> dict[int, float]:
    """Convert raw 1-5 ratings into within-language percentile ranks [0, 1] so English-rubric
    score shifts in Qwen3-0.6B never starve Korean, Russian, Arabic, or Japanese during selection
    and cooldown curriculum ordering."""
    by_grp: dict[str, list[tuple[float, float, int]]] = {}
    all_items: list[tuple[float, float, int]] = []
    for rec in cands:
        prior, doc_id = rec[0], rec[1]
        if doc_id not in scores:
            continue
        lang_grp = _cand_lang(rec)
        item = (scores[doc_id], prior, doc_id)
        by_grp.setdefault(lang_grp, []).append(item)
        all_items.append(item)
    if not all_items:
        return {}
    # Global percentile fallback for very small buckets (<20 rated docs)
    all_items.sort(key=lambda t: (t[0], t[1], -t[2]))
    n_all = len(all_items)
    global_q = {doc_id: (idx + 0.5) / n_all for idx, (_, _, doc_id) in enumerate(all_items)}

    norm: dict[int, float] = {}
    for recs in by_grp.values():
        n = len(recs)
        if n < 20:
            for _, _, doc_id in recs:
                norm[doc_id] = round(global_q[doc_id], 4)
            continue
        recs.sort(key=lambda t: (t[0], t[1], -t[2]))
        for idx, (_, _, doc_id) in enumerate(recs):
            norm[doc_id] = round((idx + 0.5) / n, 4)
    return norm


def build_script(task: Task, cands: list[tuple], scores: dict[int, float]) -> list[int]:
    """Select documents up to the token budget; order higher-rated documents last (cooldown)."""
    budget_bytes = task.train_tokens * BYTES_PER_TOKEN
    if not scores:
        # Fallback before GPU scoring: shuffled high-prior multilingual docs.
        chosen: list[int] = []
        used = 0
        for rec in cands:
            doc_id, size = rec[1], rec[2]
            if used >= budget_bytes:
                break
            chosen.append(doc_id)
            used += size
        random.Random(0).shuffle(chosen)
        return chosen

    q_scores = _quantile_normalize_by_lang(cands, scores)
    ranked = sorted(
        [(q_scores[rec[1]], scores[rec[1]], rec[0], rec[1], rec[2], rec[3]) for rec in cands if rec[1] in q_scores],
        key=lambda t: (-t[0], -t[1], -t[2], t[3]),
    )
    selected: list[tuple[float, float, int]] = []
    seen_openings: set[bytes] = set()
    used = 0
    for q, _, prior, doc_id, size, head in ranked:
        if used >= budget_bytes:
            break
        # Drop templated near-duplicate pages sharing the same normalized opening (~40 words).
        op_words = head.lower().split()[:40]
        if len(op_words) >= 12:
            op_key = hashlib.blake2b(" ".join(op_words).encode("utf-8", "replace"), digest_size=8).digest()
            if op_key in seen_openings:
                continue
            seen_openings.add(op_key)
        selected.append((q, prior, doc_id))
        used += size
    if used < budget_bytes:
        picked = {doc_id for _, _, doc_id in selected}
        for rec in cands:
            prior, doc_id, size = rec[0], rec[1], rec[2]
            if used >= budget_bytes:
                break
            if doc_id not in picked:
                selected.append((0.5, prior, doc_id))
                used += size
    # Curriculum ordering: split into 3 quality bands (balanced across every language via per-language
    # quantile normalization) and shuffle inside each band. With BYTES_PER_TOKEN=3.75 vs ~3.55 B/tok
    # actual multilingual density, stet.script.materialize() finishes squarely inside the shuffled top
    # third (Band 2) at passes=1.000 — zero wrap back to Band 0 and zero elite-tail truncation.
    selected.sort(key=lambda t: (t[0], t[1], t[2]))
    n = len(selected)
    if n <= 8:
        return [doc_id for _, _, doc_id in selected]
    rng = random.Random(0)
    out: list[int] = []
    bands = 3
    for b in range(bands):
        lo = (b * n) // bands
        hi = ((b + 1) * n) // bands
        chunk = [doc_id for _, _, doc_id in selected[lo:hi]]
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


def _vllm_gpu_memory_utilization(allow_gb: float, total_gb: float, *, headroom_gb: float = 2.0, cap: float = 0.65) -> float:
    """Compute vLLM's gpu_memory_utilization from the sandbox allowance and physical VRAM.

    vLLM reserves `gpu_memory_utilization * total_device_vram`, so a fixed 0.65 requests 31.2 GiB
    on a 48 GiB GPU and exceeds Season 3's 22 GiB sandbox allowance. Capping by `(allow_gb - headroom)
    / total_gb` keeps vLLM comfortably inside the sandbox budget on every card size (24..80+ GiB).
    """
    if total_gb <= 0:
        return cap
    if allow_gb > 0:
        budget = max(1.0, min(allow_gb - headroom_gb, total_gb * cap))
        return round(max(0.1, min(cap, budget / total_gb)), 3)
    return cap


def _score_vllm(task: Task, where: str, cands: list[tuple[float, int, int, str, str]], scores: dict[int, float]) -> bool:
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
        try:
            total_gb = torch.cuda.mem_get_info()[1] / (1 << 30)
        except Exception:
            total_gb = 0.0
        util = _vllm_gpu_memory_utilization(float(getattr(task, "gpu_memory_gb", 0.0) or 0.0), total_gb)
        llm = LLM(
            model=where,
            dtype="bfloat16",
            max_model_len=MAX_PROMPT_TOKENS + 16,
            gpu_memory_utilization=util,
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
                {"prompt_token_ids": _build_prompt_ids(tok, rec[3], MAX_PROMPT_TOKENS, prefix_ids, suffix_ids)}
                for rec in part
            ]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for rec, out in zip(part, outputs, strict=True):
                doc_id = rec[1]
                step_lps = out.outputs[0].logprobs[0] if out.outputs and out.outputs[0].logprobs else {}
                lps = _extract_logprobs(step_lps)
                scores[doc_id] = _expected_from_logprobs(lps, digit_ids)
            if len(scores) % (batch_size * 10) == 0 or len(scores) == len(cands):
                task.write(build_script(task, cands, scores))
                print(f"vLLM rated {len(scores):,} / {len(cands):,}, {task.time_left():.0f}s left", file=sys.stderr)
        return bool(scores)
    except Exception as exc:
        print(f"vLLM unavailable ({exc}); falling back to transformers", file=sys.stderr)
        return False


def _score_transformers(task: Task, where: str, cands: list[tuple[float, int, int, str, str]], scores: dict[int, float]) -> None:
    """Portable batched PyTorch/transformers path (AMD ROCm, macOS, or NVIDIA fallback)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    remaining = [rec for rec in cands if rec[1] not in scores]
    if not remaining:
        return
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
        for start in range(0, len(remaining), chunk):
            part = remaining[start : start + chunk]
            enc = [_build_prompt_ids(tok, rec[3], MAX_PROMPT_TOKENS, prefix_ids, suffix_ids) for rec in part]
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
            if scores:
                task.write(build_script(task, cands, scores))
            print(f"rated {len(scores):,} / {len(cands):,}, {task.time_left():.0f}s left", file=sys.stderr)


def main():
    task = Task.from_env()
    cands = collect_candidates(task)
    print(f"{len(cands):,} candidates, {task.time_left():.0f}s left", file=sys.stderr)
    scores: dict[int, float] = {}
    where = str(task.asset("qwen3-0.6b"))
    if not _score_vllm(task, where, cands, scores):
        _score_transformers(task, where, cands, scores)
    if not scores:
        raise RuntimeError("reader model rated zero candidates; refusing to emit an unrated fallback script")
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

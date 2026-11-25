"""Fresh harvest-derived comprehension & reasoning synthesis (Season 3 Exam C v2).

From the round's harvested fresh documents (the same three hours of writing published the day
before the draw that feed twin-context comprehension), deterministically synthesizes three
orthogonal open-ended multi-sentence prose reasoning tasks in each document's own language:

  - explain   : why/how causal or mechanism explanation (2-3 sentences, own words)
  - infer     : multi-detail analytical implication or trade-off synthesis (2-3 sentences)
  - synthesis : declarative lead-in + 2-3 sentence prose synthesis (ZERO '?' or Q:/A: syntax)

Every item is generated with greedy decoding (temperature=0) by a pinned Apache-2.0 teacher
(`Qwen/Qwen3-8B`), wrapped in 6 diverse natural templates (no rigid `Question:/Answer:` cue),
and filtered deterministically to reject verbatim copying (>50% 5-gram overlap with the excerpt),
short/empty answers, and meta-chatter.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from importlib import resources

from ...spec import prf_u64
from ..items import Item
from . import Doc
from .select import SHARES

BAD_PREFIX = ("here is", "sure,", "based on the excerpt", "certainly", "below is")
CJK_LANGS = frozenset({"zh", "ja", "ko"})
TASK_TYPES = ("explain", "infer", "synthesis")

# Sources whose text is prose/discussion/documentation suitable for natural-language reasoning.
SKIP_LANG_PREFIXES = ("code:",)


def ngram_overlap(ans: str, src: str, n: int = 5) -> float:
    """Fraction of n-grams in `ans` that appear verbatim in `src`."""
    a_toks = re.findall(r"\w+", ans.lower())
    s_toks = re.findall(r"\w+", src.lower())
    if len(a_toks) < n or len(s_toks) < n:
        return 0.0
    s_set = {tuple(s_toks[i : i + n]) for i in range(len(s_toks) - n + 1)}
    a_ng = [tuple(a_toks[i : i + n]) for i in range(len(a_toks) - n + 1)]
    return sum(1 for g in a_ng if g in s_set) / max(1, len(a_ng))


def wrap_prompt(excerpt: str, prompt_line: str, task_type: str, wrapper_id: int) -> str:
    """Six diverse natural wrappers; `synthesis` has zero '?' and zero Q:/A: surface cues."""
    if task_type == "synthesis":
        if wrapper_id % 3 == 0:
            return f"{excerpt}\n\n{prompt_line}"
        if wrapper_id % 3 == 1:
            return f"{excerpt}\n\n---\n{prompt_line}"
        return f"{excerpt}\n\n{prompt_line} "
    w = wrapper_id % 6
    if w == 0:
        return f"{excerpt}\n\n{prompt_line}\n"
    if w == 1:
        return f"{excerpt}\n\n---\n{prompt_line}\n\n"
    if w == 2:
        return f"{prompt_line}\n\n{excerpt}\n\n"
    if w == 3:
        return f"{excerpt}\n\n{prompt_line} "
    if w == 4:
        return f"Context:\n{excerpt}\n\n{prompt_line}\n"
    return f"{excerpt}\n\n• {prompt_line}\n"


def _clean_angle(text: str) -> str:
    """Strip accidental literal `<...>` wrappers if the teacher echoed placeholder brackets."""
    s = text.strip()
    while s.startswith("<") and s.endswith(">") and len(s) >= 4:
        s = s[1:-1].strip()
    return s


def build_teacher_prompt(text: str, lang: str, task_type: str) -> str:
    txt = text[:1500]
    if task_type == "explain":
        inst = (
            f"Read the excerpt below (language: {lang}). Write ONE substantive why/how explanation prompt and a clear "
            f"2-to-3 sentence prose answer in the SAME language ({lang}). The answer must explain the cause, mechanism, or "
            f"reasoning in your own words (do NOT copy sentences verbatim; do NOT give a 1-word answer). "
            f"Do not add preamble or angle brackets. Output strictly:\nPROMPT: ...\nRESPONSE: ..."
        )
    elif task_type == "infer":
        inst = (
            f"Read the excerpt below (language: {lang}). Write ONE analytical prompt that requires synthesizing multiple "
            f"details or inferring a practical implication/trade-off from the excerpt, followed by a clear 2-to-3 sentence "
            f"prose response in the SAME language ({lang}). Paraphrase in natural prose (no verbatim copying, no meta-commentary, "
            f"no angle brackets). Output strictly:\nPROMPT: ...\nRESPONSE: ..."
        )
    else:  # synthesis
        inst = (
            f"Read the excerpt below (language: {lang}). Write ONE natural declarative lead-in sentence (NOT a question, "
            f"no question mark, no 'Q:'/'A:') that introduces a 2-to-3 sentence synthesis of the excerpt's main point "
            f"and underlying rationale, followed by that 2-to-3 sentence prose synthesis, all in the SAME language ({lang}). "
            f"Use your own words (no verbatim copying, no angle brackets). Output strictly:\nPROMPT: ...\nRESPONSE: ..."
        )
    return f"<|im_start|>user\n{inst}\n\nExcerpt:\n{txt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\nPROMPT:"


def select_source_docs(
    docs: list[Doc],
    seed: bytes,
    target: int = 640,
    shares: dict[str, float] | None = None,
) -> list[dict]:
    """Deterministically pick ~`target` diverse fresh harvest docs stratified by language & source."""
    shares = shares or SHARES
    by_lang: dict[str, list[Doc]] = defaultdict(list)
    seen_prefix: set[str] = set()
    # Deterministic order before filtering so dedup never depends on input order.
    ordered = sorted(docs, key=lambda d: (prf_u64(seed, f"reason-doc/{d.source}/{d.id}", 0), d.source, d.id))
    for d in ordered:
        lang = (d.lang or "en").strip()
        if not lang or lang.startswith(SKIP_LANG_PREFIXES):
            continue
        txt = (d.text or "").strip()
        if not (450 <= len(txt) <= 6000):
            continue
        key = txt[:160].lower()
        if key in seen_prefix:
            continue
        seen_prefix.add(key)
        by_lang[lang].append(d)

    if not by_lang:
        return []

    # Allocate quotas proportional to crawl share, capped so English takes at most ~38%
    # and every language with available docs gets meaningful coverage.
    langs = sorted(by_lang)
    raw_w = {lang: max(float(shares.get(lang, 0.25)), 0.25) for lang in langs}
    if "en" in raw_w and len(langs) > 1:
        others = sum(v for k, v in raw_w.items() if k != "en")
        raw_w["en"] = min(raw_w["en"], others * 0.62)
    tot_w = sum(raw_w.values())
    quotas = {lang: min(len(by_lang[lang]), max(12, round(target * raw_w[lang] / max(tot_w, 1e-9)))) for lang in langs}

    # If quotas exceed target, trim proportionally from largest pools; if under, top up from remaining.
    selected_docs: list[Doc] = []
    for lang in langs:
        pool = by_lang[lang]
        by_src: dict[str, list[Doc]] = defaultdict(list)
        for d in pool:
            by_src[d.source.split(":", 1)[0]].append(d)
        cap = quotas[lang]
        picked: list[Doc] = []
        src_keys = sorted(by_src)
        while len(picked) < cap and src_keys:
            nxt = []
            for sk in src_keys:
                if by_src[sk] and len(picked) < cap:
                    picked.append(by_src[sk].pop(0))
                if by_src[sk]:
                    nxt.append(sk)
            src_keys = nxt
        selected_docs.extend(picked)

    # Trim or keep in deterministic seeded order
    selected_docs.sort(key=lambda d: (prf_u64(seed, f"reason-pick/{d.source}/{d.id}", 0), d.source, d.id))
    selected_docs = selected_docs[:target]

    lang_idx: dict[str, int] = defaultdict(int)
    out = []
    for d in selected_docs:
        k = lang_idx[d.lang]
        lang_idx[d.lang] += 1
        out.append({
            "source": d.source,
            "id": d.id,
            "lang": d.lang,
            "text": d.text[:2400],
            "task_type": TASK_TYPES[k % 3],
            "wrapper_id": k % 6,
        })
    return out


def parse_generated_items(
    selected: list[dict],
    generations: list[str],
    *,
    max_ngram_overlap: float = 0.50,
) -> tuple[list[Item], dict[str, int]]:
    """Deterministically parse and filter raw teacher generations into `Item` objects."""
    items: list[Item] = []
    dropped: Counter[str] = Counter()
    for d, gen_txt in zip(selected, generations, strict=True):
        raw = ("PROMPT:" + (gen_txt or "")).strip()
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1].strip()
        if "RESPONSE:" not in raw:
            dropped["no_response_tag"] += 1
            continue
        p_part, r_part = raw.split("RESPONSE:", 1)
        pr = _clean_angle(p_part.replace("PROMPT:", "", 1).strip().strip('"').strip())
        resp = " ".join(ln.strip() for ln in r_part.strip().splitlines() if ln.strip()).strip()
        for stop_tok in ("<|im_end|>", "PROMPT:", "Excerpt:"):
            if stop_tok in resp:
                resp = resp.split(stop_tok, 1)[0].strip()
        resp = _clean_angle(resp)
        if not (12 <= len(pr) <= 350):
            dropped["prompt_len"] += 1
            continue
        is_cjk = d["lang"] in CJK_LANGS
        min_bytes = 48 if is_cjk else 70
        resp_b = len(resp.encode("utf-8"))
        if not (min_bytes <= resp_b <= 900):
            dropped["resp_len"] += 1
            continue
        if resp.lower().startswith(BAD_PREFIX):
            dropped["meta_chatter"] += 1
            continue
        if d["task_type"] == "synthesis" and ("?" in pr or "？" in pr or "؟" in pr):
            # Enforce zero-question syntax on synthesis lead-ins
            pr = pr.rstrip("?？؟").strip() + "."
        if not is_cjk and ngram_overlap(resp, d["text"], n=5) > max_ngram_overlap:
            dropped["verbatim_copy"] += 1
            continue
        ctx_str = wrap_prompt(d["text"][:1400], pr, d["task_type"], d["wrapper_id"])
        cont = resp if ctx_str.endswith((" ", "\n")) else f" {resp}"
        src_prefix = d["source"].split(":", 1)[0]
        items.append(
            Item(
                kind="answer",
                corner=f"{d['task_type']}/{d['lang']}/{src_prefix}",
                fmt="prose",
                lang=d["lang"],
                source=f"{d['task_type']}:{d['source']}",
                ctx=ctx_str,
                cont=cont,
            )
        )
    return items, dict(dropped)


def _generate_via_endpoint(prompts: list[str], url: str, model_name: str, log=print) -> list[str]:
    """Generate deterministically via an OpenAI-compatible `/v1/completions` server (e.g. vLLM)."""
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/completions"):
        endpoint = endpoint + ("/completions" if endpoint.endswith("/v1") else "/v1/completions")
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("STET_EXAMINER_LLM_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    outs: list[str] = []
    batch = 32
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        body = json.dumps({
            "model": model_name,
            "prompt": chunk,
            "temperature": 0.0,
            "max_tokens": 240,
            "stop": ["<|im_end|>", "\n\n\n"],
        }).encode()
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        choices = sorted(data["choices"], key=lambda c: int(c.get("index", 0)))
        outs.extend(c.get("text", "") for c in choices)
    log(f"  reasoning synthesis via {endpoint}: {len(outs)} drafts in {time.time() - t0:.1f}s")
    return outs


def _gen_shard_local(args_tuple):
    gpu_id, repo, rev, shard_prompts = args_tuple
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    order = sorted(range(len(shard_prompts)), key=lambda i: len(shard_prompts[i]))
    raw_outs = [""] * len(shard_prompts)
    bs = 16
    with torch.inference_mode():
        for b0 in range(0, len(order), bs):
            idxs = order[b0 : b0 + bs]
            enc = tok([shard_prompts[i] for i in idxs], return_tensors="pt", padding=True, truncation=True, max_length=1400).to(device)
            in_len = enc["input_ids"].shape[1]
            gen_ids = model.generate(**enc, max_new_tokens=240, do_sample=False, pad_token_id=tok.pad_token_id)
            for row, orig_i in enumerate(idxs):
                raw_outs[orig_i] = tok.decode(gen_ids[row, in_len:], skip_special_tokens=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return raw_outs


def _generate_local_gpu(prompts: list[str], repo: str, rev: str, log=print) -> list[str]:
    """Deterministic greedy generation on local CUDA GPU(s) via pinned `transformers` SDPA."""
    import multiprocessing as mp

    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError(
            "synthesizing Season 3 reasoning exam items requires a CUDA GPU (16 GB+) "
            "or STET_EXAMINER_LLM_URL pointing at a deterministic Qwen3-8B server"
        )
    t0 = time.time()
    snapshot_download(repo, revision=rev)
    n_gpus = min(4, torch.cuda.device_count())
    if n_gpus <= 1:
        outs = _gen_shard_local((0, repo, rev, prompts))
    else:
        shards = [(g, repo, rev, prompts[g::n_gpus]) for g in range(n_gpus)]
        with mp.get_context("spawn").Pool(n_gpus) as pool:
            shard_outs = pool.map(_gen_shard_local, shards)
        outs = [""] * len(prompts)
        for g, s_out in enumerate(shard_outs):
            for k, txt in enumerate(s_out):
                outs[g + k * n_gpus] = txt
    log(f"  reasoning synthesis ({repo}@{rev[:8]} on {n_gpus} GPU): {len(outs)} drafts in {time.time() - t0:.1f}s")
    return outs


def synthesize_reasoning(
    docs: list[Doc],
    seed: bytes,
    cfg: dict,
    *,
    shares: dict[str, float] | None = None,
    log=print,
) -> list[Item]:
    """Synthesize fresh multilingual `explain`, `infer`, and `synthesis` items from harvested docs."""
    target = int(cfg.get("items", 640))
    max_ov = float(cfg.get("max_ngram_overlap", 0.50))
    repo = str(cfg.get("model", "Qwen/Qwen3-8B"))
    rev = str(cfg.get("revision", "b968826d9c46dd6066d109eabc6255188de91218"))

    selected = select_source_docs(docs, seed, target=target, shares=shares)
    if not selected:
        return []
    prompts = [build_teacher_prompt(d["text"], d["lang"], d["task_type"]) for d in selected]
    if url := os.environ.get("STET_EXAMINER_LLM_URL"):
        gens = _generate_via_endpoint(prompts, url, os.environ.get("STET_EXAMINER_LLM_MODEL") or repo, log=log)
    else:
        gens = _generate_local_gpu(prompts, repo, rev, log=log)
    items, dropped = parse_generated_items(selected, gens, max_ngram_overlap=max_ov)
    counts = Counter(it.source.split(":", 1)[0] for it in items)
    log(f"  reasoning items: {len(items)} kept of {len(selected)} ({dict(counts)}, dropped={dropped})")
    return items


def can_synthesize_locally() -> bool:
    """Whether this machine can synthesize fresh reasoning items (GPU or configured endpoint)."""
    if os.environ.get("STET_EXAMINER_LLM_URL"):
        return True
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.mem_get_info()[1] >= 18 * (1 << 30))
    except Exception:
        return False


def load_practice_pack(season, seed: bytes) -> list[Item]:
    """Load the bundled openly-licensed (CC-BY-SA Wikipedia) offline practice pack for practice/gym rounds.

    Official rounds NEVER call this: official exams are always freshly harvested and synthesized
    at the round's draw.
    """
    import zstandard

    cfg = (season.raw.get("exam") or {}).get("practice_pack") or {}
    if not cfg or not cfg.get("file"):
        return []
    blob = resources.files("stet.exam").joinpath("data", cfg["file"]).read_bytes()
    if cfg.get("sha256") and hashlib.sha256(blob).hexdigest() != cfg["sha256"]:
        raise RuntimeError(f"practice pack {cfg['file']} does not match the season's sha256")
    raw = zstandard.ZstdDecompressor().decompress(blob, max_output_size=64 << 20).decode("utf-8")
    items = [Item.from_dict(json.loads(ln)) for ln in raw.split("\n") if ln]
    # Deterministic per-round ordering from seed so every machine sees the exact same order.
    items.sort(key=lambda it: (prf_u64(seed, f"practice/{it.id}", 0), it.id))
    return items

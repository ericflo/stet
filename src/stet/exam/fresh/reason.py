"""Fresh harvest-derived comprehension & reasoning synthesis (Season 3 Exam C v2).

From the round's harvested fresh documents (the same three hours of writing published the day
before the draw that feed twin-context comprehension), writes three orthogonal open-ended
multi-sentence prose reasoning tasks in each document's own language:

  - explain   : why/how causal or mechanism explanation (2-3 sentences, own words)
  - infer     : multi-detail analytical implication or trade-off synthesis (2-3 sentences)
  - synthesis : declarative lead-in + 2-3 sentence prose synthesis (ZERO '?' or Q:/A: syntax)

Every item is written once at the draw by the pinned Apache-2.0 teacher (`Qwen/Qwen3-8B` at
greedy `temperature = 0`), receives one shared token-bounded excerpt given identically to both
teacher and student (reserving headroom so the teacher's closing `PROMPT: ` cue is never
truncated), wraps prompts across diverse natural templates (independent of task type), preserves
the immutable harvest document ID (`<task>:<source>:<doc_id>`), checks response language and
sentence completion, filters verbatim copying from both excerpt and lead-in across whitespace and
CJK/spaceless scripts, and publishes the teacher's lead-in and answer for every item at settlement.
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
ECHO_PROMPT_PREFIX = (
    "write a ",
    "write one ",
    "write two ",
    "read the excerpt",
    "prompt:",
    "response:",
    "output strictly",
)
CJK_LANGS = frozenset({"zh", "ja", "ko"})
TASK_TYPES = ("explain", "infer", "synthesis")

DEFAULT_MAX_EXCERPT_TOKENS = 380
DEFAULT_MAX_EXCERPT_CHARS = 1200
DEFAULT_MAX_NEW_TOKENS = 320
DEFAULT_MIN_KEEP_RATE = 0.50
TEACHER_MAX_PROMPT_TOKENS = 1400

# Sources whose text is prose/discussion/documentation suitable for natural-language reasoning.
SKIP_LANG_PREFIXES = ("code:",)

_SPACELESS_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u1100-\u11ff"
    r"\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff]"
)
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?。！？…؟۔।][\"'”’»)\]}」』】]*$")

# Close language equivalence clusters accepted by the response language verifier.
_LANG_CLUSTERS = (
    frozenset({"no", "nb", "nn"}),
    frozenset({"hr", "sr", "bs", "sh"}),
    frozenset({"id", "ms"}),
    frozenset({"zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant"}),
    frozenset({"pt", "gl"}),
)

try:
    import py3langid as _langid

    _LANGID_SUPPORTED = frozenset(lang for lang, _ in _langid.rank("hello"))
except Exception:  # pragma: no cover - optional dependency outside stet[run]
    _langid = None
    _LANGID_SUPPORTED = frozenset()


def _norm_compact(text: str) -> str:
    return _NON_WORD_RE.sub("", text.lower())


def _is_spaceless_script(text: str) -> bool:
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return False
    hits = len(_SPACELESS_RE.findall(text))
    return hits / len(non_ws) >= 0.25


def _dominant_script(text: str) -> str:
    """Coarse dominant writing script for zero-dependency language/script sanity checks."""
    counts: dict[str, int] = {}
    for ch in text[:1500]:
        o = ord(ch)
        if 0x0041 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF:
            s = "latin"
        elif 0x0400 <= o <= 0x052F:
            s = "cyrillic"
        elif 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
            s = "arabic"
        elif 0x0900 <= o <= 0x097F:
            s = "devanagari"
        elif 0x0980 <= o <= 0x09FF:
            s = "bengali"
        elif 0x0E00 <= o <= 0x0E7F:
            s = "thai"
        elif 0x0590 <= o <= 0x05FF:
            s = "hebrew"
        elif 0x0370 <= o <= 0x03FF:
            s = "greek"
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            s = "hangul"
        elif 0x3040 <= o <= 0x30FF:
            s = "kana"
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            s = "han"
        elif 0x10A0 <= o <= 0x10FF:
            s = "georgian"
        elif 0x0530 <= o <= 0x058F:
            s = "armenian"
        else:
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return ""
    if counts.get("kana", 0) >= 6:
        return "ja"
    top, n = max(counts.items(), key=lambda kv: kv[1])
    if n < 8:
        return ""
    return top


def _langs_compatible(want: str, got: str) -> bool:
    w = want.lower().split("-", 1)[0]
    g = got.lower().split("-", 1)[0]
    if w == g:
        return True
    return any(w in cl and g in cl for cl in _LANG_CLUSTERS)


def _non_latin_scripts(text: str) -> dict[str, int]:
    """Count characters per non-Latin script (for detecting cross-script code-switching leaks)."""
    counts: dict[str, int] = {}
    for ch in text[:2000]:
        o = ord(ch)
        if 0x0400 <= o <= 0x052F:
            s = "cyrillic"
        elif 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
            s = "arabic"
        elif 0x0900 <= o <= 0x097F:
            s = "devanagari"
        elif 0x0980 <= o <= 0x09FF:
            s = "bengali"
        elif 0x0E00 <= o <= 0x0E7F:
            s = "thai"
        elif 0x0590 <= o <= 0x05FF:
            s = "hebrew"
        elif 0x0370 <= o <= 0x03FF:
            s = "greek"
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            s = "hangul"
        elif 0x3040 <= o <= 0x30FF:
            s = "kana"
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            s = "han"
        elif 0x10A0 <= o <= 0x10FF:
            s = "georgian"
        elif 0x0530 <= o <= 0x058F:
            s = "armenian"
        else:
            continue
        counts[s] = counts.get(s, 0) + 1
    return counts


def _has_foreign_script(text: str, excerpt: str, min_chars: int = 2) -> bool:
    """Return True if `text` introduces >= `min_chars` characters of a non-Latin script absent
    from `excerpt` (treating Han and Kana as compatible for Japanese)."""
    ex_scripts = _non_latin_scripts(excerpt)
    txt_scripts = _non_latin_scripts(text)
    for script, n in txt_scripts.items():
        if n < min_chars:
            continue
        if script in ex_scripts:
            continue
        if script == "han" and "kana" in ex_scripts:
            continue
        if script == "kana" and "han" in ex_scripts:
            continue
        return True
    return False


def _text_matches_language(
    text: str,
    excerpt: str,
    lang: str,
    *,
    min_langid_chars: int = 35,
    min_en_chars: int = 35,
) -> bool:
    """Verify a teacher-written prompt or answer stays in the excerpt's language/script and
    does not leak foreign non-Latin scripts (e.g., Thai or Han tokens in Arabic/Hebrew/Russian)."""
    if _has_foreign_script(text, excerpt):
        return False
    ex_script = _dominant_script(excerpt)
    tx_script = _dominant_script(text)
    if ex_script and tx_script:
        if ex_script == "ja":
            if tx_script not in ("ja", "han"):
                return False
        elif ex_script == "han":
            if tx_script not in ("han", "ja"):
                return False
        elif ex_script != tx_script:
            return False

    if _langid is not None and len(text) >= min(min_langid_chars, min_en_chars):
        try:
            det, _ = _langid.classify(text)
        except Exception:
            det = ""
        if det:
            base_lang = lang.lower().split("-", 1)[0]
            if len(text) >= min_en_chars and base_lang != "en" and det == "en":
                return False
            if len(text) >= min_langid_chars and base_lang in _LANGID_SUPPORTED and not _langs_compatible(base_lang, det):
                return False
    return True


def _prompt_matches_language(prompt: str, excerpt: str, lang: str) -> bool:
    """Verify the teacher's lead-in prompt stays in the excerpt's script/language without code-switching."""
    return _text_matches_language(prompt, excerpt, lang, min_langid_chars=60, min_en_chars=25)


def _response_matches_language(resp: str, excerpt: str, lang: str) -> bool:
    """Verify the teacher's answer stays in the excerpt's language/script (catches English,
    neighboring-language fallback, or mid-sentence foreign-script leaks)."""
    return _text_matches_language(resp, excerpt, lang, min_langid_chars=35, min_en_chars=35)


def _char_shingle_overlap(ans_compact: str, src_compact: str, k: int) -> float:
    if not ans_compact or not src_compact:
        return 0.0
    if len(ans_compact) < k:
        return 1.0 if ans_compact in src_compact else 0.0
    if len(src_compact) < k:
        return 0.0
    s_set = {src_compact[i : i + k] for i in range(len(src_compact) - k + 1)}
    a_ng = [ans_compact[i : i + k] for i in range(len(ans_compact) - k + 1)]
    return sum(1 for g in a_ng if g in s_set) / max(1, len(a_ng))


def ngram_overlap(ans: str, src: str, n: int = 5, *, cjk_char_n: int = 8, fallback_char_n: int = 12) -> float:
    """Fraction of n-grams in `ans` that appear verbatim in `src`.

    Uses script-appropriate shingles:
      - CJK / spaceless scripts (Chinese, Japanese, Korean, Thai, etc.): character `cjk_char_n`-grams
        over normalized non-punctuation characters (default 8 chars ≈ 4–5 words).
      - Whitespace-delimited scripts with >= `n` word tokens: word `n`-grams, plus a direct
        normalized-substring check so copying a short clause verbatim cannot slip through.
      - Short answers with fewer than `n` word tokens in any script: character `fallback_char_n`-grams
        (or exact substring containment when shorter than `fallback_char_n`).
    """
    a_compact = _norm_compact(ans)
    s_compact = _norm_compact(src)
    if not a_compact or not s_compact:
        return 0.0
    if a_compact in s_compact:
        return 1.0
    if _is_spaceless_script(ans):
        return _char_shingle_overlap(a_compact, s_compact, cjk_char_n)
    a_toks = re.findall(r"\w+", ans.lower())
    s_toks = re.findall(r"\w+", src.lower())
    if len(a_toks) < n or len(s_toks) < n:
        return _char_shingle_overlap(a_compact, s_compact, fallback_char_n)
    s_set = {tuple(s_toks[i : i + n]) for i in range(len(s_toks) - n + 1)}
    a_ng = [tuple(a_toks[i : i + n]) for i in range(len(a_toks) - n + 1)]
    return sum(1 for g in a_ng if g in s_set) / max(1, len(a_ng))


def bound_excerpt(
    text: str,
    *,
    tok=None,
    max_tokens: int = DEFAULT_MAX_EXCERPT_TOKENS,
    max_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
) -> str:
    """Build ONE bounded excerpt string given identically to BOTH teacher and student.

    First bounds by `max_chars` (preferring a whitespace boundary when present), then—when the
    pinned teacher tokenizer `tok` is supplied—caps at `max_tokens` teacher tokens so even dense
    non-Latin/CJK/Thai scripts stay well below `TEACHER_MAX_PROMPT_TOKENS` with full headroom
    reserved for the instruction and closing `PROMPT: ` cue.
    """
    s = (text or "").strip()
    if len(s) > max_chars:
        cut = s.rfind(" ", 0, max_chars)
        s = s[: cut if cut > max_chars // 2 else max_chars].strip()
    if tok is not None and max_tokens > 0 and s:
        if hasattr(tok, "encode"):
            try:
                enc = tok.encode(s, add_special_tokens=False)
                ids = enc.ids if hasattr(enc, "ids") else list(enc)
            except TypeError:
                enc = tok.encode(s)
                ids = enc.ids if hasattr(enc, "ids") else list(enc)
            if len(ids) > max_tokens:
                ids = ids[:max_tokens]
                try:
                    s = tok.decode(ids, skip_special_tokens=True).strip()
                except TypeError:
                    s = tok.decode(ids).strip()
    return s


def wrap_prompt(excerpt: str, prompt_line: str, task_type: str, wrapper_id: int) -> str:
    """Diverse natural wrappers; `synthesis` uses 3 zero-'?' declarative layouts."""
    if task_type == "synthesis":
        w = wrapper_id % 3
        if w == 0:
            return f"{excerpt}\n\n{prompt_line}"
        if w == 1:
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


def split_wrapped_ctx(ctx: str, task_type: str) -> tuple[str, str, int] | None:
    """Invert `wrap_prompt()` cleanly so `public_items()` can publish the teacher's prompt line
    and wrapper_id while withholding only a third-party source excerpt."""
    if task_type == "synthesis":
        if "\n\n---\n" in ctx:
            ex, pr = ctx.rsplit("\n\n---\n", 1)
            w_id = 1
        elif ctx.endswith(" ") and "\n\n" in ctx[:-1]:
            ex, pr = ctx[:-1].rsplit("\n\n", 1)
            w_id = 2
        elif "\n\n" in ctx:
            ex, pr = ctx.rsplit("\n\n", 1)
            w_id = 0
        else:
            return None
    else:
        if ctx.startswith("Context:\n") and ctx.endswith("\n") and "\n\n" in ctx[9:-1]:
            ex, pr = ctx[9:-1].rsplit("\n\n", 1)
            w_id = 4
        elif "\n\n---\n" in ctx and ctx.endswith("\n\n"):
            ex, pr = ctx[:-2].rsplit("\n\n---\n", 1)
            w_id = 1
        elif "\n\n• " in ctx and ctx.endswith("\n"):
            ex, pr = ctx[:-1].rsplit("\n\n• ", 1)
            w_id = 5
        elif ctx.endswith(" ") and "\n\n" in ctx[:-1]:
            ex, pr = ctx[:-1].rsplit("\n\n", 1)
            w_id = 3
        elif ctx.endswith("\n\n") and "\n\n" in ctx[:-2]:
            pr, ex = ctx[:-2].split("\n\n", 1)
            w_id = 2
        elif ctx.endswith("\n") and "\n\n" in ctx[:-1]:
            ex, pr = ctx[:-1].rsplit("\n\n", 1)
            w_id = 0
        else:
            return None
    if wrap_prompt(ex, pr, task_type, w_id) != ctx:
        return None
    return ex, pr, w_id


def _clean_angle(text: str) -> str:
    """Strip accidental literal `<...>` wrappers if the teacher echoed placeholder brackets."""
    s = text.strip()
    while s.startswith("<") and s.endswith(">") and len(s) >= 4:
        s = s[1:-1].strip()
    return s


def build_teacher_prompt(excerpt: str, lang: str, task_type: str) -> str:
    """Build the teacher prompt from the already-bounded shared `excerpt`.

    Uses explicit two-line field descriptions with ZERO literal `...` placeholders (so the
    teacher never echoes `PROMPT: ...`) and ends with `PROMPT: ` to begin generation directly.
    """
    if task_type == "explain":
        inst = (
            f"Read the excerpt below (language: {lang}). Write two lines in the SAME language ({lang}). "
            f"Line 1 must start with 'PROMPT: ' followed by one substantive why/how explanation question about the excerpt. "
            f"Line 2 must start with 'RESPONSE: ' followed by a complete 2-to-3 sentence prose answer explaining the cause, "
            f"mechanism, or reasoning in your own words (no verbatim copying from the excerpt or the question, no ellipses "
            f"placeholders, no angle brackets, and finish the last sentence with proper punctuation)."
        )
    elif task_type == "infer":
        inst = (
            f"Read the excerpt below (language: {lang}). Write two lines in the SAME language ({lang}). "
            f"Line 1 must start with 'PROMPT: ' followed by one analytical question requiring synthesis of multiple details "
            f"or inferring a practical implication or trade-off from the excerpt. "
            f"Line 2 must start with 'RESPONSE: ' followed by a complete 2-to-3 sentence prose answer in your own words "
            f"(no verbatim copying from the excerpt or the question, no ellipses placeholders, no angle brackets, and finish "
            f"the last sentence with proper punctuation)."
        )
    else:  # synthesis
        inst = (
            f"Read the excerpt below (language: {lang}). Write two lines in the SAME language ({lang}). "
            f"Line 1 must start with 'PROMPT: ' followed by one concrete declarative topic sentence (10 to 25 words, NOT a "
            f"question, no question mark, no Q/A labels, never echoing these instructions) that introduces the subject of the "
            f"excerpt. Line 2 must start with 'RESPONSE: ' followed by a complete 2-to-3 sentence prose synthesis explaining "
            f"the excerpt's main point and underlying rationale in your own words (do not repeat the topic sentence verbatim, "
            f"do not copy sentences from the excerpt, never write placeholder ellipses or angle brackets, and finish the last "
            f"sentence with proper punctuation)."
        )
    return f"<|im_start|>user\n{inst}\n\nExcerpt:\n{excerpt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\nPROMPT: "


def _source_bucket(source: str) -> str:
    """Group Wikipedia variants into one round-robin bucket so Wikipedia does not take 3x turns
    against non-Wikipedia sources when other sources are present in the same language."""
    base = source.split(":", 1)[0]
    if base.startswith("wikipedia"):
        return "wikipedia"
    return base


def select_source_docs(
    docs: list[Doc],
    seed: bytes,
    target: int = 640,
    shares: dict[str, float] | None = None,
    *,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
    min_lang_share: float = 0.05,
    major_lang_share: float = 1.0,
    major_lang_floor: int = 27,
) -> list[dict]:
    """Deterministically pick ~`target` fresh harvest docs aligned with the round's corpus shares.

    - Allocates quotas strictly in proportion to crawl shares (English capped at <=38%), never
      inflating tiny micro-languages with a fixed floor.
    - Ensures every major crawl language (`share >= 1.0%`) with available docs receives at least
      `major_lang_floor` (27 = 9 per reasoning kind) docs so each major language clears `SMALL=8`
      per kind rather than falling into the pooled small-language bucket.
    - Round-robins across source families with Wikipedia variants grouped as one bucket.
    """
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
        min_len = 180 if lang in CJK_LANGS else 450
        if not (min_len <= len(txt) <= 6000) or "\ufffd" in txt:
            continue
        key = txt[:160].lower()
        if key in seen_prefix:
            continue
        seen_prefix.add(key)
        by_lang[lang].append(d)

    if not by_lang:
        return []

    # Exclude micro-languages (< min_lang_share% of the crawl) when larger languages are available,
    # so teacher slots go to real crawl languages rather than 1-doc Wikipedia micro-editions that
    # the teacher often answers in English or a neighboring language.
    eligible = [lang for lang in sorted(by_lang) if float(shares.get(lang, 0.0)) >= min_lang_share]
    if not eligible or sum(len(by_lang[lang]) for lang in eligible) < min(target, 30):
        eligible = sorted(by_lang)

    raw_w = {lang: max(float(shares.get(lang, 0.05)), 0.01) for lang in eligible}
    if "en" in raw_w and len(eligible) > 1:
        others = sum(v for k, v in raw_w.items() if k != "en")
        raw_w["en"] = min(raw_w["en"], others * 0.62)  # <= ~38.3% English

    # Exact crawl-share quota allocation with a `major_lang_floor` (27 = 9/kind) floor for major
    # crawl languages (`share >= 1.0%`), filled iteratively to `target` without truncating majors.
    quotas: dict[str, int] = {lang: 0 for lang in eligible}
    cap_of = {lang: len(by_lang[lang]) for lang in eligible}
    total_avail = sum(cap_of.values())
    want_total = min(target, total_avail)

    # Pass 1: reserve `major_lang_floor` for major languages (ordered by crawl share desc),
    # scaling the floor down cleanly if `want_total` is smaller than `len(majors) * floor`.
    majors = sorted(
        [lang for lang in eligible if float(shares.get(lang, 0.0)) >= major_lang_share],
        key=lambda k: (-float(shares.get(k, 0.0)), k),
    )
    eff_floor = min(major_lang_floor, want_total // max(1, len(majors))) if majors else 0
    used = 0
    for lang in majors:
        give = min(cap_of[lang], eff_floor, want_total - used)
        quotas[lang] = give
        used += give

    # Pass 2: distribute remaining budget proportionally to crawl share across all eligible languages.
    while used < want_total:
        rem_langs = [lang for lang in eligible if quotas[lang] < cap_of[lang]]
        if not rem_langs:
            break
        rem_need = want_total - used
        rem_w = sum(raw_w[lang] for lang in rem_langs)
        added = 0
        fracs: list[tuple[float, str]] = []
        for lang in rem_langs:
            exact = rem_need * raw_w[lang] / max(rem_w, 1e-12)
            whole = min(cap_of[lang] - quotas[lang], int(exact))
            quotas[lang] += whole
            added += whole
            if quotas[lang] < cap_of[lang]:
                fracs.append((exact - whole, lang))
        used += added
        if used < want_total and fracs:
            fracs.sort(key=lambda t: (-t[0], t[1]))
            for _, lang in fracs:
                if used >= want_total:
                    break
                if quotas[lang] < cap_of[lang]:
                    quotas[lang] += 1
                    used += 1
        elif added == 0:
            break

    selected_docs: list[Doc] = []
    for lang in eligible:
        cap = quotas[lang]
        if cap <= 0:
            continue
        by_src: dict[str, list[Doc]] = defaultdict(list)
        for d in by_lang[lang]:
            by_src[_source_bucket(d.source)].append(d)
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

    # Deterministic order across all selected docs (already bounded to <= target).
    selected_docs.sort(key=lambda d: (prf_u64(seed, f"reason-pick/{d.source}/{d.id}", 0), d.source, d.id))

    lang_idx: dict[str, int] = defaultdict(int)
    task_idx: dict[tuple[str, str], int] = defaultdict(int)
    out = []
    for d in selected_docs:
        k = lang_idx[d.lang]
        lang_idx[d.lang] += 1
        # Seeded per-language starting offset so smaller languages spread evenly across
        # explain, infer, and synthesis instead of piling up in explain.
        offset = prf_u64(seed, f"reason-task-off/{d.lang}", 0) % len(TASK_TYPES)
        ttype = TASK_TYPES[(offset + k) % len(TASK_TYPES)]
        # Decouple wrapper assignment from task assignment so every task type (including
        # `synthesis`, which wraps mod 3) cycles evenly across all of its wrappers.
        w_id = task_idx[(d.lang, ttype)] % 6
        task_idx[(d.lang, ttype)] += 1
        excerpt = bound_excerpt(d.text, max_chars=max_excerpt_chars)
        out.append({
            "source": d.source,
            "id": d.id,
            "lang": d.lang,
            "text": excerpt,
            "excerpt": excerpt,
            "task_type": ttype,
            "wrapper_id": w_id,
        })
    return out


def apply_token_budget(
    selected: list[dict],
    tok,
    *,
    max_tokens: int = DEFAULT_MAX_EXCERPT_TOKENS,
    max_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
) -> list[dict]:
    """Apply the pinned teacher tokenizer's token cap to every selected doc's shared excerpt."""
    out = []
    for d in selected:
        ex = bound_excerpt(d.get("excerpt", d["text"]), tok=tok, max_tokens=max_tokens, max_chars=max_chars)
        nd = dict(d)
        nd["excerpt"] = ex
        nd["text"] = ex
        out.append(nd)
    return out


class FatalExamError(RuntimeError):
    """A deterministic exam-synthesis failure that retrying the same inputs will not fix."""


def parse_generated_items_detailed(
    selected: list[dict],
    generations: list[str],
    *,
    max_ngram_overlap: float = 0.50,
    check_lang: bool = True,
) -> tuple[list[Item], dict[str, int], list[dict]]:
    """Deterministically parse and filter raw teacher generations into `Item` objects,
    returning `(kept_items, drop_counts, drop_records)` so every dropped draft is auditable."""
    items: list[Item] = []
    dropped: Counter[str] = Counter()
    drop_records: list[dict] = []

    def _drop(reason: str, d: dict, raw_gen: str, pr: str = "", resp: str = "", excerpt: str = "") -> None:
        dropped[reason] += 1
        drop_records.append({
            "reason": reason,
            "task_type": d.get("task_type", ""),
            "source": d.get("source", ""),
            "id": str(d.get("id") or ""),
            "lang": d.get("lang", ""),
            "prompt": pr,
            "response": resp,
            "raw": (raw_gen or "")[:1200],
            "excerpt": (excerpt or d.get("excerpt") or "")[:400],
        })

    for d, gen_txt in zip(selected, generations, strict=True):
        raw = ("PROMPT: " + (gen_txt or "")).strip()
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1].strip()
        if "RESPONSE:" not in raw:
            _drop("no_response_tag", d, gen_txt)
            continue
        p_part, r_part = raw.split("RESPONSE:", 1)
        pr_raw = p_part.replace("PROMPT:", "", 1).strip().strip('"').strip()
        pr = " ".join(ln.strip() for ln in pr_raw.splitlines() if ln.strip()).strip()
        resp = " ".join(ln.strip() for ln in r_part.strip().splitlines() if ln.strip()).strip()
        for stop_tok in ("<|im_end|>", "PROMPT:", "Excerpt:"):
            if stop_tok in resp:
                resp = resp.split(stop_tok, 1)[0].strip()
        pr = _clean_angle(pr)
        resp = _clean_angle(resp)
        if "\ufffd" in pr or "\ufffd" in resp or "\ufffd" in d.get("excerpt", ""):
            _drop("broken_utf8", d, gen_txt, pr, resp)
            continue
        is_cjk = d["lang"] in CJK_LANGS
        min_pr = 6 if is_cjk else 10
        if (
            not (min_pr <= len(pr) <= 450)
            or not pr.strip(" .…-:")
            or pr.startswith("...")
            or pr.lower().startswith(ECHO_PROMPT_PREFIX)
        ):
            _drop("prompt_len", d, gen_txt, pr, resp)
            continue
        min_bytes = 48 if is_cjk else 70
        resp_b = len(resp.encode("utf-8"))
        if not (min_bytes <= resp_b <= 1200):
            _drop("resp_len", d, gen_txt, pr, resp)
            continue
        if resp.lower().startswith(BAD_PREFIX):
            _drop("meta_chatter", d, gen_txt, pr, resp)
            continue
        if not _SENTENCE_END_RE.search(resp):
            _drop("truncated_sentence", d, gen_txt, pr, resp)
            continue
        if d["task_type"] == "synthesis" and ("?" in pr or "？" in pr or "؟" in pr):
            # Enforce zero-question syntax on synthesis lead-ins
            pr = pr.rstrip("?？؟").strip() + "."
        excerpt = d.get("excerpt") or bound_excerpt(d["text"])
        if check_lang:
            if not _prompt_matches_language(pr, excerpt, d["lang"]):
                _drop("wrong_prompt_lang", d, gen_txt, pr, resp, excerpt)
                continue
            if not _response_matches_language(resp, excerpt, d["lang"]):
                _drop("wrong_lang", d, gen_txt, pr, resp, excerpt)
                continue
        if ngram_overlap(resp, excerpt, n=5) > max_ngram_overlap:
            _drop("verbatim_copy", d, gen_txt, pr, resp, excerpt)
            continue
        if ngram_overlap(resp, pr, n=5) > max_ngram_overlap:
            _drop("prompt_copy", d, gen_txt, pr, resp, excerpt)
            continue
        ctx_str = wrap_prompt(excerpt, pr, d["task_type"], d["wrapper_id"])
        cont = resp if ctx_str.endswith((" ", "\n")) else f" {resp}"
        src_prefix = d["source"].split(":", 1)[0]
        doc_id = str(d.get("id") or "").strip()
        # Preserve the immutable harvest document ID (matching fresh/select.py's `<source>:<id>`)
        # so withheld reasoning items remain auditable under the published rules.
        full_src = f"{d['task_type']}:{d['source']}:{doc_id}" if doc_id else f"{d['task_type']}:{d['source']}"
        items.append(
            Item(
                kind="answer",
                corner=f"{d['task_type']}/{d['lang']}/{src_prefix}",
                fmt="prose",
                lang=d["lang"],
                source=full_src,
                ctx=ctx_str,
                cont=cont,
            )
        )
    return items, dict(dropped), drop_records


def parse_generated_items(
    selected: list[dict],
    generations: list[str],
    *,
    max_ngram_overlap: float = 0.50,
    check_lang: bool = True,
) -> tuple[list[Item], dict[str, int]]:
    """Deterministically parse and filter raw teacher generations into `Item` objects."""
    items, dropped, _ = parse_generated_items_detailed(
        selected, generations, max_ngram_overlap=max_ngram_overlap, check_lang=check_lang
    )
    return items, dropped


def _load_teacher_tokenizer(repo: str, rev: str, *, required: bool = True):
    """Load the pinned teacher tokenizer (without requiring PyTorch) for exact shared excerpt bounding."""
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tok_path = hf_hub_download(repo, "tokenizer.json", revision=rev)
        return Tokenizer.from_file(tok_path)
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"could not load pinned reasoning teacher tokenizer {repo}@{rev}: {exc}"
            ) from exc
        return None


def _verify_endpoint_model(url: str, repo: str, rev: str, headers: dict[str, str]) -> str:
    """Verify the configured OpenAI-compatible endpoint serves the season's pinned model & revision."""
    override = os.environ.get("STET_EXAMINER_LLM_MODEL", "").strip()
    pinned_rev = os.environ.get("STET_EXAMINER_LLM_REVISION", "").strip()
    if pinned_rev and pinned_rev != rev:
        raise RuntimeError(f"STET_EXAMINER_LLM_REVISION={pinned_rev!r} does not match season revision {rev!r}")
    if override and override not in (repo, f"{repo}@{rev}"):
        raise RuntimeError(
            f"STET_EXAMINER_LLM_MODEL={override!r} does not match the season's pinned reasoning teacher "
            f"{repo!r} (revision {rev})"
        )
    base = url.rstrip("/")
    if base.endswith("/completions"):
        base = base[: -len("/completions")]
    models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    req = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read())
    except Exception as exc:
        raise RuntimeError(
            f"could not verify reasoning endpoint model at {models_url} (expected {repo}@{rev}): {exc}"
        ) from exc
    served = {str(m.get("id", "")) for m in payload.get("data", []) if isinstance(m, dict) and m.get("id")}
    want = override or (f"{repo}@{rev}" if f"{repo}@{rev}" in served else repo)
    if not served or want not in served:
        raise RuntimeError(
            f"reasoning endpoint {models_url} serves {sorted(served)}, which does not include pinned model {want!r}"
        )
    return want


def _generate_via_endpoint(
    prompts: list[str],
    url: str,
    repo: str,
    rev: str,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    log=print,
) -> list[str]:
    """Generate greedily via an OpenAI-compatible `/v1/completions` server (e.g. vLLM)."""
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/completions"):
        endpoint = endpoint + ("/completions" if endpoint.endswith("/v1") else "/v1/completions")
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("STET_EXAMINER_LLM_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    model_name = _verify_endpoint_model(url, repo, rev, headers)
    outs: list[str] = []
    batch = 32
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        body = json.dumps({
            "model": model_name,
            "prompt": chunk,
            "temperature": float(temperature),
            "top_p": 1.0,
            "seed": 0,
            "max_tokens": int(max_new_tokens),
            "stop": ["<|im_end|>", "\n\n\n"],
        }).encode()
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        choices = sorted(data["choices"], key=lambda c: int(c.get("index", 0)))
        outs.extend(c.get("text", "") for c in choices)
    log(f"  reasoning synthesis via {endpoint} ({model_name}): {len(outs)} drafts in {time.time() - t0:.1f}s")
    return outs


def _gen_shard_local(args_tuple):
    gpu_id, repo, rev, shard_prompts, max_new_tokens = args_tuple
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    # Verify no prompt would be truncated (every excerpt is already token-bounded before prompting).
    for i, pr in enumerate(shard_prompts):
        ids = tok.encode(pr, add_special_tokens=False)
        if len(ids) > TEACHER_MAX_PROMPT_TOKENS:
            raise RuntimeError(
                f"teacher prompt {i} is {len(ids)} tokens (> {TEACHER_MAX_PROMPT_TOKENS}); "
                "refusing to truncate closing instructions"
            )
        tail = tok.decode(ids[-8:], skip_special_tokens=False)
        if not pr.endswith("PROMPT: ") or not tail.endswith("PROMPT: "):
            raise RuntimeError(f"teacher prompt {i} lost its closing PROMPT: cue (tail={tail!r})")

    order = sorted(range(len(shard_prompts)), key=lambda i: len(shard_prompts[i]))
    raw_outs = [""] * len(shard_prompts)
    bs = 16

    def _run_batch(mdl, idxs: list[int]) -> None:
        enc = tok(
            [shard_prompts[i] for i in idxs],
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        in_len = enc["input_ids"].shape[1]
        gen_ids = mdl.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        )
        for row, orig_i in enumerate(idxs):
            raw_outs[orig_i] = tok.decode(gen_ids[row, in_len:], skip_special_tokens=True)
        del enc, gen_ids

    with torch.inference_mode():
        pos = 0
        while pos < len(order):
            idxs = order[pos : pos + bs]
            try:
                _run_batch(model, idxs)
                pos += len(idxs)
            except torch.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if len(idxs) <= 1:
                    raise
                bs = max(1, len(idxs) // 2)
                continue
            if (pos // max(1, bs)) % 4 == 0:
                torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return raw_outs


def _generate_local_gpu(
    prompts: list[str],
    repo: str,
    rev: str,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    log=print,
) -> list[str]:
    """Greedy generation on local GPU(s) (NVIDIA CUDA or AMD ROCm) via pinned `transformers` SDPA."""
    import multiprocessing as mp

    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError(
            "synthesizing Season 3 reasoning exam items requires a CUDA/ROCm GPU (18 GB+) "
            "or STET_EXAMINER_LLM_URL pointing at a Qwen3-8B server"
        )
    t0 = time.time()
    snapshot_download(repo, revision=rev)
    n_gpus = min(4, torch.cuda.device_count())
    if n_gpus <= 1:
        outs = _gen_shard_local((None, repo, rev, prompts, max_new_tokens))
    else:
        vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
        gpu_ids = vis[:n_gpus] if len(vis) >= n_gpus else [str(g) for g in range(n_gpus)]
        shards = [(gpu_ids[g], repo, rev, prompts[g::n_gpus], max_new_tokens) for g in range(n_gpus)]
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
    drops_path=None,
) -> list[Item]:
    """Synthesize fresh multilingual `explain`, `infer`, and `synthesis` items from harvested docs.

    Saves every dropped draft beside the kept items when `drops_path` is provided. If one or two
    reasoning kinds fall below `min_keep_rate`, omits only those kinds (so the remaining kinds are
    still cached and scored, matching `_filter_pillar_support`). Raises `FatalExamError` only when
    no reasoning kind survives, so `stet examine --follow` cancels the round immediately rather than
    repeating an identical temp=0 synthesis for hours."""
    from pathlib import Path

    target = int(cfg.get("items", 640))
    max_ov = float(cfg.get("max_ngram_overlap", 0.50))
    repo = str(cfg.get("model", "Qwen/Qwen3-8B"))
    rev = str(cfg.get("revision", "b968826d9c46dd6066d109eabc6255188de91218"))
    max_ex_toks = int(cfg.get("max_excerpt_tokens", DEFAULT_MAX_EXCERPT_TOKENS))
    max_ex_chars = int(cfg.get("max_excerpt_chars", DEFAULT_MAX_EXCERPT_CHARS))
    max_new = int(cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    min_keep_rate = float(cfg.get("min_keep_rate", DEFAULT_MIN_KEEP_RATE))
    temp = float(cfg.get("temperature", 0.0))

    selected = select_source_docs(docs, seed, target=target, shares=shares, max_excerpt_chars=max_ex_chars)
    if not selected:
        return []
    # Bound every document's excerpt ONCE with the pinned teacher tokenizer so both teacher and
    # student see the exact same excerpt and no teacher prompt can ever overflow or lose `PROMPT: `.
    tok = _load_teacher_tokenizer(repo, rev, required=True)
    selected = apply_token_budget(selected, tok, max_tokens=max_ex_toks, max_chars=max_ex_chars)
    prompts = [build_teacher_prompt(d["excerpt"], d["lang"], d["task_type"]) for d in selected]
    if url := os.environ.get("STET_EXAMINER_LLM_URL"):
        gens = _generate_via_endpoint(
            prompts, url, repo, rev, max_new_tokens=max_new, temperature=temp, log=log
        )
    else:
        gens = _generate_local_gpu(prompts, repo, rev, max_new_tokens=max_new, log=log)
    items, dropped, drop_records = parse_generated_items_detailed(selected, gens, max_ngram_overlap=max_ov)
    if drops_path is not None:
        dp = Path(drops_path)
        dp.parent.mkdir(parents=True, exist_ok=True)
        tmp = dp.with_name(f"{dp.name}.tmp{os.getpid()}")
        try:
            payload = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in drop_records).encode("utf-8")
            tmp.write_bytes(payload)
            os.replace(tmp, dp)
        finally:
            tmp.unlink(missing_ok=True)
    attempted = Counter(d["task_type"] for d in selected)
    counts = Counter(it.source.split(":", 1)[0] for it in items)
    log(f"  reasoning items: {len(items)} kept of {len(selected)} ({dict(counts)}, attempted={dict(attempted)}, dropped={dropped})")
    if min_keep_rate > 0:
        bad_kinds = {
            k
            for k, n in attempted.items()
            if n >= 10 and counts.get(k, 0) / n < min_keep_rate
        }
        if bad_kinds:
            bad_desc = [
                f"{k}={counts.get(k, 0)}/{attempted[k]} ({100 * counts.get(k, 0) / attempted[k]:.0f}%)"
                for k in sorted(bad_kinds)
            ]
            items = [it for it in items if it.source.split(":", 1)[0] not in bad_kinds]
            if not items:
                raise FatalExamError(
                    f"reasoning synthesis keep rate fell below {100 * min_keep_rate:.0f}% on every kind "
                    f"({', '.join(bad_desc)}; dropped={dropped})"
                )
            log(
                f"  omitting reasoning kind(s) below {100 * min_keep_rate:.0f}% keep rate "
                f"(share goes to remaining reasoning kinds): {', '.join(bad_desc)}"
            )
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

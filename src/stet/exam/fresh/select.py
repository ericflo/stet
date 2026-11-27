"""From harvested documents to a round's fresh items: make the items, place them in corners,
and choose with the round's seed."""

from __future__ import annotations

import dataclasses
import re
from collections import defaultdict

from ..items import Item
from ..rng import Rng
from . import Doc

# Approximate shares of Common Crawl's text by language (percent), from Common Crawl's published
# statistics. To be replaced by a count of the round's own corpus. A language not listed gets
# OTHER.
SHARES = {
    "en": 43.0, "ru": 6.2, "de": 5.6, "ja": 5.0, "zh": 5.0, "es": 4.6, "fr": 4.4, "it": 2.5, "pt": 2.0, "nl": 1.8,
    "pl": 1.8, "vi": 1.0, "tr": 1.0, "id": 1.0, "cs": 1.0, "ko": 0.7, "sv": 0.7, "fa": 0.7, "ar": 0.6, "uk": 0.6,
    "hu": 0.5, "el": 0.5, "ro": 0.5, "da": 0.4, "fi": 0.4, "th": 0.4, "no": 0.3, "sk": 0.3, "bg": 0.3, "he": 0.2,
    "hi": 0.2, "ca": 0.2, "lt": 0.15, "hr": 0.15, "sl": 0.12, "sr": 0.12, "et": 0.1, "lv": 0.1, "bn": 0.1, "ms": 0.1,
}
OTHER = 0.05
ALIASES = {"simple": "en", "nb": "no", "zh-yue": "zh", "zh-classical": "zh", "zh-min-nan": "nan"}
PASSAGE_CHARS = 2400
ANSWER_CHARS = 800
QUESTION_CHARS = 1500


def hash_title(title: str) -> int:
    """A stable bucket for a title: corners for sources that have no topic of their own."""
    import hashlib

    return int.from_bytes(hashlib.sha256(title.encode()).digest()[:4], "big")


def lang_of(doc: Doc) -> str:
    return ALIASES.get(doc.lang, doc.lang)


def _clip(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    cut = text.rfind(" ", 0, n)
    return text[: cut if cut > n // 2 else n]


def _codeish(text: str) -> bool:
    """Whether most of a text's lines read as code or a patch. A line of code is short: a
    paragraph of prose is one long line, and a semicolon or an equals sign in it says nothing."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    code = sum(1 for ln in lines if len(ln) <= 200 and re.search(
        r"[{};=]|^\s{4}|\)\s*$|^\s*(def|class|import|return|if|for)\b|^(diff |@@ |\+\+\+ |--- |[+-]\s)", ln))
    return code / len(lines) > 0.35


def _lead(text: str) -> str:
    """The first sentence or two, up to about 300 characters."""
    m = re.match(r"(.{40,300}?[.!?。！？])(\s|$)", text, re.S)
    return (m.group(1) if m else _clip(text, 300)).strip()


def crawl_space(text: str) -> str:
    """White space as the crawl has it: one space between words, one line end between lines.

    Common Crawl's text of a page has an empty line 300 times less often than writing taken
    from its source, a tab 600 times less often, two spaces together 15 times less often. A
    student pays some twenty bits for a mark it has never met, so an exam that kept those marks
    would rank curators by whether their pages happened to hold a few of them, and a script
    cannot supply what the crawl does not have (docs/research/exam-first-principles.md, 9.9)."""
    text = re.sub(r"[ \t\u00a0]+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
    return re.sub(r" ?\n(?: ?\n)*", "\n", text)


def items_of(doc: Doc) -> list[Item]:
    """A document's items, laid out as the crawl lays out a page."""
    return [dataclasses.replace(it, text=crawl_space(it.text), ctx=crawl_space(it.ctx), cont=crawl_space(it.cont)) for it in _items_of(doc)]


def _items_of(doc: Doc) -> list[Item]:
    lang = lang_of(doc)
    base = f"fresh/{doc.source}/{lang}/{doc.topic or '-'}"
    src = f"{doc.source}:{doc.id}"
    if doc.source == "wikipedia":
        lead = _lead(doc.text)
        out = [Item(kind="passage", corner=f"{base}/article", fmt="prose", lang=lang, source=src,
                    text=_clip(f"{doc.title}\n\n{doc.text}", PASSAGE_CHARS))]
        if len(lead) >= 40 and lead != doc.text.strip():
            out.append(Item(kind="answer", corner=f"{base}/lead", fmt="prose", lang=lang, source=src,
                            ctx=f"{doc.title}\n\n", cont=lead))
        return out
    if doc.source == "wikipedia-added":
        return [Item(kind="passage", corner=f"{base}/added", fmt="prose", lang=lang, source=src,
                     text=_clip(f"{doc.title}\n\n{doc.text}", PASSAGE_CHARS))]
    if doc.source == "wikipedia-talk":
        return [Item(kind="passage", corner=f"{base}/{abs(hash_title(doc.title)) % 64}", fmt="dialogue", lang=lang, source=src,
                     text=_clip(f"{doc.title}\n\n{doc.text}", PASSAGE_CHARS))]  # the title names the talk page
    # Every other source: a reply given what it answers, or a piece of writing on its own. A source
    # without topics of its own has its documents spread over buckets of their titles, so that
    # the corner cap does not leave it a couple of items.
    bucket = doc.topic or str(hash_title(doc.title or doc.id) % 64)
    corner = f"fresh/{doc.source}/{lang}/{bucket}"
    if doc.question:
        fmt = "code" if _codeish(doc.text) else "dialogue"
        head = f"{doc.title}\n\n" if doc.title and not doc.question.startswith(doc.title) else ""  # a thread starts with its title
        return [Item(kind="answer", corner=f"{corner}/reply", fmt=fmt, lang=lang, source=src,
                     ctx=f"{head}{_tail(doc.question, QUESTION_CHARS)}\n\n", cont=_clip(doc.text, ANSWER_CHARS))]
    fmt = "code" if _codeish(doc.text) else "prose"
    return [Item(kind="passage", corner=f"{corner}/text", fmt=fmt, lang=lang, source=src,
                 text=_clip(f"{doc.title}\n\n{doc.text}" if doc.title else doc.text, PASSAGE_CHARS))]


def _tail(text: str, n: int) -> str:
    """The last n characters or so, from a word boundary: what a reply answers is nearest the end."""
    text = text.strip()
    if len(text) <= n:
        return text
    cut = text.find(" ", len(text) - n)
    return text[cut + 1 if 0 <= cut < len(text) - n // 2 else len(text) - n :]


def _interleave(items: list[Item], rng) -> list[Item]:
    """Items of one language in a seeded order that takes from each source in turn, so that a
    source with many documents cannot fill a language's quota alone."""
    by: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by[it.source.split(":", 1)[0]].append(it)
    for src in sorted(by):
        rng.child(src).shuffle(by[src])
    order = sorted(by)
    rng.shuffle(order)
    out = []
    for i in range(max(map(len, by.values()), default=0)):
        out += [by[src][i] for src in order if i < len(by[src])]
    return out


def _select_kind(
    items: list[Item],
    rng,
    n: int,
    per_corner: int,
    shares: dict,
    used: dict[str, int],
    *,
    max_source_frac: float | None = None,
) -> list[Item]:
    """About n items of one kind of writing, shared among languages by their share of the web; a
    language without enough candidates gives its unused share to the others. When `max_source_frac`
    is set and at least two distinct sources exist in `items`, caps any single source in the first
    pass so one noisy or narrow source cannot dominate the kind."""
    by_lang: dict[str, list[Item]] = defaultdict(list)
    sources_seen: set[str] = set()
    for it in items:
        by_lang[it.lang].append(it)
        sources_seen.add(it.source.split(":", 1)[0])
    for lang in sorted(by_lang):
        by_lang[lang] = _interleave(by_lang[lang], rng.child(lang))

    src_cap: int | None = None
    if max_source_frac is not None and len(sources_seen) >= 2 and n > 0:
        frac = min(max_source_frac, 0.4) if len(sources_seen) >= 3 else max_source_frac
        src_cap = max(1, int(n * frac))

    chosen: list[Item] = []
    chosen_ids: set[int] = set()
    src_used: dict[str, int] = defaultdict(int)
    for enforce_cap in ([True, False] if src_cap is not None else [False]):
        want = n - len(chosen)
        if want <= 0:
            break
        weight = {lang: shares.get(lang, OTHER) for lang in by_lang}
        pos = {lang: 0 for lang in by_lang}
        while want > 0 and weight:
            total = sum(weight.values())
            quota = {lang: max(1, round(want * w / total)) for lang, w in weight.items()}
            got = 0
            for lang in sorted(weight, key=lambda x: (-weight[x], x)):
                q = quota[lang]
                while q > 0 and pos[lang] < len(by_lang[lang]) and got < want:
                    it = by_lang[lang][pos[lang]]
                    pos[lang] += 1
                    if id(it) in chosen_ids or used[it.corner] >= per_corner:
                        continue
                    src = it.source.split(":", 1)[0]
                    if enforce_cap and src_cap is not None and src_used[src] >= src_cap:
                        continue
                    used[it.corner] += 1
                    src_used[src] += 1
                    chosen_ids.add(id(it))
                    chosen.append(it)
                    q -= 1
                    got += 1
                if pos[lang] >= len(by_lang[lang]):
                    weight.pop(lang)
            if got == 0:
                break
            want -= got
    return chosen


def select(docs: list[Doc], seed: bytes, n: int, kinds, per_corner: int = 2, shares: dict | None = None) -> list[Item]:
    """About n items of the kinds of writing the season names (`kinds`: its parts; a document of
    any other kind is passed over). Each kind an equal part (a kind without enough candidates
    gives what it leaves to the others), languages inside each kind by their share of the web,
    sources inside each language in turn, and at most per_corner from any corner."""
    from ..aggregate import kind_of

    shares = shares or SHARES
    rng = Rng(seed, "fresh/select")
    by_kind: dict[str, list[Item]] = defaultdict(list)
    for d in sorted(docs, key=lambda d: (d.source, d.id)):
        for it in items_of(d):
            if kind_of(it) in kinds:
                by_kind[kind_of(it)].append(it)
    used: dict[str, int] = defaultdict(int)
    chosen: dict[str, list[Item]] = {k: [] for k in by_kind}
    left = sorted(by_kind)
    while left and (want := n - sum(map(len, chosen.values()))) > 0:
        part = max(1, want // len(left))
        for k in list(left):
            got = _select_kind([it for it in by_kind[k] if used[it.corner] < per_corner and it not in set(chosen[k])],
                               rng.child(f"{k}/{len(chosen[k])}"), part, per_corner, shares, used)
            chosen[k] += got
            if len(got) < part:
                left.remove(k)
    return [it for k in sorted(chosen) for it in chosen[k]]


# --- Season 3: same-source twin-context comprehension (zero raw web perplexity) ------------------

CONTEXT_CTX_MIN = 80
CONTEXT_CTX_MAX = 1200
CONTEXT_CONT_MIN = 30
CONTEXT_CONT_MAX = 400
CONTEXT_WRONGS = 3

# Prose boundaries: sentence-ending punctuation followed by either a space or a paragraph newline
# (crawl_space normalizes runs of spaces to " " and runs of newlines to "\n"), plus CJK punctuation.
_PROSE_SEP = re.compile(r"(?<=[.!?…؟۔।])([ \n])|(?<=[。！？｡])([ \n]?)")


def _prose_segments(body: str) -> tuple[list[str], list[str]]:
    """Split prose into (segments, separators) preserving the actual separator (' ', '\\n', or '')."""
    parts: list[str] = []
    seps: list[str] = []
    pos = 0
    prev_sep = ""
    for m in _PROSE_SEP.finditer(body):
        seg = body[pos : m.start()].strip()
        sep = m.group(1) if m.group(1) is not None else m.group(2)
        if seg:
            if parts:
                seps.append(prev_sep)
            parts.append(seg)
            prev_sep = sep
        pos = m.end()
    tail = body[pos:].strip()
    if tail:
        if parts:
            seps.append(prev_sep)
        parts.append(tail)
    return parts, seps


def _join_with_seps(parts: list[str], seps: list[str]) -> str:
    if not parts:
        return ""
    out = [parts[0]]
    for sep, p in zip(seps, parts[1:], strict=True):
        out.append(sep)
        out.append(p)
    return "".join(out)


def _split_continuation(doc: Doc) -> tuple[str, str, str, str] | None:
    """Split a harvested document into (corner, fmt, ctx, cont) at a clean sentence or paragraph break.

    Preserves the document's actual boundary separator: space separators go at the start of `cont`
    (never at the end of `ctx`), newline separators go at the end of `ctx`, and CJK no-space breaks
    add no extra space, so `encode(ctx) + encode(cont)` matches running-text BPE tokenization
    without charging a boundary penalty to the first continuation token.
    Returns None when the document has no clean sentence/line boundary or is too short.
    """
    lang = lang_of(doc)
    base = f"fresh/{doc.source}/{lang}/{doc.topic or '-'}"
    bucket = doc.topic or str(hash_title(doc.title or doc.id) % 64)
    if doc.question:
        fmt = "code" if _codeish(doc.text) else "dialogue"
        head = f"{doc.title}\n" if doc.title and not doc.question.startswith(doc.title) else ""
        ctx = crawl_space(f"{head}{_tail(doc.question, QUESTION_CHARS)}").rstrip() + "\n"
        cont = crawl_space(_clip(doc.text, CONTEXT_CONT_MAX)).lstrip()
        if len(ctx.strip()) < 20 or len(cont) < CONTEXT_CONT_MIN:
            return None
        return f"fresh/{doc.source}/{lang}/{bucket}/reply-ctx", fmt, ctx, cont

    body = crawl_space(doc.text).strip()
    if len(body) < CONTEXT_CTX_MIN + CONTEXT_CONT_MIN:
        return None
    title = crawl_space(doc.title).strip() if doc.title else ""
    fmt = "dialogue" if doc.source == "wikipedia-talk" else ("code" if _codeish(body) else "prose")

    # Require a real sentence/paragraph (or line for code/dialogue) boundary; never cut mid-sentence.
    if fmt in ("code", "dialogue"):
        parts = [p.strip() for p in body.split("\n") if p.strip()]
        seps = ["\n"] * max(0, len(parts) - 1)
    else:
        parts, seps = _prose_segments(body)
    if len(parts) < 2:
        return None

    split_idx = 1
    while split_idx < len(parts) - 1 and len(_join_with_seps(parts[:split_idx], seps[: split_idx - 1])) < 240:
        split_idx += 1
    head_raw = _join_with_seps(parts[:split_idx], seps[: split_idx - 1])
    # Trim long openings from the left (_tail) so the end of `head_text` stays on the real boundary.
    head_text = _tail(head_raw, CONTEXT_CTX_MAX) if len(head_raw) > CONTEXT_CTX_MAX else head_raw
    boundary_sep = seps[split_idx - 1]

    end_idx = split_idx + 1
    while (
        end_idx < len(parts)
        and len(_join_with_seps(parts[split_idx:end_idx], seps[split_idx : end_idx - 1])) < 120
        and len(_join_with_seps(parts[split_idx : end_idx + 1], seps[split_idx:end_idx])) <= CONTEXT_CONT_MAX
    ):
        end_idx += 1
    tail_text = _clip(_join_with_seps(parts[split_idx:end_idx], seps[split_idx : end_idx - 1]), CONTEXT_CONT_MAX)

    if len(head_text) < 40 or len(tail_text) < CONTEXT_CONT_MIN or head_text == tail_text:
        return None
    base_ctx = f"{title}\n{head_text}" if title else head_text
    if boundary_sep == "\n":
        ctx, cont = base_ctx + "\n", tail_text
    elif boundary_sep == " ":
        ctx, cont = base_ctx, " " + tail_text
    else:
        ctx, cont = base_ctx, tail_text
    if doc.source == "wikipedia":
        corner = f"{base}/article-ctx"
    elif doc.source == "wikipedia-added":
        corner = f"{base}/added-ctx"
    elif doc.source == "wikipedia-talk":
        corner = f"{base}/{abs(hash_title(doc.title)) % 64}/talk-ctx"
    else:
        corner = f"fresh/{doc.source}/{lang}/{bucket}/cont-ctx"
    return corner, fmt, ctx, cont


def _cont_leaks_into(cont: str, wrong_ctx: str) -> bool:
    """True when a candidate wrong opening contains the scored continuation (for example, in a
    reply chain where one document's reply is another document's parent context)."""
    c = cont.strip()
    if not c:
        return False
    if c in wrong_ctx:
        return True
    if len(c) >= 32:
        mid = (len(c) - 32) // 2
        if c[:32] in wrong_ctx or c[-32:] in wrong_ctx or c[mid : mid + 32] in wrong_ctx:
            return True
    return False


def select_context(docs: list[Doc], seed: bytes, n: int, kinds, per_corner: int = 2, shares: dict | None = None) -> list[Item]:
    """Build same-source, same-language twin-context comprehension items (`kind="context"`).

    For every eligible harvested document, split it cleanly into an opening (`ctx`) and true
    continuation (`cont`), then pair `cont` with up to 3 length-matched wrong openings from
    OTHER documents in the exact same `(source, lang, boundary_style)` group. Because every
    option scores the exact same continuation `cont` under same-source contexts, unconditional
    passage ease, surface style matching, and document-start bias cancel identically.
    """
    from ..aggregate import kind_of, source_kind

    shares = shares or SHARES
    rng = Rng(seed, "fresh/select-ctx")

    # 1. Split every eligible document and group by (source, lang, ends_with_newline).
    ordered = sorted(docs, key=lambda d: (d.source, d.id))
    by_group: dict[tuple[str, str, bool], list[tuple[Doc, str, str, str, str]]] = defaultdict(list)
    for d in ordered:
        if source_kind(d.source) not in kinds:
            continue
        split = _split_continuation(d)
        if split is None:
            continue
        corner, fmt, ctx, cont = split
        by_group[(d.source, lang_of(d), ctx.endswith("\n"))].append((d, corner, fmt, ctx, cont))

    # 2. Pair each continuation with length-matched wrong openings from OTHER docs in the same group.
    by_kind: dict[str, list[Item]] = defaultdict(list)
    for (src, lang, is_nl), group in sorted(by_group.items()):
        if len(group) < 2:
            continue
        pool = [(d.id, ctx, len(ctx.encode("utf-8"))) for d, _, _, ctx, _ in group]
        for d, corner, fmt, ctx, cont in group:
            target_len = len(ctx.encode("utf-8"))
            ctx_s = ctx.strip()
            others = [
                (did, c, abs(bl - target_len))
                for did, c, bl in pool
                if did != d.id and c != ctx and not _cont_leaks_into(cont, c) and ctx_s not in c and c.strip() not in ctx
            ]
            if not others:
                continue
            others.sort(key=lambda x: (x[2], x[0]))
            band = others[: max(CONTEXT_WRONGS * 4, 12)]
            sub_rng = Rng(seed, f"fresh/wrong-ctx/{src}/{lang}/{int(is_nl)}/{d.id}")
            sub_rng.shuffle(band)
            wrongs: list[str] = []
            wrong_sources: list[str] = []
            seen_w: set[str] = set()
            for did, c, _ in band:
                if c not in seen_w:
                    seen_w.add(c)
                    wrongs.append(c)
                    wrong_sources.append(f"{src}:{did}")
                    if len(wrongs) >= CONTEXT_WRONGS:
                        break
            if not wrongs:
                continue
            it = Item(
                kind="context",
                corner=corner,
                fmt=fmt,
                lang=lang,
                source=f"{d.source}:{d.id}",
                ctx=ctx,
                cont=cont,
                wrongs=tuple(wrongs),
                wrong_sources=tuple(wrong_sources),
            )
            by_kind[kind_of(it)].append(it)

    # 3. Select across kinds, languages, sources, and corners exactly as select() does.
    used: dict[str, int] = defaultdict(int)
    chosen: dict[str, list[Item]] = {k: [] for k in by_kind}
    left = sorted(by_kind)
    while left and (want := n - sum(map(len, chosen.values()))) > 0:
        part = max(1, want // len(left))
        for k in list(left):
            got = _select_kind(
                [it for it in by_kind[k] if used[it.corner] < per_corner and it not in set(chosen[k])],
                rng.child(f"{k}/{len(chosen[k])}"),
                part,
                per_corner,
                shares,
                used,
                max_source_frac=0.5,
            )
            chosen[k] += got
            if len(got) < part:
                left.remove(k)
    return [it for k in sorted(chosen) for it in chosen[k]]

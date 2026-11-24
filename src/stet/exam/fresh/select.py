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


def _select_kind(items: list[Item], rng, n: int, per_corner: int, shares: dict, used: dict[str, int]) -> list[Item]:
    """About n items of one kind of writing, shared among languages by their share of the web; a
    language without enough candidates gives its unused share to the others."""
    by_lang: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_lang[it.lang].append(it)
    for lang in sorted(by_lang):
        by_lang[lang] = _interleave(by_lang[lang], rng.child(lang))
    weight = {lang: shares.get(lang, OTHER) for lang in by_lang}
    chosen: list[Item] = []
    pos = {lang: 0 for lang in by_lang}
    want = n
    while want > 0 and weight:
        total = sum(weight.values())
        quota = {lang: max(1, round(want * w / total)) for lang, w in weight.items()}
        got = 0
        for lang in sorted(weight, key=lambda x: (-weight[x], x)):
            q = quota[lang]
            while q > 0 and pos[lang] < len(by_lang[lang]) and got < want:
                it = by_lang[lang][pos[lang]]
                pos[lang] += 1
                if used[it.corner] < per_corner:
                    used[it.corner] += 1
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


# --- Season 3: same-source contrastive continuation (zero raw web perplexity) --------------------

CONTRASTIVE_CTX_MIN = 80
CONTRASTIVE_CTX_MAX = 1200
CONTRASTIVE_CONT_MIN = 30
CONTRASTIVE_CONT_MAX = 400
CONTRASTIVE_WRONGS = 3

_SENT_SPLIT = re.compile(r"(?<=[.!?…؟۔।]) +|(?<=[。！？｡])\s*")


def _split_continuation(doc: Doc) -> tuple[str, str, str, str] | None:
    """Split a harvested document into (corner, fmt, ctx, cont) for contrastive continuation.

    Returns None when the document is too short to form a meaningful opening + continuation.
    """
    lang = lang_of(doc)
    base = f"fresh/{doc.source}/{lang}/{doc.topic or '-'}"
    bucket = doc.topic or str(hash_title(doc.title or doc.id) % 64)
    if doc.question:
        fmt = "code" if _codeish(doc.text) else "dialogue"
        head = f"{doc.title}\n" if doc.title and not doc.question.startswith(doc.title) else ""
        ctx = crawl_space(f"{head}{_tail(doc.question, QUESTION_CHARS)}") + "\n"
        cont = crawl_space(_clip(doc.text, CONTRASTIVE_CONT_MAX))
        if len(ctx) < 20 or len(cont) < CONTRASTIVE_CONT_MIN:
            return None
        return f"fresh/{doc.source}/{lang}/{bucket}/reply-mc", fmt, ctx, cont

    body = crawl_space(doc.text).strip()
    if len(body) < CONTRASTIVE_CTX_MIN + CONTRASTIVE_CONT_MIN:
        return None
    title = crawl_space(doc.title).strip() if doc.title else ""
    fmt = "dialogue" if doc.source == "wikipedia-talk" else ("code" if _codeish(body) else "prose")

    # Split into sentences (or lines for code/dialogue) so the cut never lands mid-word or mid-sentence.
    parts = [p.strip() for p in (body.split("\n") if fmt in ("code", "dialogue") else _SENT_SPLIT.split(body)) if p.strip()]
    if len(parts) < 2:
        # Fallback: split cleanly on a space or newline near the midpoint of the opening.
        mid = min(max(len(body) // 2, CONTRASTIVE_CTX_MIN), CONTRASTIVE_CTX_MAX)
        cut = body.rfind(" ", CONTRASTIVE_CTX_MIN // 2, mid)
        if cut < 0:
            cut = mid
        head_text, tail_text = body[:cut].strip(), _clip(body[cut:].strip(), CONTRASTIVE_CONT_MAX)
        sep = "\n" if fmt in ("code", "dialogue") else (" " if body[cut : cut + 1] == " " else "")
    else:
        sep = "\n" if fmt in ("code", "dialogue") else ("" if lang in ("ja", "zh", "th") else " ")
        head_parts = [parts[0]]
        idx = 1
        while idx < len(parts) - 1 and len(sep.join(head_parts)) < 240:
            head_parts.append(parts[idx])
            idx += 1
        head_text = _clip(sep.join(head_parts), CONTRASTIVE_CTX_MAX)
        tail_parts = [parts[idx]]
        idx += 1
        while idx < len(parts) and len(sep.join(tail_parts)) < 120 and len(sep.join([*tail_parts, parts[idx]])) <= CONTRASTIVE_CONT_MAX:
            tail_parts.append(parts[idx])
            idx += 1
        tail_text = _clip(sep.join(tail_parts), CONTRASTIVE_CONT_MAX)

    if len(head_text) < 40 or len(tail_text) < CONTRASTIVE_CONT_MIN or head_text == tail_text:
        return None
    ctx = (f"{title}\n{head_text}" if title else head_text) + (sep if sep else "")
    if doc.source == "wikipedia":
        corner = f"{base}/article-mc"
    elif doc.source == "wikipedia-added":
        corner = f"{base}/added-mc"
    elif doc.source == "wikipedia-talk":
        corner = f"{base}/{abs(hash_title(doc.title)) % 64}/talk-mc"
    else:
        corner = f"fresh/{doc.source}/{lang}/{bucket}/cont-mc"
    return corner, fmt, ctx, tail_text


def select_contrastive(docs: list[Doc], seed: bytes, n: int, kinds, per_corner: int = 2, shares: dict | None = None) -> list[Item]:
    """Build same-source, same-language contrastive continuation items (`kind="choice"`).

    For every harvested document whose kind of writing is named by the season, split it into an
    opening (`ctx`) and real continuation (`cont`), then draw up to 3 length-matched wrong
    continuations from OTHER documents in the EXACT SAME `(source, lang)` group. Surface style
    matching lifts right and wrong equally and nets zero gain.
    """
    from ..aggregate import kind_of, source_kind

    shares = shares or SHARES
    rng = Rng(seed, "fresh/select-mc")

    # 1. Split every eligible document and group by (source, lang).
    ordered = sorted(docs, key=lambda d: (d.source, d.id))
    by_group: dict[tuple[str, str], list[tuple[Doc, str, str, str, str]]] = defaultdict(list)
    for d in ordered:
        if source_kind(d.source) not in kinds:
            continue
        split = _split_continuation(d)
        if split is None:
            continue
        corner, fmt, ctx, cont = split
        by_group[(d.source, lang_of(d))].append((d, corner, fmt, ctx, cont))

    # 2. Pair each candidate with length-matched wrong continuations from OTHER docs in the same (source, lang).
    by_kind: dict[str, list[Item]] = defaultdict(list)
    for (src, lang), group in sorted(by_group.items()):
        if len(group) < 2:
            continue
        # Pool of unique (doc_id, cont, byte_len) within this exact (source, lang).
        pool = [(d.id, cont, len(cont.encode("utf-8"))) for d, _, _, _, cont in group]
        for d, corner, fmt, ctx, cont in group:
            target_len = len(cont.encode("utf-8"))
            others = [(did, c, abs(bl - target_len)) for did, c, bl in pool if did != d.id and c != cont]
            if not others:
                continue
            # Sort by closeness in byte length, then deterministically sample from the closest band.
            others.sort(key=lambda x: (x[2], x[0]))
            band = others[: max(CONTRASTIVE_WRONGS * 4, 12)]
            sub_rng = Rng(seed, f"fresh/wrong/{src}/{lang}/{d.id}")
            sub_rng.shuffle(band)
            wrongs: list[str] = []
            seen_w: set[str] = set()
            for _, c, _ in band:
                if c not in seen_w:
                    seen_w.add(c)
                    wrongs.append(c)
                    if len(wrongs) >= CONTRASTIVE_WRONGS:
                        break
            if not wrongs:
                continue
            it = Item(kind="choice", corner=corner, fmt=fmt, lang=lang, source=f"{d.source}:{d.id}", ctx=ctx, cont=cont, wrongs=tuple(wrongs))
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
            )
            chosen[k] += got
            if len(got) < part:
                left.remove(k)
    return [it for k in sorted(chosen) for it in chosen[k]]

"""From harvested documents to a round's fresh items: make the items, place them in corners,
and choose with the round's seed."""

from __future__ import annotations

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
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    code = sum(1 for ln in lines if re.search(r"[{};=]|^\s{4}|\)\s*$|^\s*(def|class|import|return|if|for)\b", ln))
    return code / len(lines) > 0.35


def _lead(text: str) -> str:
    """The first sentence or two, up to about 300 characters."""
    m = re.match(r"(.{40,300}?[.!?。！？])(\s|$)", text, re.S)
    return (m.group(1) if m else _clip(text, 300)).strip()


def items_of(doc: Doc) -> list[Item]:
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
                     text=_clip(f"Talk: {doc.title}\n\n{doc.text}", PASSAGE_CHARS))]
    if doc.source == "stackexchange":
        fmt = "code" if _codeish(doc.text) else "dialogue"
        if doc.question:
            return [Item(kind="answer", corner=f"{base}/answer", fmt=fmt, lang=lang, source=src,
                         ctx=f"Question: {doc.title}\n{_clip(doc.question, QUESTION_CHARS)}\n\nAnswer:", cont=" " + _clip(doc.text, ANSWER_CHARS))]
        return [Item(kind="passage", corner=f"{base}/question", fmt=fmt, lang=lang, source=src,
                     text=_clip(f"{doc.title}\n\n{doc.text}", PASSAGE_CHARS))]
    return []


def select(docs: list[Doc], seed: bytes, n: int, per_corner: int = 2, shares: dict | None = None) -> list[Item]:
    """About n items, at most per_corner from any corner, shared among languages by their share of
    the web; a language without enough candidates gives its unused share to the others."""
    shares = shares or SHARES
    rng = Rng(seed, "fresh/select")
    by_lang: dict[str, list[Item]] = defaultdict(list)
    for d in sorted(docs, key=lambda d: (d.source, d.id)):
        for it in items_of(d):
            by_lang[it.lang].append(it)
    for lang in sorted(by_lang):
        rng.child(lang).shuffle(by_lang[lang])
    weight = {lang: shares.get(lang, OTHER) for lang in by_lang}
    chosen: list[Item] = []
    used: dict[str, int] = defaultdict(int)
    taken: dict[str, int] = defaultdict(int)
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
                    taken[lang] += 1
                    q -= 1
                    got += 1
            if pos[lang] >= len(by_lang[lang]):
                weight.pop(lang)
        if got == 0:
            break
        want -= got
    return chosen

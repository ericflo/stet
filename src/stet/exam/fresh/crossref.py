"""Abstracts of scholarly works whose DOI was registered in the window.

Crossref's public API (https://api.crossref.org/works) answers without a key or an account. It
offers a faster "polite pool" to callers who send an e-mail address; none is sent, and the public
pool's pace (one request a second) is kept. Works are listed by when their DOI was first
registered (`created`, to the second), newest first, with a cursor: an hour holds a few hundred
that carry an abstract, read whole in one or two requests.

A DOI registered in the window is not always writing published in it. Publishers register back
issues years late, new versions of preprints, and articles whose preprint has been public for
months. So a work is kept only if the record dates its publication, to the month at least, no
longer than RECENT_DAYS before the window, and names no earlier form of itself (a preprint, a
previous version, a twin under another DOI). What is left is as new as the record can say; it is
still the publisher's word, and an abstract may have been public on a page the record does not
name.

The abstract is deposited as JATS XML. It is read as plain text: paragraphs and section titles
(Background, Methods) a blank line apart, formulas as their TeX where given, no "Abstract"
label. Notices (corrections, retractions, front matter) go, and so do abstracts that are too
short, registered twice, or made of paragraphs that several works of the hour share (a book's
blurb on each of its chapters, a journal's standing text). A journal gives at most a few, and a
DOI prefix (a publisher, a preprint server) not many more.

The language is read from the abstract by fastText's lid.176. The record's `language` field is
the article's, not the abstract's (a German article with an English abstract), is seldom filled
in and sometimes wrongly; it decides only when lid.176 is unsure. The licence is the work's, as
its publisher registered it with Crossref, where one is in force; otherwise none is claimed.
"""

from __future__ import annotations

import calendar
import hashlib
import html
import re
import time
import urllib.parse
from collections import Counter

from . import Doc, get_json
from .hackernews import lid, prose_chars

API = "https://api.crossref.org/works"
NO_LICENSE = "© its authors or publisher; no licence registered with Crossref; abstract via Crossref's public API"
LICENSED = "{url} (the work's licence, as registered with Crossref); abstract via Crossref's public API"
GAP = 1.1  # seconds between requests: the public pool allows one a second, one at a time
ROWS = 500  # works a page (Crossref allows a thousand)
MAX_PAGES = 4  # of an hour that are read, from its end back
RECENT_DAYS = 31  # a work published longer than this before the window is a back issue, not news
MIN_CHARS = 300  # of abstract, links aside
PER_JOURNAL = 3  # abstracts from any one journal, book or proceedings
PER_PREFIX = 10  # and from any one DOI prefix: a publisher, a preprint server
COMMON = 3  # a paragraph found in this many of the hour's abstracts is a template's
COMMON_CHARS = 20  # and paragraphs shorter than this (Methods, Results) are too ordinary to tell
UNSURE = 0.5  # lid.176 below this gives way to the language the record names
TYPES = frozenset({"journal-article", "posted-content", "proceedings-article", "book-chapter", "report", "dissertation", "monograph",
                   "book", "edited-book", "reference-entry"})
# Relations by which a record names an earlier or other form of the same work.
EARLIER = ("has-preprint", "is-preprint-of", "is-version-of", "is-same-as", "is-identical-to", "is-translation-of")
_DATES = ("issued", "published", "published-online", "published-print", "posted")
_VERSION = {"vor": 0, "am": 1, "unspecified": 2, "tdm": 3}  # which of a work's licences to name: the published version's first
_LATER = re.compile(r"[/._]v(\d+)$", re.I)  # a DOI that ends in a version: .../v2, ..._v3
_NOTICE = re.compile(r"\s*(?:correction|corrigendum|erratum|errata|retraction|retracted|withdrawn|withdrawal|expression of concern|"
                     r"editorial board|front ?matter|back ?matter|table of contents|preliminary pages|issue information|masthead)\b", re.I)
_next = [0.0]


def _get(params: dict):
    """A request to Crossref, spaced by GAP (get_json's own gap is shorter)."""
    wait = _next[0] - time.time()
    if wait > 0:
        time.sleep(wait)
    _next[0] = time.time() + GAP
    return get_json(API, params)


def _stamp(t: int) -> str:
    """A time as Crossref's filters take it: UTC, to the second, without a zone."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))


def _window(start: int, end: int, why: Counter) -> list[dict]:
    """The records of the works with an abstract whose DOI was registered in [start, end), newest
    first, at most MAX_PAGES pages of them."""
    flt = f"from-created-date:{_stamp(start)},until-created-date:{_stamp(end - 1)},has-abstract:true"
    out: list[dict] = []
    cursor = "*"
    for _ in range(MAX_PAGES):
        msg = _get({"filter": flt, "sort": "created", "order": "desc", "rows": ROWS, "cursor": cursor}).get("message") or {}
        items = msg.get("items") or []
        out += items
        cursor = msg.get("next-cursor")
        if len(items) < ROWS or not cursor:
            break
    else:
        why[f"hours with more than {MAX_PAGES * ROWS} works"] += 1
    return out


# --- what a record says ---------------------------------------------------------------------------------


def created(work: dict) -> int | None:
    """When the work's DOI was first registered, in unix seconds."""
    ms = (work.get("created") or {}).get("timestamp")
    return int(ms) // 1000 if isinstance(ms, int | float) else None


def published(work: dict) -> int | None:
    """The earliest day the record says the work was published (online, in print, posted), a date
    without a day read as its month's last; None when no date is given to the month."""
    days = []
    for k in _DATES:
        parts = ((work.get(k) or {}).get("date-parts") or [[]])[0] or []
        if len(parts) >= 2 and all(isinstance(p, int) for p in parts[:3]) and 1000 <= parts[0] < 9999 and 1 <= parts[1] <= 12:
            y, m = parts[0], parts[1]
            if len(parts) >= 3:
                days.append(calendar.timegm((y, m, parts[2], 0, 0, 0)))
            else:
                days.append(calendar.timegm((y + (m == 12), m % 12 + 1, 1, 0, 0, 0)) - 86400)
    return min(days) if days else None


def licence(work: dict) -> str:
    """The licence the record gives for the work and that is in force, or that none is claimed."""
    found = [x for x in work.get("license") or []
             if isinstance(x, dict) and str(x.get("URL") or "").startswith(("http://", "https://")) and not x.get("delay-in-days")]
    if not found:
        return NO_LICENSE
    best = min(found, key=lambda x: (_VERSION.get(x.get("content-version"), len(_VERSION)), x["URL"]))
    return LICENSED.format(url=best["URL"])


# --- what is prose --------------------------------------------------------------------------------------

_FORMULA = re.compile(r"<(?:jats:)?(inline-formula|disp-formula)\b[^>]*>(.*?)</(?:jats:)?\1>", re.S | re.I)
_TEX = re.compile(r"<(?:jats:)?tex-math\b[^>]*>(.*?)</(?:jats:)?tex-math>", re.S | re.I)
_MATHML = re.compile(r"<(?:mml:)?math\b.*?</(?:mml:)?math>", re.S | re.I)
_BLOCK = re.compile(r"</?(?:jats:)?(?:p|title|sec|list|list-item|disp-quote|break|br|div|li|ul|ol|table|tr|h[1-6])\b[^<>]*>", re.I)
_TAG = re.compile(r"</?[A-Za-z][\w:.-]*(?:\s[^<>]*)?/?>")
# Some publishers deposit HTML escaped inside the text (&lt;i&gt;, &lt;p&gt;), some escape it twice (&amp;lt;): entities
# are read twice, and then the tags HTML has are taken out, and only those, so that "a < b" stays.
_HTML_BLOCK = re.compile(r"</?(?:p|br|div|li|ul|ol|tr|h[1-6])\b[^<>]*>", re.I)
_HTML = re.compile(r"</?(?:b|i|u|em|strong|sub|sup|span|font|a|small|italic|bold|sc|scp)\b[^<>]*>", re.I)
_LABELS = r"a\s?bstract|summary|resumen|resumo|résumé|zusammenfassung|abstrak|аннотация|резюме|özet|摘要|要旨"
_LABEL = re.compile(rf"(?:{_LABELS})\s*[:.]?", re.I)  # a paragraph that is only the label
_LABELLED = re.compile(rf"(?:{_LABELS})\s*[:.]\s*(?=\S)", re.I)  # or one that starts with it: "Abstract: ...", "ABSTRACT ..."
_SHOUTED = re.compile(r"ABSTRACT\s+(?=\S)")


def _math(m: re.Match) -> str:
    """A formula as text, where it stands in its sentence: its TeX where the record has it, or
    else what its MathML holds."""
    tex = _TEX.search(m.group(0))
    return tex.group(1).strip() if tex else "".join(_TAG.sub(" ", m.group(0)).split())


def blocks(abstract: str) -> list[str]:
    """A JATS abstract as plain paragraphs and section titles, in order: no tags, entities read,
    runs of space as one, and no "Abstract" before it."""
    t = _MATHML.sub(_math, _FORMULA.sub(_math, abstract or ""))
    t = _TAG.sub("", _BLOCK.sub("\n\n", _TEX.sub(lambda m: m.group(1).strip(), t)))
    t = _HTML.sub("", _HTML_BLOCK.sub("\n\n", html.unescape(html.unescape(t))))
    out = [" ".join(b.split()) for b in re.split(r"\n\s*\n", t)]
    out = [b for b in out if b]
    while out and _LABEL.fullmatch(out[0]):
        out.pop(0)
    if out and (m := _LABELLED.match(out[0]) or _SHOUTED.match(out[0])):
        out[0] = out[0][m.end() :]
    return out


def _key(block: str) -> str:
    return " ".join(block.lower().split())


def _plain(s) -> str:
    """A title, or a journal's name, as Crossref lists it (the first of a list), without markup."""
    s = (s[0] if s else "") if isinstance(s, list) else (s or "")
    return " ".join(_HTML.sub("", _HTML_BLOCK.sub(" ", html.unescape(html.unescape(_TAG.sub("", str(s)))))).split())


def _tag(work: dict) -> str:
    """The language the record names, as a bare code ("en" for "en-US"), or nothing."""
    code = re.split(r"[-_]", str(work.get("language") or "").strip().lower())[0]
    return code if re.fullmatch(r"[a-z]{2,3}", code) else ""


def _fit(work: dict, start: int, end: int, why: Counter) -> bool:
    """Whether a record is of a work first registered in the window and, by all it says, new."""
    at = created(work)
    doi = str(work.get("DOI") or "")
    if at is None or not start <= at < end or not doi:
        why["outside the window"] += 1
    elif work.get("type") not in TYPES:
        why[f"a {work.get('type') or 'work of no type'}"] += 1
    elif work.get("update-to") or _NOTICE.match(_plain(work.get("title"))):
        why["a notice"] += 1
    elif (m := _LATER.search(doi)) and int(m.group(1)) > 1 or any((work.get("relation") or {}).get(k) for k in EARLIER):
        why["public earlier in another form"] += 1
    elif (day := published(work)) is None:
        why["no publication date to the month"] += 1
    elif day < start - RECENT_DAYS * 86400:
        why["published long before"] += 1
    else:
        return True
    return False


# --- the harvest ----------------------------------------------------------------------------------------


def harvest(start: int, end: int, cap: int, rng, log=print) -> list[Doc]:
    """Up to cap abstracts of works whose DOI was registered in [start, end) and that are newly
    published, chosen with rng: at most PER_JOURNAL from a journal and PER_PREFIX from a DOI prefix."""
    t0 = time.time()
    why: Counter = Counter()
    try:
        found = _window(start, end, why)
    except Exception as e:
        log(f"  crossref: {type(e).__name__}: {e}")
        return []
    first: dict[str, dict] = {}
    for w in found:  # a record listed twice (a cursor's pages can overlap) is read once
        first.setdefault(str(w.get("DOI") or "").lower(), w)
    works = [(doi, w, blocks(w.get("abstract") or "")) for doi, w in sorted(first.items()) if _fit(w, start, end, why)]
    shared = Counter(k for _, _, bs in works for k in {_key(b) for b in bs} if len(k) >= COMMON_CHARS)
    pool, seen = [], set()
    for doi, w, bs in works:
        text = "\n\n".join(b for b in bs if shared[_key(b)] < COMMON)
        if prose_chars(text) < MIN_CHARS:
            why["short"] += 1
            continue
        key = hashlib.sha256(_key(text).encode()).hexdigest()
        if key in seen:
            why["registered twice"] += 1
            continue
        seen.add(key)
        pool.append((doi, w, text))
    rng.shuffle(pool)
    docs: list[Doc] = []
    per_journal: Counter = Counter()
    per_prefix: Counter = Counter()
    for doi, w, text in pool:
        if len(docs) >= cap:
            break
        journal = _plain(w.get("container-title"))
        prefix = str(w.get("prefix") or doi.split("/", 1)[0])
        if journal and per_journal[journal.lower()] >= PER_JOURNAL:
            why["journal full"] += 1
            continue
        if per_prefix[prefix] >= PER_PREFIX:
            why["publisher full"] += 1
            continue
        per_journal[journal.lower()] += 1
        per_prefix[prefix] += 1
        tag = _tag(w)
        label, sure = lid(text)
        docs.append(Doc(source="crossref", id=doi, url=f"{API}/{urllib.parse.quote(doi)}", lang=tag if tag and tag != label and sure < UNSURE else label,
                        license=licence(w), published=created(w), title=_plain(w.get("title")), text=text,
                        extra={"doi": f"https://doi.org/{urllib.parse.quote(doi)}", "type": w.get("type"), "journal": journal,
                               "publisher": _plain(w.get("publisher")), "lang_tag": tag, "lid": label, "lid_p": sure}))
    langs = Counter(d.lang for d in docs)
    skipped = ", ".join(f"{k} {v}" for k, v in why.most_common())
    log(f"  crossref: {len(docs)} abstracts of {len(pool)} newly published, among {len(found)} works registered in the window, in "
        f"{dict(langs.most_common(8))} ({time.time() - t0:.0f}s; skipped: {skipped or 'none'})")
    return docs

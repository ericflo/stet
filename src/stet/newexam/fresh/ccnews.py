"""News articles, in every language Common Crawl's news crawl (CC-NEWS) reaches, published in the window.

CC-NEWS fetches articles from news sites' feeds and sitemaps around the clock and writes them, as
they are fetched, into WARC files of about a gigabyte, each named for the moment it was started
(CC-NEWS-YYYYMMDDHHMMSS-NNNNN.warc.gz): a new one every one and a half to three hours. A file is
uploaded, and its month's listing rewritten, at five past the first hour after it is closed.

A window reads every file whose records were fetched in it: those started before the window
closed whose successor started after it opened. Fetched in the window is not enough, as news sites
are fetched again and again: an article is kept only if its own publication time (its
article:published_time, its JSON-LD datePublished, ...), given to the minute with a time zone, is
inside the window and not after the page was fetched; where a page gives several, the earliest.
Its main text is the block of the page with the most paragraph text, without navigation, link
lists and furniture; its language is fastText's guess.

No one outlet may fill the source: at most PER_SITE articles from any site, a story syndicated to
many sites is kept once, and the cap is shared among languages by their share of the web. Each
record's chance comes from the seed and its WARC-Record-ID alone, so the choice samples the whole of
each file and does not depend on the order it is read in.
"""

from __future__ import annotations

import calendar
import codecs
import email.utils
import gzip
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

from . import UA, Doc, html_text, iso

BASE = "https://data.commoncrawl.org/"
LICENSE = "Common Crawl CC-NEWS; the articles' copyright stays with their publishers"
PER_SITE = 3  # articles from any one site
MIN_CHARS = 500  # of an article's main text
MIN_LANG_P = 0.5  # fastText's confidence in the article's language
SKEW = 900  # seconds a page's publication time may be after it was fetched: the clocks differ
MAX_RECORD = 8 << 20  # bytes of a record, decompressed; a larger one is passed over
MAX_HTML = 2 << 20  # characters of a page that are read
READ_LANG = 1000  # characters of an article its language is guessed from
FETCH_WAITS = (60, 120, 240, 480)  # the data host says "slow down" when many machines fetch at once
NAME = re.compile(r"CC-NEWS-(\d{14})-(\d{5})\.warc\.gz$")


# --- the files ---------------------------------------------------------------------------------


def _get(url: str, tries: int = 6, pause: float = 2.0, limit: int = 16 << 20) -> bytes:
    """A small file from the data host, waiting out a busy one."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read(limit + 1)
            if len(data) > limit:
                raise OSError(f"{url} is larger than {limit} bytes")
            return data
        except urllib.error.HTTPError as e:
            if i == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            asked = e.headers.get("Retry-After", "")
            time.sleep(min(120.0, float(asked)) if asked.isdigit() else pause * 2**i)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(pause * 2**i)
    raise AssertionError("unreachable")


def started(path: str) -> int:
    """When a file was started, from its name."""
    m = NAME.search(path)
    if not m:
        raise ValueError(f"{path!r} is not a CC-NEWS file name")
    return calendar.timegm(time.strptime(m.group(1), "%Y%m%d%H%M%S"))


def listing(year: int, month: int) -> list[str]:
    """Every file of a month, by path. A month not begun has none."""
    try:
        raw = _get(f"{BASE}crawl-data/CC-NEWS/{year:04d}/{month:02d}/warc.paths.gz")
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):  # the host answers 403 for what is not there
            return []
        raise
    return [p for p in gzip.decompress(raw).decode().split() if NAME.search(p)]


def _months(start: int, end: int) -> list[tuple[int, int]]:
    out = []
    t = start - 3 * 86400  # the file a window starts in may have been started in the month before
    while t < end + 86400:
        g = time.gmtime(t)
        if (g.tm_year, g.tm_mon) not in out:
            out.append((g.tm_year, g.tm_mon))
        t += 86400
    return out


def window_files(paths: list[str], start: int, end: int, after: int = 0) -> tuple[list[str], bool]:
    """The files whose records were fetched in [start, end), and the first `after` files started
    after it (for articles published in the window and fetched later); and whether the listing
    is known to hold them all. A file is listed only once it is closed, so it does when it lists
    a file started at or after end, and the `after` files that are to be read."""
    paths = sorted(set(paths), key=started)
    out = []
    later = 0
    for i, p in enumerate(paths):
        nxt = started(paths[i + 1]) if i + 1 < len(paths) else None
        if started(p) < end and (nxt is None or nxt > start):
            out.append(p)
        elif started(p) >= end:
            if later < after:
                out.append(p)
            later += 1
    return out, later >= max(1, after)


def _local(path: str, log=print, quiet: bool = True) -> Path:
    """A file in the cache, downloaded whole first and checked to the end of its gzip stream."""
    from ...data import FetchError, TooBig, cache_dir, fetch

    name = path.rsplit("/", 1)[-1]
    if not NAME.fullmatch(name) or not path.startswith("crawl-data/CC-NEWS/"):
        raise ValueError(f"{path!r} is not a CC-NEWS file")
    for wait in (*FETCH_WAITS, None):
        try:
            return fetch([BASE + path], "", cache_dir() / "ccnews" / name, name, quiet=quiet, max_bytes=2 << 30, gzipped=True)
        except FetchError as e:
            if wait is None or isinstance(e, TooBig):
                raise
            log(f"  ccnews {name}: {e}; trying again in {wait // 60} min")
            time.sleep(wait)
    raise AssertionError("unreachable")


# --- WARC ----------------------------------------------------------------------------------------

CHUNK = 1 << 20  # bytes read from the file at a time
FEED = 1 << 16  # bytes given to the decompressor at a time: what is left over is copied, once per member
OUT = 1 << 22  # bytes it may give back at a time


def members(f, limit: int = MAX_RECORD):
    """Each gzip member of an open binary file: (offset, length, data), where its compressed
    bytes start and how many there are, and what it holds (None when that is more than limit)."""
    buf, i, base = b"", 0, 0  # base: where buf starts in the file; i: how far into it this is
    while True:
        if len(buf) - i < 2:
            more = f.read(CHUNK)
            buf, base, i = buf[i:] + more, base + i, 0
            if not buf:
                return
            if not more and len(buf) < 2:
                raise ValueError(f"no gzip member at offset {base}")
        if buf[i : i + 2] != b"\x1f\x8b":
            if not buf[i:].strip(b"\x00") and not f.read(CHUNK).strip(b"\x00"):
                return  # padding at the end
            raise ValueError(f"no gzip member at offset {base + i}")
        offset = base + i
        d = zlib.decompressobj(31)
        parts: list[bytes] | None = []
        size = 0
        while True:
            piece = memoryview(buf)[i : i + FEED]
            out = d.decompress(piece, OUT)
            size += len(out)
            if parts is not None and size <= limit:
                parts.append(out)
            else:
                parts = None
            i += len(piece) - len(d.unused_data if d.eof else d.unconsumed_tail)
            if d.eof:
                break
            if i >= len(buf) and len(out) < OUT:  # all of it read, and nothing held back
                buf, base, i = f.read(CHUNK), base + len(buf), 0
                if not buf:
                    raise EOFError(f"the gzip member at offset {offset} is cut off")
        yield offset, base + i - offset, (b"".join(parts) if parts is not None else None)


def parse_record(data: bytes) -> tuple[dict[str, str], bytes] | None:
    """A WARC record's headers (names lower-cased) and its block."""
    if not data.startswith(b"WARC/"):
        return None
    end = data.find(b"\r\n\r\n")
    sep = 4
    if end < 0:
        end, sep = data.find(b"\n\n"), 2
        if end < 0:
            return None
    headers: dict[str, str] = {}
    for line in data[:end].split(b"\n")[1:]:
        k, colon, v = line.partition(b":")
        if colon:
            headers[k.strip().decode("latin-1").lower()] = v.strip().decode("utf-8", "replace")
    try:
        n = int(headers.get("content-length", ""))
    except ValueError:
        n = len(data) - end - sep
    return headers, data[end + sep : end + sep + n]


def records(f, limit: int = MAX_RECORD):
    """Every WARC record in a file of gzip members, one to a member: (offset, length, headers, block)."""
    for offset, length, data in members(f, limit):
        if data is not None and (rec := parse_record(data)):
            yield offset, length, rec[0], rec[1]


def parse_http(block: bytes) -> tuple[int, dict[str, str], bytes] | None:
    """An HTTP response's status, headers (names lower-cased) and body."""
    end = block.find(b"\r\n\r\n")
    sep = 4
    if end < 0:
        end, sep = block.find(b"\n\n"), 2
        if end < 0:
            return None
    lines = block[:end].split(b"\n")
    m = re.match(rb"HTTP/[\d.]+\s+(\d{3})", lines[0])
    if not m:
        return None
    headers: dict[str, str] = {}
    for line in lines[1:]:
        k, colon, v = line.partition(b":")
        if colon:
            headers[k.strip().decode("latin-1").lower()] = v.strip().decode("latin-1")
    return int(m.group(1)), headers, block[end + sep :]


_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([\w.:-]+)""", re.I)


def html_of(block: bytes) -> str | None:
    """The page an HTTP response carries, as text, if it is a whole HTML page (status 200)."""
    r = parse_http(block)
    if not r:
        return None
    status, headers, body = r
    ctype = headers.get("content-type", "").lower()
    if status != 200 or not ("text/html" in ctype or "application/xhtml" in ctype):
        return None
    enc = headers.get("content-encoding", "identity").lower()  # the crawler stores pages decoded
    if enc in ("gzip", "x-gzip", "deflate"):
        try:
            body = zlib.decompress(body, 47 if "gzip" in enc else 15)
        except zlib.error:
            return None
    elif enc not in ("", "identity", "none"):
        return None
    m = re.search(r"charset\s*=\s*[\"']?([\w.:-]+)", ctype) or _META_CHARSET.search(body[:4096])
    charset = m.group(1) if m else "utf-8"
    if isinstance(charset, bytes):
        charset = charset.decode("latin-1")
    try:
        codecs.lookup(charset)
    except LookupError:
        charset = "utf-8"
    if charset.lower() in ("iso-8859-1", "latin-1", "us-ascii", "ascii"):
        charset = "cp1252"  # as browsers read them
    return body[: MAX_HTML * 2].decode(charset, "replace")[:MAX_HTML]


# --- when it was published -----------------------------------------------------------------------

_TAG = re.compile(r"<meta\b[^>]*>", re.I)
_ATTR = re.compile(r"""([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""")
_LD = re.compile(r"""<script\b[^>]*type\s*=\s*["']?application/ld\+json[^>]*>(.*?)</script>""", re.I | re.S)
_TIME = re.compile(r"<time\b[^>]*>", re.I)
_ISO = re.compile(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s]+(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,]\d+)?)?\s*(Z|UTC|GMT|[+-]\d{2}:?\d{2}|[+-]\d{2}(?!\d))?)?", re.I)
# Meta tags that name when an article was first published (lower-cased property, name or itemprop).
PUBLISHED_META = ("article:published_time", "og:article:published_time", "article:published", "datepublished", "published_time",
                  "article.published", "publishdate", "publish-date", "publish_date", "pubdate", "parsely-pub-date", "sailthru.date",
                  "dc.date.issued", "dcterms.issued", "dc.date.created", "dcterms.created", "datecreated", "original-publish-date",
                  "cxenseparse:recs:publishtime", "article_date_original")  # not a bare "date": some pages give the time they were served
_ARTICLE_LD = re.compile(r"article|posting|report|blog|liveblog|webpage", re.I)


def parse_time(s: str) -> tuple[int, bool] | None:
    """A time as unix seconds, and whether it is exact: given to the minute, with its zone."""
    s = (s or "").strip()
    if not s:
        return None
    if re.fullmatch(r"\d{10}(\d{3})?", s):  # seconds or milliseconds since the epoch
        return int(s[:10]), True
    m = _ISO.match(s)
    if m:
        y, mo, d, hh, mm, ss, zone = m.groups()
        try:
            t = calendar.timegm((int(y), int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0), 0, 0, 0))
        except (ValueError, OverflowError):
            return None
        if not (1 <= int(mo) <= 12 and 1 <= int(d) <= 31 and int(hh or 0) < 24 and int(mm or 0) < 60):
            return None
        if hh is None or zone is None:
            return t, False
        z = zone.upper()
        if z not in ("Z", "UTC", "GMT"):
            sign = -1 if z[0] == "-" else 1
            digits = z[1:].replace(":", "")
            t -= sign * (int(digits[:2]) * 3600 + int(digits[2:4] or 0) * 60)
        return t, True
    p = email.utils.parsedate_tz(s)  # "Thu, 19 Nov 2026 10:23:00 +0300"
    if not p:
        return None
    try:
        if p[9] is not None and ":" in s:
            return int(email.utils.mktime_tz(p)), True
        return calendar.timegm(p[:6] + (0, 0, 0)), False
    except (OverflowError, ValueError, TypeError):
        return None


def _attrs(tag: str) -> dict[str, str]:
    out = {}
    for k, a, b, c in _ATTR.findall(tag):
        out.setdefault(k.lower(), html.unescape(a or b or c))
    return out


def _ld_articles(obj, depth: int = 0):
    """Every article (or web page) a JSON-LD block describes, as a dict."""
    if depth > 6:
        return
    if isinstance(obj, list):
        for x in obj:
            yield from _ld_articles(x, depth + 1)
    elif isinstance(obj, dict):
        kind = obj.get("@type", "")
        kind = " ".join(map(str, kind)) if isinstance(kind, list) else str(kind)
        if _ARTICLE_LD.search(kind):
            yield obj
        for k in ("@graph", "mainEntity", "mainEntityOfPage"):
            if isinstance(obj.get(k), (list, dict)):
                yield from _ld_articles(obj[k], depth + 1)


def _ld(page: str):
    """The articles of each JSON-LD block of a page; a block that is not JSON, as its text."""
    for block in _LD.findall(page):
        try:
            yield list(_ld_articles(json.loads(block.strip().rstrip(";"))))
        except (ValueError, RecursionError):
            yield block


def published_of(page: str) -> tuple[int | None, str]:
    """When a page says it was first published, as unix seconds, and where that came from.

    Every exact time the page gives for it (meta tags, JSON-LD, a <time> marked as the publication
    date) is read, and the earliest wins: an article republished or "updated" keeps its first
    date somewhere, and an old one must not pass for new. A page that gives only a day, or a
    time without its zone, has no time (None) and says "vague"; one that gives none, "none"."""
    found: list[tuple[int, str]] = []
    vague = False
    for tag in _TAG.findall(page[:400_000]):
        a = _attrs(tag)
        key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
        if key in PUBLISHED_META and "content" in a:
            r = parse_time(a["content"])
            if r and r[1]:
                found.append((r[0], f"meta:{key}"))
            elif r:
                vague = True
    for block in _ld(page):
        if isinstance(block, str):
            dates = re.findall(r'"date(?:Published|Created)"\s*:\s*"([^"]+)"', block)
        else:
            dates = [str(a[k]) for a in block for k in ("datePublished", "dateCreated") if isinstance(a.get(k), (str, int))]
        for s in dates:
            r = parse_time(s)
            if r and r[1]:
                found.append((r[0], "jsonld"))
            elif r:
                vague = True
    if not found:
        for tag in _TIME.findall(page):
            a = _attrs(tag)
            if "datetime" in a and (re.search(r"\spubdate\b", tag, re.I) or a.get("itemprop", "").lower() in ("datepublished", "datecreated")):
                r = parse_time(a["datetime"])
                if r and r[1]:
                    found.append((r[0], "time"))
                elif r:
                    vague = True
    if found:
        return min(found)
    return None, "vague" if vague else "none"


# --- the article's text --------------------------------------------------------------------------

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr", "keygen"}
SKIP = {"script", "style", "noscript", "template", "svg", "math", "iframe", "object", "canvas", "select", "button", "textarea", "label",
        "nav", "footer", "aside", "header", "figure", "figcaption", "menu", "dialog", "video", "audio", "picture", "head"}
SKIP_ROLES = {"navigation", "banner", "contentinfo", "complementary", "menu", "menubar", "search", "dialog", "alert", "toolbar"}
BLOCK = {"p", "div", "section", "article", "main", "li", "ul", "ol", "dl", "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote",
         "pre", "table", "tbody", "thead", "tr", "td", "th", "br", "hr", "center", "address", "body", "form", "details", "summary", "caption"}
CONTAINER = {"div", "section", "article", "main", "td", "body", "form", "blockquote", "center", "details"}
PROSE = {"p", "div", "section", "article", "main", "td", "blockquote", "pre", "center", "body", "form", "dd"}
HEAD = {"h2", "h3", "h4", "h5", "h6"}
# Class and id words that mark furniture inside a page, and those that mark the article itself.
# Only words that say plainly what a thing is: "video-wrapper" or "meta" may well hold the article.
NEG = re.compile(r"(?:^|[-_ ])(?:comments?|related|share|sharing|social|newsletter|subscri\w*|promo\w*|advert\w*|ads?|ad-?slot|sponsor\w*|"
                 r"cookies?|breadcrumbs?|sidebar|widgets?|recommend\w*|popular|trending|most-?read|tags?|tag-?list|author-?bio|byline|"
                 r"caption|signup|paywall|outbrain|taboola|more-?stories|read-?more|also-?read|see-?also|footer|masthead|navbar|navigation|"
                 r"nav|menu|modal|popup|disclaimer|copyright)(?:$|[-_ ])", re.I)
POS = re.compile(r"(?:^|[-_ ])(?:article|story|entry|post|content|body|text|main|news|detail|prose|rich-?text|single|blog)(?:$|[-_ ])"
                 r"|articlebody|storybody|postbody|entrycontent|newstext", re.I)
BOILER = re.compile(r"^(?:read (?:more|also)|also read|see also|related:|advertisement|subscribe|sign up|follow us|share (?:this|on)|"
                    r"click here|all rights reserved|copyright|©|photo:|image:|credit:|getty images|читайте также|leia também|"
                    r"lesen sie auch|lire aussi|lee también|leggi anche|czytaj (?:też|także)|lees ook|läs också|læs også|les også|"
                    r"listen to (?:this|the) article|add us as a preferred source|join the conversation)", re.I)
# A line that only says when the article was published or updated.
DATELINE = re.compile(r"^(?:published|updated|posted|last updated|publicado|actualizado|pubblicato|aggiornato|veröffentlicht|aktualisiert|"
                      r"publié|mis à jour|опубликовано|обновлено|yayınlanma|opublikowano)\b[^.!?]{0,60}\d", re.I)
# What an article's last lines ask the reader to do, when they are not the article.
CALL = re.compile(r"whatsapp|telegram|google news|newsletter|subscri|podcast|\bapp\b|facebook|instagram|twitter|youtube|tiktok|"
                  r"follow us|sign up|download|abonn|suscr|iscriviti|подпис|канал|takip edin", re.I)
SENTENCE_END = re.compile(r"[.!?…。！？\"'”’»)\]」』؟۔।॥]$")


def _tail(line: str) -> bool:
    """Whether an article's last line is the page's, not the article's: a call to follow or
    subscribe, a lead-in to something taken away ("Read also:"), or a short label."""
    return (len(line) < 300 and bool(CALL.search(line))) or line.endswith(":") or (len(line) < 60 and not SENTENCE_END.search(line))


_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u2060\ufeff"))  # soft hyphens and zero-width spaces (not ZWNJ/ZWJ: words need them)


class _Page(HTMLParser):
    """The pieces of a page an article is found among: each run of text with the elements it
    sits in, how much of it is links, and the elements' marks."""

    def __init__(self, hidden: bool = True):
        super().__init__(convert_charrefs=True)
        self.hidden = hidden  # pass over what the page hides (until its scripts show it)
        self.stack: list[tuple[str, int]] = []  # (tag, node id)
        self.parent: dict[int, int] = {}
        self.tag: dict[int, str] = {}
        self.hint: dict[int, int] = {}  # +1 the article's mark, -1 furniture's
        self.skip = 0  # how deep in the stack an element skipped began, 0 for none
        self.nodes = 0
        self.blocks: list[tuple[str, int, tuple[int, ...], str, int]] = []  # (kind, its element, the ids of all it is in, text, chars in links)
        self._text: list[str] = []
        self._links = 0
        self._in_a = 0
        self.title = ""
        self._in_title = False

    def _flush(self):
        if self._text:
            text = re.sub(r"\s+", " ", "".join(self._text).translate(_INVISIBLE)).strip()
            if text:
                ids = tuple(i for _, i in self.stack)
                kind, node = next(((t, i) for t, i in reversed(self.stack) if t in BLOCK), ("body", 0))
                self.blocks.append((kind, node, ids, text, min(self._links, len(text))))
        self._text, self._links = [], 0

    def _pop_to(self, depth: int):
        while len(self.stack) > depth:
            tag, _ = self.stack.pop()
            if tag == "a":
                self._in_a = max(0, self._in_a - 1)
        if self.skip and len(self.stack) < self.skip:
            self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == "title" and not self.title:
            self._in_title = True
            return
        if tag in BLOCK:
            self._flush()
        if tag in VOID:
            return
        if tag == "body":  # a head left open ends where the body begins
            for i, (t, _) in enumerate(self.stack):
                if t == "head":
                    self._pop_to(i)
                    break
        # an open paragraph ends where a block begins; a list item, where the next one does
        if tag in BLOCK and self.stack and self.stack[-1][0] == "p":
            self._pop_to(len(self.stack) - 1)
        if tag in ("li", "dt", "dd", "tr", "td", "th", "p"):
            stop = {"li": ("ul", "ol"), "dt": ("dl",), "dd": ("dl",), "tr": ("table", "tbody", "thead"), "td": ("tr",), "th": ("tr",),
                    "p": tuple(CONTAINER)}[tag]
            same = ("dt", "dd") if tag in ("dt", "dd") else ("td", "th") if tag in ("td", "th") else (tag,)
            for i in range(len(self.stack) - 1, -1, -1):
                t = self.stack[i][0]
                if t in same:
                    self._pop_to(i)
                    break
                if t in stop:
                    break
        self.nodes += 1
        n = self.nodes
        self.parent[n] = self.stack[-1][1] if self.stack else 0
        self.tag[n] = tag
        a = dict(attrs)
        marks = f"{a.get('class') or ''} {a.get('id') or ''}"
        if tag in ("article", "main") or (a.get("itemprop") or "").lower() == "articlebody" or (a.get("role") or "") == "main":
            self.hint[n] = 1
        else:  # marked as both, as "post-content has-sidebar" is: neither
            self.hint[n] = (1 if POS.search(marks) else 0) - (1 if NEG.search(marks) else 0)
        self.stack.append((tag, n))
        if tag == "a":
            self._in_a += 1
        if not self.skip and (tag in SKIP or (a.get("role") or "").lower() in SKIP_ROLES or self.hidden and (
                "hidden" in a or re.search(r"display\s*:\s*none", a.get("style") or ""))):
            self.skip = len(self.stack)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack and self.stack[-1][0] == tag:
            self._pop_to(len(self.stack) - 1)

    def handle_endtag(self, tag):
        if tag == "title" and self._in_title:
            self._in_title = False
            return
        if tag in BLOCK:
            self._flush()
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                self._pop_to(i)
                return

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self.skip:
            return
        self._text.append(data)
        if self._in_a:
            self._links += len(data.strip())

    def close(self):
        super().close()
        self._flush()


def _density(text: str, links: int) -> float:
    return links / max(1, len(text))


def extract(page: str) -> tuple[str, str, str]:
    """A news page's title (its <title>), the article's main text, a paragraph to a line, and
    where the text came from: "html"; "hidden", what the page hides until its scripts run; or
    "jsonld", the articleBody it describes itself with, for a page its scripts write."""
    title, text = main_text(page)
    how = "html"
    if len(text) < MIN_CHARS:
        title2, text2 = main_text(page, hidden=False)
        if len(text2) > len(text):
            title, text, how = title2 or title, text2, "hidden"
    if len(text) < MIN_CHARS:
        body = article_body(page)
        if len(body) > len(text):
            text, how = body, "jsonld"
    return title, text, how


def article_body(page: str) -> str:
    """The articleBody of the first article a page's JSON-LD describes, as plain text."""
    for block in _ld(page):
        for a in block if isinstance(block, list) else []:
            body = a.get("articleBody")
            if isinstance(body, str) and body.strip():
                text = html_text(html.unescape(body)).translate(_INVISIBLE)
                lines = (re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.split("\n"))
                return "\n".join(ln for ln in lines if ln)
    return ""


def main_text(page: str, hidden: bool = True) -> tuple[str, str]:
    """The page's <title> and the article's main text, from its markup.

    The article is the element holding the most prose: each paragraph (not a link, not a list
    item, not a heading) counts its length to the element that holds it and half to the one
    above, more within an element marked as the article and less within one marked as
    furniture. Its paragraphs and subheadings are the text, without those in furniture inside
    it, those that are mostly links, and short or stock lines."""
    p = _Page(hidden)
    try:
        p.feed(page)
        p.close()
    except (AssertionError, ValueError):  # html.parser on markup it cannot follow
        p._flush()
    score: dict[int, float] = defaultdict(float)
    for kind, node, _, text, links in p.blocks:
        if kind not in PROSE or len(text) < 25 or _density(text, links) > 0.3 or not node:
            continue
        own = node if p.tag.get(node) in CONTAINER else p.parent.get(node, 0)
        weight = len(text) + 20 * text.count(",") + 20 * text.count("，") + 20 * text.count("、")
        score[own] += weight
        score[p.parent.get(own, 0)] += weight / 2
    title = re.sub(r"\s+", " ", p.title).strip()
    if not score:
        return title, ""

    def adjusted(n: int) -> float:
        f = 1.0
        m = n
        while m:  # an element inside furniture is furniture; inside the article, the article
            h = p.hint.get(m, 0)
            if h < 0:
                f *= 0.2
            elif h > 0 and m == n:
                f *= 1.25
            m = p.parent.get(m, 0)
        return score[n] * f

    best = max(score, key=lambda n: (adjusted(n), -n))
    lines: list[str] = []
    for kind, _, ids, text, links in p.blocks:
        if best not in ids:
            continue
        below = ids[ids.index(best) + 1 :]
        if any(p.hint.get(i, 0) < 0 for i in below):
            continue
        if _density(text, links) > (0.2 if kind in ("li", "td", "th", "dt") else 0.5):
            continue
        if len(text) < (8 if kind in HEAD else 20) or BOILER.match(text) or not re.search(r"[^\W\d_]", text):
            continue
        if len(text) < 100 and DATELINE.match(text):
            continue
        if lines and lines[-1] == text:
            continue
        lines.append(text)
    while len(lines) > 1 and _tail(lines[-1]):
        lines.pop()
    return title, "\n".join(lines)


def og_title(page: str) -> str:
    for tag in _TAG.findall(page[:400_000]):
        a = _attrs(tag)
        if (a.get("property") or a.get("name") or "").lower() in ("og:title", "twitter:title") and a.get("content", "").strip():
            return re.sub(r"\s+", " ", a["content"]).strip()
    return ""


# --- the harvest ---------------------------------------------------------------------------------

_SLD = {"co", "com", "net", "org", "gov", "ac", "edu", "ne", "or", "go", "gob", "gouv", "nic", "mil", "info", "biz", "ltd", "plc", "sch",
        "nom", "web", "gen", "ind", "firm", "in", "nhs", "lg", "ed", "gr", "res", "nat"}


def site_of(url: str) -> str:
    """The site a URL is on, roughly its registered domain: news.example.co.uk is example.co.uk."""
    host = (urllib.parse.urlsplit(url).hostname or "").lower().rstrip(".")
    parts = host.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in _SLD:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _lang(text: str) -> tuple[str, float]:
    from .langs import model

    (label,), (prob,) = model().predict(text[:READ_LANG].replace("\n", " "), k=1)
    return label.removeprefix("__label__"), float(prob)


def quotas(counts: dict[str, int], cap: int, shares: dict[str, float], other: float) -> dict[str, int]:
    """cap shared among languages by their weight, at least one each while it lasts; a language
    with fewer candidates than its share gives the rest to the others."""
    w = {lang: max(1, round(100 * shares.get(lang, other))) for lang in counts}
    q = dict.fromkeys(counts, 0)
    open_ = {lang for lang, n in counts.items() if n > 0}
    left = cap
    while left > 0 and open_:
        total = sum(w[lang] for lang in open_)
        given = 0
        for lang in sorted(open_, key=lambda x: (-w[x], x)):
            take = min(max(1, left * w[lang] // total), counts[lang] - q[lang], left - given)
            q[lang] += take
            given += take
            if q[lang] >= counts[lang]:
                open_.discard(lang)
            if given >= left:
                break
        if not given:
            break
        left -= given
    return q


def _paragraph_keys(text: str) -> set[bytes]:
    return {hashlib.sha1(re.sub(r"\W+", "", line.lower()).encode()).digest()[:10] for line in text.split("\n") if len(line) >= 150}


class _Stats(Counter):
    def line(self, name: str) -> str:
        return (f"{name}: {self['records']} records, {self['html']} HTML pages, {self['exact']} with an exact date "
                f"({self['vague']} only a day or no zone), {self['window']} published in the window, {self['capped']} passed over "
                f"for their site, {self['short']} too short, {self['unsure']} of no clear language, {self['candidates']} articles, "
                f"{self['syndicated']} of them another site's story")


def scan(path: Path, warc: str, start: int, end: int, rng, per_site: int = PER_SITE, sites: dict | None = None,
         stats: Counter | None = None) -> dict[str, list]:
    """Read one file: the articles published in [start, end), by site, at most per_site of each,
    those with the smallest keys. Pass sites to add to an earlier file's."""
    sites = {} if sites is None else sites
    st = Counter() if stats is None else stats
    with open(path, "rb") as f:
        for offset, length, h, block in records(f):
            if h.get("warc-type") != "response":
                continue
            st["records"] += 1
            fetched = parse_time(h.get("warc-date", ""))
            if fetched:
                st["last_fetched"] = max(st["last_fetched"], fetched[0])
            page = html_of(block)
            if page is None:
                continue
            st["html"] += 1
            t, how = published_of(page)
            if t is None:
                st[how] += 1
                continue
            st["exact"] += 1
            if fetched and t > fetched[0] + SKEW:  # published after it was fetched: the page's clock is wrong
                st["future"] += 1
                continue
            if not start <= t < end:
                st["outside"] += 1
                continue
            st["window"] += 1
            url = h.get("warc-target-uri", "")
            rid = h.get("warc-record-id", "").strip("<>")
            site = site_of(url)
            key = (rng.child(rid).next64(), rid)
            have = sites.setdefault(site, [])
            same = next((c for c in have if c["url"] == url), None)
            if (len(have) >= per_site and key >= max(c["key"] for c in have)) or (same and same["key"] <= key):
                st["capped"] += 1
                continue
            title, text, text_from = extract(page)
            if len(text) < MIN_CHARS:
                st["short"] += 1
                continue
            lang, prob = _lang(text)
            if prob < MIN_LANG_P:
                st["unsure"] += 1
                continue
            st["candidates"] += 1
            doc = Doc(source="ccnews", id=rid, url=url, lang=lang, license=LICENSE, published=t,
                      title=og_title(page) or title, text=text, topic=site,
                      extra={"warc": warc, "offset": offset, "length": length, "fetched": fetched[0] if fetched else 0,
                             "date_from": how, "text_from": text_from, "lang_p": round(prob, 3)})
            if same:
                have.remove(same)
            have.append({"key": key, "url": url, "doc": doc})
            have.sort(key=lambda c: c["key"])
            del have[per_site:]
    return sites


def choose(sites: dict[str, list], cap: int, shares: dict | None = None, stats: Counter | None = None) -> list[Doc]:
    """From the articles each site kept, a story once (the first by key of those that share a long
    paragraph), then cap of them shared among languages, by key."""
    from .select import OTHER, SHARES

    pool = sorted((c for have in sites.values() for c in have), key=lambda c: c["key"])
    seen: set[bytes] = set()
    by_lang: dict[str, list[Doc]] = defaultdict(list)
    for c in pool:
        keys = _paragraph_keys(c["doc"].text)
        if keys & seen:
            if stats is not None:
                stats["syndicated"] += 1
            continue
        seen |= keys
        by_lang[c["doc"].lang].append(c["doc"])
    q = quotas({lang: len(v) for lang, v in by_lang.items()}, cap, shares or SHARES, OTHER)
    return [d for lang in sorted(by_lang) for d in by_lang[lang][: q[lang]]]


def harvest(start: int, end: int, cap: int, rng, log=print, per_site: int = PER_SITE, after: int = 0, keep: bool = False,
            shares: dict | None = None) -> list[Doc]:
    """At most cap news articles published in [start, end), from the CC-NEWS files fetched then,
    and from the `after` files that follow them. A file is deleted from the cache once read,
    unless keep. A window whose files are not all published yet is harvested from those that
    are, and says so."""
    paths = [p for ym in _months(start, end + after * 4 * 3600) if ym <= time.gmtime()[:2] for p in listing(*ym)]
    files, closed = window_files(paths, start, end, after)
    if not files:
        log("  ccnews: no file of the window is listed yet")
        return []
    sites: dict[str, list] = {}
    stats: Counter = _Stats()
    for p in files:
        t0 = time.time()
        local = _local(p, log=log)
        try:
            before = stats["candidates"]
            scan(local, p.rsplit("/", 1)[-1], start, end, rng, per_site, sites, stats)
            log(f"  ccnews {p.rsplit('/', 1)[-1]}: {stats['candidates'] - before} articles in {time.time() - t0:.0f}s")
        finally:
            if not keep:
                local.unlink(missing_ok=True)
    if not closed and (after or stats["last_fetched"] < end):  # the last file read is closed, and may still end before the window
        log(f"  ccnews: the window is not all published yet: its files so far were fetched until {iso(stats['last_fetched'])}")
    docs = choose(sites, cap, shares, stats)
    log("  " + stats.line("ccnews") + f"; kept {len(docs)} from {len({d.topic for d in docs})} sites "
        f"in {len({d.lang for d in docs})} languages")
    return docs

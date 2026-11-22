"""Fresh text: written and published in the day before a round's draw, in hours the draw picks,
so nobody could have known which of it a round would hold.

A harvest collects documents from sources that publish all the time and need no key, over one
hour of the clock (harvest.py). Each document keeps an immutable id and a URL that shows it as
it was harvested (a revision id, a post id), so anyone can check it against its source. The
harvest is written as a snapshot, and its sha256 is public before any draw that picks it.

The documents are then turned into exam items (passages, and answers given their question),
placed in corners (source, language, topic), and selected with the round's seed: at most a few
items from any corner, languages in proportion to their share of the web.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

UA = {"User-Agent": "stet-exam-harvester/0.1 (https://www.stetrun.com)"}


@dataclass(frozen=True)
class Doc:
    source: str  # "wikipedia", "ccnews", "hackernews", ...
    id: str  # immutable within the source: "en:1234567" (a Wikipedia revision), a post's id
    url: str  # shows this document as it was harvested
    lang: str
    license: str
    published: int  # unix seconds
    title: str
    text: str
    topic: str = ""  # a source's own coarse topic: a Wikidata class, a site and tag
    question: str = ""  # for an answer: the question it answers
    extra: dict = field(default_factory=dict)


GAP = 0.15  # seconds between requests to one host: the sources are shared, and say so when pressed
_last: dict[str, float] = {}
_pace = threading.Lock()  # the harvest reads its sources at once, and some share a host


MAX_BODY = 32 << 20  # bytes of an answer that are read: the largest real one is a small part of this
MAX_NAMED = 2 << 20  # of an answer from a server its poster named
NAMED_SECONDS = 30.0  # and the time such an answer is given to arrive in
_NAMED = re.compile(r"https://([A-Za-z0-9.-]+)(?::\d{1,5})?(?:/[!-~]*)?")


class Refused(ValueError):
    """An address that is not asked, or an answer that is not read to its end."""


def public(url: str) -> str:
    """The host of an https URL that only names a public machine; Refused for any other. A URL a
    poster chose (their server's, in their DID document) is fetched only if it passes this: every
    address its host resolves to must be one anybody can reach, so none on this machine or its
    network."""
    m = _NAMED.fullmatch(url)
    if not m or "." not in m.group(1).strip("."):
        raise Refused(f"not a public https address: {url[:200]!r}")
    host = m.group(1)
    try:
        found = {info[4][0] for info in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    except (OSError, UnicodeError) as e:
        raise Refused(f"{host} does not resolve: {e}") from None
    for a in found:
        ip = ipaddress.ip_address(a.split("%")[0])
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if not ip.is_global or ip.is_multicast:  # loopback, private, link-local, reserved, shared, unspecified
            raise Refused(f"{host} resolves to {ip}, which is not a public address")
    if not found:
        raise Refused(f"{host} does not resolve")
    return host


class _SameHost(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to the same https host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = urllib.parse.urlsplit(newurl)
        if new.scheme != "https" or new.netloc.lower() != req.host.lower():
            raise Refused(f"{req.host} redirects elsewhere: {newurl[:200]!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_named = urllib.request.build_opener(_SameHost)


def _body(r, limit: int, seconds: float | None) -> bytes:
    """An answer's bytes: Refused past limit, or when seconds have passed and it is not all here."""
    if seconds is None:
        data = r.read(limit + 1)
    else:
        until, data = time.monotonic() + seconds, bytearray()
        while len(data) <= limit and (piece := r.read1(min(1 << 16, limit + 1 - len(data)))):
            data += piece
            if time.monotonic() > until:
                raise Refused(f"an answer still arriving after {seconds:.0f}s")
    if len(data) > limit:
        raise Refused(f"an answer larger than {limit} bytes")
    return data


def get_json(url: str, params: dict, tries: int = 6, pause: float = 2.0, *, limit: int = MAX_BODY, named: bool = False):
    """GET a JSON document politely: one request at a time per host, a gap between them, and
    when a host says it is too busy (429, 503), as long a wait as it asks for. An answer larger
    than limit bytes is Refused. named is for a URL a poster chose: it is asked only if public()
    passes it, is not followed to another host, and is given NAMED_SECONDS to arrive."""
    host = urllib.parse.urlsplit(url).netloc
    q = urllib.parse.urlencode(params)
    for i in range(tries):
        if named:
            public(url)  # each try: what a name resolves to can change
        with _pace:  # this request's turn at the host, a gap after the last one's
            at = max(time.time(), _last.get(host, 0) + GAP)
            _last[host] = at
        if (wait := at - time.time()) > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(f"{url}?{q}", headers={**UA, "Accept-Encoding": "identity"})
            with (_named.open(req, timeout=15) if named else urllib.request.urlopen(req, timeout=60)) as r:
                return json.loads(_body(r, limit, NAMED_SECONDS if named else None))
        except Refused:
            raise
        except urllib.error.HTTPError as e:
            if i == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            asked = e.headers.get("Retry-After", "")
            time.sleep(min(120.0, float(asked)) if asked.isdigit() else pause * 2**i)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(pause * 2**i)


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "pre", "h1", "h2", "h3", "h4", "tr", "blockquote"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_text(html: str) -> str:
    p = _Text()
    p.feed(html or "")
    text = "".join(p.parts)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text)).strip()


def iso(t: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def write_snapshot(docs: list[Doc], path: Path, meta: dict) -> str:
    """Write a harvest and return its sha256: one JSON line of metadata, then one per document."""
    lines = [json.dumps(meta, sort_keys=True, ensure_ascii=False)]
    lines += [json.dumps(asdict(d), sort_keys=True, ensure_ascii=False) for d in sorted(docs, key=lambda d: (d.source, d.id))]
    blob = ("\n".join(lines) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return hashlib.sha256(blob).hexdigest()


def read_snapshot(path: Path) -> tuple[dict, list[Doc]]:
    # Cut at the newline write_snapshot wrote and at nothing else. A document's text may hold
    # U+2028, U+2029 or U+0085, which json writes as they are and str.splitlines() cuts at.
    lines = [x for x in path.read_bytes().decode().split("\n") if x]
    return json.loads(lines[0]), [Doc(**json.loads(x)) for x in lines[1:]]

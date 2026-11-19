"""Short public posts on Mastodon, in many languages, and replies to them, written in the window.

A harvest reads the local public timelines of servers that let anyone read them without an
account, covering the window by status id (Mastodon's ids encode the moment a post was made, so
a page can start at any time). A public timeline shows a thread's later posts but not replies
from other people, so replies come from threads: those of posts made in the window or in the hour
before it, keeping only the replies written in the window. A reply there may come from any server
in the fediverse, and is kept with the post it answers. Servers are read side by side, each by
one worker, so no server sees more than one request at a time.

Servers whose terms or robots.txt forbid this use are not read, and their users' replies seen
on other servers are left out too. So is a post that is boosted, marked sensitive, edited or
written by a bot, and any post by an author who asked not to be indexed or said no in their bio.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import UA, Doc, get_json, html_text
from .bluesky import OPT_OUT, PER_AUTHOR, Candidate, choose, language, parse_time, rank, readable, width

# Large servers, and the largest open ones in languages the large ones barely write in, whose
# local timeline anyone may read (checked in November 2026: mastodon.social, mastodon.online,
# mstdn.social, masto.ai and others need an account). pawoo.net is left out for what it hosts, and
# mastodon.eus for taking a minute to answer each request (Basque has no share of its own anyway).
INSTANCES = (
    "mastodon.world", "mas.to", "fosstodon.org", "hachyderm.io", "universeodon.com", "mastodonapp.uk", "techhub.social",
    "mathstodon.xyz", "mastodon.ie", "aus.social", "kolektiva.social", "mastodon.green",  # English, mostly
    "mstdn.jp", "social.vivaldi.net", "social.mikutter.hachune.net", "mstdn.maud.io",  # Japanese
    "norden.social", "ruhr.social", "berlin.social",  # German
    "piaille.fr", "mastodon.top",  # French
    "mastodon.nl", "mastodon.uno", "masto.es", "tkz.one", "masto.pt", "bolha.us", "mastodon.com.br",
    "mastodon.gal", "101010.pl", "qdon.space", "mastodon.com.tr", "mastodon.sk", "mstdn.dk",
    "witter.cz", "mast.dragon-fly.club",
)
# Servers whose own terms or robots.txt say no to this: not read, and their users' posts seen
# elsewhere left out. Read in November 2026; a server not listed was not read beyond its robots.txt.
EXCLUDED = {
    "chaos.social": "its terms forbid machine-learning or research use of posts without the users' consent",
    "mastodon.bida.im": "its about page forbids research or machine-learning use without the users' consent",
    "infosec.exchange": "its terms forbid scrapers and data mining",
    "mastodon.art": "its about page forbids crawling it into a database for analysis",
    "mementomori.social": "its terms forbid bulk automated collection",
    "troet.cafe": "its robots.txt turns away research collectors of its public timeline",
    "muenchen.social": "its robots.txt turns away research collectors of its public timeline",
    "social.tchncs.de": "its robots.txt disallows the API to every robot",
    "planet.moe": "its terms forbid AI use of posts without the users' consent",
}
LICENSE = "author's copyright (public fediverse post; no license granted to others)"
AGENT = UA["User-Agent"].split("/")[0]  # this harvester's name, as robots.txt files address it
MIN_QUESTION = 20
STRATA = 4  # the window is read backwards in this many parts,
PAGES = 8  # each at most this many pages of 40, so a busy server is still read across the whole window
REPLY_SHARE = 0.5
LOOKBACK = 3600  # seconds before the window whose posts' threads are also opened, for the replies written in the window
LOOKBACK_PAGES = 2
REPLY_VISIBILITY = ("public",)  # not "unlisted": its author asked for it to be kept out of public view
WORKERS = 8  # servers read at once
BATCH = 32  # threads fetched at once, then read in their seeded order
WANT = 2  # candidate replies from other people wanted per reply kept, before threads stop being opened
MAX_THREADS = 400  # threads opened at most in one harvest
MAX_DELAY = 10.0  # a server asking robots to wait longer than this between requests is not read
TRIES = 2  # tries per request, and failures before a server is given up for the rest of a harvest
SLOW = 30.0  # seconds: a request that takes longer counts as a failure, though it was answered
REQUIRE_INDEXABLE = True  # only authors who opted in to their server's full-text search: who said yes, not only who did not say no
_LEADING = re.compile(r"^(?:\s*@[\w.-]+(?:@[\w-]+(?:\.[\w-]+)+)?[\s,:]*)+")
_delay: dict[str, float] = {}
_last: dict[str, float] = {}
_failed: Counter = Counter()


def snowflake(t: int) -> int:
    """The smallest status id a server would give a post made at unix second t."""
    return (t * 1000) << 16


def robots(host: str) -> tuple[bool, float]:
    """Whether the server's robots.txt lets this harvester read the API (and does not say no to
    putting its content into AI models), and the crawl delay it asks for. A robots.txt that is
    missing allows everything; one that cannot be read allows nothing (RFC 9309)."""
    try:
        req = urllib.request.Request(f"https://{host}/robots.txt", headers=UA)
        with urllib.request.urlopen(req, timeout=15) as r:
            text = r.read(1 << 20).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return 400 <= e.code < 500, 0.0
    except Exception:
        return False, 0.0
    return allowed(host, text)


def allowed(host: str, text: str) -> tuple[bool, float]:
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(text.splitlines())
    ok = all(rp.can_fetch(AGENT, f"https://{host}{p}") for p in ("/api/v1/timelines/public", "/api/v1/statuses/1/context"))
    if re.search(r"^\s*content-signal\s*:.*\bai-input\s*=\s*no\b", text, re.I | re.M):
        ok = False
    return ok, float(rp.crawl_delay(AGENT) or 0.0)


def _get(host: str, path: str, params: dict):
    """A request to a server, as often as its robots.txt allows. A server that fails TRIES
    requests (not counting a status that is gone) is given up for the rest of the harvest: there
    are many servers, and a dead one would otherwise cost minutes a request."""
    if _failed[host] >= TRIES:
        raise ConnectionError(f"{host} has failed {_failed[host]} requests; given up")
    wait = _last.get(host, 0.0) + _delay.get(host, 0.0) - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[host] = time.time()
    t0 = time.time()
    try:
        out = get_json(f"https://{host}{path}", params, tries=TRIES)
    except urllib.error.HTTPError as e:
        if e.code not in (404, 410):
            _failed[host] += 1
        raise
    except Exception:
        _failed[host] += 1
        raise
    if time.time() - t0 > SLOW:
        _failed[host] += 1
    return out


def timeline(host: str, start: int, end: int, strata: int = STRATA, pages: int = PAGES) -> list[dict]:
    """The server's local public timeline over [start, end), read backwards from the end: each of
    strata parts of the window gets up to pages pages, and a part the reading has already passed
    (as it soon does on a quiet server) is not read again."""
    out = []
    floor = snowflake(end)  # every status from floor to the window's end has been read
    for k in reversed(range(strata)):
        a, b = start + (end - start) * k // strata, start + (end - start) * (k + 1) // strata
        if b <= a or floor <= snowflake(a):
            continue
        max_id = min(floor, snowflake(b))
        unbroken = max_id == floor  # carrying on from where the reading got to, rather than starting this part afresh
        for _ in range(pages):
            page = _get(host, "/api/v1/timelines/public", {"local": "true", "limit": 40, "max_id": max_id})
            if not page:
                if unbroken:
                    floor = 0  # there is nothing older
                break
            out += page
            low = min(int(s["id"]) for s in page)
            if low >= max_id:
                break
            max_id = low
            if unbroken:
                floor = low
            if low < snowflake(a):
                break
    return out


def text_of(status: dict) -> str:
    """A status as plain text, without the mentions a client puts before a reply."""
    return _LEADING.sub("", html_text(status.get("content") or "")).strip()


def home(account: dict, host: str) -> str:
    acct = account.get("acct") or ""
    return acct.split("@", 1)[1].lower() if "@" in acct else host


def declined(account: dict, host: str) -> str:
    """Why an author's posts may not be used, or ""."""
    where = home(account, host)
    if where in EXCLUDED:
        return "from a server that forbids this"
    if account.get("noindex"):
        return "author asked not to be indexed"
    if REQUIRE_INDEXABLE and account.get("indexable") is not True:
        return "author not opted in to search"
    bio = " ".join([html_text(account.get("note") or "")] +
                   [f"{f.get('name', '')} {html_text(f.get('value') or '')}" for f in account.get("fields") or []])
    if OPT_OUT.search(bio):
        return "author said no in their bio"
    return ""


def quotable(status: dict, host: str) -> str:
    """Why a status may not be shown as the post a reply answers, or ""."""
    if status.get("visibility") != "public":
        return "answers a post that is not public"
    if status.get("sensitive") or status.get("spoiler_text"):
        return "answers a sensitive post"
    why = declined(status.get("account") or {}, host)
    if why:
        return f"answers a post that may not be used ({why})"
    if width(text_of(status)) < MIN_QUESTION:
        return "answers a post with too little text"
    return ""


def candidate(host: str, s: dict, start: int, end: int, salt: int, parent: dict | None = None, lid=None):
    """A status as (Candidate, "") if it might be kept, else (None, the reason). A reply's parent
    is given if it is at hand; if not, it is fetched once the reply is about to be chosen."""
    try:
        return _candidate(host, s, start, end, salt, parent, lid)
    except (TypeError, AttributeError, ValueError, KeyError):
        return None, "malformed"


def _candidate(host, s, start, end, salt, parent, lid):
    if s.get("reblog"):
        return None, "a boost"
    if s.get("visibility") not in (REPLY_VISIBILITY if s.get("in_reply_to_id") else ("public",)):
        return None, "not public"
    if s.get("sensitive") or s.get("spoiler_text"):
        return None, "sensitive"
    if s.get("edited_at"):
        return None, "edited"
    a = s.get("account") or {}
    if a.get("bot") or a.get("group"):
        return None, "a bot or group"
    why = declined(a, host)
    if why:
        return None, why
    created = parse_time(s.get("created_at") or "")
    if created is None or not start <= created < end:
        return None, "written outside the window"
    text = text_of(s)
    if not readable(text):
        return None, "short, or mostly links and tags"
    lang = language(text, [s.get("language") or ""], lid or identify)
    if lang is None:
        return None, "language not clear"
    question = ""
    if s.get("in_reply_to_id") and parent is not None:
        why = quotable(parent, host)
        if why:
            return None, why
        question = text_of(parent)
    key = s.get("uri") or f"https://{host}/statuses/{s['id']}"
    return Candidate(key=key, source="mastodon-reply" if s.get("in_reply_to_id") else "mastodon", lang=lang,
                     author=a.get("url") or f"{a.get('acct', '')}@{host}", text=text, rank=rank(salt, key),
                     extra={"host": host, "id": str(s["id"]), "url": s.get("url") or key, "created": created, "home": home(a, host),
                            "parent_id": str(s.get("in_reply_to_id") or ""), "question": question,
                            "parent_url": (parent or {}).get("url") or (parent or {}).get("uri") or "",
                            "self_reply": bool(s.get("in_reply_to_id")) and s.get("in_reply_to_account_id") == a.get("id"),
                            "visibility": s.get("visibility"), "indexable": a.get("indexable"),
                            "discoverable": a.get("discoverable")}), ""


def _server(host: str, start: int, end: int):
    """One server's window, and the hour before it (for threads): (statuses, earlier, what to log)."""
    try:
        ok, delay = robots(host)
        if not ok or delay > MAX_DELAY:
            return [], [], f"  mastodon {host}: " + ("robots.txt says no" if not ok else f"asks robots to wait {delay:.0f}s")
        _delay[host] = delay
        got = timeline(host, start, end)
        try:
            earlier = timeline(host, start - LOOKBACK, start, strata=1, pages=LOOKBACK_PAGES)
        except Exception:
            earlier = []
        return got, earlier, ""
    except Exception as e:  # one server failing does not stop the harvest
        return None, [], f"  mastodon {host}: {type(e).__name__}: {e}"


def _fetch(jobs: list[tuple[str, str]]) -> list:
    """GET each (server, path), side by side across servers but one at a time on any server; each
    result, or the exception it raised, in the order asked."""
    by_host: dict[str, list[int]] = defaultdict(list)
    for i, (host, _) in enumerate(jobs):
        by_host[host].append(i)
    out: list = [None] * len(jobs)

    def run(host):
        for i in by_host[host]:
            try:
                out[i] = _get(host, jobs[i][1], {})
            except Exception as e:
                out[i] = e

    with ThreadPoolExecutor(max(1, min(WORKERS, len(by_host)))) as pool:
        list(pool.map(run, list(by_host)))
    return out


def identify(text: str):  # looked up when called, so that a test can stand in for lid.176
    from .bluesky import identify as lid

    return lid(text)


def harvest(start: int, end: int, cap: int, rng, log=print, *, instances=INSTANCES, shares: dict | None = None) -> list[Doc]:
    """Up to cap posts and replies written on Mastodon servers in [start, end), chosen with rng."""
    t0 = time.time()
    end = min(end, int(t0))
    if cap <= 0 or end <= start:
        return []
    salt = rng.child("order").next64()
    _failed.clear()
    seen: dict[str, tuple[str, dict]] = {}  # by ActivityPub uri: the server it was read from, and the status
    earlier: dict[str, tuple[str, dict]] = {}  # posts from the hour before the window, whose threads may have replies in it
    read = []
    hosts = [h for h in instances if h not in EXCLUDED]
    with ThreadPoolExecutor(max(1, min(WORKERS, len(hosts)))) as workers:
        results = list(workers.map(lambda h: _server(h, start, end), hosts))
    for host, (got, before, said) in zip(hosts, results, strict=True):  # in the list's order, however they finished
        if said:
            log(said)
        if got is None:
            continue
        if got or not said:
            read.append(host)
        for s in got:
            seen.setdefault(s.get("uri") or f"{host}/{s['id']}", (host, s))
        for s in before:
            earlier.setdefault(s.get("uri") or f"{host}/{s['id']}", (host, s))
    by_id = {(host, str(s["id"])): s for host, s in seen.values()}
    pool: dict[str, Candidate] = {}
    why: Counter = Counter()
    for host, s in seen.values():
        p = by_id.get((host, str(s.get("in_reply_to_id")))) if s.get("in_reply_to_id") else None
        c, reason = candidate(host, s, start, end, salt, parent=p)
        why[reason or "candidate"] += 1
        if c is not None:
            pool.setdefault(c.key, c)

    # Replies from other people: from the threads of posts made in the window or the hour before it.
    roots = {**earlier, **seen}
    threads = sorted((rank(salt, "thread " + uri), host, s) for uri, (host, s) in roots.items()
                     if (s.get("replies_count") or 0) > 0 and not s.get("reblog") and not quotable(s, host)
                     and start - LOOKBACK <= (parse_time(s.get("created_at") or "") or 0) < end)[:MAX_THREADS]
    want = WANT * REPLY_SHARE * cap
    opened = 0
    for b in range(0, len(threads), BATCH):
        if sum(c.source == "mastodon-reply" and not c.extra["self_reply"] for c in pool.values()) >= want:
            break
        batch = threads[b:b + BATCH]
        for (_, host, s), ctx in zip(batch, _fetch([(h, f"/api/v1/statuses/{x['id']}/context") for _, h, x in batch]), strict=True):
            if sum(c.source == "mastodon-reply" and not c.extra["self_reply"] for c in pool.values()) >= want:
                break  # what was fetched beyond this point is not used, so the outcome does not depend on BATCH
            if isinstance(ctx, Exception):
                log(f"  mastodon {host}: thread {s['id']}: {type(ctx).__name__}: {ctx}")
                continue
            opened += 1
            statuses = {str(x["id"]): x for x in [s, *ctx.get("ancestors", []), *ctx.get("descendants", [])]}
            for x in ctx.get("descendants", []):
                parent = statuses.get(str(x.get("in_reply_to_id")))
                key = x.get("uri") or ""
                if parent is None or key in pool or key in seen:
                    continue
                c, reason = candidate(host, x, start, end, salt, parent=parent)
                why[(reason or "candidate") + " (reply)"] += 1
                if c is not None:
                    pool[c.key] = c
    log(f"  mastodon: {len(seen)} statuses from {len(read)} servers, {opened} threads opened, {time.time() - t0:.0f}s; "
        f"{len(pool)} candidates: {dict(why.most_common())}")

    dropped: Counter = Counter()

    def check(todo: list[Candidate]) -> set[str]:
        """Fetch the post a thread's later part answers, where it was not at hand."""
        ok = set()
        for c in todo:
            x = c.extra
            if c.source == "mastodon-reply" and not x["question"]:
                try:
                    p = _get(x["host"], f"/api/v1/statuses/{x['parent_id']}", {})
                except Exception:
                    dropped["answers a post that is gone"] += 1
                    continue
                reason = quotable(p, x["host"])
                if reason:
                    dropped[reason] += 1
                    continue
                x["question"], x["parent_url"] = text_of(p), p.get("url") or p.get("uri") or ""
            ok.add(c.key)
        return ok

    chosen = choose(list(pool.values()), cap, check, shares=shares, per_author=PER_AUTHOR,
                    split={"mastodon": 1 - REPLY_SHARE, "mastodon-reply": REPLY_SHARE})
    docs = []
    for c in chosen:
        x = c.extra
        extra = {"uri": c.key, "visibility": x["visibility"], "indexable": x["indexable"], "discoverable": x["discoverable"]}
        if c.source == "mastodon-reply":
            extra.update(parent=x["parent_url"], self_reply=x["self_reply"])
        docs.append(Doc(source=c.source, id=f"{x['host']}:{x['id']}", url=x["url"], lang=c.lang, license=LICENSE,
                        published=x["created"], title="", text=c.text, topic=x["home"], question=x["question"], extra=extra))
    kinds = Counter(d.source for d in docs)
    log(f"  mastodon: {kinds['mastodon']} posts and {kinds['mastodon-reply']} replies in {len(set(d.lang for d in docs))} languages "
        f"({time.time() - t0:.0f}s); checked and left out: {dict(dropped.most_common())}")
    return docs

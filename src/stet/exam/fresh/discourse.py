"""Posts on public Discourse forums written in the window, each reply with the thread it answers.

A Discourse forum serves its public posts as JSON to anyone, without a key or an account:
/posts.json gives the newest fifty with their authors' own Markdown (`raw`), `before=<post id>`
the fifty below an id, and /posts/by_number/<topic>/<number>.json one post of a topic. A few dozen
large forums are read, technical and not, in several languages. A forum that fails is left out
and the rest kept.

Most forums write fewer than fifty posts in the hours a harvest looks back over, and one request
lists a whole window. On a busier one the window's end is found among the post ids (they grow
with time) by where its time falls between posts already seen, and the window is read back from
its end, at most MAX_POSTS posts of it: a forum busier than that gives the end of its window.

A reply is kept with its question, the thread so far: the topic's title and opening post, then
the posts it answers, nearest last. Those are the post it was written in reply to and what that
one replied to in turn; for a reply to the topic at large, the posts it quotes, or else the post
just before it. Only the reply is scored. A topic's opening post is kept on its own.

What is kept is people's own prose: what a post quotes is taken out (it is in the question), and
polls, uploads, HTML comments and the marks of Markdown and BBCode go; code and links do not count
towards its length, nor do lines that several of a forum's posts share (a category's template).
Posts that are hidden, deleted, a moderator's action, a wiki, edited after the window closed, by
the forum's own system accounts or accounts named as bots, or posted twice are left out, and so is
a reply to a post that is gone. A topic, an author and a forum each give at most a few.

community.openai.com and forum.cursor.com are busy and open, but much of what is posted there is
pasted from a model, and nothing in the JSON tells which: they are left out. So is
forums.unrealengine.com, whose posts are nearly all a marketplace's product listings.
"""

from __future__ import annotations

import calendar
import hashlib
import re
import time
import urllib.error
from collections import Counter

from . import Doc, get_json
from .hackernews import BOT, CONTEXT_CHARS, MIN_CHARS, PER_THREAD, lid, prose_chars, thread
from .lemmy import md_text

# Forums open to anonymous reads that are written in most hours of the day (checked November 2026:
# posts an hour, averaged over a day, from 460 on the first down to one or two on the programming
# languages' own). A forum installed under a path is named with it. Each host is one forum for
# PER_FORUM, so the four languages of Blizzard's European forum share one allowance.
FORUMS = (
    # general: conversation, games, hobbies, study
    "boards.straightdope.com", "us.forums.blizzard.com/en/wow", "talk.collegeconfidential.com", "forum.quartertothree.com",
    "forums.eveonline.com", "forums.flightsimulator.com", "community.infiniteflight.com", "forum.slowtwitch.com",
    "community.roonlabs.com", "forum.inaturalist.org", "community.wanikani.com", "community.openstreetmap.org",
    # technical: home automation, electronics, networks, operating systems, programming
    "community.home-assistant.io", "forum.arduino.cc", "forum.openwrt.org", "devforum.roblox.com", "community.hubitat.com",
    "forum.bambulab.com", "discussion.fedoraproject.org", "forum.manjaro.org", "forum.level1techs.com", "discuss.python.org",
    "users.rust-lang.org",
    # not in English: German, French, Spanish, Russian, Chinese (OpenStreetMap's and Arduino's above have many more)
    "community.simon42.com", "community.jeedom.com", "forum.openstreetmap.fr", "meta.appinn.net",
    "eu.forums.blizzard.com/de/wow", "eu.forums.blizzard.com/fr/wow", "eu.forums.blizzard.com/es/wow", "eu.forums.blizzard.com/ru/wow",
)
# Each forum has its own terms; some license what is posted (CC BY-NC-SA is Discourse's old default),
# most do not say. None is read here, so none is claimed.
LICENSE = "© its author; no open licence assumed (Discourse, {host}'s terms)"
GAP = 0.35  # seconds between requests to one host: Discourse allows an address 200 a minute by default
TRIES = 2  # of a request: a forum that does not answer is left out, not waited for
PAGE = 50  # posts in a full page of /posts.json
MAX_POSTS = 150  # of a forum's window that are read, from its end back
MAX_SEEK = 8  # pages read to find where a window ends among a busy forum's ids, or one long past
PER_FORUM = 12  # documents from any one host
PER_AUTHOR = 2  # and from any one account on it
DEPTH = 4  # posts up a chain of replies that a question may hold
COMMON = 3  # a line found in this many of a forum's posts in the window is a template's
COMMON_CHARS = 20  # and lines shorter than this are too ordinary to tell
_next: dict[str, float] = {}


def _get(site: str, path: str, params: dict):
    """A request to one forum, spaced by GAP on its host (get_json's own gap is shorter)."""
    host = site.split("/", 1)[0]
    wait = _next.get(host, 0.0) - time.time()
    if wait > 0:
        time.sleep(wait)
    _next[host] = time.time() + GAP
    return get_json(f"https://{site}/{path}", params, tries=TRIES)


def _ts(s: str) -> int:
    """Discourse's timestamps ("2026-11-21T18:17:56.251Z") as unix seconds."""
    return calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))


def _when(p: dict) -> int:
    return _ts(p["created_at"])


# --- a forum's window -----------------------------------------------------------------------------------


def _listing(site: str, start: int, end: int, seen: dict, why: Counter) -> list[dict]:
    """The forum's public posts created in [start, end), at most the last MAX_POSTS of them. Every
    post read on the way is put in seen by its topic and number, for the threads."""

    def page(before: int | None) -> list[dict]:
        got = _get(site, "posts.json", {"before": before} if before else {}).get("latest_posts") or []
        for p in got:
            seen[(p["topic_id"], p["post_number"])] = p
        return sorted(got, key=lambda p: -p["id"])

    ps = page(None)
    asked, lo = None, 0  # the id the page was asked below; every post up to lo is known to be before end
    up = down = None  # the earliest post known to be at end or later, and the latest known to be before it
    for step in range(MAX_SEEK):
        if not ps and asked is None:
            return []
        late = [p for p in ps if _when(p) >= end]
        early = [p for p in ps if _when(p) < end]
        if late and (up is None or late[-1]["id"] < up["id"]):
            up = late[-1]
        if early and (late or asked is None or asked == up["id"]):
            break  # this page holds the window's end
        if early:
            down = max(down or early[0], early[0], key=lambda p: p["id"])
        if not late:
            lo = max(lo, asked - 1)  # nothing below what was asked is as late as the end
        if up["id"] - lo <= PAGE:
            asked = up["id"]
        else:
            if down is None and ps:  # every post so far is too new: as far below as this page's pace says the end is
                span = max(1, _when(ps[0]) - _when(ps[-1]))
                guess = up["id"] - (ps[0]["id"] - ps[-1]["id"] + 1) * (_when(up) - end) // span
            elif down is not None and step < MAX_SEEK // 2:  # between the two, where the end's time falls
                guess = down["id"] + (up["id"] - down["id"]) * (end - _when(down)) // max(1, _when(up) - _when(down))
            else:  # or, when posts come too unevenly for that, halfway
                guess = (lo + up["id"]) // 2
            asked = max(lo + 2, min(up["id"], guess + PAGE // 2))
        ps = page(asked)
    else:
        why["window not found"] += 1
        return []
    hi = up["id"] if up is not None else ps[0]["id"] + 1  # the window's posts are below this id
    got = {p["id"]: p for p in ps if p["id"] < hi}
    while len(got) < MAX_POSTS and min(map(_when, ps)) >= start:
        ps = page(ps[-1]["id"])
        if not ps:
            break
        got.update((p["id"], p) for p in ps)
    last = sorted(got.values(), key=lambda p: -p["id"])[:MAX_POSTS]
    if len(got) >= MAX_POSTS and min(map(_when, last)) >= start:
        why["windows longer than what is read"] += 1
    return [p for p in last if start <= _when(p) < end]


# --- what is prose --------------------------------------------------------------------------------------

_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.S)
_QUOTE = re.compile(r"\[quote\b[^\]]*\](?:(?!\[quote\b).)*?\[/quote\]", re.S | re.I)  # the innermost first
_QUOTED = re.compile(r"\[quote=\"[^\"\]]*?post:\s*(\d+),\s*topic:\s*(\d+)", re.I)
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\n(.*?)(?:^[ \t]*\1[^\n]*(?:\n|\Z)|\Z)", re.S | re.M)
_INLINE = re.compile(r"(`+)([^\n]+?)\1")
_POLL = re.compile(r"\[poll\b[^\]]*\].*?\[/poll\]", re.S | re.I)
_UPLOAD = re.compile(r"\[[^\]\n]*\|attachment\]\(upload://[^)\s]*\)(?:[ \t]*\([\d.,]+[ \t]*[KMG]?B\))?", re.I)
_BB_IMG = re.compile(r"\[img\b[^\]]*\].*?\[/img\]", re.S | re.I)
_BB_URL = re.compile(r"\[url=[^\]]*\](.*?)\[/url\]", re.S | re.I)
_BB_CODE = re.compile(r"[ \t]*\[/?code\b[^\]]*\][ \t]*", re.I)
_BB = re.compile(r"\[/?(?:b|i|u|s|url|quote|details|spoiler|center|left|right|size|color|font|list|wrap|grid|date|event)\b[^\]\n]*\]|"
                 r"\[\*\]", re.I)
_BREAK = re.compile(r"<(?:br|/p|/div|/li|/tr|/h[1-6])\b[^<>]*>", re.I)
_TAG = re.compile(r"</?[a-zA-Z][\w-]*(?:\s[^<>]*)?/?>")  # as the page does, which shows no tag it does not know
_EMOJI = re.compile(r"[ \t]*(?<![\w:/])(?::(?:[a-z][a-z0-9_+-]*|\+1|-1|100)(?::t[2-6])?:)+(?!\w)")
_HELD = re.compile(r"\x00(\d+)\x00")


def _prose(s: str) -> str:
    """A stretch of a post outside its code blocks, as plain text: no quoted lines, polls, uploads
    or HTML comments, no marks of BBCode, HTML, Markdown or emoji. Inline code stays as written."""
    s = _UPLOAD.sub("", _POLL.sub("\n", _COMMENT.sub("", s)))
    s = "\n".join(ln for ln in s.split("\n") if not ln.lstrip().startswith(">"))
    held: list[str] = []

    def hold(m: re.Match) -> str:
        held.append(m.group(2).strip())
        return f"\x00{len(held) - 1}\x00"

    s = _INLINE.sub(hold, s)
    s = _BB_URL.sub(r"\1", _BB_IMG.sub("", s))
    s = _TAG.sub("", _BREAK.sub("\n", _BB.sub("", s)))
    return _HELD.sub(lambda m: held[int(m.group(1))] if len(m.group(1)) < 9 and int(m.group(1)) < len(held) else "",
                     _EMOJI.sub("", md_text(s)))


def raw_text(raw: str, code: bool = True) -> str:
    """A post's Markdown as plain text, roughly as it reads on the page, without what it quotes
    (quoted posts, quoted lines). Code blocks are kept as they were written; without code, they go."""
    t = (raw or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")  # no NULs: _prose marks what it holds with them
    for _ in range(20):  # quotes inside quotes
        t, n = _QUOTE.subn("\n", t)
        if not n:
            break
    t = _BB_CODE.sub("\n```\n", t)  # BBCode's code block as Markdown's
    out, pos = [], 0
    for m in _FENCE.finditer(t):
        out.append(_prose(t[pos : m.start()]))
        if code:
            out.append(m.group(2).strip("\n"))
        pos = m.end()
    out.append(_prose(t[pos:]))
    t = "\n\n".join(x for x in out if x.strip())
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", t)).strip()


def _key(line: str) -> str:
    return " ".join(line.lower().split())


def _shared(texts: list[str]) -> set[str]:
    """The lines several posts share: a category's template, a form, a signature."""
    count = Counter(k for t in texts for k in {_key(ln) for ln in t.split("\n")} if len(k) >= COMMON_CHARS)
    return {k for k, n in count.items() if n >= COMMON}


def _without(text: str, shared: set[str]) -> str:
    if not shared:
        return text
    return re.sub(r"\n{3,}", "\n\n", "\n".join(ln for ln in text.split("\n") if _key(ln) not in shared)).strip()


def _gone(p: dict | None) -> bool:
    return not p or bool(p.get("hidden") or p.get("user_deleted") or p.get("deleted_at"))


def _fit(p: dict, end: int, why: Counter) -> bool:
    """Whether a post may be kept, before its words are read: a person's own, shown, and left
    alone since the window closed."""
    if p.get("post_type", 1) != 1:
        why["a moderator's action or a note"] += 1
    elif _gone(p):
        why["hidden or deleted"] += 1
    elif p.get("wiki"):
        why["wiki"] += 1
    elif (p.get("user_id") or 0) <= 0 or BOT.search(p.get("username") or ""):
        why["bot"] += 1  # Discourse's own accounts (system, discobot, its AI helpers) have ids below one
    elif (p.get("version") or 1) > 1 and _ts(p.get("updated_at") or p["created_at"]) >= end:
        why["edited after the window"] += 1
    else:
        return True
    return False


# --- the thread a reply answers -------------------------------------------------------------------------


def _post(site: str, topic: int, number: int, seen: dict) -> dict | None:
    """A topic's post by its number, as the forum shows it now; None for one it no longer shows."""
    key = (topic, number)
    if key not in seen:
        try:
            seen[key] = _get(site, f"posts/by_number/{topic}/{number}.json", {})
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404, 410):
                raise
            seen[key] = None
    return seen[key]


def _parents(site: str, p: dict, seen: dict) -> list[dict] | None:
    """The posts p answers besides the topic's first, nearest first; None if one of them is gone."""
    topic, number = p["topic_id"], p["post_number"]
    out: list[dict] = []
    to = p.get("reply_to_post_number")
    if to:  # written in reply to a post: that post, and what it replied to in turn, while there is room
        size = 0
        while to and to > 1 and len(out) < DEPTH and size < CONTEXT_CHARS:
            up = _post(site, topic, to, seen)
            if _gone(up):
                return None
            out.append(up)
            size += len(up.get("raw") or "")
            to = up.get("reply_to_post_number") if (up.get("reply_to_post_number") or 0) < to else None
        return out
    quoted = []  # a reply to the topic at large: the posts of the topic that it quotes
    for m in _QUOTED.finditer(p.get("raw") or ""):
        n = int(m.group(1))
        if int(m.group(2)) == topic and 1 < n < number and n not in quoted:
            quoted.append(n)
    if quoted:
        for n in sorted(quoted, reverse=True)[:DEPTH]:
            up = _post(site, topic, n, seen)
            if _gone(up):
                return None
            out.append(up)
        return out
    if number > 2:  # or else the post just before it, if it is there to be read
        up = _post(site, topic, number - 1, seen)
        if not _gone(up) and up.get("post_type", 1) == 1:
            out.append(up)
    return out


def _doc(site: str, p: dict, text: str, own: str, seen: dict) -> Doc | None:
    host = site.split("/", 1)[0]
    label, sure = lid(own)
    title = " ".join((p.get("topic_title") or "").split())
    extra = {"site": site, "topic": p["topic_id"], "number": p["post_number"], "lid": sure}
    common = dict(source="discourse", id=f"{site}:{p['id']}", url=f"https://{site}/p/{p['id']}", lang=label,
                  license=LICENSE.format(host=host), published=_when(p), title=title, text=text,
                  topic=f"{site}/c{p.get('category_id') or 0}")
    if p["post_number"] == 1:
        return Doc(extra=extra, **common)
    first = _post(site, p["topic_id"], 1, seen)
    if _gone(first):
        return None
    chain = _parents(site, p, seen)
    if chain is None:
        return None
    blocks = [raw_text(first.get("raw") or "")] + [raw_text(c.get("raw") or "") for c in reversed(chain)]
    extra["parents"] = [c["post_number"] for c in reversed(chain)]
    return Doc(question=thread(title, blocks), extra=extra, **common)


# --- the harvest ----------------------------------------------------------------------------------------


def harvest(start: int, end: int, cap: int, rng, log=print, forums: tuple[str, ...] | list[str] = FORUMS) -> list[Doc]:
    """Up to cap replies and opening posts written in [start, end) on the forums, chosen with rng:
    at most PER_THREAD from a topic, PER_AUTHOR from an account and PER_FORUM from a host."""
    t0 = time.time()
    why: Counter = Counter()
    seen_text: set[str] = set()
    cands = []  # (site, post, its text, its prose alone)
    caches: dict[str, dict] = {}
    for site in forums:
        seen = caches.setdefault(site, {})
        mine, keys = [], set()  # the forum's own candidates and texts: kept once all of its posts are read
        try:
            posts = _listing(site, start, end, seen, why)
            posts = [p for p in sorted(posts, key=lambda p: p["id"]) if _fit(p, end, why)]
            shared = _shared([raw_text(p.get("raw") or "", code=False) for p in posts])
            for p in posts:
                own = _without(raw_text(p.get("raw") or "", code=False), shared)
                if prose_chars(own) < MIN_CHARS:
                    why["short"] += 1
                    continue
                text = _without(raw_text(p.get("raw") or ""), shared)
                key = hashlib.sha256(" ".join(text.split()).lower().encode()).hexdigest()
                if key in seen_text or key in keys:
                    why["posted twice"] += 1
                    continue
                keys.add(key)
                mine.append((site, p, text, own))
        except Exception as e:  # one forum failing, or sending what cannot be read, does not stop the harvest
            log(f"  discourse {site}: {type(e).__name__}: {e}")
            del caches[site]
            continue
        seen_text |= keys
        cands += mine
    cands.sort(key=lambda c: (c[0], c[1]["id"]))
    rng.shuffle(cands)
    docs: list[Doc] = []
    per_forum: Counter = Counter()
    per_thread: Counter = Counter()
    per_author: Counter = Counter()
    for site, p, text, own in cands:
        if len(docs) >= cap:
            break
        host = site.split("/", 1)[0]
        thread_key, author = (site, p["topic_id"]), (host, (p.get("username") or "").lower())
        if per_forum[host] >= PER_FORUM:
            why["forum full"] += 1
            continue
        if per_thread[thread_key] >= PER_THREAD:
            why["thread full"] += 1
            continue
        if per_author[author] >= PER_AUTHOR:
            why["author full"] += 1
            continue
        try:
            doc = _doc(site, p, text, own, caches[site])
        except Exception as e:  # a post of the thread that the forum would not give
            why[f"error {type(e).__name__}"] += 1
            continue
        if doc is None:
            why["reply to a post that is gone"] += 1
            continue
        per_forum[host] += 1
        per_thread[thread_key] += 1
        per_author[author] += 1
        docs.append(doc)
    langs = Counter(d.lang for d in docs)
    skipped = ", ".join(f"{k} {v}" for k, v in why.most_common())
    log(f"  discourse: {len(docs)} posts of {len(cands)} candidates from {len(caches)} of {len(forums)} forums, in "
        f"{dict(langs.most_common(8))} ({time.time() - t0:.0f}s; skipped: {skipped or 'none'})")
    return docs

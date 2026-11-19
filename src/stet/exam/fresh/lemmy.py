"""Comments and text posts on Lemmy written in the window, each comment with the thread it replies to.

Lemmy's public API (v3, no key or account) is read on a few large, general instances. Each is read
for what is written in its own communities (type_=Local: a community lives on one instance, so a
comment is listed by one of them, whoever wrote it and wherever they did), newest first, back
through the window; ap_id, which names a comment across the federation, keeps any that two
instances list to one. A comment is kept with its question, the thread so far: its post's title
and text, then the comments it replies to, nearest last, as its own instance shows them. Only the
comment is scored. A text post written in the window is kept on its own; a link post is not, as
its text is most often the article it links to, pasted, and nor is a post that starts with its
own title, as the news feeds that post from Mastodon and Friendica do.

What is kept is what its author wrote and left alone: anything edited since, deleted or removed
goes, and so do bots (accounts that say so, or are named so), banned accounts, moderators' notes,
NSFW posts, text posted twice, and replies to comments that are gone. Lemmy writes Markdown; it is
read as plain text (links as their words, no images), quotes are taken out, and only prose counts
towards its length. Lemmy's own language tag is checked against fastText's lid.176, which wins
unless it is unsure. A post gives at most a few documents. Writers who mark their comments with a
licence against AI use are left out, as they asked.

feddit.org refuses the harvester's User-Agent and is left out; it is not disguised.
"""

from __future__ import annotations

import calendar
import hashlib
import html
import re
import time
from collections import Counter

from . import Doc, get_json
from .hackernews import BOT, CONTEXT_CHARS, MIN_CHARS, PER_THREAD, lid, prose_chars, thread, unquoted

# General instances whose own communities are written in every hour or so, open to anonymous
# reads (checked November 2026). hexbear.net and lemmygrad.ml are as busy, but each is given to one
# politics, and are left out.
INSTANCES = ("lemmy.world", "lemmy.ml", "sh.itjust.works", "lemmy.ca", "sopuli.xyz", "lemmy.blahaj.zone", "programming.dev",
             "lemmy.dbzer0.com", "lemmy.zip", "lemmy.today", "lemmy.sdf.org", "reddthat.com", "discuss.tchncs.de", "feddit.nl",
             "feddit.it", "jlai.lu", "lemmy.eco.br", "szmer.info", "feddit.uk", "lemmy.nz")
LICENSE = "© its author; no open licence (Lemmy, {host}'s terms)"
GAP = 0.35  # seconds between requests to one instance: Lemmy's default limit is 180 a minute
PAGE = 50  # Lemmy's largest page
MAX_PAGES = 80  # of comments, or of posts, read through the window on one instance
FAR = 4096  # pages back the search for the window's first page may go
UNSURE = 0.5  # lid.176 below this gives way to the language the author tagged
_AI_LICENCE = re.compile(r"commercial[- ]ai|creativecommons\.org/licenses/by-nc", re.I)
_next: dict[str, float] = {}


def _get(host: str, path: str, params: dict):
    """A request to one instance, spaced by GAP (get_json's own gap is shorter)."""
    wait = _next.get(host, 0.0) - time.time()
    if wait > 0:
        time.sleep(wait)
    _next[host] = time.time() + GAP
    return get_json(f"https://{host}/api/v3/{path}", params)


def _ts(s: str) -> int:
    """Lemmy's timestamps ("2026-11-19T14:34:35.424696Z") as unix seconds."""
    return calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))


_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\(\s*<?([^)\s>]*)>?(?:\s+\"[^\"]*\")?\s*\)")
_FENCED = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$", re.S | re.M)
_MENTIONS = re.compile(r"^(?:@[\w.-]+(?:@[\w.-]+)?[,:]?\s+)+")


def md_text(md: str, code: bool = True) -> str:
    """Lemmy's Markdown as plain text, roughly as it reads on the page: links as their words,
    images and the marks of emphasis, headings, spoilers and rules gone, and the mentions it
    starts with (as replies from Mastodon do). Without code, fenced code goes too."""
    t = html.unescape(md or "").replace("\r\n", "\n")
    t = _FENCED.sub("\n", t) if not code else re.sub(r"^[ \t]*(```|~~~).*$", "", t, flags=re.M)
    t = _IMG.sub("", t)
    t = _LINK.sub(lambda m: m.group(1).strip() or m.group(2), t)
    t = re.sub(r"^[ \t]*:::[ \t]*(?:spoiler\b)?[ \t]*(.*)$", r"\1", t, flags=re.M)
    t = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]+", "", t, flags=re.M)
    t = re.sub(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", "", t, flags=re.M)
    t = re.sub(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", r"\2", t)
    t = re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"\1", t)
    t = re.sub(r"(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])", r"\1", t)
    t = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"\1", t)
    t = re.sub(r"`([^`\n]+)`", r"\1", t)
    t = _MENTIONS.sub("", t.lstrip())
    t = re.sub(r"[ \t]+\n", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _languages(host: str) -> dict[int, str]:
    site = _get(host, "site", {})
    return {int(x["id"]): x.get("code", "und") for x in site.get("all_languages", [])}


def _listing(host: str, what: str, start: int, end: int, why: Counter) -> list[dict]:
    """The views of an instance's local comments or posts written in [start, end): pages newest
    first, from the first that reaches back before end (found by doubling, then bisecting, so that
    a harvest run late does not read everything since) until one reaches back before start."""
    key = "comments" if what == "comment" else "posts"
    pages: dict[int, list[dict]] = {}

    def page(n: int) -> list[dict]:
        if n not in pages:
            pages[n] = _get(host, f"{what}/list", {"sort": "New", "type_": "Local", "limit": PAGE, "page": n}).get(key, [])
        return pages[n]

    def oldest(n: int) -> float:  # a post pinned to the instance's front page is listed first, however old
        times = [_ts(v[what]["published"]) for v in page(n) if not v[what].get("featured_local")]
        return min(times) if times else float("-inf")

    lo, hi = 0, 1  # every page up to lo is newer than end; hi reaches back before it
    while oldest(hi) >= end and hi < FAR:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if oldest(mid) >= end:
            lo = mid
        else:
            hi = mid
    out = []
    n = hi
    while True:
        out += [v for v in page(n) if start <= _ts(v[what]["published"]) < end and not v[what].get("featured_local")]
        if oldest(n) < start or len(page(n)) < PAGE:
            break
        if n - hi >= MAX_PAGES:
            why[f"{what}s past {MAX_PAGES} pages"] += 1
            break
        n += 1
    return out


def _gone(c: dict) -> bool:
    return bool(c.get("deleted") or c.get("removed"))


def _plain(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def _fit(v: dict, what: str, seen: set[str], why: Counter) -> str | None:
    """The view's text, cleaned, if it may be kept: written and left alone by a person, long enough."""
    own, post, community, creator = v[what], v["post"], v["community"], v.get("creator", {})
    if _gone(own) or own.get("updated"):
        why["edited, deleted or removed"] += 1
        return None
    if creator.get("bot_account") or BOT.search(creator.get("name", "")):
        why["bot"] += 1
        return None
    if creator.get("banned") or v.get("creator_banned_from_community") or own.get("distinguished"):
        why["banned or a moderator's note"] += 1
        return None
    if _gone(post) or post.get("nsfw") or community.get("nsfw") or community.get("visibility", "Public") != "Public":
        why["post gone, NSFW or not public"] += 1
        return None
    if what == "post" and post.get("url") and not (post.get("url_content_type") or "").startswith("image/"):
        why["link post"] += 1  # its text is most often the article it links to, pasted
        return None
    raw = own.get("content", "") if what == "comment" else (post.get("body") or "")
    if _AI_LICENCE.search(raw):
        why["licence against AI use"] += 1
        return None
    text = unquoted(md_text(raw))
    if prose_chars(unquoted(md_text(raw, code=False))) < MIN_CHARS:
        why["short"] += 1
        return None
    if what == "post" and _plain(text.split("\n", 1)[0]) == _plain(post.get("name") or ""):
        why["a feed's post"] += 1  # a headline, then its story: what news feeds post from Mastodon and Friendica
        return None
    key = hashlib.sha256(" ".join(text.split()).lower().encode()).hexdigest()
    if key in seen:
        why["posted twice"] += 1
        return None
    seen.add(key)
    return text


def harvest(start: int, end: int, cap: int, rng, log=print, instances: tuple[str, ...] | list[str] = INSTANCES) -> list[Doc]:
    """Up to cap comments and text posts written in [start, end) on the instances' own
    communities, chosen with rng, at most PER_THREAD from a post."""
    t0 = time.time()
    why: Counter = Counter()
    seen_ap: set[str] = set()
    seen_text: set[str] = set()
    cands = []  # (host, "comment" | "post", view, text)
    views: dict[tuple[str, int], dict] = {}  # comments by instance and id, for the threads
    codes: dict[str, dict[int, str]] = {}
    for host in instances:
        try:
            langs = _languages(host)
            listed = [("comment", v) for v in _listing(host, "comment", start, end, why)]
            listed += [("post", v) for v in _listing(host, "post", start, end, why)]
        except Exception as e:  # one instance failing does not stop the harvest
            log(f"  lemmy {host}: {type(e).__name__}: {e}")
            continue
        codes[host] = langs
        for what, v in listed:
            if what == "comment":
                views[(host, v["comment"]["id"])] = v
            ap = v[what].get("ap_id", f"{host}/{what}/{v[what]['id']}")
            if ap in seen_ap:
                why["listed twice"] += 1
                continue
            seen_ap.add(ap)
            text = _fit(v, what, seen_text, why)
            if text is not None:
                cands.append((host, what, v, text))
    cands.sort(key=lambda c: (c[0], c[1], c[2][c[1]]["id"]))
    rng.shuffle(cands)
    docs: list[Doc] = []
    per: Counter = Counter()
    for host, what, v, text in cands:
        if len(docs) >= cap:
            break
        post = v["post"]
        if per[(host, post["id"])] >= PER_THREAD:
            why["thread full"] += 1
            continue
        try:
            doc = _doc(host, what, v, text, codes.get(host, {}), views)
        except Exception as e:  # a parent the instance would not give
            why[f"error {type(e).__name__}"] += 1
            continue
        if doc is None:
            why["reply to a comment that is gone"] += 1
            continue
        per[(host, post["id"])] += 1
        docs.append(doc)
    langs = Counter(d.lang for d in docs)
    skipped = ", ".join(f"{k} {v}" for k, v in why.most_common())
    log(f"  lemmy: {len(docs)} comments and posts of {len(cands)} candidates from {len(codes)} instances, in "
        f"{dict(langs.most_common(8))} ({time.time() - t0:.0f}s; skipped: {skipped or 'none'})")
    return docs


def _parents(host: str, v: dict, views: dict[tuple[str, int], dict], room: int) -> list[dict] | None:
    """The comments v replies to, nearest first, until there is room's worth of them; None if
    one of them is gone."""
    path = [int(x) for x in v["comment"].get("path", "").split(".")[1:-1]]
    out, size = [], 0
    for cid in reversed(path):
        if size >= room:
            break
        if (host, cid) not in views:
            views[(host, cid)] = _get(host, "comment", {"id": cid})["comment_view"]
        c = views[(host, cid)]["comment"]
        if _gone(c):
            return None
        out.append(c)
        size += len(c.get("content", ""))
    return out


def _doc(host: str, what: str, v: dict, text: str, codes: dict[int, str], views) -> Doc | None:
    own, post, community = v[what], v["post"], v["community"]
    tag = codes.get(own.get("language_id", 0), "und")
    label, p = lid(unquoted(md_text(own.get("content", "") if what == "comment" else post.get("body") or "", code=False)))
    lang = tag if tag not in ("und", label) and p < UNSURE else label
    topic = f"{community.get('name', '')}@{host}"
    extra = {"instance": host, "post": post["id"], "ap_id": own.get("ap_id", ""), "lang_tag": tag, "lid": label, "lid_p": p}
    common = dict(source="lemmy", lang=lang, license=LICENSE.format(host=host), published=_ts(own["published"]),
                  title=" ".join((post.get("name") or "").split()), text=text, topic=topic)
    if what == "post":
        return Doc(id=f"{host}:p{post['id']}", url=f"https://{host}/post/{post['id']}", extra=extra, **common)
    chain = _parents(host, v, views, CONTEXT_CHARS)
    if chain is None:
        return None
    blocks = [md_text(post.get("body") or "")] + [md_text(c.get("content", "")) for c in reversed(chain)]
    extra["parents"] = [c["id"] for c in reversed(chain)]
    return Doc(id=f"{host}:c{own['id']}", url=f"https://{host}/comment/{own['id']}", question=thread(common["title"], blocks),
               extra=extra, **common)

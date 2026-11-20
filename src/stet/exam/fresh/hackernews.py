"""Comments on Hacker News written in the window, each with the thread it replies to.

The official API (https://github.com/HackerNews/API: Firebase, no key, no stated rate limit) serves
one item a request, and item ids grow with time: the window's ids are found by bisecting on the
items' times, and a seeded sample of them is read until there are enough. A comment is kept with
its question, the thread so far: its story's title and text, then the comments it replies to,
nearest last. Only the comment is scored. A text post written in the window (Ask HN and the like)
is kept on its own.

Only the author's own prose counts towards its length: what it quotes (lines that start with ">")
is taken out, as it is in the question already, and code and links do not count. Dead and deleted
items go, and so do replies to them, text posted twice, and accounts named as bots. A thread gives
at most a few comments, so that one big discussion cannot fill the harvest.

Hacker News lets a comment be edited, or deleted, for two hours after it is posted, and the API
does not say whether it was. The harvest runs an hour after the window closes, so a comment from
late in the window may be harvested as first written, and one deleted since is not.

The helpers below the source (the thread as a question, prose, language) serve the other
discussion sources too.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter

from . import Doc, get_json, html_text

API = "https://hacker-news.firebaseio.com/v0"
LICENSE = "© its author; Hacker News terms of use (no open licence), via the official API"
MIN_CHARS = 200  # of the author's own prose: no quotes, code or links
PER_THREAD = 3  # documents from any one story
CONTEXT_CHARS = 1500  # of the thread a comment replies to, the farthest dropped first
EDIT_SECONDS = 2 * 3600  # how long Hacker News lets a comment be edited
BOT = re.compile(r"bot\d*$", re.I)

_A = re.compile(r"<a\s[^>]*?href=\"([^\"]*)\"[^>]*>.*?</a>", re.S | re.I)
_PRE = re.compile(r"<pre>.*?</pre>", re.S | re.I)


def hn_text(html: str) -> str:
    """Hacker News's markup as plain text: paragraphs a blank line apart, and links in full, as
    their authors wrote them (the site shows a long one cut short)."""
    html = _A.sub(lambda m: m.group(1), html or "")
    return html_text(re.sub(r"<p>", "<br><br>", html, flags=re.I))


def _item(get, i: int) -> dict | None:
    return get(f"{API}/item/{i}.json")


def _when(get, i: int, top: int) -> float:
    """When item i was created; a missing item takes the time of the next one there is."""
    for j in range(i, min(i + 10, top + 1)):
        it = _item(get, j)
        if it and "time" in it:
            return it["time"]
    return float("inf")


def _first(get, t: int, top: int) -> int:
    """The first id created at t or later, at most top + 1: back from top in growing steps until
    an item is older than t, then a bisection."""
    if _when(get, top, top) < t:
        return top + 1
    b, step = top, 1024
    a = max(0, b - step)
    while a > 0 and _when(get, a, top) >= t:
        b, step = a, step * 2
        a = max(0, b - step)
    while b - a > 1:
        m = (a + b) // 2
        if _when(get, m, top) < t:
            a = m
        else:
            b = m
    return b


def window(get, start: int, end: int) -> range:
    """The ids of the items created in [start, end), give or take the few whose clocks disagree
    with their ids (each item read is checked against the window again)."""
    top = int(get(f"{API}/maxitem.json"))
    hi = _first(get, end, top)
    lo = _first(get, start, min(hi, top))
    return range(lo, max(lo, hi))


def harvest(start: int, end: int, cap: int, rng, log=print) -> list[Doc]:
    """Up to cap comments and text posts written in [start, end), chosen with rng from a seeded
    order of the window's ids, at most PER_THREAD from a story."""
    t0 = time.time()
    if time.time() < end + EDIT_SECONDS:
        log("  hackernews: the window closed less than two hours ago, so some of its comments may still be edited")
    cache: dict[str, object] = {}
    calls = [0]

    def get(url: str):
        if url not in cache:
            calls[0] += 1
            cache[url] = get_json(url, {})
        return cache[url]

    try:
        ids = list(window(get, start, end))
    except Exception as e:
        log(f"  hackernews: {type(e).__name__}: {e}")
        return []
    rng.shuffle(ids)
    docs: list[Doc] = []
    per: Counter = Counter()
    seen: set[str] = set()
    why: Counter = Counter()
    read = 0
    for i in ids:
        if len(docs) >= cap:
            break
        read += 1
        try:
            got = _doc(get, _item(get, i), start, end, seen, why)
        except Exception as e:  # an item the API would not give: the next one
            why[f"error {type(e).__name__}"] += 1
            continue
        if got is None:
            continue
        story, doc = got
        if per[story] >= PER_THREAD:
            why["thread full"] += 1
            continue
        per[story] += 1
        docs.append(doc)
    skipped = ", ".join(f"{k} {v}" for k, v in why.most_common())
    log(f"  hackernews: {len(docs)} comments and posts from {read} of the window's {len(ids)} items "
        f"({calls[0]} requests, {time.time() - t0:.0f}s; skipped: {skipped or 'none'})")
    return docs


def _doc(get, it: dict | None, start: int, end: int, seen: set[str], why: Counter) -> tuple[int, Doc] | None:
    """An item as a document, and the story it belongs to; None for an item that is not kept."""
    if not it or it.get("deleted") or it.get("dead"):
        why["dead or deleted"] += 1
        return None
    kind = it.get("type")
    if not start <= it.get("time", 0) < end:
        why["outside the window"] += 1
        return None
    if kind not in ("comment", "story"):
        why[kind or "no type"] += 1
        return None
    if BOT.search(it.get("by", "")):
        why["bot"] += 1
        return None
    html = it.get("text", "")
    if kind == "story" and not html:
        why["link post"] += 1
        return None
    text = unquoted(hn_text(html))
    own = unquoted(hn_text(_PRE.sub(" ", html)))
    if prose_chars(own) < MIN_CHARS:
        why["short"] += 1
        return None
    key = hashlib.sha256(" ".join(text.split()).lower().encode()).hexdigest()
    if key in seen:
        why["posted twice"] += 1
        return None
    seen.add(key)
    label, p = lid(own)
    common = dict(source="hackernews", id=str(it["id"]), url=f"https://news.ycombinator.com/item?id={it['id']}", lang=label,
                  license=LICENSE, published=int(it["time"]), text=text)
    if kind == "story":
        return it["id"], Doc(title=it.get("title", ""), extra={"lid": p}, **common)
    chain = []  # what it replies to, nearest first
    parent = it.get("parent")
    for _ in range(1000):
        up = _item(get, parent) if parent else None
        if not up or up.get("deleted") or up.get("dead"):
            why["reply to a dead or deleted item"] += 1
            return None
        if up.get("type") != "comment":
            break
        chain.append(up)
        parent = up.get("parent")
    else:
        return None
    if up.get("type") not in ("story", "poll"):
        why[f"under a {up.get('type')}"] += 1
        return None
    blocks = [hn_text(up.get("text", ""))] + [hn_text(c.get("text", "")) for c in reversed(chain)]
    return up["id"], Doc(title=up.get("title", ""), question=thread(up.get("title", ""), blocks),
                         extra={"story": up["id"], "parents": [c["id"] for c in reversed(chain)], "lid": p}, **common)


# --- helpers for discussion sources ----------------------------------------------------------------

URL = re.compile(r"\b(?:https?://|www\.)\S+")


def unquoted(text: str) -> str:
    """Text without the lines it quotes: a reply's quotes are not its own, and are in its question."""
    lines = [ln for ln in text.split("\n") if not ln.lstrip().startswith(">")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def prose_chars(text: str) -> int:
    """How much prose there is: characters, links aside and runs of space as one."""
    return len(" ".join(URL.sub(" ", text).split()))


def lid(text: str) -> tuple[str, float]:
    """The language fastText's lid.176 reads in a text, and how sure it is."""
    from .langs import model

    (label,), (p,) = model().predict(" ".join(URL.sub(" ", text).split())[:1000], k=1)
    return label.removeprefix("__label__"), round(float(p), 3)


def _tail(text: str, n: int) -> str:
    """About the last n characters, from a word boundary."""
    if len(text) <= n:
        return text
    s = text[len(text) - n :]
    cut = s.find(" ")
    return s[cut + 1 :] if 0 <= cut < n // 3 else s


def thread(title: str, blocks: list[str], n: int = CONTEXT_CHARS) -> str:
    """A thread as a reply's question: its title, then its blocks (the post, then the comments
    replied to, nearest last), blank lines apart. Past n characters the farthest blocks go first,
    and the last one kept may keep only its end; an ellipsis shows where text was left out."""
    title = " ".join(title.split())[:300]
    blocks = [b.strip() for b in blocks if b and b.strip()]
    room = n - len(title) - 5  # what each block costs is its length and the blank line before it; 3 to spare for an ellipsis
    kept: list[str] = []
    for b in reversed(blocks):
        if len(b) + 2 <= room:
            kept.append(b)
            room -= len(b) + 2
            continue
        kept.append("…" + _tail(b, room + 1).lstrip() if room >= 120 else "…")
        break
    return "\n\n".join([title] * bool(title) + kept[::-1])

"""Lemmy as a fresh source, against a stand-in for two instances' API: no request leaves the machine."""

import time
import urllib.parse

import pytest

from stet.exam.fresh import lemmy
from stet.exam.rng import Rng

START = 1_800_000_000
END = START + 3600
PROSE = "and it goes on for a while in plain words, as people write when they have something to say. " * 3
LANGS = [{"id": 0, "code": "und"}, {"id": 37, "code": "en"}, {"id": 45, "code": "de"}]


def say(what: str) -> str:
    return f"{what} {PROSE}"


def iso(t: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.123456Z", time.gmtime(t))


def post(host, pid, t, name, body="", url=None, **kw):
    return {"id": pid, "name": name, "body": body, "url": url, "published": iso(t), "deleted": False, "removed": False,
            "nsfw": False, "ap_id": f"https://{host}/post/{pid}", "language_id": 0, **kw}


def community(name="tech", **kw):
    return {"name": name, "nsfw": False, "visibility": "Public", **kw}


def person(name="someone", **kw):
    return {"name": name, "bot_account": False, "banned": False, **kw}


def comment(host, cid, p, t, content, parents=(), creator=None, comm=None, **kw):
    c = {"id": cid, "content": content, "published": iso(t), "deleted": False, "removed": False, "distinguished": False,
         "ap_id": f"https://{host}/comment/{cid}", "path": ".".join(map(str, (0, *parents, cid))), "language_id": 37, **kw}
    return {"comment": c, "creator": creator or person(), "post": p, "community": comm or community(),
            "creator_banned_from_community": False}


def post_view(p, creator=None, comm=None):
    return {"post": p, "creator": creator or person(), "community": comm or community(), "creator_banned_from_community": False}


def instances():
    """Two instances. The first has a few hundred short comments newer than the window, the cases
    a harvest must tell apart inside it, and more short ones older than it."""
    a = "a.example"
    old = post(a, 1, START - 7200, "An old post", say("The post's own words,"))
    big = post(a, 2, START - 3600, "A big thread")
    cs = [comment(a, 5000 + i, old, END + 10 * i, "ok") for i in range(300)]  # newer than the window
    parent = comment(a, 90, old, START - 600, "The parent, written before the window.")
    gone = comment(a, 91, old, START - 500, "", removed=True)
    inside = [
        comment(a, 100, old, START + 100, "> The parent\n\n" + say("A [reply](https://example.org/x) to **it**,"), parents=(90,)),
        comment(a, 101, old, START + 101, say("Edited later,"), updated=iso(END + 50)),
        comment(a, 102, old, START + 102, say("Deleted,"), deleted=True),
        comment(a, 103, old, START + 103, say("Removed,"), removed=True),
        comment(a, 104, old, START + 104, say("Says it is a bot,"), creator=person(bot_account=True)),
        comment(a, 105, old, START + 105, say("Named as a bot,"), creator=person("newsbot")),
        comment(a, 106, old, START + 106, say("A reply to a removed comment,"), parents=(91,)),
        comment(a, 107, old, START + 107, say("Under a licence,") + "\n\n[Anti Commercial-AI license](https://creativecommons.org/licenses/by-nc-sa/4.0/)"),
        comment(a, 108, old, START + 108, "Too short " + "https://example.org/" + "x" * 300),
        comment(a, 109, old, START + 109, say("Not safe for work,"), comm=community("nsfwthings", nsfw=True)),
        comment(a, 110, old, START + 110, say("Tagged German, written in English,"), language_id=45),
        comment(a, 111, old, START + 111, say("Tagged German, unsure,"), language_id=45),
        comment(a, 112, old, START + 112, say("A moderator speaks,"), distinguished=True),
        comment(a, 113, old, START - 1, say("Just before the window,")),
        comment(a, 114, old, END, say("As the window closed,")),
    ]
    inside += [comment(a, 200 + i, big, START + 200 + i, say(f"Comment {i} in the big thread,")) for i in range(10)]
    inside += [comment(a, 300 + i, post(a, 30 + i, START - 100, f"Post {i}"), START + 300 + i, say(f"The only reply to post {i},"))
               for i in range(20)]
    older = [comment(a, 10 + i, old, START - 1000 - i, "ok") for i in range(60)]
    a_comments = sorted(cs + inside + older, key=lambda v: v["comment"]["published"], reverse=True)
    a_posts = [
        post_view(post(a, 3, START + 400, "Ask: what do you read?", say("I have been wondering,"))),
        post_view(post(a, 4, START + 401, "A link", say("The article, pasted,"), url="https://news.example/a", url_content_type="text/html")),
        post_view(post(a, 6, START + 402, "Headline here", "Headline here\n\n" + say("The story,"))),
        post_view(post(a, 7, START - 50, "Before the window", say("Too early,"))),
    ]
    a_posts.sort(key=lambda v: v["post"]["published"], reverse=True)
    a_posts.insert(0, post_view(post(a, 8, START - 90000, "Pinned to the front page", say("Rules,"), featured_local=True)))
    b = "b.example"
    bpost = post(b, 1, START - 100, "Elsewhere")
    twin = comment(b, 7, bpost, START + 100, say("Listed twice,"))
    twin["comment"]["ap_id"] = "https://a.example/comment/100"  # the federation's name for a's comment 100
    b_comments = [twin, comment(b, 8, bpost, START + 200, say("A comment on the second instance,"))]
    return {a: {"comments": a_comments, "posts": a_posts, "byid": {v["comment"]["id"]: v for v in a_comments + [parent, gone]}},
            b: {"comments": b_comments, "posts": [], "byid": {}}}


class FakeAPI:
    def __init__(self, hosts):
        self.hosts, self.calls = hosts, []

    def __call__(self, url, params, **kw):
        parts = urllib.parse.urlsplit(url)
        path = parts.path.removeprefix("/api/v3/")
        self.calls.append((parts.netloc, path, params.get("page")))
        h = self.hosts[parts.netloc]
        if path == "site":
            return {"all_languages": LANGS}
        if path in ("comment/list", "post/list"):
            assert params["sort"] == "New" and params["type_"] == "Local"
            key = "comments" if path == "comment/list" else "posts"
            n, size = params["page"], params["limit"]
            return {key: h[key][(n - 1) * size : n * size]}
        if path == "comment":
            return {"comment_view": h["byid"][params["id"]]}
        raise AssertionError(url)


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI(instances())
    monkeypatch.setattr(lemmy, "get_json", fake)
    monkeypatch.setattr(lemmy, "GAP", 0)
    monkeypatch.setattr(lemmy, "PAGE", 20)
    monkeypatch.setattr(lemmy, "lid", lambda text: ("en", 0.3 if "unsure" in text else 0.95))
    return fake


def run(cap=1000, seed=b"seed"):
    return lemmy.harvest(START, END, cap, Rng(seed, "fresh/lemmy"), log=lambda *a: None, instances=["a.example", "b.example"])


def test_only_what_was_written_in_the_window_and_left_alone_is_kept(api):
    docs = run()
    ids = {d.id for d in docs}
    assert all(START <= d.published < END for d in docs)
    assert {"a.example:c100", "a.example:c110", "a.example:c111", "b.example:c8", "a.example:p3"} <= ids
    for gone in (101, 102, 103, 104, 105, 106, 107, 108, 109, 112, 113, 114):
        assert f"a.example:c{gone}" not in ids
    for gone in (4, 6, 7, 8):  # a link post, a feed's post, one before the window, one pinned
        assert f"a.example:p{gone}" not in ids
    assert "b.example:c7" not in ids  # the same comment, listed by a second instance
    assert all(d.source == "lemmy" and d.license.startswith("©") for d in docs)


def test_the_window_is_found_without_reading_everything_since(api):
    run()
    pages = [p for host, path, p in api.calls if host == "a.example" and path == "comment/list"]
    assert len(pages) <= 10  # of 21 pages, 15 newer than the window: doubled and bisected past, not read through


def test_a_reply_is_kept_with_its_thread(api):
    doc = next(d for d in run() if d.id == "a.example:c100")
    assert doc.title == "An old post" and doc.topic == "tech@a.example" and doc.url == "https://a.example/comment/100"
    assert doc.question == f"An old post\n\n{say('The post' + chr(39) + 's own words,').strip()}\n\nThe parent, written before the window."
    assert doc.text.startswith("A reply to it, and it goes on")  # its quote is gone, its Markdown read
    assert doc.extra["parents"] == [90] and doc.extra["ap_id"] == "https://a.example/comment/100"
    ask = next(d for d in run() if d.id == "a.example:p3")
    assert ask.question == "" and ask.title == "Ask: what do you read?"


def test_the_language_is_read_and_the_tag_only_breaks_doubt(api):
    docs = {d.id: d for d in run()}
    assert docs["a.example:c110"].lang == "en" and docs["a.example:c110"].extra["lang_tag"] == "de"
    assert docs["a.example:c111"].lang == "de"  # lid.176 unsure, so the author's tag
    assert docs["a.example:c100"].lang == "en"


def test_a_thread_gives_at_most_a_few(api):
    docs = run()
    assert sum(d.extra["post"] == 2 for d in docs) == lemmy.PER_THREAD
    assert sum(d.extra["post"] in range(30, 50) for d in docs) == 20


def test_the_choice_is_seeded(api):
    a = run(cap=10)
    assert len(a) == 10 and [d.id for d in run(cap=10)] == [d.id for d in a]
    assert {d.id for d in run(cap=10, seed=b"another")} != {d.id for d in a}


def test_markdown_becomes_text():
    md = ("[@ann@social.example](https://social.example/@ann) Here is **bold**, *it*, ~~gone~~ and `code`, "
          "a [link](https://example.org \"t\") and ![a picture](https://example.org/p.png).\n\n# A heading\n\n"
          "::: spoiler Hidden\nsecret\n:::\n\n```\nx = 1\n```\n\nsnake_case_name stays")
    assert lemmy.md_text(md) == "Here is bold, it, gone and code, a link and .\n\nA heading\n\nHidden\nsecret\n\nx = 1\n\nsnake_case_name stays"
    assert lemmy.md_text(md, code=False) == "Here is bold, it, gone and code, a link and .\n\nA heading\n\nHidden\nsecret\n\nsnake_case_name stays"

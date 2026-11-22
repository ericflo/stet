"""Discourse forums as a fresh source, against a stand-in for their JSON: no request leaves the machine."""

import re
import time
import urllib.error
import urllib.parse

import pytest

from stet.exam.fresh import discourse
from stet.exam.rng import Rng

START = 1_800_000_000
END = START + 3600
PROSE = "and it goes on for a while in plain words, as people write when they have something to say. " * 3
SLOW, BUSY, DOWN = "slow.example", "busy.example", "down.example"
DE, FR = "games.example/de/wow", "games.example/fr/wow"


def say(what: str) -> str:
    return f"{what} {PROSE}"


def iso(t: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.123Z", time.gmtime(t))


def post(pid, t, topic, number, raw, title="A topic", reply_to=None, user=None, updated=None, **kw):
    return {"id": pid, "created_at": iso(t), "updated_at": iso(updated or t), "topic_id": topic, "post_number": number,
            "reply_to_post_number": reply_to, "raw": raw, "username": user or f"user{pid}", "user_id": 7, "post_type": 1, "version": 1,
            "hidden": False, "user_deleted": False, "deleted_at": None, "wiki": False, "topic_title": title, "category_id": 5, **kw}


def slow() -> list[dict]:
    """A forum that writes less than a page in the hours around the window: the cases a harvest
    must tell apart."""
    old, ask, big = "An old topic", "Ask: what do you read?", "A big thread"
    return [
        post(1, START - 9000, 1, 1, "The topic's own words.", old),
        post(2, START - 8000, 1, 2, "The first reply, written before the window.", old),
        post(3, START - 7000, 1, 3, "", old, hidden=True),
        post(4, START - 6000, 1, 4, "A reply to the first reply.", old, reply_to=2),
        post(5, START - 5000, 3, 1, "The big thread opens.", big),
        post(99, START - 1, 1, 5, say("Just before the window,"), old),
        post(100, START + 100, 1, 6, '[quote="user4, post:4, topic:1"]\nA reply to the first reply.\n[/quote]\n'
             + say("A [reply](https://example.org/x) to **it** :slight_smile:,"), old, reply_to=4),
        post(101, START + 101, 1, 7, say("A reply to a hidden post,"), old, reply_to=3),
        post(102, START + 102, 1, 8, say("Hidden by flags,"), old, hidden=True),
        post(103, START + 103, 1, 9, say("Deleted by its author,"), old, user_deleted=True),
        post(104, START + 104, 1, 10, say("A moderator's note,"), old, post_type=2),
        post(105, START + 105, 1, 11, say("The forum's own account,"), old, user="system", user_id=-1),
        post(106, START + 106, 1, 12, say("Named as a bot,"), old, user="helperbot"),
        post(107, START + 107, 1, 13, say("A wiki,"), old, wiki=True),
        post(108, START + 108, 1, 14, say("Edited after the window closed,"), old, version=2, updated=END + 60),
        post(109, START + 109, 1, 15, say("Edited before it closed,"), old, reply_to=1, version=2, updated=START + 900),
        post(110, START + 110, 1, 16, "Too short https://example.org/" + "x" * 300, old),
        post(111, START + 111, 2, 1, say("I have been wondering,"), ask),
        post(112, START + 112, 2, 2, say("The first answer,"), ask),
        post(113, START + 113, 2, 3, say("The second answer,"), ask),
        post(114, START + 114, 1, 17, '[quote="user2, post:2, topic:1"]\nThe first reply\n[/quote]\n' + say("To the topic, quoting,"), old),
        post(115, START + 115, 4, 1, say("Said twice,"), "Twice"),
        post(116, START + 116, 5, 1, say("Said twice,"), "Twice again"),
        *[post(120 + i, START + 120 + i, 10 + i, 1, f"What did you expect to happen here?\n\n{say(f'Report {i},')}", f"Report {i}")
          for i in range(3)],
        *[post(130 + i, START + 130 + i, 3, 2 + i, say(f"Reply {i} in the big thread,"), big, reply_to=1) for i in range(10)],
        *[post(150 + i, START + 150 + i, 20 + i, 1, say(f"Post {i} by one author,"), f"By one author {i}", user="prolific") for i in range(5)],
        post(199, END, 1, 18, say("As the window closed,"), old),
        post(200, END + 500, 1, 19, say("After the window,"), old),
    ]


def busy(first: int, every: int, late: int, n: int = 600) -> list[dict]:
    """A forum with a new topic every few seconds, from before the window until late seconds
    after it."""
    t0 = END + late - (n - 1) * every
    return [post(first + i, t0 + i * every, first + i, 1, say(f"Topic {i},"), f"Topic {i}") for i in range(n)]


class FakeForums:
    def __init__(self, sites: dict[str, list[dict]]):
        self.sites, self.calls = sites, []

    def __call__(self, url, params, **kw):
        rest = url.removeprefix("https://")
        site = next(s for s in sorted(self.sites, key=len, reverse=True) if rest.startswith(s + "/"))
        path = rest[len(site) + 1 :]
        self.calls.append((site, path, params.get("before")))
        if site == DOWN:
            raise OSError("no answer")
        posts = self.sites[site]
        if path == "posts.json":
            below = [p for p in posts if "before" not in params or p["id"] < params["before"]]
            return {"latest_posts": sorted(below, key=lambda p: (p["created_at"], p["id"]), reverse=True)[: discourse.PAGE]}
        m = re.fullmatch(r"posts/by_number/(\d+)/(\d+)\.json", path)
        found = [p for p in posts if (p["topic_id"], p["post_number"]) == (int(m.group(1)), int(m.group(2))) and not p["hidden"]]
        if not found:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return found[0]

    def pages(self, site: str) -> int:
        return sum(1 for s, path, _ in self.calls if s == site and path == "posts.json")


@pytest.fixture
def api(monkeypatch):
    fake = FakeForums({SLOW: slow(), BUSY: busy(1000, 30, 7200), DOWN: [],
                       DE: [post(i, START + i, i, 1, say(f"Thema {i},"), f"Thema {i}") for i in range(1, 9)],
                       FR: [post(i, START + i, i, 1, say(f"Sujet {i},"), f"Sujet {i}") for i in range(1, 9)]})
    monkeypatch.setattr(discourse, "get_json", fake)
    monkeypatch.setattr(discourse, "GAP", 0)
    monkeypatch.setattr(discourse, "PER_FORUM", 1000)
    monkeypatch.setattr(discourse, "lid", lambda text: ("de" if "Thema" in text else "en", 0.95))
    return fake


def run(forums=(SLOW,), cap=1000, seed=b"seed", log=lambda *a: None):
    return discourse.harvest(START, END, cap, Rng(seed, "fresh/discourse"), log=log, forums=forums)


def test_only_what_people_wrote_in_the_window_and_left_alone_is_kept(api):
    docs = run()
    ids = {int(d.id.split(":")[1]) for d in docs}
    assert all(START <= d.published < END for d in docs)
    assert {100, 109, 111, 112, 113, 114} <= ids
    for gone in (99, 101, 102, 103, 104, 105, 106, 107, 108, 110, 199, 200):
        assert gone not in ids  # outside the window, a reply to a hidden post, hidden, deleted, a note, bots, a wiki, edited late, short
    assert len(ids & {115, 116}) == 1  # the same words twice are kept once
    assert len(ids) == len(docs) and all(d.source == "discourse" and d.license.startswith("©") and SLOW in d.license for d in docs)
    assert api.pages(SLOW) == 1  # the whole window was in the newest page


def test_a_reply_is_kept_with_its_thread(api):
    docs = {int(d.id.split(":")[1]): d for d in run()}
    doc = docs[100]
    assert doc.url == "https://slow.example/p/100" and doc.title == "An old topic" and doc.topic == "slow.example/c5" and doc.lang == "en"
    assert doc.question == ("An old topic\n\nThe topic's own words.\n\nThe first reply, written before the window.\n\n"
                            "A reply to the first reply.")
    assert doc.extra["parents"] == [2, 4] and doc.extra["topic"] == 1 and doc.extra["number"] == 6
    assert doc.text.startswith("A reply to it, and it goes on")  # its quote is gone, its Markdown and emoji read
    assert docs[109].question == "An old topic\n\nThe topic's own words." and docs[109].extra["parents"] == []  # a reply to the first post
    assert docs[114].extra["parents"] == [2]  # a reply to the topic at large answers the post it quotes
    assert docs[113].extra["parents"] == [2] and docs[113].question.endswith(say("The first answer,").strip())  # or the post before it
    assert docs[112].extra["parents"] == [] and docs[112].question.startswith("Ask: what do you read?\n\nI have been wondering,")
    ask = docs[111]
    assert ask.question == "" and ask.title == "Ask: what do you read?" and ask.text.startswith("I have been wondering,")


def test_a_line_that_several_posts_share_is_a_template(api):
    reports = [d for d in run() if d.title.startswith("Report")]
    assert len(reports) == 3 and all("expect to happen" not in d.text and d.text.startswith("Report") for d in reports)


def test_a_thread_an_author_and_a_host_give_at_most_a_few(api, monkeypatch):
    docs = run()
    assert sum(d.extra["topic"] == 3 for d in docs) == discourse.PER_THREAD
    assert sum(d.title.startswith("By one author") for d in docs) == discourse.PER_AUTHOR
    monkeypatch.setattr(discourse, "PER_FORUM", 5)
    docs = run(forums=(DE, FR))  # two forums on one host share its allowance
    assert len(docs) == 5 and {d.extra["site"] for d in docs} == {DE, FR}
    assert all(d.url.startswith(f"https://{d.extra['site']}/p/") and d.id.startswith(d.extra["site"] + ":") for d in docs)
    assert {d.lang for d in docs} == {"de", "en"}


def test_a_busy_forum_is_read_from_the_end_of_its_window(api, monkeypatch):
    inside = {p["id"] for p in api.sites[BUSY] if START <= discourse._when(p) < END}
    assert len(inside) == 120
    docs = run(forums=(BUSY,))
    assert {int(d.id.split(":")[1]) for d in docs} == inside
    assert api.pages(BUSY) <= 5  # of twelve pages: the five newer than the window are jumped over, not read through
    monkeypatch.setattr(discourse, "MAX_POSTS", 60)
    docs = run(forums=(BUSY,))
    assert {int(d.id.split(":")[1]) for d in docs} == set(sorted(inside)[-60:])


@pytest.mark.parametrize("every,late,n", [(5, 300, 2000), (7, 40_000, 9000), (240, 90_000, 2000), (30, -1800, 2000),
                                          (30, -7200, 2000), (3600, 86_400, 2000), (1, 7200, 20_000)])
def test_the_window_is_found_whenever_the_harvest_runs(monkeypatch, every, late, n):
    """Fast forums and slow, harvests an hour late and a day late, a forum silent since the
    window, each under a burst of posts whose ids are far above the rest."""
    posts = busy(1000, every, late, n=n)
    posts += [post(10**6 + i, END + late + 1 + i, 10**6 + i, 1, say(f"Burst {i},"), f"Burst {i}") for i in range(180)]
    fake = FakeForums({BUSY: posts})
    monkeypatch.setattr(discourse, "get_json", fake)
    monkeypatch.setattr(discourse, "GAP", 0)
    got = discourse._listing(BUSY, START, END, {}, why := discourse.Counter())
    inside = sorted(p["id"] for p in posts if START <= discourse._when(p) < END)
    assert [p["id"] for p in got] == inside[::-1][: discourse.MAX_POSTS] and not why["window not found"]
    assert fake.pages(BUSY) <= 8  # a few to find the window's end, a few to read back from it


def test_a_forum_that_fails_is_left_out(api):
    said = []
    docs = run(forums=(DOWN, SLOW), log=said.append)
    assert docs and all(d.extra["site"] == SLOW for d in docs)
    assert any("down.example" in line and "OSError" in line for line in said)
    assert [d.id for d in run(forums=(SLOW,))] == [d.id for d in docs]


def test_a_post_with_the_marks_the_reader_uses_inside_it_is_read(api):
    for raw in ("Held \x005\x00 and `code` and \x000\x00 here.", "\x00" + "9" * 5000 + "\x00 and `code`", "\x00-1\x00 `a` \x001\x00"):
        assert "\x00" not in discourse.raw_text(raw) and "code" in discourse.raw_text(raw + " `code`")
    assert discourse.raw_text("Held \x005\x00 and `code` and \x000\x00 here.") == "Held 5 and code and 0 here."
    assert discourse._prose("Held \x005\x00 and `code` and \x000\x00 here.") == "Held  and code and code here."
    api.sites["nul.example"] = [post(i, START + i, i, 1, say(f"Marked \x00{i + 3}\x00 `x` post {i},"), f"Marked {i}") for i in range(1, 4)]
    docs = run(forums=("nul.example", SLOW))
    assert sum(d.extra["site"] == "nul.example" for d in docs) == 3 and not any("\x00" in d.text for d in docs)
    assert any(d.extra["site"] == SLOW for d in docs)  # and the other forums with it


@pytest.mark.parametrize("bad", [{"raw": 5}, {"raw": ["a list"]}, {"username": 7}, {"user_id": "seven"}, {"created_at": "yesterday"},
                                 {"id": "one"}, {"version": 2, "updated_at": 12}])
def test_a_forum_that_sends_what_cannot_be_read_is_left_out_and_the_others_kept(api, bad):
    api.sites["odd.example"] = [post(i, START + i, i, 1, say(f"Said twice, {i % 2}"), f"Odd {i}") for i in range(1, 5)]
    api.sites["odd.example"][2].update(bad)
    said = []
    docs = run(forums=(SLOW, "odd.example", DE), log=said.append)
    assert {d.extra["site"] for d in docs} == {SLOW, DE} and any("odd.example" in line for line in said)
    assert [d.id for d in docs] == [d.id for d in run(forums=(SLOW, DE))]  # as if the forum had not answered


def test_the_choice_is_seeded(api):
    a = run(forums=(SLOW, BUSY, DE), cap=12)
    assert len(a) == 12 and [d.id for d in run(forums=(SLOW, BUSY, DE), cap=12)] == [d.id for d in a]
    assert {d.id for d in run(forums=(SLOW, BUSY, DE), cap=12, seed=b"another")} != {d.id for d in a}


def test_markup_becomes_text():
    raw = ('[quote="ann, post:3, topic:12"]\nouter [quote="bob, post:2, topic:12"]inner[/quote] words\n[/quote]\n'
           "I **agree** about `__init__` and <kbd>Ctrl</kbd>+C :slight_smile: :+1:t2: see [the docs](https://example.org/d) at 10:30:45\n"
           "> a quoted line\n<!-- a template's hint -->\n![shot|690x388](upload://abc.png)\n"
           "[log.txt|attachment](upload://xyz.txt) (1.2 MB)\n[poll type=regular]\n* yes\n* no\n[/poll]\n"
           '[details="More"]\nfolded words\n[/details]\n```python\ndef f(*args, **kw):\n    return x[:n:]  # <b>kept</b>\n> kept too\n```\n'
           "After it, `Vec<String>` stays.\n[code]\nx = __name__\n[/code]\n")
    assert discourse.raw_text(raw) == ("I agree about __init__ and Ctrl+C see the docs at 10:30:45\n\nfolded words\n\n"
                                       "def f(*args, **kw):\n    return x[:n:]  # <b>kept</b>\n> kept too\n\n"
                                       "After it, Vec<String> stays.\n\nx = __name__")
    assert discourse.raw_text(raw, code=False) == ("I agree about __init__ and Ctrl+C see the docs at 10:30:45\n\nfolded words\n\n"
                                                   "After it, Vec<String> stays.")

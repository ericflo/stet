"""Hacker News as a fresh source, against a stand-in for its API: no request leaves the machine."""

import pytest

from stet.exam.fresh import hackernews as hn
from stet.exam.rng import Rng

T0 = 1_800_000_000
START, END = T0 + 10 * 1000, T0 + 10 * 1100  # ids 1000 to 1099
PROSE = "and it goes on for a while in plain words, as people write when they have something to say. " * 3


def say(what: str) -> str:
    return f"{what} {PROSE}"


def at(i: int) -> int:
    return T0 + 10 * i


def forum() -> dict[int, dict]:
    """Two thousand items, one every ten seconds: mostly short replies to story 1, and around the
    window the cases a harvest must tell apart."""
    items = {i: {"id": i, "type": "comment", "by": "someone", "time": at(i), "parent": 1, "text": "ok"} for i in range(1, 2001)}
    items[1] = {"id": 1, "type": "story", "by": "someone", "time": at(1), "title": "Busy story", "url": "https://example.org"}
    items[2] = {"id": 2, "type": "story", "by": "someone", "time": at(2), "title": "Big thread", "url": "https://example.org/big"}
    items[5] = {"id": 5, "type": "story", "by": "ann", "time": at(5), "title": "Old story", "text": "The story&#x27;s own words."}
    items[6] = {"id": 6, "type": "comment", "by": "bob", "time": at(6), "parent": 5, "text": "The first parent, written before the window."}
    items[7] = {"id": 7, "type": "comment", "deleted": True, "time": at(7), "parent": 5}
    items[8] = {"id": 8, "type": "comment", "by": "cat", "dead": True, "time": at(8), "parent": 5, "text": "[flagged]"}
    items[999] = {**items[999], "parent": 5, "text": say("Written just before the window opened,")}
    items[1100] = {**items[1100], "parent": 5, "text": say("Written as the window closed,")}
    items[1000] = {"id": 1000, "type": "story", "by": "dan", "time": at(1000), "title": "Ask HN: How do you keep notes?",
                   "text": say("I have tried many apps<p>")}
    items[1001] = {**items[1001], "parent": 6,
                   "text": "&gt; The first parent<p>" + say("A reply to it, with a link <a href=\"https:&#x2F;&#x2F;example.org&#x2F;a&#x2F;long&#x2F;path\" "
                                                          "rel=\"nofollow\">https:&#x2F;&#x2F;example.org&#x2F;a&#x2F;lo...</a>")}
    items[1002] = {**items[1002], "dead": True, "text": say("Killed by flags,")}
    items[1003] = {"id": 1003, "type": "comment", "deleted": True, "time": at(1003), "parent": 5}
    items[1004] = {**items[1004], "parent": 7, "text": say("A reply to a deleted comment,")}
    items[1005] = {**items[1005], "parent": 8, "text": say("A reply to a dead comment,")}
    items[1006] = {**items[1006], "parent": 5, "text": "Too short to keep, even with a link https://example.org/" + "x" * 300}
    items[1007] = {**items[1007], "parent": 5, "by": "summarybot", "text": say("A bot says,")}
    items[1008] = {**items[1008], "parent": 5, "text": say("Said twice,")}
    items[1009] = {**items[1009], "parent": 1, "text": say("Said twice,")}
    for i in range(1010, 1030):  # one big thread
        items[i] = {**items[i], "parent": 2, "text": say(f"Comment {i} in the big thread,")}
    for i in range(1030, 1060):  # and many small ones
        items[i] = {"id": i, "type": "story", "by": "someone", "time": at(i) - 1, "title": f"Story {i}", "url": "https://example.org"}
        items[i + 30] = {**items[i + 30], "parent": i, "text": say(f"The only reply to story {i},")}
    items[1095] = None  # the API has no item at all for a few ids
    items[1500] = None
    return items


class FakeAPI:
    def __init__(self, items):
        self.items, self.calls = items, 0

    def __call__(self, url, params, **kw):
        self.calls += 1
        if url.endswith("/maxitem.json"):
            return max(self.items)
        return self.items.get(int(url.rsplit("/", 1)[1].removesuffix(".json")))


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI(forum())
    monkeypatch.setattr(hn, "get_json", fake)
    monkeypatch.setattr(hn, "lid", lambda text: ("en", 0.99))
    return fake


def run(cap=1000, seed=b"seed"):
    return hn.harvest(START, END, cap, Rng(seed, "fresh/hackernews"), log=lambda *a: None)


def test_the_window_is_found_by_time(api):
    def get(url):
        return api(url, {})

    assert hn.window(get, START, END) == range(1000, 1100)
    assert api.calls < 30  # a bisection, not a walk
    assert hn.window(get, START + 5, END + 5) == range(1001, 1101)
    assert hn.window(get, at(2001), at(2100)) == range(2001, 2001)  # nothing written yet


def test_only_what_was_written_in_the_window_is_kept(api):
    docs = run()
    ids = {d.id for d in docs}
    assert all(START <= d.published < END for d in docs)
    assert "999" not in ids and "1100" not in ids
    assert {"1000", "1001"} <= ids
    for gone in ("1002", "1003", "1004", "1005", "1006", "1007"):  # dead, deleted, replies to them, short, a bot
        assert gone not in ids
    assert len(ids & {"1008", "1009"}) == 1  # the same words twice are kept once
    assert len(ids) == len(docs) and all(d.source == "hackernews" and d.lang == "en" for d in docs)


def test_a_reply_is_kept_with_its_thread(api):
    doc = next(d for d in run() if d.id == "1001")
    assert doc.title == "Old story" and doc.url == "https://news.ycombinator.com/item?id=1001"
    assert doc.question == "Old story\n\nThe story's own words.\n\nThe first parent, written before the window."
    assert doc.extra["story"] == 5 and doc.extra["parents"] == [6]
    assert doc.text.startswith("A reply to it, with a link https://example.org/a/long/path and it goes on")  # its quote is gone, its link whole
    ask = next(d for d in run() if d.id == "1000")
    assert ask.question == "" and ask.title == "Ask HN: How do you keep notes?" and ask.text.startswith("I have tried many apps")


def test_a_long_thread_keeps_its_title_and_its_nearest_end():
    blocks = ["The post. " * 100, "A first reply. " * 50, "The nearest reply ends here."]
    q = hn.thread("A title", blocks, n=400)
    assert len(q) <= 400 and q.startswith("A title\n\n…") and q.endswith("A first reply.\n\nThe nearest reply ends here.")
    assert "The post" not in q
    assert hn.thread("T", ["short", "x" * 1000], n=100) == "T\n\n…"
    assert hn.thread("T", ["a", "b"]) == "T\n\na\n\nb"
    for n in range(60, 400, 7):
        assert len(hn.thread("A title " * 5, blocks, n=n)) <= n


def test_a_thread_gives_at_most_a_few(api):
    docs = run()
    assert sum(d.extra.get("story") == 2 for d in docs) == hn.PER_THREAD
    assert sum(d.extra.get("story") in range(1030, 1060) for d in docs) == 30


def test_the_choice_is_seeded(api):
    a = run(cap=12)
    assert len(a) == 12 and [d.id for d in run(cap=12)] == [d.id for d in a]
    assert {d.id for d in run(cap=12, seed=b"another")} != {d.id for d in a}


def test_markup_becomes_text():
    assert hn.hn_text("One<p>Two &amp; <i>three</i><p><pre><code>  x = 1\n</code></pre>") == "One\n\nTwo & three\n\n  x = 1"
    assert hn.unquoted("&gt; no\n\n> quoted\n\nmine\n  > also quoted\nmine too") == "&gt; no\n\nmine\nmine too"
    assert hn.prose_chars("see https://example.org/x   and    www.example.com now") == len("see and now")

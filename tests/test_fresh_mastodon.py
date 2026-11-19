"""Mastodon as a fresh-text source: reading a window by status id, what a post must be to be
kept, replies with the posts they answer, and a seeded choice. get_json stands in for every
server: no test touches the network."""

import collections
import time
import urllib.error

import pytest

from stet.exam.fresh import mastodon
from stet.exam.fresh.mastodon import allowed, candidate, declined, snowflake, text_of, timeline
from stet.exam.rng import Rng

START, END = 1_790_000_000, 1_790_003_600
LONG = "The river rose all night and by morning the old footbridge was gone, so we walked the long way round to school."
JA = "今朝は川の水かさが増して、古い木の橋が流されてしまったので、私たちは遠回りして学校へ行きました。"


def lid(text):
    if any("぀" <= c <= "ヿ" for c in text):
        return "ja", 0.99
    return "en", 0.97


@pytest.fixture(autouse=True)
def no_lid_model(monkeypatch):
    monkeypatch.setattr(mastodon, "identify", lid)
    monkeypatch.setattr(mastodon, "_last", {})
    monkeypatch.setattr(mastodon, "_delay", {})
    monkeypatch.setattr(mastodon, "_failed", collections.Counter())


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(t))


def account(host, who, **kw):
    local = "@" not in who
    name = who if local else who.split("@")[0]
    base = f"https://{host if local else who.split('@')[1]}"
    return {"id": f"id-{who}", "acct": who, "url": f"{base}/@{name}", "bot": False, "group": False,
            "noindex": False if local else None, "note": "<p>I like rivers.</p>", "fields": [], "indexable": True, "discoverable": True,
            **kw}


def status(host, sid, t, text=LONG, who="ann", lang="en", reply_to=None, **kw):
    acc = kw.pop("account", None) or account(host, who)
    uri_base = acc["url"].rsplit("/@", 1)[0]
    return {"id": str(sid), "uri": f"{uri_base}/users/{acc['acct'].split('@')[0]}/statuses/{sid}", "url": f"{acc['url']}/{sid}",
            "created_at": iso(t), "content": f"<p>{text}</p>", "language": lang, "visibility": "public", "sensitive": False,
            "spoiler_text": "", "edited_at": None, "reblog": None, "in_reply_to_id": str(reply_to) if reply_to else None,
            "in_reply_to_account_id": kw.pop("reply_acct", acc["id"] if reply_to else None), "replies_count": 0, "account": acc, **kw}


def test_status_ids_encode_the_moment():
    assert snowflake(START) >> 16 == START * 1000
    assert snowflake(START) < int(status("h", snowflake(START) + 5, START)["id"]) < snowflake(START + 1)


# --- reading a window -------------------------------------------------------------------------------


class Servers:
    """Servers whose timelines answer max_id paging, and threads and statuses by id."""

    def __init__(self):
        self.timelines: dict[str, list[dict]] = collections.defaultdict(list)
        self.contexts: dict[tuple[str, str], list[dict]] = {}
        self.statuses: dict[tuple[str, str], dict] = {}
        self.requests: collections.Counter = collections.Counter()

    def post(self, host, s, context=None):
        self.timelines[host].append(s)
        self.statuses[(host, s["id"])] = s
        if context is not None:
            self.contexts[(host, s["id"])] = context
            s["replies_count"] = len(context)
        return s

    def get_json(self, url, params, tries=6, pause=2.0):
        host, path = url.removeprefix("https://").split("/", 1)
        self.requests[host] += 1
        if path == "api/v1/timelines/public":
            assert params["local"] == "true" and params["limit"] == 40
            older = [s for s in self.timelines[host] if int(s["id"]) < int(params["max_id"])]
            return sorted(older, key=lambda s: -int(s["id"]))[:40]
        if path.endswith("/context"):
            sid = path.split("/")[3]
            return {"ancestors": [], "descendants": self.contexts.get((host, sid), [])}
        if path.startswith("api/v1/statuses/"):
            sid = path.split("/")[3]
            if (host, sid) not in self.statuses:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return self.statuses[(host, sid)]
        raise AssertionError(url)


def test_a_busy_window_is_read_across_all_of_it(monkeypatch):
    net = Servers()
    for t in range(START - 100, END + 100):  # a post a second: far more than the page budget
        net.post("busy.test", status("busy.test", snowflake(t) + 1, t))
    monkeypatch.setattr(mastodon, "get_json", net.get_json)
    got = timeline("busy.test", START, END)
    assert net.requests["busy.test"] == mastodon.STRATA * mastodon.PAGES
    assert len({(mastodon.parse_time(s["created_at"]) - START) * 4 // (END - START) for s in got}) == 4
    assert all(int(s["id"]) < snowflake(END) for s in got)
    net2 = Servers()
    for t in range(START - 100, END + 100, 300):  # a quiet server: one page covers the window
        net2.post("quiet.test", status("quiet.test", snowflake(t) + 1, t))
    monkeypatch.setattr(mastodon, "get_json", net2.get_json)
    got = timeline("quiet.test", START, END)
    assert net2.requests["quiet.test"] == 1
    assert {mastodon.parse_time(s["created_at"]) for s in got} >= set(range(START + 200, END, 300))
    net3 = Servers()
    for t in range(START - 100, END + 100, 30):  # in between: one unbroken reading back, a little over a page a part
        net3.post("mid.test", status("mid.test", snowflake(t) + 1, t))
    monkeypatch.setattr(mastodon, "get_json", net3.get_json)
    got = timeline("mid.test", START, END)
    assert {mastodon.parse_time(s["created_at"]) for s in got} >= set(range(START + 20, END, 30))
    assert net3.requests["mid.test"] == 4  # 120 posts in the window: three pages and the one that passes its start
    assert len(got) == len({s["id"] for s in got})  # nothing read twice


# --- what a status must be ----------------------------------------------------------------------------


def good(**kw):
    return status("big.test", snowflake(START + 100) + 1, START + 100, **kw)


@pytest.mark.parametrize("s, reason", [
    (good(reblog={"id": "1"}), "a boost"),
    (good(visibility="unlisted"), "not public"),
    (good(sensitive=True), "sensitive"),
    (good(spoiler_text="politics"), "sensitive"),
    (good(edited_at=iso(START + 200)), "edited"),
    (good(account=account("big.test", "ann", bot=True)), "a bot or group"),
    (good(account=account("big.test", "ann", group=True)), "a bot or group"),
    (good(account=account("big.test", "ann", noindex=True)), "author asked not to be indexed"),
    (good(account=account("big.test", "ann", note="<p>Artist. #NoBot #nofediblock</p>")), "author said no in their bio"),
    (good(account=account("big.test", "ann", fields=[{"name": "AI", "value": "No AI training, please"}])), "author said no in their bio"),
    (good(account=account("big.test", "eve@chaos.social")), "from a server that forbids this"),
    (status("big.test", 1, START - 1), "written outside the window"),
    (status("big.test", 1, END), "written outside the window"),
    (good(text="Good morning!"), "short, or mostly links and tags"),
    (good(text="#photography #nature #river #bridge #morning #mist #autumn #landscape #walk #fog #trees"), "short, or mostly links and tags"),
    (good(lang="ja"), "language not clear"),
    (good(visibility="unlisted", reply_to=5), "not public"),  # its author asked for it to be kept out of public view
    (good(visibility="private", reply_to=5), "not public"),
    (good(lang=None), ""),
    (good(text=JA, lang="ja"), ""),
    (good(account=account("big.test", "ann", indexable=False, discoverable=False)), "author not opted in to search"),
])
def test_what_keeps_a_status_out(s, reason):
    assert candidate("big.test", s, START, END, 7)[1] == reason


def test_opting_in_to_search_can_be_required(monkeypatch):
    s = good(account=account("big.test", "ann", indexable=False))
    monkeypatch.setattr(mastodon, "REQUIRE_INDEXABLE", True)
    assert candidate("big.test", s, START, END, 7)[1] == "author not opted in to search"
    assert declined(account("big.test", "ann"), "big.test") == ""


def test_a_status_becomes_a_candidate_and_a_reply_carries_its_question():
    c, why = candidate("big.test", good(), START, END, 7)
    assert why == "" and c.source == "mastodon" and c.lang == "en" and c.text == LONG and c.extra["home"] == "big.test"
    parent = status("big.test", 5, START - 500, text="Has anyone seen what happened to the footbridge?", who="bob@other.example")
    reply = good(reply_to=5, reply_acct=parent["account"]["id"], who="ann")
    r, _ = candidate("big.test", reply, START, END, 7, parent=parent)
    assert r.source == "mastodon-reply" and r.extra["question"] == "Has anyone seen what happened to the footbridge?"
    assert r.extra["self_reply"] is False and r.extra["parent_url"] == parent["url"]
    shy = status("big.test", 6, START - 500, text="Hm?", who="bob@other.example")
    assert candidate("big.test", reply, START, END, 7, parent=shy)[1] == "answers a post with too little text"
    hidden = status("big.test", 6, START - 500, who="bob@other.example", sensitive=True)
    assert candidate("big.test", reply, START, END, 7, parent=hidden)[1] == "answers a sensitive post"
    forbidden = status("big.test", 6, START - 500, who="eve@chaos.social")
    assert candidate("big.test", reply, START, END, 7, parent=forbidden)[1] == "answers a post that may not be used (from a server that forbids this)"


def test_the_mentions_a_reply_starts_with_are_not_its_text():
    html = ('<p><span class="h-card"><a href="https://a.test/@alice" class="u-url mention">@<span>alice</span></a></span> '
            '<span class="h-card"><a href="https://b.test/@bob" class="u-url mention">@<span>bob@b.test</span></a></span> '
            "Thanks, I will ask @carol later.</p><p>Second paragraph.</p>")
    assert text_of({"content": html}) == "Thanks, I will ask @carol later.\nSecond paragraph."


@pytest.mark.parametrize("robots, verdict", [
    ("User-agent: *\nDisallow: /media_proxy/\nDisallow: /interact/\n", (True, 0.0)),
    ("User-agent: GPTBot\nDisallow: /\n\nUser-agent: *\nDisallow: /media_proxy/\n", (True, 0.0)),
    ("User-agent: *\nDisallow: /api/\n", (False, 0.0)),
    ("User-agent: stet-exam-harvester\nDisallow: /\n", (False, 0.0)),
    ("User-agent: *\nCrawl-delay: 3\nDisallow: /interact/\n", (True, 3.0)),
    ("User-agent: *\nContent-Signal: search=yes, ai-input=no, ai-train=no\nAllow: /\n", (False, 0.0)),
    ("User-agent: *\nContent-Signal: search=yes,ai-train=no,use=reference\nAllow: /\n", (True, 0.0)),
])
def test_robots_txt_is_honoured(robots, verdict):
    assert allowed("big.test", robots) == verdict


# --- a whole harvest ----------------------------------------------------------------------------------


def world():
    net = Servers()
    sid = 0
    for t in range(START - 600, END + 600, 20):  # big.test: English, a post every 20 seconds, with threads
        sid += 1
        n = sid
        who = f"user{n % 40}"
        parent = None
        if n % 5 == 0:  # a thread's next part, answering its author's last post
            parent, who = snowflake(t - 20) + 1, f"user{(n - 1) % 40}"
        elif n % 7 == 0:  # or a post of theirs from yesterday, which is not in the window: fetched when needed
            parent = snowflake(t - 86400) + 1
            if n % 3:  # (and one whose post has since been deleted)
                net.statuses[("big.test", str(parent))] = status("big.test", parent, t - 86400, who=who, text=f"Post old {n}: {LONG}")
        reps = []
        if n % 4 == 0:  # someone answered: from another server, from one that forbids this, and a sensitive answer
            to = {"reply_to": snowflake(t) + 1, "reply_acct": f"id-{who}"}
            lag = 30 if t >= START else 700  # a post from before the window answered in it
            reps = [status("big.test", snowflake(t + lag) + 7, t + lag, who=f"bob{n % 30}@other.example", **to,
                           text=f"Reply {n}: I saw it go, the water took the middle span first and then the rest of it."),
                    status("big.test", snowflake(t + 40) + 8, t + 40, who="eve@chaos.social", **to,
                           text=f"Reply {n}: from a server whose users have not agreed to anything like this at all."),
                    status("big.test", snowflake(t + 50) + 9, t + 50, who=f"cid{n % 30}@third.example", **to,
                           text=f"Reply {n}: a reply hidden behind a content warning, which is left out of the sample.",
                           spoiler_text="spoilers")]
        net.post("big.test", status("big.test", snowflake(t) + 1, t, who=who, text=f"Post {n}: {LONG}", reply_to=parent),
                 context=reps or None)
    for t in range(START - 600, END + 600, 60):
        net.post("ja.test", status("ja.test", snowflake(t) + 3, t, who=f"taro{t % 7}", lang="ja", text=f"{t}番: {JA}"))
        net.post("chaos.social", status("chaos.social", snowflake(t) + 3, t, who="eve"))
        net.post("closed.test", status("closed.test", snowflake(t) + 3, t, who="zed"))
    return net


def run(monkeypatch, seed=b"round", cap=40, log=None):
    net = world()
    monkeypatch.setattr(mastodon, "get_json", net.get_json)
    monkeypatch.setattr(mastodon, "robots", lambda host: (host != "closed.test", 0.0))
    docs = mastodon.harvest(START, END, cap, Rng(seed, "fresh/mastodon"), log=log or (lambda *a: None),
                            instances=("big.test", "ja.test", "chaos.social", "closed.test"))
    return docs, net


def test_a_harvest_is_repeatable_capped_and_reads_only_what_it_may(monkeypatch):
    docs, net = run(monkeypatch)
    again, _ = run(monkeypatch)
    other, _ = run(monkeypatch, seed=b"another round")
    assert docs == again and {d.id for d in docs} != {d.id for d in other}
    assert len(docs) == 40 and len({d.id for d in docs}) == 40
    assert net.requests["chaos.social"] == 0 and net.requests["closed.test"] == 0
    assert {d.topic for d in docs} <= {"big.test", "ja.test", "other.example"}
    assert collections.Counter(d.source for d in docs) == {"mastodon": 20, "mastodon-reply": 20}
    assert {d.lang for d in docs} == {"en", "ja"}
    authors = collections.Counter(d.url.rsplit("/", 1)[0] for d in docs)
    assert max(authors.values()) <= mastodon.PER_AUTHOR
    assert len({(d.published - START) * 4 // (END - START) for d in docs}) == 4
    assert all(START <= d.published < END for d in docs)


def test_replies_come_with_the_posts_they_answer(monkeypatch):
    docs, net = run(monkeypatch)
    replies = [d for d in docs if d.source == "mastodon-reply"]
    assert any(d.extra["self_reply"] for d in replies) and any(not d.extra["self_reply"] for d in replies)
    for d in replies:
        assert d.question.startswith("Post ") and d.extra["parent"].startswith("https://big.test/@user")
        assert "chaos.social" not in d.url and "content warning" not in d.text
        if not d.extra["self_reply"]:
            assert d.topic == "other.example" and d.url.startswith("https://other.example/@bob")
    assert any(d.question.startswith("Post old ") for d in replies)  # fetched, as it was not in the window
    for d in docs:
        host, sid = d.id.split(":")
        assert host in ("big.test", "ja.test")
        assert (host, sid) in net.statuses or any(x["id"] == sid for ctx in net.contexts.values() for x in ctx)
        assert d.license.startswith("author's copyright") and d.title == "" and d.extra["uri"].startswith("https://")


def test_threads_from_just_before_the_window_give_replies_written_in_it(monkeypatch):
    net = Servers()
    for i, t in enumerate(range(START - 3000, END, 60)):  # only posts from before the window were answered in it
        root = status("early.test", snowflake(t) + 1, t, who=f"ann{i % 9}", text=f"Post {i}: {LONG}")
        answers = []
        if t < START:
            answers = [status("early.test", snowflake(START + 60 * i) + 2, START + 60 * i, who=f"bob{i}@other.example",
                              reply_to=root["id"], reply_acct=root["account"]["id"], visibility="public",
                              text=f"Reply {i}: the water took the middle span first, and the rest of it went an hour later.")]
        net.post("early.test", root, context=answers or None)
    monkeypatch.setattr(mastodon, "get_json", net.get_json)
    monkeypatch.setattr(mastodon, "robots", lambda host: (True, 0.0))
    docs = mastodon.harvest(START, END, 20, Rng(b"round", "fresh/mastodon"), log=lambda *a: None, instances=("early.test",))
    replies = [d for d in docs if d.source == "mastodon-reply"]
    assert len(replies) == 10 and all(START <= d.published < END and d.extra["visibility"] == "public" for d in replies)
    assert all(d.question.startswith("Post ") and not d.extra["self_reply"] for d in replies)


def test_the_outcome_does_not_depend_on_how_requests_are_spread(monkeypatch):
    docs, _ = run(monkeypatch)
    monkeypatch.setattr(mastodon, "BATCH", 1)
    monkeypatch.setattr(mastodon, "WORKERS", 1)
    one_at_a_time, _ = run(monkeypatch)
    monkeypatch.setattr(mastodon, "BATCH", 1000)
    monkeypatch.setattr(mastodon, "WORKERS", 16)
    all_at_once, _ = run(monkeypatch)
    assert docs == one_at_a_time == all_at_once


def test_a_server_that_fails_does_not_stop_the_harvest(monkeypatch):
    net = world()

    def flaky(url, params, tries=6, pause=2.0):
        if "ja.test" in url:
            raise urllib.error.HTTPError(url, 503, "Unavailable", {}, None)
        return net.get_json(url, params)

    said = []
    monkeypatch.setattr(mastodon, "get_json", flaky)
    monkeypatch.setattr(mastodon, "robots", lambda host: (True, 0.0))
    docs = mastodon.harvest(START, END, 20, Rng(b"round", "fresh/mastodon"), log=said.append, instances=("big.test", "ja.test"))
    assert len(docs) == 20 and {d.lang for d in docs} == {"en"}
    assert any("ja.test" in line and "HTTPError" in line for line in said)


def test_a_server_too_slow_to_wait_for_is_given_up(monkeypatch):
    clock = iter(range(0, 10_000, 40))  # each request takes 40 seconds
    monkeypatch.setattr(mastodon.time, "time", lambda: float(next(clock)))
    monkeypatch.setattr(mastodon, "get_json", lambda *a, **k: [])
    for _ in range(mastodon.TRIES):
        assert mastodon._get("slow.test", "/api/v1/timelines/public", {}) == []
    with pytest.raises(ConnectionError):
        mastodon._get("slow.test", "/api/v1/timelines/public", {})


def test_a_dead_server_is_given_up_after_a_few_failures(monkeypatch):
    calls = []

    def dead(url, params, tries=6, pause=2.0):
        calls.append((url, tries))
        raise TimeoutError("timed out")

    monkeypatch.setattr(mastodon, "get_json", dead)
    for _ in range(5):
        with pytest.raises((TimeoutError, ConnectionError)):
            mastodon._get("dead.test", "/api/v1/statuses/1", {})
    assert len(calls) == mastodon.TRIES and all(t == mastodon.TRIES for _, t in calls)
    gone = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    monkeypatch.setattr(mastodon, "get_json", lambda *a, **k: (_ for _ in ()).throw(gone))
    for _ in range(5):  # a status that is gone says nothing about its server
        with pytest.raises(urllib.error.HTTPError):
            mastodon._get("alive.test", "/api/v1/statuses/1", {})

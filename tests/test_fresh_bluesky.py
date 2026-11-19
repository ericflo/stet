"""Bluesky as a fresh-text source: the websocket reader, what a post must be to be kept, and a
seeded choice that comes out the same every time. No test touches the network."""

import base64
import bisect
import collections
import json

import pytest

from stet.exam.fresh import bluesky
from stet.exam.fresh.bluesky import (
    CLOSE,
    CONT,
    PING,
    PONG,
    TEXT,
    Candidate,
    Frames,
    Messages,
    WebSocket,
    WebSocketError,
    accept_key,
    candidate,
    choose,
    flag,
    frame,
    plan,
    readable,
    says_no,
    unfaceted,
    width,
)
from stet.exam.rng import Rng

# --- frames, from RFC 6455's own examples (section 5.7) --------------------------------------------


def test_the_rfcs_example_frames():
    f = Frames()
    assert f.feed(bytes([0x81, 0x05]) + b"Hello") == [(True, TEXT, b"Hello")]
    masked = bytes([0x81, 0x85, 0x37, 0xFA, 0x21, 0x3D, 0x7F, 0x9F, 0x4D, 0x51, 0x58])
    assert f.feed(masked) == [(True, TEXT, b"Hello")]
    assert frame(TEXT, b"Hello", key=bytes([0x37, 0xFA, 0x21, 0x3D])) == masked
    assert f.feed(bytes([0x89, 0x05]) + b"Hello") == [(True, PING, b"Hello")]
    assert f.feed(bytes([0x82, 0x7E, 0x01, 0x00]) + bytes(256)) == [(True, 0x2, bytes(256))]
    assert f.feed(bytes([0x82, 0x7F]) + (65536).to_bytes(8, "big") + b"x" * 65536) == [(True, 0x2, b"x" * 65536)]
    assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_frames_arrive_in_pieces_and_lengths_of_every_size_round_trip():
    payloads = [b"", b"a" * 125, b"b" * 126, "é".encode() * 40000, b"c" * 70000]
    stream = b"".join(frame(TEXT, p, key=b"\x01\x02\x03\x04") for p in payloads)
    f = Frames()
    got = []
    for i in range(0, len(stream), 997):  # byte boundaries that fall inside headers and payloads alike
        got += f.feed(stream[i:i + 997])
    assert [p for _, _, p in got] == payloads and not f.buf
    assert frame(TEXT, b"x" * 126)[1] == 0x80 | 126 and frame(TEXT, b"x" * 65536)[1] == 0x80 | 127  # clients always mask


def test_fragmented_messages_join_and_control_frames_pass_between_them():
    m = Messages()
    stream = bytes([0x01, 0x03]) + b"Hel" + bytes([0x89, 0x00]) + bytes([0x80, 0x02]) + b"lo"
    assert m.feed(stream[:4]) == []
    assert m.feed(stream[4:]) == [(PING, b""), (TEXT, b"Hello")]
    assert m.feed(frame(TEXT, b"one", fin=False) + frame(CONT, b" two", fin=False) + frame(CONT, b" three")) == [(TEXT, b"one two three")]


@pytest.mark.parametrize("data", [
    bytes([0x80, 0x01]) + b"x",  # a continuation with nothing to continue
    bytes([0x01, 0x01]) + b"x" + bytes([0x81, 0x01]) + b"y",  # a new message inside a fragmented one
    bytes([0x09, 0x00]),  # a fragmented ping
    bytes([0x89, 0x7E, 0x00, 0x7E]) + bytes(126),  # a ping too long to be a control frame
    bytes([0xC1, 0x01]) + b"x",  # a compressed frame, though no compression was agreed
    bytes([0x83, 0x00]),  # an opcode the protocol does not have
])
def test_malformed_streams_are_refused(data):
    with pytest.raises(WebSocketError):
        Messages().feed(data)


def test_oversized_frames_are_refused():
    with pytest.raises(WebSocketError):
        Frames(max_size=10).feed(bytes([0x81, 0x0B]) + b"x" * 11)


class FakeSocket:
    """A server that answers the handshake, then sends what it is given."""

    def __init__(self, then: bytes, accept=None):
        self.sent = b""
        self.then, self.at, self.accept, self.answered, self.closed = then, 0, accept, False, False

    def sendall(self, data):
        self.sent += data

    def recv(self, n):
        if not self.answered:
            self.answered = True
            key = next(ln.split(": ")[1] for ln in self.sent.decode().split("\r\n") if ln.startswith("Sec-WebSocket-Key"))
            head = ("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {self.accept or accept_key(key)}\r\n\r\n")
            self.at = 5
            return head.encode() + self.then[:5]  # the first frame's start comes with the handshake
        out = self.then[self.at:self.at + min(n, 3)]  # and the rest a few bytes at a time
        self.at += len(out)
        return out

    def close(self):
        self.closed = True


def test_a_connection_handshakes_answers_pings_and_closes(monkeypatch):
    server = bytes([0x89, 0x02]) + b"hi" + bytes([0x81, 0x05]) + b"first" + bytes([0x81, 0x06]) + b"second" + bytes([0x88, 0x02, 0x03, 0xE8])
    sock = FakeSocket(server)
    monkeypatch.setattr(bluesky.socket, "create_connection", lambda addr, timeout: sock)
    ws = WebSocket("ws://jetstream.test/subscribe?cursor=5")
    request = sock.sent.decode()
    assert request.startswith("GET /subscribe?cursor=5 HTTP/1.1\r\nHost: jetstream.test\r\n") and "User-Agent: stet-exam-harvester" in request
    assert base64.b64decode(request.split("Sec-WebSocket-Key: ")[1].split("\r\n")[0]).__len__() == 16
    sent_before = len(sock.sent)
    assert ws.recv() == "first"
    pong = Frames().feed(sock.sent[sent_before:])
    assert pong == [(True, PONG, b"hi")] and sock.sent[sent_before + 1] & 0x80  # answered, masked
    assert ws.recv() == "second"
    assert ws.recv() is None and ws.closed
    assert Frames().feed(sock.sent[sent_before + 8:]) == [(True, CLOSE, b"\x03\xe8")]  # the close echoed back
    ws.close()
    assert sock.closed


def test_a_server_that_does_not_upgrade_properly_is_refused(monkeypatch):
    monkeypatch.setattr(bluesky.socket, "create_connection", lambda addr, timeout: FakeSocket(b"", accept="wrong"))
    with pytest.raises(WebSocketError):
        WebSocket("ws://jetstream.test/subscribe")


# --- what a post must be --------------------------------------------------------------------------

START, END = 1_790_000_000, 1_790_003_600
LONG = "The river rose all night and by morning the old footbridge was gone, so we walked the long way round to school."


def lid(text):
    """A stand-in for lid.176: Japanese if it has kana, German if it says so, else English."""
    if any("぀" <= c <= "ヿ" for c in text):
        return "ja", 0.99
    return ("de", 0.95) if "Brücke" in text else ("en", 0.97)


def event(n=1, t=START + 100, text=LONG, did=None, langs=("en",), reply=None, created=None, **record):
    rec = {"$type": bluesky.POST, "createdAt": created or iso_ms(t), "text": text, "langs": list(langs), **record}
    if reply:
        rec["reply"] = {"parent": {"uri": reply, "cid": "bafyparent"}, "root": {"uri": reply, "cid": "bafyparent"}}
    return {"did": did or f"did:plc:author{n}", "time_us": t * 1_000_000 + 1234, "kind": "commit",
            "commit": {"rev": "r", "operation": "create", "collection": bluesky.POST, "rkey": f"rk{n}", "record": rec, "cid": f"bafy{n}"}}


def iso_ms(t):
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S.123Z", time.gmtime(t))


def test_a_good_post_becomes_a_candidate():
    c, why = candidate(event(), START, END, 7, lid=lid)
    assert why == "" and c.source == "bluesky" and c.lang == "en" and c.key == "at://did:plc:author1/app.bsky.feed.post/rk1"
    assert c.extra["created"] == START + 100 and c.extra["cid"] == "bafy1" and c.text == LONG
    r, _ = candidate(event(2, reply="at://did:plc:x/app.bsky.feed.post/p"), START, END, 7, lid=lid)
    assert r.source == "bluesky-reply" and r.extra["parent"] == "at://did:plc:x/app.bsky.feed.post/p"
    assert candidate(event(), START, END, 7, lid=lid)[0].rank != candidate(event(), START, END, 8, lid=lid)[0].rank


@pytest.mark.parametrize("ev, reason", [
    (event(t=START - 10), "written outside the window"),
    (event(t=END), "written outside the window"),
    (event(created="2020-01-01T00:00:00Z"), "written outside the window"),  # backdated: an import of old posts
    (event(created=iso_ms(START + 1000)), "dated far from when it was sent"),
    (event(created="yesterday"), "written outside the window"),
    (event(text="Too short to say much."), "short, or mostly links and tags"),
    (event(text="#art #painting #oil #canvas #landscape #sunset #artist #artwork #gallery #paint #color #nature #sky"), "short, or mostly links and tags"),
    (event(text="🔥" * 50 + " wow " + "🔥" * 50), "short, or mostly links and tags"),
    (event(langs=("ja",)), "language not clear"),  # declared Japanese, written in English
    (event(langs=()), ""),  # declared nothing, and lid.176 is sure
    (event(labels={"$type": "com.atproto.label.defs#selfLabels", "values": [{"val": "porn"}]}), "labelled by its author"),
    (event(labels={"$type": "com.atproto.label.defs#selfLabels", "values": []}), ""),
    (event(bridgyOriginalText="<p>from the fediverse</p>"), "bridged from another network"),
    (event(labels=[{"val": "porn"}]), "malformed"),  # not every server checks what it writes
    (event(text=12345), "malformed"),
    ({"did": "d", "time_us": START * 10**6, "kind": "identity"}, "not a post"),
])
def test_what_keeps_a_post_out(ev, reason):
    assert candidate(ev, START, END, 7, lid=lid)[1] == reason


def test_declared_languages_of_the_wrong_shape_count_as_none_declared():
    e = event()
    e["commit"]["record"]["langs"] = "en"
    c, why = candidate(e, START, END, 7, lid=lid)
    assert why == "" and c.extra["langs"] == []


def test_reposts_deletions_and_edits_are_not_new_posts():
    for op, col in (("delete", bluesky.POST), ("update", bluesky.POST), ("create", "app.bsky.feed.repost")):
        e = event()
        e["commit"].update(operation=op, collection=col)
        assert candidate(e, START, END, 7, lid=lid) == (None, "not a new post")


def test_links_mentions_and_tags_marked_as_facets_do_not_count_as_prose():
    link = "example.com/a-very-long-path-to-an-article-about-rivers-and-bridges..."
    text = f"Read this, it is good: {link} @someone.bsky.social"
    start = text.encode().index(b"example")
    facets = [{"index": {"byteStart": start, "byteEnd": start + len(link)}, "features": [{"$type": "app.bsky.richtext.facet#link"}]}]
    assert "example.com" not in unfaceted(text, facets)
    assert not readable(text, unfaceted(text, facets))
    multi = "Größe 🌊 über die Brücke #fluss"
    s = multi.encode().index(b"#")
    assert unfaceted(multi, [{"index": {"byteStart": s, "byteEnd": len(multi.encode())}}]).strip() == "Größe 🌊 über die Brücke"
    assert unfaceted("abc", [{"index": {"byteStart": "x"}}, {"index": {"byteStart": 5, "byteEnd": 2}}]) == "abc"


def test_a_wide_letter_counts_twice():
    ja = "今朝は川の水かさが増して、古い木の橋が流されてしまったので、私たちは遠回りして学校へ行きました。"
    assert width(ja) == 2 * len(ja) - 3  # its three punctuation marks are not letters
    assert readable(ja) and not readable(ja[:20])
    assert candidate(event(text=ja, langs=("ja",)), START, END, 7, lid=lid)[1] == ""


def test_labels_and_preferences_that_keep_an_author_out():
    assert flag({"labels": [], "author": {"labels": [{"val": "!no-unauthenticated"}]}}) == "!no-unauthenticated"
    assert flag({"labels": [{"val": "porn", "neg": True}], "author": {"labels": []}}) == ""
    assert flag({"author": {"labels": [{"val": "bridged-from-bridgy-fed-activitypub"}]}}).startswith("bridged-from")
    assert flag({"author": {"labels": [{"val": "dr-kris-fictionkin"}]}}) == ""
    assert says_no({"preferences": {"training": {"allow": True}, "inference": {"allow": False}}})
    assert not says_no({"preferences": {"training": {"allow": True}}}) and not says_no(None)
    for bio in ("NO re-upload / NO AI training", "#noai", "AI学習禁止", "no-ai please", "I do not consent to AI scraping"):
        assert bluesky.OPT_OUT.search(bio), bio
    for bio in ("I play the piano and air guitar", "casino AI researcher", "notes on AI safety"):
        assert not bluesky.OPT_OUT.search(bio), bio


# --- a seeded choice ------------------------------------------------------------------------------


def test_slices_spread_over_the_whole_window_and_depend_only_on_the_seed():
    a = plan(START, END, Rng(b"s", "x"))
    assert a == plan(START, END, Rng(b"s", "x")) != plan(START, END, Rng(b"t", "x"))
    width_ = (END - START) // bluesky.SLICES
    for k, (begin, stop) in enumerate(a):
        assert START + k * width_ <= begin and begin + bluesky.MAX_DEPTH <= stop == START + (k + 1) * width_
    assert plan(START, START + 6, Rng(b"s")) == [(START + k // 2, START + (k + 1) // 2) for k in range(12) if (k + 1) // 2 > k // 2]


def cand(i, lang="en", author=None, source="bluesky", text=None):
    return Candidate(key=f"k{i}", source=source, lang=lang, author=author or f"a{i}", text=text or f"text {i}",
                     rank=bluesky.rank(1, f"k{i}"))


def test_choice_follows_languages_shares_and_caps_authors_and_repeats():
    cands = [cand(i) for i in range(100)] + [cand(100 + i, "de") for i in range(100)] + [cand(200 + i, "kaa") for i in range(3)]
    cands += [cand(300 + i, author="spammer") for i in range(10)] + [cand(400 + i, text="same text") for i in range(10)]
    checked = []
    out = choose(cands, 40, lambda todo: (checked.extend(todo), {c.key for c in todo})[1], shares={"en": 4, "de": 1, "kaa": 1})
    assert len(out) == 40 and len(checked) == 40  # only what would be chosen is checked
    langs = collections.Counter(c.lang for c in out)
    assert langs["kaa"] == 3 and langs["en"] > 2.5 * langs["de"]  # a short language leaves its share to the rest
    assert sum(c.author == "spammer" for c in out) <= 2 and sum(c.text == "same text" for c in out) <= 1
    assert out == choose(list(reversed(cands)), 40, lambda todo: {c.key for c in todo}, shares={"en": 4, "de": 1, "kaa": 1})


def test_choice_replaces_what_fails_its_checks_and_honours_the_split():
    cands = [cand(i) for i in range(50)] + [cand(100 + i, source="bluesky-reply") for i in range(50)]
    failing = {c.key for c in cands[::3]}
    seen = []

    def check(todo):
        assert not set(c.key for c in todo) & set(seen)  # nothing is checked twice
        seen.extend(c.key for c in todo)
        return {c.key for c in todo} - failing

    out = choose(cands, 20, check, split={"bluesky": 0.5, "bluesky-reply": 0.5})
    assert len(out) == 20 and not {c.key for c in out} & failing
    assert collections.Counter(c.source for c in out) == {"bluesky": 10, "bluesky-reply": 10}
    assert len(choose(cands[:5], 20, lambda todo: {c.key for c in todo})) == 5


# --- a whole harvest, against a stream and services that are not there ------------------------------


class Network:
    """A replayable stream with a post every second (and a machine posting every second too), and
    answers to the checks."""

    def __init__(self):
        self.events, self.parents = [], {}
        for t in range(START - 300, END + 300):
            self.events.append(event(10**6 + t, t=t, did="did:plc:feedbot", text=f"Update {t}: {LONG}"))
            n = t - START
            de = n % 5 == 0
            text = f"Post {n}: " + ("Die Brücke über den Fluss war heute Morgen weg, also gingen wir den langen Weg zur Schule."
                                    if de else LONG)
            reply = f"at://did:plc:parent{n}/app.bsky.feed.post/p{n}" if n % 3 == 0 else None
            self.events.append(event(n, t=t, text=text, did=f"did:plc:author{n % 400}", langs=("de",) if de else ("en",), reply=reply))
            if reply:
                self.parents[reply] = n
        self.times = [e["time_us"] for e in self.events]
        self.requests = collections.Counter()

    def stream(self, host, cursor):
        self.requests["stream " + host] += 1
        if host == "down.test":
            raise ConnectionResetError("reset by peer")
        yield from self.events[bisect.bisect_left(self.times, cursor):]

    def view(self, uri):
        did, rkey = uri.removeprefix("at://").split("/app.bsky.feed.post/")
        if uri in self.parents:
            n = self.parents[uri]
            return {"uri": uri, "cid": "bafyparent", "author": {"did": did, "labels": []}, "labels": [],
                    "record": {"text": "" if n % 9 == 0 else f"What happened to the bridge, number {n}?"}}
        n = int(rkey.removeprefix("rk"))
        assert did != "did:plc:feedbot"  # never even checked
        if n % 7 == 1:
            return None  # deleted since
        labels = [{"val": "!no-unauthenticated"}] if author_no(did) % 11 == 4 else []
        return {"uri": uri, "cid": f"bafy{n}", "author": {"did": did, "labels": labels}, "labels": [], "record": {"text": "..."}}

    def get_json(self, url, params, tries=6, pause=2.0):
        self.requests[url.split("/xrpc/")[-1]] += 1
        if url.endswith("app.bsky.feed.getPosts"):
            uris = [v for k, v in params if k == "uris"]
            assert len(uris) <= 25
            return {"posts": [v for v in map(self.view, uris) if v]}
        if url.endswith("app.bsky.actor.getProfiles"):
            dids = [v for k, v in params if k == "actors"]
            assert len(dids) <= 25
            return {"profiles": [{"did": d, "description": "#noai" if author_no(d) % 13 == 5 else "I like rivers."} for d in dids]}
        if url.endswith("com.atproto.repo.describeRepo"):
            did = params["repo"]
            cols = ["app.bsky.feed.post"] + (["community.lexicon.preference.ai"] if author_no(did) % 17 == 6 else [])
            return {"collections": cols, "didDoc": {"service": [{"id": "#atproto_pds", "serviceEndpoint": "https://pds.test"}]}}
        if url.endswith("com.atproto.repo.getRecord"):
            return {"value": {"preferences": {"training": {"allow": False}}}}
        raise AssertionError(url)


def author_no(did):
    return int(did.removeprefix("did:plc:author")) if did.startswith("did:plc:author") else -1


def run(monkeypatch, seed=b"round", cap=30, hosts=("jet.test",), log=None):
    net = Network()
    monkeypatch.setattr(bluesky, "_events", net.stream)
    monkeypatch.setattr(bluesky, "get_json", net.get_json)
    monkeypatch.setattr(bluesky, "identify", lid)
    docs = bluesky.harvest(START, END, cap, Rng(seed, "fresh/bluesky"), log=log or (lambda *a: None), hosts=hosts)
    return docs, net


def test_a_harvest_is_repeatable_capped_and_spread_over_the_window(monkeypatch):
    docs, net = run(monkeypatch)
    again, _ = run(monkeypatch)
    other, _ = run(monkeypatch, seed=b"another round")
    assert docs == again and {d.id for d in docs} != {d.id for d in other}
    assert len(docs) == 30
    assert max(collections.Counter(d.id.split("/")[2] for d in docs).values()) <= bluesky.PER_AUTHOR
    assert len({(d.published - START) * bluesky.SLICES // (END - START) for d in docs}) >= 8  # from all over the window
    assert all(START <= d.published < END for d in docs)
    assert collections.Counter(d.source for d in docs) == {"bluesky": 15, "bluesky-reply": 15}
    assert {d.lang for d in docs} == {"en", "de"}
    # enough was read after the first round: nowhere near the whole window
    assert net.requests["stream jet.test"] == bluesky.SLICES


def test_a_harvest_keeps_what_the_checks_pass_and_says_where_it_came_from(monkeypatch):
    said = []
    docs, _ = run(monkeypatch, log=said.append)
    left_out = said[-1].split("checked and left out: ")[1]
    for reason in ("deleted", "labelled !no-unauthenticated", "author said no to AI use", "answers a post with too little text"):
        assert reason in left_out, left_out  # each check had something to catch
    for d in docs:
        n = int(d.id.rsplit("/rk", 1)[1])
        a = author_no(d.id.split("/")[2])
        assert n % 7 != 1  # deleted since
        assert a % 11 != 4 and a % 13 != 5 and a % 17 != 6  # asked logged-out apps not to show them, #noai, said no to AI
        assert d.url == f"https://bsky.app/profile/{d.id.split('/')[2]}/post/rk{n}" and d.title == d.topic == ""
        assert d.license.startswith("author's copyright") and d.extra["cid"] == f"bafy{n}"
        if d.source == "bluesky-reply":
            assert d.question == f"What happened to the bridge, number {n}?" and n % 9 != 0  # a parent with no text is no question
            assert d.extra["parent"].endswith(f"/p{n}") and d.extra["self_reply"] is False
        else:
            assert d.question == "" and n % 3 != 0


def test_a_host_that_fails_is_left_for_the_next(monkeypatch):
    docs, net = run(monkeypatch, hosts=("down.test", "jet.test"))
    assert len(docs) == 30
    assert net.requests["stream down.test"] == 1  # once it failed, the others started with the host that worked


def test_replay_moves_on_from_a_host_that_no_longer_holds_the_moment(monkeypatch):
    net = Network()

    def stream(host, cursor):
        if host == "short.test":  # its replay starts long after the cursor asked for
            cursor = max(cursor, (START + 1000) * 1_000_000)
        return net.stream(host, cursor)

    monkeypatch.setattr(bluesky, "_events", stream)
    got = []
    hosts = ["short.test", "jet.test"]
    out = bluesky.replay(START * 1_000_000, (START + 10) * 1_000_000, got.append, hosts, log=lambda *a: None)
    assert out == (START + 10) * 1_000_000 and hosts == ["jet.test", "short.test"]
    assert [e["time_us"] // 1_000_000 for e in got if e["did"] != "did:plc:feedbot"] == list(range(START, START + 10))


def test_events_are_read_as_plain_json_from_the_cursor(monkeypatch):
    msgs = iter([json.dumps({"time_us": 5}), b"\x28\xb5\x2f\xfd", json.dumps({"time_us": 6}), None])

    class FakeWS:
        def __init__(self, url, **kw):
            assert url == f"wss://h.test/subscribe?wantedCollections={bluesky.POST}&cursor=5"

        def recv(self):
            return next(msgs)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(bluesky, "WebSocket", FakeWS)
    events = bluesky._events("h.test", 5)
    assert [next(events), next(events)] == [{"time_us": 5}, {"time_us": 6}]
    with pytest.raises(ConnectionError):
        next(events)

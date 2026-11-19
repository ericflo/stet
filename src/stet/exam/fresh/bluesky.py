"""Short public posts on Bluesky, in many languages, and replies to them, written in the window.

Every post on Bluesky's network goes out on a public stream as it is made. Jetstream, Bluesky's
keyless JSON relay of that stream, replays it from any moment it still holds (about a day and a
half back), so a window that has closed can be read again. A harvest reads a few short slices
spread over the window, each from its own cursor, and deepens them all together until it has
enough. A post is kept as the record that was written (its cid pins the exact bytes), and a
reply is kept with the post it answers.

Posts are held to what their authors asked of apps like this one. A post is left out if its
author asked that logged-out viewers not see them, labels themselves a bot, is bridged in from
another network, or has said no to AI use of their posts (in an AI-preference record or in
their bio). So is a post deleted or changed since, or labelled adult or graphic, and the posts
of an author who posts faster than a person does.

The helpers at the bottom (what counts as prose, which language a post is in, and how a seeded
sample is chosen) are shared with the other social sources.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import socket
import ssl
import time
import unicodedata
import urllib.parse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

from . import UA, Doc, get_json
from .select import OTHER, SHARES

JETSTREAMS = ("jetstream2.us-east.bsky.network", "jetstream1.us-east.bsky.network",
              "jetstream1.us-west.bsky.network", "jetstream2.us-west.bsky.network")
POST = "app.bsky.feed.post"
APPVIEW = "https://public.api.bsky.app/xrpc"
ENTRYWAY = "https://bsky.social/xrpc"  # answers describeRepo for the accounts on Bluesky's own servers
PLC = "https://plc.directory"
# Bluesky's terms (14 August 2025): "You retain ownership of your Content"; the license users
# grant goes to Bluesky alone, and "Bluesky does not control how others may use your Content".
LICENSE = "author's copyright (public Bluesky post; no license granted to others)"

MIN_CHARS = 80  # characters of prose once links, mentions and hashtags are taken out (a wide CJK letter counts as two)
MIN_PROSE = 0.6  # of a post's characters, at least this share must be prose rather than links, mentions and tags
MIN_LETTERS = 0.5  # and of its prose, at least this share letters rather than emoji, digits and symbols
MIN_QUESTION = 20  # characters of the post a reply answers
MAX_SKEW = 600  # seconds between a post's createdAt and the moment the network relayed it
PER_AUTHOR = 2
MAX_RATE = 120  # posts an hour: an author posting faster than this in the stream read is a machine, labelled or not
REPLY_SHARE = 0.5  # of what is kept, replies (while there are enough)
SLICES = 12  # slices of the window, one in each twelfth of it
STEP = 20  # seconds of stream every slice reads per round
MAX_DEPTH = 240  # seconds of stream a slice reads at most
OVERSAMPLE = 6  # candidates wanted per post kept, before the checks that need the network
LID_AGREE = 0.4  # lid.176's confidence needed when it agrees with a language the author declared
LID_ALONE = 0.8  # and when the author declared none
TIMEOUT = 8.0  # seconds without a byte before a Jetstream host is given up for the next (a healthy one answers in under one)
HOLE = 120  # seconds: a replay that starts later than this after its cursor no longer holds that moment

# Labels on a post or its author that keep it out: the author asked logged-out apps not to show
# them, is a bot, or the post (or author) is adult, graphic or hidden by moderation.
SKIP_LABELS = {"!no-unauthenticated", "!hide", "!warn", "!takedown", "!suspend", "bot", "porn", "sexual", "nudity",
               "sexual-figurative", "graphic-media", "gore", "spam", "rude", "intolerant", "threat", "self-harm", "extremist",
               "impersonation", "misleading"}
# Records in which an author states what AI may do with their posts. Any "allow": false in them
# keeps the author out: scoring a model on a post is putting it into an AI model.
AI_PREFS = ("community.lexicon.preference.ai", "org.user-intents.declaration", "org.user-intents.demo.declaration")
# What an author's bio says when they do not want their posts collected or fed to AI.
OPT_OUT = re.compile(r"#no(?:ai|bots?|index|search|archive|train(?:ing)?|scrap(?:e|ing)|crawl(?:ing)?)\b"
                     r"|\bno[\s-]*(?:gen[\s-]*)?a\.?i\b|\bno[\s-]+(?:ai[\s-]+)?(?:training|scraping|machine[\s-]+learning)\b"
                     r"|\bnot? consent to (?:ai|scraping|training)|AI\s*(?:学習|利用|使用|学习|転載)|AI\s*학습", re.I)


# --- a minimal websocket client (RFC 6455), enough to read Jetstream -----------------------------

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
CONT, TEXT, BINARY, CLOSE, PING, PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA
MAX_MESSAGE = 1 << 24


class WebSocketError(Exception):
    pass


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def mask(data: bytes, key: bytes) -> bytes:
    """XOR with the 4-byte key, repeated: masking and unmasking are the same."""
    n = len(data)
    if not n:
        return b""
    k = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "big") ^ int.from_bytes(k, "big")).to_bytes(n, "big")


def frame(opcode: int, payload: bytes = b"", fin: bool = True, key: bytes | None = None) -> bytes:
    """A frame as a client sends it: always masked, as the protocol requires of clients."""
    key = os.urandom(4) if key is None else key
    n = len(payload)
    head = bytearray([(0x80 if fin else 0) | opcode])
    if n < 126:
        head.append(0x80 | n)
    elif n < 1 << 16:
        head.append(0x80 | 126)
        head += n.to_bytes(2, "big")
    else:
        head.append(0x80 | 127)
        head += n.to_bytes(8, "big")
    return bytes(head) + key + mask(payload, key)


class Frames:
    """Bytes in, whole frames out as (fin, opcode, payload), unmasked if they came masked."""

    def __init__(self, max_size: int = MAX_MESSAGE):
        self.buf = bytearray()
        self.max_size = max_size

    def feed(self, data: bytes) -> list[tuple[bool, int, bytes]]:
        self.buf += data
        out = []
        while (f := self._one()) is not None:
            out.append(f)
        return out

    def _one(self):
        b = self.buf
        if len(b) < 2:
            return None
        fin, rsv, op = bool(b[0] & 0x80), b[0] & 0x70, b[0] & 0x0F
        masked, n, at = bool(b[1] & 0x80), b[1] & 0x7F, 2
        if rsv:
            raise WebSocketError("reserved bits set, though no extension was agreed")
        if op not in (CONT, TEXT, BINARY, CLOSE, PING, PONG):
            raise WebSocketError(f"unknown opcode {op:#x}")
        if n == 126:
            if len(b) < 4:
                return None
            n, at = int.from_bytes(b[2:4], "big"), 4
        elif n == 127:
            if len(b) < 10:
                return None
            n, at = int.from_bytes(b[2:10], "big"), 10
        if op >= CLOSE and (not fin or n > 125):
            raise WebSocketError("a control frame must be whole and short")
        if n > self.max_size:
            raise WebSocketError(f"a frame of {n} bytes")
        key = b""
        if masked:
            if len(b) < at + 4:
                return None
            key, at = bytes(b[at:at + 4]), at + 4
        if len(b) < at + n:
            return None
        payload = bytes(b[at:at + n])
        del b[:at + n]
        return fin, op, mask(payload, key) if masked else payload


class Messages:
    """Frames to messages, as (opcode, payload): a fragmented message's continuation frames are
    joined, and control frames, which may come between them, are passed straight on."""

    def __init__(self, max_size: int = MAX_MESSAGE):
        self.frames = Frames(max_size)
        self.max_size = max_size
        self._op: int | None = None
        self._parts: list[bytes] = []
        self._size = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        out = []
        for fin, op, payload in self.frames.feed(data):
            if op >= CLOSE:
                out.append((op, payload))
                continue
            if op == CONT:
                if self._op is None:
                    raise WebSocketError("a continuation frame with no message to continue")
            elif self._op is not None:
                raise WebSocketError("a new message before the last one ended")
            else:
                self._op = op
            self._parts.append(payload)
            self._size += len(payload)
            if self._size > self.max_size:
                raise WebSocketError(f"a message of more than {self.max_size} bytes")
            if fin:
                out.append((self._op, b"".join(self._parts)))
                self._op, self._parts, self._size = None, [], 0
        return out


class WebSocket:
    """A client connection: recv() gives the next text (str) or binary (bytes) message, and None
    once the server has closed. Pings are answered as they come."""

    def __init__(self, url: str, timeout: float = TIMEOUT, headers: dict | None = None):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("ws", "wss") or not u.hostname:
            raise ValueError(f"not a websocket address: {url}")
        port = u.port or (443 if u.scheme == "wss" else 80)
        sock = socket.create_connection((u.hostname, port), timeout=timeout)
        if u.scheme == "wss":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=u.hostname)
        self.sock = sock
        self.closed = False
        key = base64.b64encode(os.urandom(16)).decode()
        target = (u.path or "/") + (f"?{u.query}" if u.query else "")
        host = u.hostname if u.port is None else f"{u.hostname}:{u.port}"
        lines = [f"GET {target} HTTP/1.1", f"Host: {host}", "Upgrade: websocket", "Connection: Upgrade",
                 f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"]
        lines += [f"{k}: {v}" for k, v in {**UA, **(headers or {})}.items()]
        try:
            sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
            rest = self._handshake(key)
        except BaseException:
            sock.close()
            raise
        self.messages = Messages()
        self.pending: deque[tuple[int, bytes]] = deque(self.messages.feed(rest))

    def _handshake(self, key: str) -> bytes:
        buf = b""
        while b"\r\n\r\n" not in buf:
            data = self.sock.recv(4096)
            if not data or len(buf) > 65536:
                raise WebSocketError("no handshake from the server")
            buf += data
        head, rest = buf.split(b"\r\n\r\n", 1)
        status, *fields = head.decode("latin-1").split("\r\n")
        if status.split(" ")[1:2] != ["101"]:
            raise WebSocketError(f"the server answered {status!r}")
        got = {k.strip().lower(): v.strip() for k, _, v in (f.partition(":") for f in fields)}
        if got.get("upgrade", "").lower() != "websocket" or "upgrade" not in got.get("connection", "").lower():
            raise WebSocketError("the server did not upgrade the connection")
        if got.get("sec-websocket-accept") != accept_key(key):
            raise WebSocketError("the server's accept key is wrong")
        return rest

    def recv(self):
        while True:
            while self.pending:
                op, payload = self.pending.popleft()
                if op == TEXT:
                    return payload.decode("utf-8", "replace")
                if op == BINARY:
                    return payload
                if op == PING:
                    self._send(PONG, payload)
                elif op == CLOSE:
                    if not self.closed:
                        self._send(CLOSE, payload[:2])  # echo the status code back
                        self.closed = True
                    return None
            if self.closed:
                return None
            data = self.sock.recv(1 << 16)
            if not data:
                self.closed = True
                return None
            self.pending.extend(self.messages.feed(data))

    def _send(self, op: int, payload: bytes = b""):
        try:
            self.sock.sendall(frame(op, payload))
        except OSError:
            self.closed = True

    def close(self):
        if not self.closed:
            self._send(CLOSE, (1000).to_bytes(2, "big"))
            self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- reading the stream -----------------------------------------------------------------------------

def _events(host: str, cursor: int):
    """Jetstream's posts from cursor (unix microseconds) on, as plain JSON events (compress=false,
    Jetstream's default), until the caller stops reading."""
    url = f"wss://{host}/subscribe?wantedCollections={POST}&cursor={cursor}"
    with WebSocket(url) as ws:
        while True:
            msg = ws.recv()
            if msg is None:
                raise ConnectionError(f"{host} closed the stream")
            if isinstance(msg, str):  # binary frames are zstd-compressed events, which were not asked for
                yield json.loads(msg)


def replay(cursor: int, until: int, take, hosts: list[str], log=print) -> int:
    """Hand take() every event from cursor to until (unix microseconds), from the first host that
    holds them; a host that stalls or fails is moved to the back of hosts (so the next read starts
    with one that worked), and the next picks up from where it stopped. Returns until, which is
    where the next read of this slice starts."""
    at = cursor
    for _ in range(2 * len(hosts)):
        host = hosts[0]
        events = _events(host, at)
        try:
            first = True
            for e in events:
                t = int(e.get("time_us") or 0)
                if first and t > at + HOLE * 1_000_000:
                    raise LookupError(f"holds nothing from {at // 1_000_000} until {t // 1_000_000}")
                first = False
                if t >= until:
                    return until
                take(e)
                at = max(at, t)
        except (OSError, ValueError, LookupError, WebSocketError) as e:
            log(f"  bluesky: {host}: {type(e).__name__}: {e}")
            hosts.append(hosts.pop(0))
        finally:
            events.close()
    log(f"  bluesky: no host could replay {cursor // 1_000_000}..{until // 1_000_000}; that part is left out")
    return until


def plan(start: int, end: int, rng, slices: int = SLICES, depth: int = MAX_DEPTH) -> list[tuple[int, int]]:
    """Where each slice begins, and the end of its share of the window: one slice in every
    slices-th of the window, at a seeded point that leaves it room to read depth seconds."""
    out = []
    width = end - start
    for k in range(slices):
        a, b = start + width * k // slices, start + width * (k + 1) // slices
        if b <= a:
            continue
        room = (b - a) - min(depth, b - a)
        out.append((a + rng.below(room + 1), b))
    return out


# --- posts --------------------------------------------------------------------------------------

def parse_time(s: str) -> int | None:
    try:
        d = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    if d.tzinfo is None:
        return None
    return math.floor(d.timestamp())


def unfaceted(text: str, facets) -> str:
    """The text without its facets (Bluesky marks links, mentions and hashtags by byte range)."""
    b = text.encode()
    spans = []
    for f in facets or []:
        try:
            s, e = int(f["index"]["byteStart"]), int(f["index"]["byteEnd"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= s < e:
            spans.append((s, min(e, len(b))))
    out, at = [], 0
    for s, e in sorted(spans):
        s = max(s, at)
        if e <= s:
            continue
        out += [b[at:s], b" "]
        at = e
    out.append(b[at:])
    return b"".join(out).decode("utf-8", "replace")


def candidate(event: dict, start: int, end: int, salt: int, lid=None):
    """A new post from Jetstream as (Candidate, "") if it might be kept, else (None, the reason).
    Any server may write to the stream, and not every one checks what it writes."""
    try:
        return _candidate(event, start, end, salt, lid)
    except (TypeError, AttributeError, ValueError, KeyError):
        return None, "malformed"


def _candidate(event: dict, start: int, end: int, salt: int, lid=None):
    if event.get("kind") != "commit":
        return None, "not a post"
    c = event.get("commit") or {}
    if c.get("operation") != "create" or c.get("collection") != POST:
        return None, "not a new post"
    r = c.get("record") or {}
    did, rkey = event.get("did", ""), c.get("rkey", "")
    created = parse_time(r.get("createdAt", ""))
    if created is None or not start <= created < end:
        return None, "written outside the window"
    if abs(int(event.get("time_us") or 0) / 1e6 - created) > MAX_SKEW:
        return None, "dated far from when it was sent"
    if any(k.startswith("bridgy") for k in r) or "fediverseId" in r:
        return None, "bridged from another network"
    if (r.get("labels") or {}).get("values"):
        return None, "labelled by its author"
    text = r.get("text") or ""
    if not isinstance(text, str):
        return None, "malformed"
    text = text.strip()
    bare = unfaceted(text, r.get("facets"))
    if not readable(text, bare):
        return None, "short, or mostly links and tags"
    langs = r.get("langs") or []
    lang = language(bare, [x for x in langs if isinstance(x, str)] if isinstance(langs, list) else [], lid)
    if lang is None:
        return None, "language not clear"
    reply = r.get("reply") or {}
    parent = (reply.get("parent") or {}).get("uri", "")
    uri = f"at://{did}/{POST}/{rkey}"
    return Candidate(key=uri, source="bluesky-reply" if parent else "bluesky", lang=lang, author=did, text=text,
                     rank=rank(salt, uri), extra={"did": did, "rkey": rkey, "cid": c.get("cid", ""), "created": created,
                                                  "langs": [x for x in langs if isinstance(x, str)] if isinstance(langs, list) else [],
                                                  "parent": parent if isinstance(parent, str) else ""}), ""


def _batches(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _views(uris: list[str]) -> dict[str, dict]:
    """Posts as the network shows them now, by uri: a deleted post is missing."""
    out = {}
    for batch in _batches(uris, 25):
        d = get_json(f"{APPVIEW}/app.bsky.feed.getPosts", [("uris", u) for u in batch])  # a list of pairs: uris repeats
        out.update({p["uri"]: p for p in d.get("posts", [])})
    return out


def _profiles(dids: list[str]) -> dict[str, dict]:
    out = {}
    for batch in _batches(dids, 25):
        d = get_json(f"{APPVIEW}/app.bsky.actor.getProfiles", [("actors", x) for x in batch])
        out.update({p["did"]: p for p in d.get("profiles", [])})
    return out


def flag(view: dict) -> str:
    """The first label on a post or its author that keeps it out, or ""."""
    for lab in list(view.get("labels") or []) + list((view.get("author") or {}).get("labels") or []):
        val = lab.get("val", "")
        if not lab.get("neg") and (val in SKIP_LABELS or val.startswith("bridged-from")):
            return val
    return ""


def says_no(value) -> bool:
    """Whether a preference record says "allow": false anywhere in it."""
    if isinstance(value, dict):
        return value.get("allow") is False or any(says_no(v) for v in value.values())
    if isinstance(value, list):
        return any(says_no(v) for v in value)
    return False


def _pds(did: str) -> str:
    if did.startswith("did:plc:"):
        doc = get_json(f"{PLC}/{did}", {}, tries=2)
    elif did.startswith("did:web:"):
        doc = get_json(f"https://{did.removeprefix('did:web:')}/.well-known/did.json", {}, tries=2)
    else:
        raise ValueError(f"cannot resolve {did}")
    return next(s["serviceEndpoint"] for s in doc.get("service", []) if s.get("id", "").endswith("#atproto_pds"))


_unreachable: set[str] = set()  # servers that failed once: their other authors are not tried again


def refuses_ai(did: str) -> bool:
    """Whether the author has said no to AI use of their posts in an AI-preference record. An
    author whose records cannot be read is treated as having said no."""
    pds = ""
    try:
        try:
            repo = get_json(f"{ENTRYWAY}/com.atproto.repo.describeRepo", {"repo": did}, tries=2)
            pds = next(s["serviceEndpoint"] for s in repo["didDoc"]["service"] if s.get("id", "").endswith("#atproto_pds"))
        except Exception:  # an account on a server of its own: find that server
            pds = _pds(did).rstrip("/")
            if pds in _unreachable:
                return True
            repo = get_json(f"{pds}/xrpc/com.atproto.repo.describeRepo", {"repo": did}, tries=2)
        for col in AI_PREFS:
            if col in repo.get("collections", []):
                rec = get_json(f"{pds.rstrip('/')}/xrpc/com.atproto.repo.getRecord", {"repo": did, "collection": col, "rkey": "self"}, tries=2)
                if says_no(rec.get("value")):
                    return True
        return False
    except Exception:
        if pds:
            _unreachable.add(pds)
        return True


def harvest(start: int, end: int, cap: int, rng, log=print, *, hosts=JETSTREAMS, shares: dict | None = None) -> list[Doc]:
    """Up to cap posts and replies written on Bluesky in [start, end), chosen with rng."""
    t0 = time.time()
    end = min(end, int(t0) - 5)  # nothing after now has been written yet
    if cap <= 0 or end <= start:
        return []
    slices = plan(start, end, rng.child("slices"))
    salt = rng.child("order").next64()
    _unreachable.clear()
    pool: dict[str, Candidate] = {}
    why: Counter = Counter()
    posts: Counter = Counter()  # by author, every new post read, kept or not
    events = 0

    def take(e):
        nonlocal events
        events += 1
        if e.get("kind") == "commit" and (e.get("commit") or {}).get("operation") == "create":
            posts[e.get("did", "")] += 1
        c, reason = candidate(e, start, end, salt)
        why[reason or "candidate"] += 1
        if c is not None:
            pool.setdefault(c.key, c)

    def machines(depth: int) -> set[str]:
        """Authors who posted faster than MAX_RATE in the stream read so far."""
        read = sum(min(depth, b - a) for a, b in slices)
        return {d for d, n in posts.items() if n > max(4, MAX_RATE * read / 3600)}

    at = [a * 1_000_000 for a, _ in slices]
    hosts = list(hosts)
    depth = 0
    while depth < MAX_DEPTH:
        depth = min(depth + STEP, MAX_DEPTH)
        for i, (a, b) in enumerate(slices):
            until = min(a + depth, b) * 1_000_000
            if at[i] < until:
                at[i] = replay(at[i], until, take, hosts, log)
        if all(at[i] >= b * 1_000_000 for i, (_, b) in enumerate(slices)):
            break
        busy = machines(depth)
        if sum(c.author not in busy for c in pool.values()) >= OVERSAMPLE * cap:
            break
    read = sum(min(depth, b - a) for a, b in slices)
    busy = machines(depth)
    machines = sum(c.author in busy for c in pool.values())
    pool = {k: c for k, c in pool.items() if c.author not in busy}
    log(f"  bluesky: {events} events over {read}s of stream in {len(slices)} slices, {time.time() - t0:.0f}s; "
        f"{len(pool)} candidates ({machines} more from {len(busy)} authors posting like machines): {dict(why.most_common())}")

    views: dict[str, dict] = {}
    people: dict[str, dict] = {}
    ai: dict[str, bool] = {}
    questions: dict[str, str] = {}
    dropped: Counter = Counter()

    def check(todo: list[Candidate]) -> set[str]:
        parents = [c.extra["parent"] for c in todo if c.extra["parent"]]
        views.update(_views([u for u in dict.fromkeys([c.key for c in todo] + parents) if u not in views]))
        live = []
        for c in todo:
            v, p = views.get(c.key), views.get(c.extra["parent"]) if c.extra["parent"] else None
            reason = ("deleted" if v is None else "changed since" if v.get("cid") != c.extra["cid"] else
                      f"labelled {flag(v)}" if flag(v) else "")
            if not reason and c.extra["parent"]:
                text = ((p or {}).get("record") or {}).get("text", "").strip()
                reason = ("answers a deleted post" if p is None else f"answers a post labelled {flag(p)}" if flag(p) else
                          "answers a post with too little text" if width(text) < MIN_QUESTION else "")
                if not reason:
                    questions[c.key] = text
                    c.extra["parent_cid"] = p.get("cid", "")
                    c.extra["parent_author"] = (p.get("author") or {}).get("did", "")
            if reason:
                dropped[reason] += 1
            else:
                live.append(c)
        dids = list(dict.fromkeys(x for c in live for x in (c.author, c.extra.get("parent_author", "")) if x))
        people.update(_profiles([d for d in dids if d not in people]))
        for d in dids:
            if d not in ai:
                ai[d] = OPT_OUT.search((people.get(d) or {}).get("description") or "") is not None or refuses_ai(d)
        ok = set()
        for c in live:
            if any(ai.get(d, True) for d in (c.author, c.extra.get("parent_author")) if d):
                dropped["author said no to AI use, or could not be checked"] += 1
            else:
                ok.add(c.key)
        return ok

    chosen = choose(list(pool.values()), cap, check, shares=shares, split={"bluesky": 1 - REPLY_SHARE, "bluesky-reply": REPLY_SHARE})
    docs = []
    for c in chosen:
        x = c.extra
        extra = {"cid": x["cid"], "langs": x["langs"]}
        if x["parent"]:
            extra.update(parent=x["parent"], parent_cid=x.get("parent_cid", ""), self_reply=x.get("parent_author") == c.author)
        docs.append(Doc(source=c.source, id=c.key, url=f"https://bsky.app/profile/{x['did']}/post/{x['rkey']}", lang=c.lang,
                        license=LICENSE, published=x["created"], title="", text=c.text, question=questions.get(c.key, ""),
                        extra=extra))
    kinds = Counter(d.source for d in docs)
    log(f"  bluesky: {kinds['bluesky']} posts and {kinds['bluesky-reply']} replies in {len(set(d.lang for d in docs))} languages "
        f"({time.time() - t0:.0f}s); checked and left out: {dict(dropped.most_common())}")
    return docs


# --- shared by the social sources: prose, language, and a seeded choice ------------------------------

_URL = re.compile(r"https?://\S+|\bwww\.\S+|\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}/\S*", re.I)
_MENTION = re.compile(r"(?<![\w@])@[\w.-]+(?:@[\w-]+(?:\.[\w-]+)+)?")
_TAG = re.compile(r"(?<![\w#])#[^\s#]+")
ALIASES = {"nb": "no", "iw": "he", "in": "id", "ji": "yi", "jw": "jv", "fil": "tl"}


def prose(text: str) -> str:
    """A post without its links, mentions and hashtags, its spaces collapsed."""
    for rx in (_URL, _MENTION, _TAG):
        text = rx.sub(" ", text)
    return " ".join(text.split())


def width(text: str) -> int:
    """Characters, a wide (CJK) letter counting as two: a short line of Japanese says as much as a
    longer one of English."""
    return sum(2 if c.isalpha() and unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def readable(text: str, bare: str | None = None, min_chars: int = MIN_CHARS) -> bool:
    """Long enough, mostly prose, and mostly letters. bare is the text with the links, mentions
    and tags a source marks itself already taken out."""
    whole = " ".join(text.split())
    p = prose(text if bare is None else bare)
    if width(p) < min_chars or len(p) < MIN_PROSE * len(whole):
        return False
    chars = [c for c in p if not c.isspace()]
    return sum(c.isalpha() for c in chars) >= MIN_LETTERS * len(chars)


def lang_code(tag: str) -> str:
    """A language tag's primary code, as lid.176 names it: "pt-BR" is "pt"."""
    code = tag.strip().lower().replace("_", "-").split("-")[0]
    return ALIASES.get(code, code)


def identify(text: str) -> tuple[str, float]:
    """lid.176's best guess at the text's language, and how sure it is."""
    from .langs import model

    (label,), (p,) = model().predict(text.replace("\n", " "), k=1)
    return label.removeprefix("__label__"), float(p)


def language(text: str, declared, lid=None) -> str | None:
    """The language a post is in, when lid.176 agrees with a language its author declared (or,
    when they declared none, is sure); else None."""
    got, p = (lid or identify)(prose(text))
    claimed = {lang_code(x) for x in declared if x and x.strip()}
    if claimed:
        return got if got in claimed and p >= LID_AGREE else None
    return got if p >= LID_ALONE else None


def rank(salt: int, key: str) -> int:
    """A candidate's place in the seeded order: the same for the same seed whatever order the
    stream delivered things in."""
    return int.from_bytes(hashlib.sha256(salt.to_bytes(8, "big") + key.encode()).digest()[:8], "big")


@dataclass
class Candidate:
    key: str  # unique within the source: a post's uri
    source: str  # the Doc's source: a post, or a reply
    lang: str
    author: str
    text: str
    rank: int
    extra: dict = field(default_factory=dict)  # what the source needs to make its Doc


def choose(cands: list[Candidate], cap: int, check, shares: dict | None = None, per_author: int = PER_AUTHOR,
           split: dict | None = None) -> list[Candidate]:
    """Up to cap candidates, best rank first: each source in proportion to its split, and within
    it languages in proportion to their share of the web, a language or source short of
    candidates leaving its share to the rest; at most per_author from any author, and no text twice.

    check(candidates) returns the keys of those that pass the checks that need the network. Only
    candidates that would be chosen are checked: those that fail are replaced, and the
    replacements checked in turn."""
    shares = SHARES if shares is None else shares
    split = split or {}
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in cands:
        groups[(c.source, c.lang)].append(c)
    for g in groups.values():
        g.sort(key=lambda c: (c.rank, c.key))
    within: dict[str, float] = defaultdict(float)  # each source's split is shared among its languages by their shares
    for src, lang in groups:
        within[src] += shares.get(lang, OTHER)
    weight = {g: max(1e-12, split.get(g[0], 1.0) * shares.get(g[1], OTHER) / max(within[g[0]], 1e-12)) for g in groups}
    good: set[str] = set()
    bad: set[str] = set()
    while True:
        picked = _draft(groups, weight, cap, per_author, bad)
        todo = [c for c in picked if c.key not in good]
        if not todo:
            return picked
        passed = set(check(todo))
        for c in todo:
            (good if c.key in passed else bad).add(c.key)


def _draft(groups, weight, cap, per_author, bad) -> list[Candidate]:
    """The choice if every candidate not known to be bad passed: each next pick goes to the group
    furthest below its share (the D'Hondt rule), ties to the first group in sorted order."""
    pos = dict.fromkeys(groups, 0)
    taken = dict.fromkeys(groups, 0)
    authors: Counter = Counter()
    texts: set[str] = set()
    out: list[Candidate] = []
    live = sorted(groups)
    while len(out) < cap and live:
        g = min(live, key=lambda g: (taken[g] + 1) / weight[g])
        found = None
        while pos[g] < len(groups[g]):
            c = groups[g][pos[g]]
            pos[g] += 1
            norm = " ".join(c.text.lower().split())
            if c.key not in bad and authors[c.author] < per_author and norm not in texts:
                found = c
                break
        if found is None:
            live.remove(g)
            continue
        out.append(found)
        taken[g] += 1
        authors[found.author] += 1
        texts.add(" ".join(found.text.lower().split()))
    return out

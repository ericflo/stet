"""The harvest: what the sources published, an hour of the clock at a time.

Season 1 harvested the hour after each draw, and every round then waited two hours for its exam.
From season 2 the examiner harvests every hour of the clock as soon as it has settled, whether
or not a round wants it, and posts each harvest's hash to the server at once. A round's exam is
made at its draw from a few of the day's hours, chosen by the draw. Nobody can aim at them:

- which hours a round gets follows from its seed, which nobody has before the draw;
- what a harvest keeps of an hour follows from the hour's own seed, a public random number
  emitted when the hour ends, so nobody writing in that hour can tell what will be kept;
- and each harvest's hash is with the server before the draw, so the organisers cannot make a
  harvest again to suit a round.
"""

from __future__ import annotations

import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..rng import Rng
from . import (
    bluesky,
    ccnews,
    crossref,
    discourse,
    github,
    hackernews,
    lemmy,
    lore,
    mastodon,
    read_snapshot,
    wikipedia,
    write_snapshot,
)
from .select import SHARES

HOUR = 3600  # a harvest is an hour of the clock: [hour, hour + HOUR), hour a multiple of HOUR
# An hour is harvested this long after it ends. Every source but two has all of it at once.
# Common Crawl's news files are public 1-2 hours after the hour closes (5 at worst), and a
# file that is late leaves the news to those that are not: the harvest keeps PER_SOURCE of the
# thousands it finds. Hacker News allows edits for two hours, so an item from late in the hour
# can be harvested as first written.
SETTLE = 3600  # the season's settle_seconds
TARGET = 900  # fresh items a round would like
ADDED_MIN = 20  # edits a harvest reads at least, per language, for text added to existing articles:
TALK_EN, TALK_OTHER = 60, 8  # and for comments on talk pages
PER_SOURCE = 150  # documents a harvest keeps from each of the other sources, chosen with the seed


def _github(start, end, cap, rng, log):
    """GitHub's issues, pull requests and release notes. Its comments are left out: in 2026 about
    half of those in the archive are written by coding agents under people's own accounts, and
    nothing in the archive tells them apart."""
    return [d for d in github.harvest(start, end, cap, rng, log=log) if d.source == "github"]


# Sources besides Wikipedia, each with the kind of writing it gives (see aggregate.SOURCE_KIND).
# Stack Exchange was tried and left out: in 2026 its whole network is asked about 200 questions a
# day, four an hour, and without a key it allows 300 requests a day.
OTHERS = {"ccnews": ccnews.harvest, "github": _github, "hackernews": hackernews.harvest, "lemmy": lemmy.harvest,
          "bluesky": bluesky.harvest, "mastodon": mastodon.harvest, "discourse": discourse.harvest,
          "crossref": crossref.harvest, "lore": lore.harvest}


def added_cap(lang: str) -> int:
    """Twice the language's share of the round's fresh items, so that the languages the web writes
    most in are not left resting on a handful of items."""
    return max(ADDED_MIN, math.ceil(2 * TARGET * SHARES.get(lang, 0.0) / 100))


def harvest_window(seed: bytes, start: int, end: int, dest: Path, langs: list[str] | None = None, log=print) -> tuple[dict, str]:
    """Harvest [start, end) into a snapshot at dest; returns (metadata, sha256). Every source in
    OTHERS gives up to PER_SOURCE documents. No source needs a key or an account. A source that
    fails is left out of the snapshot, and the rest is kept.

    The sources are read at the same time, each by itself: what each gives depends only on the
    window and its own stream of the seed, and they are put together in the same order every time."""
    t0 = time.time()
    tongues = sorted(SHARES) if langs is None else langs

    def new_articles():
        try:
            return wikipedia.harvest(start, end, langs, log=log)
        except Exception as e:
            log(f"  wikipedia: {type(e).__name__}: {e}")
            return []

    def additions(talk: bool):
        name = "wikipedia-talk" if talk else "wikipedia-added"
        got = []
        for lang in tongues:
            cap = (TALK_EN if lang == "en" else TALK_OTHER) if talk else added_cap(lang)
            try:
                got += wikipedia.additions(lang, start, end, cap, Rng(seed, f"fresh/{'talk' if talk else 'added'}/{lang}"), log=log, talk=talk)
            except Exception as e:
                log(f"  {name} {lang}: {type(e).__name__}: {e}")
        log(f"  {name}: {len(got)} " + ("comments on talk pages" if talk else "additions to existing articles"))
        return got

    def other(name, fn):
        try:
            return fn(start, end, PER_SOURCE, Rng(seed, f"fresh/{name}"), log)
        except Exception as e:
            log(f"  {name}: {type(e).__name__}: {e}; left out")
            return None

    with ThreadPoolExecutor(max_workers=3 + len(OTHERS)) as pool:
        wiki = [pool.submit(new_articles), pool.submit(additions, False), pool.submit(additions, True)]
        rest = {name: pool.submit(other, name, fn) for name, fn in OTHERS.items()}
        docs = [d for f in wiki for d in f.result()]
        others = {}
        for name, f in rest.items():
            if (got := f.result()) is not None:
                others[name] = len(got)
                docs += got
    meta = {"window": [start, end], "harvested_at": int(time.time()), "sources": {"wikipedia": langs or "all",
            **others}, "documents": len(docs)}
    sha = write_snapshot(docs, dest, meta)
    log(f"  harvested {len(docs)} documents in {time.time() - t0:.0f}s -> {dest} ({sha[:12]})")
    return meta, sha


def load(dest: Path):
    return read_snapshot(dest)


# --- an hour of the clock ---------------------------------------------------------------------


def settled(season, now: float) -> int:
    """The latest hour that can be harvested at `now`: it ended at least the season's settle ago."""
    t = int(now) - int(season.raw["exam"]["fresh"]["settle_seconds"]) - HOUR
    return t - t % HOUR


def hour_seed(season, hour: int, beacon_round: int, randomness: str) -> bytes:
    """What an hour's harvest is chosen with: the public random number emitted as the hour ends."""
    return hashlib.sha256(f"stet|{season.id}|harvest|{int(hour)}|{int(beacon_round)}|{randomness.lower()}".encode()).digest()


def harvest_hour(season, hour: int, dest: Path, randomness: str | None = None, log=print) -> tuple[dict, str]:
    """Harvest the hour [hour, hour + HOUR) into a snapshot at dest: (metadata, sha256). The
    random number is the first the beacon emits at or after the hour's end, read from the beacon
    itself unless it is given."""
    if hour % HOUR:
        raise ValueError("a harvest is an hour of the clock")
    number = season.beacon_round_at(hour + HOUR)
    if randomness is None:
        from ...client import beacon_randomness

        randomness = beacon_randomness(season, number)
    meta, sha = harvest_window(hour_seed(season, hour, number, randomness), hour, hour + HOUR, dest, log=log)
    return {**meta, "hour": hour, "beacon_round": number}, sha


def choose_hours(seed: bytes, hours, k: int) -> list[int]:
    """The hours a round's exam is made from: of those it could be given, the k whose sha256 with
    the round's seed is smallest. The server chooses them the same way at the draw."""
    order = sorted(set(int(h) for h in hours), key=lambda h: hashlib.sha256(seed + f"|fresh/hour|{h}".encode()).digest())
    return sorted(order[: max(0, k)])

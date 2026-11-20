"""A round's harvest: everything the sources published in the window after its draw."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..rng import Rng
from . import (
    bluesky,
    ccnews,
    github,
    hackernews,
    lemmy,
    mastodon,
    read_snapshot,
    wikipedia,
    write_snapshot,
)
from .select import SHARES

WINDOW = 2 * 3600  # seconds after the draw
# And then this long before the harvest. Every source but two has all of the window at once.
# Common Crawl's news files are public 1-2 hours after the window closes (5 at worst), and a
# file that is late leaves the news to those that are not: the harvest keeps PER_SOURCE of the
# thousands it finds. Hacker News allows edits for two hours, so an item from late in the window
# can be harvested as first written. Waiting for all of both cost every round two hours more;
# the season file's settle_seconds (10800) is what season 1 began with.
SETTLE = 3600
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
OTHERS = {"ccnews": ccnews.harvest, "github": _github, "hackernews": hackernews.harvest, "lemmy": lemmy.harvest,
          "bluesky": bluesky.harvest, "mastodon": mastodon.harvest}


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

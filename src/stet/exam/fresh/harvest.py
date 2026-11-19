"""A round's harvest: everything the sources published in the window after its draw."""

from __future__ import annotations

import math
import time
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
# And then this long, for the sources to have published all of it and for its text to settle:
# Common Crawl's news files are public 1-2 hours after the window closes (5 at worst), and Hacker
# News allows edits for two hours.
SETTLE = 3 * 3600
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
    fails is left out of the snapshot, and the rest is kept."""
    t0 = time.time()
    docs = []
    try:
        docs += wikipedia.harvest(start, end, langs, log=log)
    except Exception as e:
        log(f"  wikipedia: {type(e).__name__}: {e}")
    added = []
    for lang in sorted(SHARES) if langs is None else langs:
        try:
            added += wikipedia.additions(lang, start, end, added_cap(lang), Rng(seed, f"fresh/added/{lang}"), log=log)
        except Exception as e:
            log(f"  wikipedia-added {lang}: {type(e).__name__}: {e}")
    log(f"  wikipedia-added: {len(added)} additions to existing articles")
    docs += added
    talk = []
    for lang in sorted(SHARES) if langs is None else langs:
        try:
            talk += wikipedia.additions(lang, start, end, TALK_EN if lang == "en" else TALK_OTHER, Rng(seed, f"fresh/talk/{lang}"), log=log, talk=True)
        except Exception as e:
            log(f"  wikipedia-talk {lang}: {type(e).__name__}: {e}")
    log(f"  wikipedia-talk: {len(talk)} comments on talk pages")
    docs += talk
    others = {}
    for name, fn in OTHERS.items():
        try:
            got = fn(start, end, PER_SOURCE, Rng(seed, f"fresh/{name}"), log)
            others[name] = len(got)
            docs += got
        except Exception as e:
            log(f"  {name}: {type(e).__name__}: {e}; left out")
    meta = {"window": [start, end], "harvested_at": int(time.time()), "sources": {"wikipedia": langs or "all",
            **others}, "documents": len(docs)}
    sha = write_snapshot(docs, dest, meta)
    log(f"  harvested {len(docs)} documents in {time.time() - t0:.0f}s -> {dest} ({sha[:12]})")
    return meta, sha


def load(dest: Path):
    return read_snapshot(dest)

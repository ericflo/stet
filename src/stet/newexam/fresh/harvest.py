"""A round's harvest: everything the sources published in the window after its draw."""

from __future__ import annotations

import math
import time
from pathlib import Path

from ..rng import Rng
from . import read_snapshot, stackexchange, wikipedia, write_snapshot
from .select import SHARES

WINDOW = 2 * 3600  # seconds after the draw
SE_SITES = 30  # sites a harvest without a StackExchange key visits, the largest always among them
SE_MAJOR = ["stackoverflow", "math", "superuser", "serverfault", "askubuntu", "unix", "physics", "electronics", "english", "stats",
            "softwareengineering", "apple", "tex", "dba", "codereview", "mathoverflow.net", "security", "academia", "gis", "diy",
            "es.stackoverflow", "ru.stackoverflow", "pt.stackoverflow", "ja.stackoverflow"]
TARGET = 900  # fresh items a round would like
ADDED_MIN = 20  # edits a harvest reads at least, per language, for text added to existing articles:
TALK_EN, TALK_OTHER = 60, 8  # and for comments on talk pages


def added_cap(lang: str) -> int:
    """Twice the language's share of the round's fresh items, so that the languages the web writes
    most in are not left resting on a handful of items."""
    return max(ADDED_MIN, math.ceil(2 * TARGET * SHARES.get(lang, 0.0) / 100))


def harvest_window(seed: bytes, start: int, end: int, dest: Path, langs: list[str] | None = None, se_sites: int | None = 0,
                   log=print) -> tuple[dict, str]:
    """Harvest [start, end) into a snapshot at dest; returns (metadata, sha256).

    StackExchange is left out unless se_sites says otherwise: without a key it allows a few
    hundred requests a day from an address, and no source here may need a key. A source that
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
    names: list[str] = []
    if se_sites != 0:
        try:
            sites = stackexchange.sites()
            names = sorted(sites)
            if se_sites is not None and se_sites < len(names):
                major = [n for n in SE_MAJOR if n in sites][:se_sites]
                rest = [n for n in names if n not in major]
                names = sorted(major + Rng(seed, "fresh/stackexchange-sites").sample(rest, max(0, se_sites - len(major))))
            docs += stackexchange.harvest(start, end, {k: sites[k] for k in names}, log=log)
        except Exception as e:
            log(f"  stackexchange: {type(e).__name__}: {e}; left out")
            names = []
    meta = {"window": [start, end], "harvested_at": int(time.time()), "sources": {"wikipedia": langs or "all",
            "stackexchange": names}, "documents": len(docs)}
    sha = write_snapshot(docs, dest, meta)
    log(f"  harvested {len(docs)} documents in {time.time() - t0:.0f}s -> {dest} ({sha[:12]})")
    return meta, sha


def load(dest: Path):
    return read_snapshot(dest)

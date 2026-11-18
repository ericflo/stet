"""A round's harvest: everything the sources published in the window after its draw."""

from __future__ import annotations

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
ADDED_EN, ADDED_OTHER = 150, 15  # edits a harvest reads per language, for text added to existing articles


def harvest_window(seed: bytes, start: int, end: int, dest: Path, langs: list[str] | None = None, se_sites: int | None = SE_SITES,
                   log=print) -> tuple[dict, str]:
    """Harvest [start, end) into a snapshot at dest; returns (metadata, sha256)."""
    t0 = time.time()
    docs = wikipedia.harvest(start, end, langs, log=log)
    added = []
    for lang in sorted(SHARES) if langs is None else langs:
        try:
            added += wikipedia.additions(lang, start, end, ADDED_EN if lang == "en" else ADDED_OTHER, Rng(seed, f"fresh/added/{lang}"), log=log)
        except Exception as e:
            log(f"  wikipedia-added {lang}: {type(e).__name__}: {e}")
    log(f"  wikipedia-added: {len(added)} additions to existing articles")
    docs += added
    sites = stackexchange.sites()
    names = sorted(sites)
    if se_sites is not None and se_sites < len(names):
        major = [n for n in SE_MAJOR if n in sites]
        rest = [n for n in names if n not in major]
        names = sorted(major + Rng(seed, "fresh/stackexchange-sites").sample(rest, max(0, se_sites - len(major))))
    docs += stackexchange.harvest(start, end, {k: sites[k] for k in names}, log=log)
    meta = {"window": [start, end], "harvested_at": int(time.time()), "sources": {"wikipedia": langs or "all",
            "stackexchange": names}, "documents": len(docs)}
    sha = write_snapshot(docs, dest, meta)
    log(f"  harvested {len(docs)} documents in {time.time() - t0:.0f}s -> {dest} ({sha[:12]})")
    return meta, sha


def load(dest: Path):
    return read_snapshot(dest)

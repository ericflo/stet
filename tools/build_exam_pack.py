#!/usr/bin/env python3
"""Build a season's exam pack from pinned Common Pile files.

The exam is what "good" means in Stet: openly licensed text that people wrote and edited
on purpose for other people to learn from. Six sources, scored separately and averaged, so
no single style can be gamed.

Every upstream file is pinned by repository revision. The output is a single .jsonl.zst
whose sha256 goes into the season file; after that the pack, not this script, is the
source of truth.

    python tools/build_exam_pack.py --upstream ~/.cache/stet/upstream --out exam-s0.jsonl.zst

A season's *held-out* exam is built the same way from the same files, with a seed that is
kept secret. It shares no document and no sentence opening with the public pack, and its
books come from other shards of the same collection:

    python tools/build_exam_pack.py --upstream ~/.cache/stet/upstream --out heldout.jsonl.zst \\
        --seed "$(cat seed.txt)" --exclude-pack exam-s0.jsonl.zst
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import zstandard

# source name -> list of (repo, revision, filename)
UPSTREAM = {
    "encyclopedia": [("common-pile/wikimedia_filtered", "0641bb84bd9b7162bcddf8be7836822161a9a342", "wikimedia-0027.json.gz")],
    "textbooks": [
        ("common-pile/libretexts_filtered", "70388bca52b4a93515e14b1d56618fd7944988fd", "libretexts-0000.json.gz"),
        ("common-pile/pressbooks_filtered", "1a1d3b50d77f834370f8eb4c0d174668dd1676bb", "pressbooks-0000.json.gz"),
        ("common-pile/oercommons_filtered", "506b6159dadcbc0dc67611cea024eedb04232fb2", "oercommons-0000.json.gz"),
    ],
    "science": [("common-pile/arxiv_abstracts_filtered", "dc1ceab4755eb037ec61e49cf1350dab7ceee6e7", "arxiv-abstracts-dolma-0001.json.gz")],
    "qa": [("common-pile/stackexchange_filtered", "c0ac7373830c688a43fc12d1988c4b19ccd884ab", "stackexchange-dolma-0008.json.gz")],
    "books": [("common-pile/project_gutenberg_filtered", "3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7", "project_gutenberg-dolma-0014.json.gz")],
    "news": [("common-pile/news_filtered", "59aaa8f104e189e6fb8033f0ed319c5c343a03b1", "news-dolma-0000.json.gz")],
}

# The public pack draws from every book in its one shard, so a held-out pack has no book
# left there to use. Its books come from two other shards of the same collection, picked
# by its secret seed. Which two is not said until the season closes.
BOOK_SHARDS = [f"project_gutenberg-dolma-{i:04d}.json.gz" for i in range(14)]


def heldout_upstream(seed: str) -> dict:
    repo, revision, _ = UPSTREAM["books"][0]
    n = len(BOOK_SHARDS)
    first = int.from_bytes(hashlib.sha256(f"{seed}:books:shard".encode()).digest()[:8], "big") % n
    step = int.from_bytes(hashlib.sha256(f"{seed}:books:second".encode()).digest()[:8], "big") % (n - 1)
    picks = sorted({first, (first + 1 + step) % n})
    return {**UPSTREAM, "books": [(repo, revision, BOOK_SHARDS[i]) for i in picks]}


MAX_CHARS = 4000  # a passage is at most this long
MIN_CHARS = 600
NON_ENGLISH_SITE = re.compile(r"^(pt|es|ru|ja)\.|\.(pt|es|ru|ja)\.")
STOPWORDS = (" the ", " of ", " and ", " to ", " in ", " is ", " that ", " for ")


def local_name(repo: str, filename: str) -> str:
    return repo.replace("/", "_") + "__" + filename


def looks_english(text: str) -> bool:
    sample = text[:3000]
    if not sample:
        return False
    letters = sum(c.isascii() and (c.isalpha() or c == " ") for c in sample)
    if letters / len(sample) < 0.72:
        return False
    low = sample.lower()
    return sum(low.count(w) for w in STOPWORDS) >= max(3, len(sample) // 400)


def keep_record(source: str, rec: dict) -> bool:
    md = rec.get("metadata") or {}
    if source == "encyclopedia":
        return md.get("wiki") == "wikipedia.com" and str(md.get("namespace")) == "0"
    if source == "qa":
        site = md.get("site") or ""
        return not NON_ENGLISH_SITE.search(site)
    return True


def passage(text: str, rng: random.Random, source: str) -> str | None:
    """A contiguous window of at most MAX_CHARS, cut on line boundaries."""
    text = text.replace("\r\n", "\n").strip()
    if len(text) < MIN_CHARS:
        return None
    if len(text) <= MAX_CHARS:
        return text
    lo, hi = 0, len(text)
    if source == "books":  # skip front and back matter
        lo, hi = int(len(text) * 0.08), int(len(text) * 0.92)
    start = rng.randrange(lo, max(lo + 1, hi - MAX_CHARS))
    nl = text.find("\n", start)
    start = nl + 1 if 0 <= nl < start + 800 else start
    chunk = text[start : start + MAX_CHARS]
    cut = chunk.rfind("\n")
    if cut > MAX_CHARS // 2:
        chunk = chunk[:cut]
    chunk = chunk.strip()
    return chunk if len(chunk) >= MIN_CHARS else None


def pack_marks(path: Path, season_id: str):
    """What a held-out pack must not share with an existing one: its documents, and its sentences.

    Returns the upstream documents the pack drew from, and a function that says whether a
    passage opens a sentence the way one of the pack's passages does (the same test that
    removes exam text from a corpus).
    """
    from stet.data import sentence_starts
    from stet.spec import load_season

    cfg = load_season(season_id).raw["decontam"]
    k, need = int(cfg["key_bytes"]), int(cfg["min_letters"])
    with open(path, "rb") as f:
        text = zstandard.ZstdDecompressor().stream_reader(f).read().decode("utf-8")
    items = [json.loads(line) for line in text.split("\n") if line]  # not splitlines: passages hold U+2028
    keys: set[bytes] = set()
    for it in items:
        keys.update(sentence_starts(it["text"], k, need))
    return frozenset(it["origin"] for it in items), lambda p: not keys.isdisjoint(sentence_starts(p, k, need))


def read_source(source: str, upstream: Path, want: int, rng: random.Random, exclude: frozenset[str] = frozenset(), files=None, shares=None):
    """Reservoir-sample `want` passages across this source's files."""
    files = files or UPSTREAM[source]
    out: list[dict] = []
    seen = 0
    dropped = Counter()
    for repo, revision, filename in files:
        path = upstream / local_name(repo, filename)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    dropped["bad_json"] += 1
                    continue
                if not keep_record(source, rec):
                    dropped["filtered"] += 1
                    continue
                if f"{repo}@{revision[:12]}/{filename}#{rec.get('id')}" in exclude:
                    dropped["in_other_pack"] += 1
                    continue
                text = rec.get("text") or ""
                # One passage per document, more for very long ones (books).
                n_pass = 1 if len(text) < 200_000 else min(6, len(text) // 150_000)
                for _ in range(n_pass):
                    p = passage(text, rng, source)
                    if p is None:
                        dropped["short"] += 1
                        continue
                    if not looks_english(p):
                        dropped["not_english"] += 1
                        continue
                    if shares is not None and shares(p):
                        dropped["shares_a_sentence"] += 1
                        continue
                    md = rec.get("metadata") or {}
                    item = {
                        "source": source,
                        "text": p,
                        "license": md.get("license") or "",
                        "url": md.get("url") or "",
                        "origin": f"{repo}@{revision[:12]}/{filename}#{rec.get('id')}",
                    }
                    seen += 1
                    if len(out) < want:
                        out.append(item)
                    else:
                        j = rng.randrange(seen)
                        if j < want:
                            out[j] = item
    return out, seen, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-split", type=int, default=4000, help="passages per source per split")
    ap.add_argument("--seed", default="20261116")
    ap.add_argument("--exclude-pack", type=Path, help="build a held-out pack: share no document and no sentence opening with this pack")
    ap.add_argument("--season", default="s0", help="whose rule for what a sentence opening is (with --exclude-pack)")
    ap.add_argument("--missing", action="store_true", help="print the upstream files this build needs and does not have (url, tab, local name), then stop")
    args = ap.parse_args()

    exclude, shares = pack_marks(args.exclude_pack, args.season) if args.exclude_pack and not args.missing else (frozenset(), None)
    upstream = heldout_upstream(args.seed) if args.exclude_pack else UPSTREAM
    if args.missing:
        for files in upstream.values():
            for repo, revision, filename in files:
                if not (args.upstream / local_name(repo, filename)).exists():
                    print(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}\t{local_name(repo, filename)}")
        return
    items: list[dict] = []
    for source in UPSTREAM:
        rng = random.Random(f"{args.seed}:{source}")
        got, seen, dropped = read_source(source, args.upstream, 2 * args.per_split, rng, exclude, upstream[source], shares)
        if not got:
            sys.exit(f"{source}: no passages")
        rng.shuffle(got)
        half = len(got) // 2
        for i, it in enumerate(got):
            it["split"] = "practice" if i < half else "official"
        items.extend(got)
        chars = sum(len(it["text"]) for it in got)
        print(f"{source:13s} kept {len(got):6d} of {seen:8d} candidates, {chars / 1e6:6.1f} MB, dropped {dict(dropped)}", file=sys.stderr)

    # Stable order: by split, source, then content hash. Index within (split, source) is
    # what a round's seed selects from.
    for it in items:
        it["id"] = hashlib.sha256(it["text"].encode()).hexdigest()[:16]
    items.sort(key=lambda it: (it["split"], it["source"], it["id"]))
    raw = "".join(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n" for it in items).encode()
    blob = zstandard.ZstdCompressor(level=19).compress(raw)
    args.out.write_bytes(blob)
    print(f"wrote {args.out}: {len(items)} passages, {len(raw) / 1e6:.1f} MB raw, {len(blob) / 1e6:.1f} MB zst", file=sys.stderr)
    print(hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main()

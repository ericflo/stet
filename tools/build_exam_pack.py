#!/usr/bin/env python3
"""Build a season's exam pack from pinned Common Pile files.

The exam is what "good" means in Stet: openly licensed text that people wrote and edited
on purpose for other people to learn from. Six sources, scored separately and averaged, so
no single style can be gamed.

Every upstream file is pinned by repository revision. The output is a single .jsonl.zst
whose sha256 goes into the season file; after that the pack, not this script, is the
source of truth.

    python tools/build_exam_pack.py --upstream ~/.cache/stet/upstream --out exam-s0.jsonl.zst
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


def read_source(source: str, upstream: Path, want: int, rng: random.Random):
    """Reservoir-sample `want` passages across this source's files."""
    files = UPSTREAM[source]
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
    ap.add_argument("--seed", type=int, default=20261116)
    args = ap.parse_args()

    items: list[dict] = []
    for source in UPSTREAM:
        rng = random.Random(f"{args.seed}:{source}")
        got, seen, dropped = read_source(source, args.upstream, 2 * args.per_split, rng)
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

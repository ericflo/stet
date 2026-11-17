#!/usr/bin/env python3
"""Build a pool manifest (sha256, size, path per shard) from a Hugging Face dataset repo.

A manifest pins every shard by content hash, so a round's corpus is the same bytes for
everyone, forever, wherever the shard is fetched from.

    python tools/build_pool_manifest.py mlfoundations/dclm-pool-400m-1x --limit 1024 > out.txt
"""
import argparse
import json
import sys
import urllib.request


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "stet-manifest/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r), r.headers.get("Link", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--limit", type=int, default=1024)
    ap.add_argument("--suffix", default=".jsonl.zst")
    ap.add_argument("--exclude", action="append", default=[], help="manifest whose shards (by sha256) must not appear in this one")
    args = ap.parse_args()

    info, _ = get(f"https://huggingface.co/api/datasets/{args.repo}")
    revision = info["sha"]
    files = []
    url = f"https://huggingface.co/api/datasets/{args.repo}/tree/{revision}?limit=1000"
    while url and (args.exclude or len(files) < args.limit):
        page, link = get(url)
        for f in page:
            if f["type"] == "file" and f["path"].endswith(args.suffix) and "lfs" in f:
                files.append((f["path"], f["lfs"]["oid"], f["size"]))
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.index("<") + 1 : part.index(">")]
    banned = set()
    for path in args.exclude:
        banned |= {line.split()[0] for line in open(path) if line.strip() and not line.startswith("#")}
    files = sorted(f for f in files if f[1] not in banned)[: args.limit]
    print(f"# repo {args.repo}")
    print(f"# revision {revision}")
    print(f"# license {' '.join(t for t in info.get('tags', []) if t.startswith('license:')) or 'see dataset card'}")
    for path, sha, size in files:
        print(f"{sha} {size} {path}")
    print(f"{len(files)} shards, {sum(s for _, _, s in files) / 1e9:.1f} GB", file=sys.stderr)


if __name__ == "__main__":
    main()

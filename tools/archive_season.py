#!/usr/bin/env python3
"""Save a finished season's pages, for /seasons/<id>/ on the live site.

The live server only ever runs the current season. A finished season keeps its pages as
they stood when it closed: its last server build is run on a copy of its records (see
docs/OPERATING.md, "Archiving a season"), and this crawls that server and keeps what each
page says: its title, its description and its content, with links rewritten to stay inside
the archive. The live site shows them inside its own masthead and footer, under a bar that
names the season.

    python tools/archive_season.py --server http://127.0.0.1:18080 --season s0 \\
        --title "Season 0" --out server/archive/s0.jsonl.zst

What is kept: the leaderboard (the season's front page), the record, the rounds and each
round, each curator with each of its files, each account, the verifiers, the glossary and
the documents (the rules as they were). What is not: the home page, the guides to taking
part, the API and its downloads, and anything else that only made sense while the season
ran. Links to the guides go to the live site; links to the rest are left as plain text.

The archive is zstd-compressed JSON Lines: the season first, then one line a page.
Nothing outside the standard library but zstandard, and nothing fetched from anywhere but
--server.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

# Pages kept in the archive. A query is kept only where a page is chosen by it (a curator's files).
KEPT = re.compile(
    r"^/(leaderboard|record|rounds|rounds/[^/]+|editors/[^/]+|editors/[^/]+/[^/]+|u/[^/]+|verifiers|glossary|docs|docs/[^/]+)$"
)
KEPT_QUERY = {"file"}
SEEDS = ["/leaderboard", "/record", "/rounds", "/verifiers", "/glossary", "/docs"]
# Pages of the live site a link may still go to: how to take part now, and its stylesheet's files.
LIVE = re.compile(r"^/(enter|verify|run|static/.*)$")
FORK = re.compile(
    r'<div class="cmd"[^>]*><pre><span class="p">\$ </span><span data-copy-text>stet fork [^<]*</span></pre>'
    r'<button type="button" data-copy>copy</button></div>'
)
ATTR = re.compile(r'\b(href|src)="([^"]*)"')
TITLE = re.compile(r"<title>(.*?)</title>", re.S)
DESC = re.compile(r'<meta name="description" content="([^"]*)">')
MAX_PAGES = 5000


def fetch(base: str, path: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(base + path, timeout=60) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), b""


def canonical(url: str) -> str | None:
    """The archive's name for an internal link (path and kept query), or None if it is not kept."""
    u = urllib.parse.urlsplit(url)
    if u.scheme or u.netloc or not KEPT.match(u.path):
        return None
    q = [(k, v) for k, v in urllib.parse.parse_qsl(u.query) if k in KEPT_QUERY]
    return u.path + ("?" + urllib.parse.urlencode(q) if q else "")


def split_page(text: str) -> dict:
    """A rendered page's title, description and content: what lies between masthead and footer."""
    start, end = text.find("</header>"), text.rfind("<footer>")
    if start < 0 or end < start:
        raise ValueError("a page without the masthead and footer it was rendered with")
    title = TITLE.search(text)
    title = html.unescape(title.group(1)).strip() if title else ""
    title = title.removesuffix(" · Stet") if " · " in title else ""  # the home page's has no page title
    desc = DESC.search(text)
    return {
        "title": title,
        "desc": html.unescape(desc.group(1)) if desc else "",
        "html": text[start + len("</header>") : end].strip() + "\n",
    }


def pack(season: dict, pages: dict[str, dict], out: str) -> int:
    """Write the archive; the same season and pages give the same bytes."""
    try:
        import zstandard
    except ImportError:
        sys.exit("needs the zstandard module: run it with the stet CLI's Python")
    lines = [json.dumps({"season": season}, sort_keys=True, ensure_ascii=False)]
    for key in sorted(pages):
        lines.append(json.dumps({"page": key, **pages[key]}, sort_keys=True, ensure_ascii=False))
    packed = zstandard.ZstdCompressor(level=19).compress(("\n".join(lines) + "\n").encode("utf-8"))
    with open(out, "wb") as f:
        f.write(packed)
    return len(packed)


class Archiver:
    def __init__(self, base: str, season: str):
        self.base = base.rstrip("/")
        self.prefix = "/seasons/" + season

    def rewrite_url(self, url: str, here: str) -> str | None:
        """Where a link in an archived page should point, or None to make it plain text."""
        raw = html.unescape(url)
        if raw.startswith(("#", "mailto:")):
            return url
        u = urllib.parse.urlsplit(urllib.parse.urljoin(here, raw))
        if u.netloc != "archive":
            if "stetrun.com" in u.netloc:
                return None  # the live site's API or files: not this season's
            return url  # elsewhere on the web
        fragment = "#" + u.fragment if u.fragment else ""
        if u.path == "/":
            return self.prefix + "/" + fragment  # the season's front page
        page = canonical(u.path + ("?" + u.query if u.query else ""))
        if page is not None:
            return html.escape(self.prefix + page + fragment, quote=True)
        if LIVE.match(u.path):
            return url
        return None

    def rewrite(self, page: str, text: str) -> str:
        here = "http://archive" + page
        text = FORK.sub("", text)  # a finished season's curators are not forked from here
        text = text.replace(" data-live>", ">").replace(" data-live ", " ")

        def attr(m: re.Match) -> str:
            new = self.rewrite_url(m.group(2), here)
            return f'{m.group(1)}="{new}"' if new is not None else ""

        return ATTR.sub(attr, text)

    def links(self, page: str, text: str) -> list[str]:
        out = []
        for _, url in ATTR.findall(text):
            u = urllib.parse.urlsplit(urllib.parse.urljoin("http://archive" + page, html.unescape(url)))
            if u.netloc == "archive" and (p := canonical(u.path + ("?" + u.query if u.query else ""))):
                out.append(p)
        return out

    def run(self) -> tuple[dict[str, dict], list]:
        pages: dict[str, dict] = {}
        seen, queue, missing = set(SEEDS), list(SEEDS), []
        while queue:
            page = queue.pop(0)
            status, ctype, body = fetch(self.base, page)
            if status != 200 or "text/html" not in ctype:
                missing.append((page, status))
                continue
            text = body.decode("utf-8")
            for nxt in self.links(page, text):
                if nxt not in seen:
                    if len(seen) >= MAX_PAGES:
                        sys.exit(f"more than {MAX_PAGES} pages: is something generating links?")
                    seen.add(nxt)
                    queue.append(nxt)
            pages[page[1:]] = split_page(self.rewrite(page, text))
            if len(pages) % 50 == 0:
                print(f"  {len(pages)} pages", file=sys.stderr)
        return pages, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="the finished season's server, running on a copy of its records")
    ap.add_argument("--season", required=True, help="its id, as in /seasons/<id>/")
    ap.add_argument("--title", required=True, help='what it is called, e.g. "Season 0"')
    ap.add_argument("--about", default="", help="one line for the list of seasons")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    status, _, body = fetch(args.server, "/api/v1/season")
    if status != 200:
        sys.exit(f"{args.server} does not answer /api/v1/season ({status})")
    season = json.loads(body)
    sid = season.get("id") or season.get("season")
    if sid != args.season:
        sys.exit(f"{args.server} runs season {sid!r}, not {args.season!r}")
    status, _, body = fetch(args.server, "/api/v1/rounds?limit=1000")
    listed = json.loads(body).get("rounds", []) if status == 200 else []
    if any(r.get("status") != "settled" for r in listed):
        sys.exit("a round is not settled: archive a season only once it is over, and nothing is scheduled")
    rounds = []
    for r in listed:
        status, _, body = fetch(args.server, "/api/v1/rounds/" + r["id"])
        rounds.append(json.loads(body)["round"] if status == 200 else r)
    status, _, body = fetch(args.server, "/api/v1/leaderboard")
    board = json.loads(body) if status == 200 else {}

    pages, missing = Archiver(args.server, args.season).run()
    meta = {
        "id": args.season,
        "title": args.title,
        "about": args.about,
        "rounds": len(rounds),
        "first": min((r.get("drawn_at") or r.get("draw_at") or 0 for r in rounds), default=0),
        "last": max((r.get("settled_at") or 0 for r in rounds), default=0),
        "scale": season.get("scale") or "",
        "curators": len(board.get("editors") or []) + len(board.get("provisional") or []),
        "pages": len(pages),
    }
    size = pack(meta, pages, args.out)
    raw = sum(len(p["html"]) for p in pages.values())
    print(f"{args.out}: {len(pages)} pages, {raw / 1e6:.1f} MB of content, {size / 1e6:.2f} MB packed")
    for path, status in missing:
        print(f"  not kept: {path} ({status})", file=sys.stderr)


if __name__ == "__main__":
    main()

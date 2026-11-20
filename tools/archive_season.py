#!/usr/bin/env python3
"""Save a finished season's site as static pages, for /seasons/<id>/.

The live server only ever runs the current season. A finished season keeps its pages as
they stood when it closed: its last server build is run on a copy of its records (see
docs/OPERATING.md, "Archiving a season"), and this crawls that server and writes every
page it can reach, with links rewritten to stay inside the archive.

    python tools/archive_season.py --server http://127.0.0.1:18080 --season s0 \\
        --title "Season 0" --out server/archive/s0.tar.zst

What is kept: the home page, leaderboard, record, rounds and each round, each curator with
each of its files, each account, the verifiers, the glossary and the documents (the rules
as they were). What is not: the guides to taking part, the API and its downloads, and
anything else that only made sense while the season ran. Links to the guides go to the
live site; links to the rest are left as plain text.

Nothing outside the standard library, and nothing fetched from anywhere but --server.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request

# Pages kept in the archive. A query is kept only where a page is chosen by it (a curator's files).
KEPT = re.compile(
    r"^/(|leaderboard|record|rounds|rounds/[^/]+|editors/[^/]+|editors/[^/]+/[^/]+|u/[^/]+|verifiers|glossary|docs|docs/[^/]+)$"
)
KEPT_QUERY = {"file"}
# Pages of the live site a link may still go to: how to take part now.
LIVE = {"/enter", "/verify", "/run"}
# Links taken out of the navigation altogether: they lead nowhere in an archive.
NAV_DROP = re.compile(r'<a href="(?:/next|/enter|/verify|/run|https?://[^"]*/api/v1/[^"]*)"[^>]*>[^<]*</a>')
FORK = re.compile(
    r'<div class="cmd"[^>]*><pre><span class="p">\$ </span><span data-copy-text>stet fork [^<]*</span></pre><button type="button" data-copy>copy</button></div>'
)
ATTR = re.compile(r'\b(href|src)="([^"]*)"')
CSS_URL = re.compile(r"url\((['\"]?)(/static/[^)'\"]+)\1\)")
MAX_PAGES = 5000

BANNER_CSS = """
/* archive */
.archived { background: var(--paper-2); border-bottom: 1px solid var(--rule); color: var(--ink-2); font-size: 15px; padding: 8px 0; }
.archived a { color: inherit; }
"""


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


def key_of(page: str) -> str:
    """The name a page is stored under: its path without the leading slash, and its query."""
    return page[1:] or "index"


class Archiver:
    def __init__(self, base: str, season: str, title: str, site: str, live: str):
        self.base, self.season, self.title, self.site, self.live = (
            base.rstrip("/"),
            season,
            title,
            site.rstrip("/"),
            live,
        )
        self.prefix = "/seasons/" + season
        self.static: set[str] = set()

    def rewrite_url(self, url: str, here: str) -> str | None:
        """Where a link in an archived page should point, or None to make it plain text."""
        raw = html.unescape(url)
        if raw.startswith("#") or raw.startswith("mailto:"):
            return url
        absolute = urllib.parse.urljoin(here, raw)
        u = urllib.parse.urlsplit(absolute)
        if u.scheme in ("http", "https") and not raw.startswith(("/", "?")):
            if u.netloc and not self.site.endswith(u.netloc) and "stetrun.com" not in u.netloc:
                return url  # elsewhere on the web
            return None  # the live site's API or files: not this season's
        if u.path.startswith("/static/"):
            self.static.add(u.path)
            return self.prefix + u.path
        page = canonical(raw if raw.startswith("/") else absolute)
        if page is not None:
            return html.escape(
                self.prefix + (page if page != "/" else "/") + ("#" + u.fragment if u.fragment else ""), quote=True
            )
        if u.path in LIVE:
            return url
        return None

    def rewrite(self, page: str, text: str) -> str:
        here = "http://archive" + page
        # The navigation: what leads nowhere in an archive goes.
        for nav in re.findall(r'<nav aria-label="(?:Main|More)">.*?</nav>', text, flags=re.S):
            text = text.replace(nav, NAV_DROP.sub("", nav))
        text = FORK.sub("", text)
        text = text.replace(" data-live>", ">").replace(" data-live ", " ")

        def attr(m: re.Match) -> str:
            new = self.rewrite_url(m.group(2), here)
            return f'{m.group(1)}="{new}"' if new is not None else ""

        text = ATTR.sub(attr, text)
        # Then what the archive adds, whose links are not the archived season's own.
        text = re.sub(
            r'<link rel="canonical" href="[^"]*">',
            f'<link rel="canonical" href="{self.site}{self.prefix}{page}">',
            text,
        )
        text = re.sub(r'<meta property="og:url" content="[^"]*">', "", text)
        # The pill that said what was happening now says the season is over.
        text = re.sub(
            r'<span id="status">.*?</span>\s*(?=<button class="theme")',
            f'<span id="status"><a class="status" href="{self.prefix}/rounds"><span class="dot done" aria-hidden="true"></span>{html.escape(self.title)} is over</a></span>\n    ',
            text,
            count=1,
            flags=re.S,
        )
        return text.replace(
            "<body>",
            f'<body>\n<div class="archived"><div class="wrap">{html.escape(self.title)} is over. '
            f"These are its pages as they stood when it closed. "
            f'<a href="/">The competition goes on: {html.escape(self.live)} →</a> · <a href="/seasons">Every season</a></div></div>',
            1,
        )

    def links(self, page: str, text: str) -> list[str]:
        here = "http://archive" + page
        out = []
        for _, url in ATTR.findall(text):
            raw = html.unescape(url)
            if raw.startswith("#"):
                continue
            absolute = urllib.parse.urljoin(here, raw)
            u = urllib.parse.urlsplit(absolute)
            if u.netloc != "archive":
                continue
            p = canonical(u.path + ("?" + u.query if u.query else ""))
            if p is not None:
                out.append(p)
        return out

    def run(self, seeds: list[str]) -> tuple[dict[str, bytes], dict]:
        files: dict[str, bytes] = {}
        seen, queue, missing = set(seeds), list(seeds), []
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
            files[key_of(page)] = self.rewrite(page, text).encode("utf-8")
            if len(files) % 50 == 0:
                print(f"  {len(files)} pages", file=sys.stderr)
        # The stylesheet, script and what the stylesheet itself loads.
        pending = sorted(self.static)
        done: set[str] = set()
        while pending:
            path = pending.pop()
            if path in done:
                continue
            done.add(path)
            status, _, body = fetch(self.base, path)
            if status != 200:
                missing.append((path, status))
                continue
            if path.endswith(".css"):
                css = body.decode("utf-8")
                for _, ref in CSS_URL.findall(css):
                    pending.append(ref.split("?")[0])
                css = CSS_URL.sub(lambda m: f"url({m.group(1)}{self.prefix}{m.group(2)}{m.group(1)})", css)
                body = (css + BANNER_CSS).encode("utf-8")
            files[path[1:]] = body
        return files, {"missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="the finished season's server, running on a copy of its records")
    ap.add_argument("--season", required=True, help="its id, as in /seasons/<id>/")
    ap.add_argument("--title", required=True, help='what it is called, e.g. "Season 0"')
    ap.add_argument("--about", default="", help="one line for the list of seasons")
    ap.add_argument("--live", default="the current season", help="what the live site is, for the banner")
    ap.add_argument("--site", default="https://www.stetrun.com")
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
    curators = len(board.get("editors") or []) + len(board.get("provisional") or [])

    a = Archiver(args.server, args.season, args.title, args.site, args.live)
    seeds = ["/", "/leaderboard", "/record", "/rounds", "/verifiers", "/glossary", "/docs"]
    files, report = a.run(seeds)
    meta = {
        "id": args.season,
        "title": args.title,
        "about": args.about,
        "rounds": len(rounds),
        "first": min((r.get("drawn_at") or r.get("draw_at") or 0 for r in rounds), default=0),
        "last": max((r.get("settled_at") or 0 for r in rounds), default=0),
        "scale": season.get("scale") or "",
        "curators": curators,
        "pages": sum(1 for k in files if not k.startswith("static/")),
    }
    files["meta.json"] = json.dumps(meta, indent=1).encode()
    try:
        import zstandard
    except ImportError:
        sys.exit("needs the zstandard module: run it with the stet CLI's Python")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            tar.addfile(info, io.BytesIO(files[name]))
    packed = zstandard.ZstdCompressor(level=19).compress(buf.getvalue())
    with open(args.out, "wb") as f:
        f.write(packed)
    raw = sum(len(v) for v in files.values())
    print(
        f"{args.out}: {meta['pages']} pages, {len(files) - meta['pages'] - 1} static files, {raw / 1e6:.1f} MB as pages, {len(packed) / 1e6:.1f} MB packed"
    )
    for path, status in report["missing"]:
        print(f"  not kept: {path} ({status})", file=sys.stderr)


if __name__ == "__main__":
    main()

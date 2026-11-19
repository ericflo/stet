"""New questions and answers on the StackExchange network's sites, posted in the window.

A question is kept as a passage. An answer posted in the window is kept with the question it
answers, which may be older: only the answer is scored, and the answer is new. The site and the
question's first tag are its topic. Without an API key the network allows 300 requests a day, so
a harvest without one visits a seeded sample of sites; with STET_SE_KEY set it visits them all.
"""

from __future__ import annotations

import os

from . import Doc, get_json, html_text

API = "https://api.stackexchange.com/2.3"
LICENSE = "CC BY-SA 4.0"
MIN_CHARS = 200
LANGS = {"es.stackoverflow": "es", "ru.stackoverflow": "ru", "pt.stackoverflow": "pt", "ja.stackoverflow": "ja",
         "rus": "ru", "spanish": "es", "french": "fr", "german": "de", "japanese": "ja", "portuguese": "pt"}


def _params(extra: dict) -> dict:
    key = os.environ.get("STET_SE_KEY")
    return {**extra, **({"key": key} if key else {})}


def sites() -> dict[str, str]:
    """Every main site (no meta sites): its api name and its address. Kept for a week, as the list
    rarely changes and a harvest without a key has few requests to spare."""
    import json
    import time

    from ...data import cache_dir

    path = cache_dir() / "stackexchange-sites.json"
    try:
        if time.time() - path.stat().st_mtime < 7 * 86400:
            return json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    found = _sites()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found))
    return found


def _sites() -> dict[str, str]:
    out = {}
    page = 1
    while True:
        d = get_json(f"{API}/sites", _params({"pagesize": 500, "page": page}))
        for s in d.get("items", []):
            if s.get("site_type") == "main_site":
                out[s["api_site_parameter"]] = s["site_url"]
        if not d.get("has_more"):
            return dict(sorted(out.items()))
        page += 1


def _lang(site: str) -> str:
    return LANGS.get(site, "en")


def _all(path: str, params: dict, cap: int = 3) -> list[dict]:
    out, page = [], 1
    while page <= cap:
        d = get_json(f"{API}/{path}", _params({**params, "page": page, "pagesize": 100}))
        out += d.get("items", [])
        if not d.get("has_more"):
            break
        page += 1
    return out


def harvest(start: int, end: int, site_urls: dict[str, str], log=print) -> list[Doc]:
    docs = []
    for site, base in site_urls.items():
        try:
            qs = _all("questions", {"site": site, "fromdate": start, "todate": end, "sort": "creation", "order": "asc", "filter": "withbody"})
            for q in qs:
                text = html_text(q.get("body", ""))
                if len(text) < MIN_CHARS or q.get("last_edit_date", 0) >= end:  # edited since the window closed
                    continue
                docs.append(Doc(source="stackexchange", id=f"{site}:q{q['question_id']}", url=q.get("link", ""), lang=_lang(site),
                                license=LICENSE, published=int(q["creation_date"]), title=html_text(q.get("title", "")), text=text,
                                topic=f"{site}/{(q.get('tags') or ['-'])[0]}"))
            ans = _all("answers", {"site": site, "fromdate": start, "todate": end, "sort": "creation", "order": "asc", "filter": "withbody"})
            if ans:
                ids = sorted({a["question_id"] for a in ans})
                parents = {}
                for i in range(0, len(ids), 100):
                    for q in _all(f"questions/{';'.join(map(str, ids[i:i + 100]))}", {"site": site, "filter": "withbody"}, cap=1):
                        parents[q["question_id"]] = q
                for a in ans:
                    q = parents.get(a["question_id"])
                    text = html_text(a.get("body", ""))
                    if not q or len(text) < MIN_CHARS or a.get("last_edit_date", 0) >= end:
                        continue
                    docs.append(Doc(source="stackexchange", id=f"{site}:a{a['answer_id']}", lang=_lang(site), license=LICENSE,
                                    url=f"{base}/a/{a['answer_id']}",
                                    published=int(a["creation_date"]), title=html_text(q.get("title", "")), text=text,
                                    question=html_text(q.get("body", "")), topic=f"{site}/{(q.get('tags') or ['-'])[0]}"))
        except Exception as e:
            log(f"  stackexchange {site}: {type(e).__name__}: {e}")
    log(f"  stackexchange: {len(docs)} questions and answers from {len(site_urls)} sites")
    return docs

"""New Wikipedia articles, in every language edition, created by people (not bots) in the window.

Edits do not count: most of an edited article's text was there before the draw. A new article is
kept as the revision that was current when it was harvested, and its Wikidata class (what it is
an instance of: a person, a village, a species) is its topic.
"""

from __future__ import annotations

import calendar
import html
import re
import time
from html.parser import HTMLParser

from . import Doc, get_json, iso

LICENSE = "CC BY-SA 4.0"
MIN_CHARS = 400


def languages() -> list[str]:
    """Every open Wikipedia's language code."""
    d = get_json("https://meta.wikimedia.org/w/api.php", {"action": "sitematrix", "smtype": "language", "smsiteprop": "code|url",
                                                           "smstate": "all", "format": "json", "smlimit": 5000})
    out = []
    for k, v in d["sitematrix"].items():
        if k == "count":
            continue
        for site in v.get("site", []):
            if site.get("code") == "wiki" and "closed" not in site and "private" not in site:
                out.append(site["url"].split("//")[1].split(".")[0])
    return sorted(set(out))


def _new_pages(lang: str, start: int, end: int) -> list[dict]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "recentchanges", "rctype": "new", "rcnamespace": 0, "rcshow": "!bot|!redirect",
              "rcstart": iso(end), "rcend": iso(start), "rclimit": 500, "rcprop": "title|ids|timestamp|sizes", "format": "json"}
    out = []
    while True:
        d = get_json(api, params)
        out += d.get("query", {}).get("recentchanges", [])
        cont = d.get("continue")
        if not cont:
            return out
        params.update(cont)


def _pages(lang: str, pageids: list[int]) -> list[dict]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    out = []
    for i in range(0, len(pageids), 20):
        d = get_json(api, {"action": "query", "prop": "extracts|revisions|pageprops", "explaintext": 1, "exlimit": 20,
                           "exsectionformat": "plain", "rvprop": "ids|timestamp", "ppprop": "wikibase_item",
                           "pageids": "|".join(map(str, pageids[i : i + 20])), "format": "json"})
        out += list(d.get("query", {}).get("pages", {}).values())
    return out


def _classes(qids: list[str]) -> dict[str, str]:
    """What each Wikidata item is an instance of (its first P31), by item id."""
    out = {}
    for i in range(0, len(qids), 50):
        d = get_json("https://www.wikidata.org/w/api.php", {"action": "wbgetentities", "ids": "|".join(qids[i : i + 50]),
                                                           "props": "claims", "format": "json"})
        for q, ent in d.get("entities", {}).items():
            claims = ent.get("claims", {}).get("P31") or []
            try:
                out[q] = claims[0]["mainsnak"]["datavalue"]["value"]["id"]
            except (KeyError, IndexError, TypeError):
                pass
    return out


def harvest(start: int, end: int, langs: list[str] | None = None, log=print) -> list[Doc]:
    docs = []
    pending = []
    for lang in langs or languages():
        try:
            new = [rc for rc in _new_pages(lang, start, end) if start <= _ts(rc["timestamp"]) < end]
            if not new:
                continue
            for p in _pages(lang, [rc["pageid"] for rc in new]):
                text = (p.get("extract") or "").strip()
                rev = (p.get("revisions") or [{}])[0]
                if len(text) < MIN_CHARS or "revid" not in rev:
                    continue
                created = next(_ts(rc["timestamp"]) for rc in new if rc["pageid"] == p["pageid"])
                pending.append((lang, p, rev, text, created))
        except Exception as e:  # one language's API failing does not stop the harvest
            log(f"  wikipedia {lang}: {type(e).__name__}: {e}")
    classes = _classes(sorted({p.get("pageprops", {}).get("wikibase_item") for _, p, *_ in pending} - {None}))
    for lang, p, rev, text, created in pending:
        q = p.get("pageprops", {}).get("wikibase_item", "")
        docs.append(Doc(source="wikipedia", id=f"{lang}:{rev['revid']}", lang=lang, license=LICENSE, published=created,
                        url=f"https://{lang}.wikipedia.org/w/index.php?oldid={rev['revid']}", title=p.get("title", ""), text=text,
                        topic=classes.get(q, ""), extra={"wikidata": q}))
    log(f"  wikipedia: {len(docs)} new articles")
    return docs


def _ts(s: str) -> int:
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


# --- text added to existing articles -------------------------------------------------------------

ADD_MIN_BYTES = 600  # an edit must add at least this much (before markup is stripped)
ADD_MIN_CHARS = 300  # and leave at least this much prose
TALK_MIN_BYTES = 400  # a comment on a talk page


class _Diff(HTMLParser):
    """The lines a MediaWiki diff shows as added, where nothing was taken away on the same row."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[tuple[str, str]]] = []
        self._cell: list[str] | None = None
        self._cls = ""

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag == "td" and self.rows:
            self._cls, self._cell = dict(attrs).get("class", ""), []

    def handle_endtag(self, tag):
        if tag == "td" and self._cell is not None and self.rows:
            self.rows[-1].append((self._cls, "".join(self._cell)))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def added_lines(diff_html: str) -> list[str]:
    p = _Diff()
    p.feed(diff_html)
    out = []
    for row in p.rows:
        classes = " ".join(c for c, _ in row)
        if "diff-empty" in classes and "diff-addedline" in classes:
            out += [t for c, t in row if "diff-addedline" in c]
    return out


def strip_wikitext(text: str) -> str:
    """Wikitext to plain prose, roughly: references, templates, tables, files and markup go."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    for _ in range(5):  # templates nest
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)
    text = re.sub(r"\[\[(?:File|Image|Category|Datei|Fichier|Archivo|Kategorie|Catégorie|Categoría):[^\]]*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"^[*#;|!].*$", "", text, flags=re.M)  # list markup, table rows
    text = re.sub(r"^:+\s*", "", text, flags=re.M)  # indentation, as replies on talk pages are indented
    text = re.sub(r"^[\s{}|\[\]=]*$", "", text, flags=re.M)  # what is left of markup that spanned lines
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip()


def additions(lang: str, start: int, end: int, cap: int, rng, log=print, talk: bool = False) -> list[Doc]:
    """Prose added to existing articles by people in the window, from up to cap of the largest edits.
    With talk, comments added to articles' talk pages instead: people discussing, in every language."""
    api = f"https://{lang}.wikipedia.org/w/api.php"
    # Registered editors only, and nothing already reverted: vandalism is the usual big edit that is
    # neither.
    params = {"action": "query", "list": "recentchanges", "rctype": "edit", "rcnamespace": 1 if talk else 0, "rcshow": "!bot|!minor|!anon",
              "rcstart": iso(end), "rcend": iso(start), "rclimit": 500, "rcprop": "title|ids|timestamp|sizes|tags", "format": "json"}
    edits = []
    while True:
        d = get_json(api, params)
        edits += [rc for rc in d.get("query", {}).get("recentchanges", [])
                  if rc.get("newlen", 0) - rc.get("oldlen", 0) >= (TALK_MIN_BYTES if talk else ADD_MIN_BYTES) and "mw-reverted" not in rc.get("tags", [])]
        if "continue" not in d or len(edits) > 20 * cap:
            break
        params.update(d["continue"])
    if len(edits) > cap:
        edits = rng.sample(edits, cap)
    kept = []
    for rc in edits:
        try:
            c = get_json(api, {"action": "compare", "fromrev": rc["old_revid"], "torev": rc["revid"], "prop": "diff", "format": "json"})
        except Exception:
            continue
        text = strip_wikitext("\n".join(added_lines(c.get("compare", {}).get("*", ""))))
        if len(text) >= ADD_MIN_CHARS:
            kept.append((rc, text))
    if talk:
        return [Doc(source="wikipedia-talk", id=f"{lang}:{rc['revid']}", lang=lang, license=LICENSE, published=_ts(rc["timestamp"]),
                    url=f"https://{lang}.wikipedia.org/w/index.php?diff={rc['revid']}&oldid={rc['old_revid']}", title=rc.get("title", ""),
                    text=text, topic="talk") for rc, text in kept]
    items = {}
    ids = sorted({rc["pageid"] for rc, _ in kept})
    for i in range(0, len(ids), 50):
        d = get_json(api, {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item", "pageids": "|".join(map(str, ids[i : i + 50])),
                           "format": "json"})
        for p in d.get("query", {}).get("pages", {}).values():
            items[p.get("pageid")] = p.get("pageprops", {}).get("wikibase_item", "")
    classes = _classes(sorted(set(items.values()) - {""}))
    return [Doc(source="wikipedia-added", id=f"{lang}:{rc['revid']}", lang=lang, license=LICENSE, published=_ts(rc["timestamp"]),
                url=f"https://{lang}.wikipedia.org/w/index.php?diff={rc['revid']}&oldid={rc['old_revid']}", title=rc.get("title", ""),
                text=text, topic=classes.get(items.get(rc["pageid"], ""), ""), extra={"wikidata": items.get(rc["pageid"], "")})
            for rc, text in kept]

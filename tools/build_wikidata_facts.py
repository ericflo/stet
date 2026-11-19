#!/usr/bin/env python3
"""Build the season's list of facts the web knows, for the exam's knowledge items.

Well-known subjects (most Wikipedia editions first) and one fact each, with labels in the
exam's knowledge languages, from Wikidata through QLever's public endpoint (no key). A fact is
kept only when its subject has exactly one value for it, so its answer is not in doubt.

    python tools/build_wikidata_facts.py src/stet/exam/data/facts-s1.jsonl.zst
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

ENDPOINT = "https://qlever.dev/api/wikidata"
UA = "stet-exam-builder/0.1 (https://www.stetrun.com)"
LANGS = ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "ja", "zh"]
PREFIX = ("PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
          "PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> ")
# (name, class, property, how many subjects, whether the object is a year)
FACTS = [
    ("birthplace", "Q5", "P19", 4000, False),
    ("birthyear", "Q5", "P569", 4000, True),
    ("capital", "Q6256", "P36", 300, False),
    ("currency", "Q6256", "P38", 300, False),
    ("city-country", "Q515", "P17", 2500, False),
    ("author", "Q7725634", "P50", 2000, False),
    ("director", "Q11424", "P57", 2000, False),
    ("painter", "Q3305213", "P170", 800, False),
    ("mountain-country", "Q8502", "P17", 800, False),
    ("symbol", "Q11344", "P246", 130, None),  # a string, the same in every language
]


def query(q: str, tries: int = 4) -> list[dict]:
    body = urllib.parse.urlencode({"query": PREFIX + q}).encode()  # POST: a list of ids is too long for a URL
    for i in range(tries):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers={"Accept": "application/sparql-results+json", "User-Agent": UA,
                                                                          "Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:
            print(f"  query failed ({e}); trying again", file=sys.stderr)
            time.sleep(10 * (i + 1))
    raise RuntimeError("the endpoint did not answer")


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[1]


def main():
    out_path = sys.argv[1]
    rows = []
    for name, cls, prop, n, year in FACTS:
        b = query(f"SELECT ?s ?o ?n WHERE {{ ?s wdt:P31 wd:{cls} . ?s wdt:{prop} ?o . ?s wikibase:sitelinks ?n }} ORDER BY DESC(?n) LIMIT {n * 2}")
        objs = defaultdict(set)
        order, links = [], {}
        for x in b:
            s = qid(x["s"]["value"])
            o = x["o"]["value"]
            o = (o[:4] if o[:4].isdigit() else None) if year else (o if year is None else qid(o))
            if o is None:
                continue
            if s not in links:
                order.append(s)
                links[s] = int(x["n"]["value"])
            objs[s].add(o)
        kept = [(s, next(iter(objs[s]))) for s in order if len(objs[s]) == 1][:n]
        print(f"{name}: {len(kept)} facts", file=sys.stderr)
        rows += [{"fact": name, "s": s, "o": o, "sitelinks": links[s]} for s, o in kept]
    ids = sorted({r["s"] for r in rows} | {r["o"] for r in rows if r["o"].startswith("Q")})
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    langs = ", ".join(f'"{lang}"' for lang in LANGS)
    for i in range(0, len(ids), 1500):
        part = " ".join(f"wd:{x}" for x in ids[i : i + 1500])
        for x in query(f"SELECT ?x ?l WHERE {{ VALUES ?x {{ {part} }} ?x rdfs:label ?l FILTER(LANG(?l) IN ({langs})) }}"):
            labels[qid(x["x"]["value"])][x["l"]["xml:lang"]] = x["l"]["value"]
        print(f"  labels: {min(len(ids), i + 1500)} of {len(ids)}", file=sys.stderr)
    import zstandard

    lines = []
    for r in rows:
        s_labels = labels.get(r["s"], {})
        o_labels = labels.get(r["o"], {}) if r["o"].startswith("Q") else {lang: r["o"] for lang in LANGS}
        both = {lang: [s_labels[lang], o_labels[lang]] for lang in LANGS if lang in s_labels and lang in o_labels}
        # A label that is only its id, or the same as the subject's, asks nothing.
        both = {lang: v for lang, v in both.items() if not v[0].startswith("Q") and v[0] != v[1]}
        if both:
            lines.append(json.dumps({"fact": r["fact"], "s": r["s"], "o": r["o"], "sitelinks": r["sitelinks"], "labels": both},
                                    ensure_ascii=False, sort_keys=True))
    raw = ("\n".join(lines) + "\n").encode()
    with open(out_path, "wb") as f:
        f.write(zstandard.ZstdCompressor(level=19).compress(raw))
    print(f"{len(lines)} facts, {len(raw) / 1e6:.1f} MB raw -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""CC-NEWS: reading WARC files, a page's publication time and main text, and choosing articles."""

import calendar
import gzip
import io
import json
import zlib

import pytest

from stet.newexam.fresh import ccnews
from stet.newexam.rng import Rng

T0 = calendar.timegm((2026, 11, 19, 10, 0, 0, 0, 0, 0))  # the window: 10:00 to 12:00 UTC
T1 = T0 + 2 * 3600
WARC = "CC-NEWS-20261119100000-00002.warc.gz"


def iso(t: int, zone: str = "Z") -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + zone


def record(kind: str, block: bytes, uri: str = "", date: int = T0, rid: str = "") -> bytes:
    head = [b"WARC/1.0", f"WARC-Type: {kind}".encode(), f"WARC-Date: {iso(date)}".encode(),
            f"WARC-Record-ID: <urn:uuid:{rid or kind}>".encode()]
    if uri:
        head.append(f"WARC-Target-URI: {uri}".encode())
    head += [b"Content-Type: application/http; msgtype=response", f"Content-Length: {len(block)}".encode()]
    return b"\r\n".join(head) + b"\r\n\r\n" + block + b"\r\n\r\n"


def response(body: str, status: int = 200, ctype: str = "text/html; charset=utf-8", charset: str = "utf-8") -> bytes:
    return (f"HTTP/1.1 {status} \r\ncontent-type: {ctype}\r\nX-Crawler-content-encoding: gzip\r\n\r\n").encode() + body.encode(charset)


def warc(records: list[bytes]) -> bytes:
    """A WARC file as Common Crawl writes them: one gzip member to a record."""
    return b"".join(gzip.compress(r) for r in records)


def prose(tag: str, n: int = 4) -> list[str]:
    return [f"{tag}: the council met on Thursday to discuss paragraph {i} of the plan, which sets out how the town will pay "
            f"for its new bridge, its schools and the roads between them over the next ten years." for i in range(n)]


def page(title: str, paragraphs: list[str], published: str | None = None, ld: dict | None = None, extra_head: str = "") -> str:
    head = f"<title>{title} | The Paper</title><meta property=\"og:title\" content=\"{title}\">{extra_head}"
    if published:
        head += f'<meta property="article:published_time" content="{published}">'
    if ld:
        head += f'<script type="application/ld+json">{json.dumps(ld)}</script>'
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f"""<!doctype html><html><head>{head}<style>p {{ color: red }}</style></head><body>
<header><a href="/">The Paper</a><nav><ul><li><a href="/news">News</a></li><li><a href="/sport">Sport</a></li></ul></nav></header>
<div class="cookie-banner">We use cookies to make this site work. Accept them all, or choose which ones you allow.</div>
<main><article><h1>{title}</h1><div class="byline">By A. Writer, 19 November 2026</div>
<div class="article-body">{body}
<div class="share-tools"><p>Share this story with your friends on every social network you use.</p></div>
<p><a href="/a">Another story</a> <a href="/b">And another one that is also only a link</a></p>
</div></article>
<aside><h3>Most read</h3><p>A sidebar paragraph that is long enough to count as prose, about something else altogether.</p></aside>
<section class="comments"><p>A reader's comment, long enough to look like a paragraph of the article to a careless reader.</p></section>
</main><footer><p>Copyright The Paper 2026. All rights reserved. Registered in England, company number 123456.</p></footer>
<script>var x = "<p>not text</p>";</script></body></html>"""


def test_members_give_each_record_with_where_it_is(tmp_path, monkeypatch):
    big = b"x" * 300_000
    recs = [record("warcinfo", b"software: test"), record("request", b"GET / HTTP/1.1\r\n\r\n", "https://a.test/1"),
            record("response", response("<p>" + "word " * 20000 + "</p>"), "https://a.test/1", rid="r1"), record("metadata", big)]
    data = warc(recs) + b"\x00" * 16  # some writers pad the end
    path = tmp_path / WARC
    path.write_bytes(data)
    monkeypatch.setattr(ccnews, "CHUNK", 4096)  # members cross the reads' boundaries
    monkeypatch.setattr(ccnews, "FEED", 1000)
    with open(path, "rb") as f:
        got = list(ccnews.members(f, limit=200_000))
    assert len(got) == 4
    for (offset, length, content), raw in zip(got, recs, strict=True):
        assert zlib.decompress(data[offset : offset + length], 31) == raw  # anyone can fetch the record by its offset
        assert content == (raw if len(raw) <= 200_000 else None)  # a record past the limit is passed over
    with open(path, "rb") as f:
        rs = list(ccnews.records(f, limit=200_000))
    assert [h["warc-type"] for _, _, h, _ in rs] == ["warcinfo", "request", "response"]
    h, block = rs[2][2], rs[2][3]
    assert h["warc-target-uri"] == "https://a.test/1" and h["warc-record-id"] == "<urn:uuid:r1>"
    assert block.startswith(b"HTTP/1.1 200") and block.endswith(b"</p>")  # Content-Length bounds it: the record's \r\n\r\n is not in it
    with open(path, "rb") as f:  # a file cut off in the middle of a record says so
        cut = io.BytesIO(f.read()[: got[2][0] + 100])
    with pytest.raises(EOFError):
        list(ccnews.members(cut))


def test_html_responses_only():
    assert ccnews.html_of(response("<p>hello</p>")) == "<p>hello</p>"
    assert ccnews.html_of(response("<p>gone</p>", status=404)) is None
    assert ccnews.html_of(response("{}", ctype="application/json")) is None
    assert ccnews.html_of(response("<p>Привет</p>", ctype="text/html; charset=windows-1251", charset="cp1251")) == "<p>Привет</p>"
    meta = '<meta charset="iso-8859-1"><p>café</p>'
    assert ccnews.html_of(response(meta, ctype="text/html", charset="latin-1")) == meta
    assert ccnews.html_of(b"not http at all") is None


def test_times_are_exact_only_with_a_zone():
    t = calendar.timegm((2026, 11, 19, 10, 23, 0, 0, 0, 0))
    for s in ("2026-11-19T10:23:00Z", "2026-11-19T13:23:00+03:00", "2026-11-19T13:23:00+0300", "2026-11-19T05:23:00.123-05:00",
              "2026-11-19 10:23:00 UTC", "Thu, 19 Nov 2026 10:23:00 +0000", "Thu, 19 Nov 2026 11:23:00 +0100", str(t), str(t * 1000)):
        assert ccnews.parse_time(s) == (t, True), s
    assert ccnews.parse_time("2026-11-19T10:23:00") == (t, False)  # local time somewhere
    assert ccnews.parse_time("2026-11-19")[1] is False and not (ccnews.parse_time("Thu, 19 Nov 2026") or (0, False))[1]
    for s in ("", "yesterday", "2026-13-45T99:00:00Z"):
        assert ccnews.parse_time(s) is None, s


def test_the_earliest_publication_time_a_page_gives():
    t = T0 + 600
    assert ccnews.published_of(page("A", [], iso(t))) == (t, "meta:article:published_time")
    ld = {"@context": "https://schema.org", "@graph": [{"@type": "WebSite", "datePublished": "2001-01-01T00:00:00Z"},
                                                       {"@type": "NewsArticle", "datePublished": iso(t + 60, "+00:00")}]}
    assert ccnews.published_of(page("A", [], ld=ld)) == (t + 60, "jsonld")  # the site's own date is not the article's
    # an article "updated" in the window whose JSON-LD remembers it is a week old: not new
    old = {"@type": "NewsArticle", "datePublished": iso(t - 7 * 86400)}
    assert ccnews.published_of(page("A", [], iso(t), ld=old))[0] == t - 7 * 86400
    assert ccnews.published_of(page("A", [], "2026-11-19")) == (None, "vague")
    assert ccnews.published_of(page("A", [], "2026-11-19T10:10:00")) == (None, "vague")
    assert ccnews.published_of(page("A", [])) == (None, "none")
    assert ccnews.published_of(f'<time pubdate datetime="{iso(t)}">today</time>') == (t, "time")
    assert ccnews.published_of(f'<time datetime="{iso(t)}">now</time>') == (None, "none")  # a clock, not a date
    assert ccnews.published_of(f'<meta name="date" content="{iso(t)}">') == (None, "none")  # some pages give the time they were served
    broken = f'<script type="application/ld+json">{{"@type": "NewsArticle", "datePublished": "{iso(t)}",}}</script>'
    assert ccnews.published_of(broken) == (t, "jsonld")


def test_the_article_without_the_page_around_it():
    paras = prose("Bridge")
    title, text, how = ccnews.extract(page("Town votes on bridge", paras))
    assert how == "html" and title == "Town votes on bridge | The Paper"
    assert text.split("\n") == paras
    html_ = page("Town votes on bridge", paras)
    assert ccnews.og_title(html_) == "Town votes on bridge"
    for junk in ("News", "Sport", "cookies", "By A. Writer", "Share this", "Another story", "sidebar", "comment", "Copyright", "not text",
                 "color"):
        assert junk not in text, junk


def test_subheadings_stay_and_the_page_furniture_at_the_end_goes():
    body = (f"<div class='story-body'><p>{prose('Lead', 1)[0]}</p><h2>What happens next</h2><p>{prose('Next', 1)[0]}</p>"
            f"<p>{prose('More', 1)[0]}</p><p>Follow us on WhatsApp for more news like this, every day.</p><p>More to read</p>"
            "<p>Read also:</p></div>")
    _, text, _ = ccnews.extract(f"<html><body><div class='wrap'>{body}</div></body></html>")
    lines = text.split("\n")
    assert lines[1] == "What happens next" and len(lines) == 4 and lines[-1].startswith("More:")


def test_a_page_its_scripts_write_is_read_from_what_it_hides_or_its_json_ld():
    paras = prose("Hidden")
    hidden = f"<html><body><div style='display:none'><article>{''.join(f'<p>{p}</p>' for p in paras)}</article></div></body></html>"
    assert ccnews.extract(hidden)[1:] == ("\n".join(paras), "hidden")
    body = "<p>" + "</p><p>".join(prose("Json")) + "</p>"
    shell = f'<html><head><script type="application/ld+json">{json.dumps({"@type": "NewsArticle", "articleBody": body})}</script></head>' \
            '<body><div id="app"></div></body></html>'
    assert ccnews.extract(shell)[1:] == ("\n".join(prose("Json")), "jsonld")
    # a head left open does not hide the body
    assert ccnews.extract(f"<html><head><title>T</title><body><p>{prose('Open', 1)[0]}</p></body></html>")[1] == prose("Open", 1)[0]


def test_sites_and_quotas():
    assert ccnews.site_of("https://news.bbc.co.uk/x") == "bbc.co.uk"
    assert ccnews.site_of("https://www.example.com/a") == "example.com"
    assert ccnews.site_of("https://g1.globo.com.br/a") == "globo.com.br"
    q = ccnews.quotas({"en": 100, "de": 100, "xx": 100}, 50, {"en": 40.0, "de": 10.0}, 0.05)
    assert sum(q.values()) == 50 and q["en"] > q["de"] > q["xx"] >= 1
    q = ccnews.quotas({"en": 3, "de": 100}, 50, {"en": 40.0, "de": 10.0}, 0.05)
    assert q == {"en": 3, "de": 47}  # what a language cannot use goes to the others
    assert ccnews.quotas({"en": 10}, 0, {"en": 1.0}, 0.05) == {"en": 0}


def test_which_files_a_window_reads():
    names = ["CC-NEWS-20261119064201-00000.warc.gz", "CC-NEWS-20261119083547-00001.warc.gz", "CC-NEWS-20261119101623-00002.warc.gz",
             "CC-NEWS-20261119113923-00003.warc.gz", "CC-NEWS-20261119130000-00004.warc.gz"]
    paths = [f"crawl-data/CC-NEWS/2026/11/{n}" for n in names]
    files, closed = ccnews.window_files(paths, T0, T1)  # 10:00 to 12:00
    assert files == paths[1:4] and closed
    files, closed = ccnews.window_files(paths[:4], T0, T1)  # 00003 may still end before noon
    assert files == paths[1:4] and not closed
    files, closed = ccnews.window_files(paths, T0, T1, after=1)
    assert files == paths[1:5] and closed
    assert ccnews.window_files(paths, T0, T1, after=2) == (paths[1:5], False)


def _articles(tmp_path, order=1):
    """A file holding: five articles from one site, a page fetched twice, a story two sites carry,
    one published before the window, one with no date, one dated after it was fetched, and a 404."""
    recs = [record("warcinfo", b"software: test")]

    def add(url, html_, fetched=T0 + 3600, status=200, rid=None):
        recs.append(record("request", b"GET / HTTP/1.1\r\n\r\n", url, fetched))
        recs.append(record("response", response(html_, status), url, fetched, rid=rid or f"{url}@{fetched}"))

    for i in range(5):
        add(f"https://www.big.test/{i}", page(f"Big {i}", prose(f"Big {i}"), iso(T0 + 60 * i)))
    again = page("Again", prose("Again"), iso(T0 + 100))
    add("https://small.test/again", again, T0 + 1000)
    add("https://small.test/again", again, T0 + 5000)
    wire = prose("Wire story")
    add("https://one.test/wire", page("Wire", wire, iso(T0 + 200)))
    add("https://two.test/wire", page("Wire, too", ["An introduction of our own, longer than a short line."] + wire, iso(T0 + 300)))
    add("https://old.test/a", page("Old", prose("Old"), iso(T0 - 60)))
    add("https://nodate.test/a", page("Undated", prose("Undated")))
    add("https://future.test/a", page("Future", prose("Future"), iso(T0 + 3600 + 3 * 3600)))
    add("https://gone.test/a", page("Gone", prose("Gone"), iso(T0 + 60)), status=404)
    path = tmp_path / str(order) / WARC
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(warc(recs[:1] + recs[1:][::order]))
    return path


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(ccnews, "_lang", lambda text: ("de" if text.startswith("Again") else "en", 0.99))


def test_a_file_gives_each_site_a_few_articles_and_a_story_once(tmp_path, no_model):
    path = _articles(tmp_path)
    stats = ccnews._Stats()
    sites = ccnews.scan(path, WARC, T0, T1, Rng(b"seed", "fresh/ccnews"), per_site=3, stats=stats)
    assert len(sites["big.test"]) == 3 and len(sites["small.test"]) == 1
    assert "old.test" not in sites and "nodate.test" not in sites and "future.test" not in sites and "gone.test" not in sites
    assert stats["records"] == 13 and stats["html"] == 12 and stats["none"] == 1 and stats["outside"] == 1 and stats["future"] == 1
    docs = ccnews.choose(sites, 100, stats=stats)
    assert stats["syndicated"] == 1 and sum(d.topic in ("one.test", "two.test") for d in docs) == 1
    assert len(docs) == 3 + 1 + 1
    d = next(d for d in docs if d.topic == "small.test")
    assert d.source == "ccnews" and d.url == "https://small.test/again" and d.title == "Again" and d.published == T0 + 100
    assert d.text == "\n".join(prose("Again")) and d.lang == "de" and "copyright stays with their publishers" in d.license
    assert d.extra["warc"] == WARC and d.extra["fetched"] in (T0 + 1000, T0 + 5000) and d.id.startswith("urn:uuid:")
    raw = path.read_bytes()
    rec = zlib.decompress(raw[d.extra["offset"] : d.extra["offset"] + d.extra["length"]], 31)
    assert d.id.encode() in rec  # the record can be found again from what the document keeps


def test_the_choice_follows_the_seed_not_the_order_of_the_file(tmp_path, no_model):
    files = {order: _articles(tmp_path, order) for order in (1, -1)}

    def run(order, seed):
        sites = ccnews.scan(files[order], WARC, T0, T1, Rng(seed, "fresh/ccnews"), per_site=2)
        return sorted((d.id, d.text) for d in ccnews.choose(sites, 100))

    a = run(1, b"a")
    assert a == run(1, b"a") == run(-1, b"a")  # again, and read backwards: the same articles
    for i in range(20):  # and some other seed takes others of the big site
        if run(1, str(i).encode()) != a:
            break
    else:
        raise AssertionError("every seed chose the same articles")


def test_harvest_reads_the_window_s_files(tmp_path, no_model, monkeypatch):
    path = _articles(tmp_path)
    listed = [f"crawl-data/CC-NEWS/2026/11/{WARC}", "crawl-data/CC-NEWS/2026/11/CC-NEWS-20261119130000-00003.warc.gz"]
    monkeypatch.setattr(ccnews, "listing", lambda y, m: listed if (y, m) == (2026, 11) else [])
    fetched = []

    def local(p, log=print):
        fetched.append(p)
        return path

    monkeypatch.setattr(ccnews, "_local", local)
    lines = []
    docs = ccnews.harvest(T0, T1, 4, Rng(b"seed", "fresh/ccnews"), log=lines.append, keep=True)
    assert fetched == listed[:1] and path.exists()
    assert len(docs) == 4 and {d.lang for d in docs} == {"en", "de"}  # each language gets a share
    assert "kept 4" in lines[-1] and not any("not all published" in x for x in lines)
    ccnews.harvest(T0, T1, 4, Rng(b"seed", "fresh/ccnews"), log=lines.append)
    assert not path.exists()  # read once, then gone from the cache

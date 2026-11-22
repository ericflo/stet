"""Crossref as a fresh source, against a stand-in for its API: no request leaves the machine."""

import time

import pytest

from stet.exam.fresh import crossref
from stet.exam.rng import Rng

START = 1_800_000_000  # 2027-01-15T08:00:00Z
END = START + 3600
PROSE = "and it goes on for a while in plain words, as people write when they report what they have found. " * 4
CC_BY = "https://creativecommons.org/licenses/by/4.0/"


def say(what: str) -> str:
    return f"<jats:p>{what} {PROSE.strip()}</jats:p>"


def date(t: int, parts: int = 3) -> dict:
    g = time.gmtime(t)
    return {"date-parts": [[g.tm_year, g.tm_mon, g.tm_mday][:parts]]}


def work(doi: str, abstract: str, at: int = START + 100, title: str = "A study of things", journal: str | None = None, issued: dict | None = None,
         kind: str = "journal-article", **kw) -> dict:
    prefix = doi.split("/", 1)[0]
    return {"DOI": doi, "prefix": prefix, "type": kind, "title": [title], "abstract": abstract, "publisher": f"Publisher {prefix}",
            "container-title": [journal if journal is not None else f"Journal of {doi}"], "created": {"timestamp": at * 1000},
            "issued": issued if issued is not None else date(START - 86_400), **kw}


def works() -> list[dict]:
    """An hour of registrations: the cases a harvest must tell apart."""
    blurb = "<jats:p>This volume gathers the papers read at the fourteenth meeting of the society, held last spring.</jats:p>"
    return [
        work("10.1000/good", "<jats:title>Abstract</jats:title><jats:sec><jats:title>Background</jats:title>" + say("We measure <jats:italic>p</jats:italic> &lt; 0.05,")
             + "</jats:sec><jats:sec><jats:title>Results</jats:title><jats:p>It &amp;amp; they hold in H<jats:sub>2</jats:sub>O.</jats:p></jats:sec>",
             title="A study of &lt;i&gt;E. coli&lt;/i&gt;\n  in water", journal="Journal of Things", language="en",
             license=[{"URL": "https://example.org/tdm", "content-version": "tdm", "delay-in-days": 0},
                      {"URL": CC_BY, "content-version": "vor", "delay-in-days": 0}]),
        work("10.1000/early", say("Registered just before the window,"), at=START - 1),
        work("10.1000/late", say("Registered as the window closed,"), at=END),
        work("10.1000/backfile", say("From an issue of long ago,"), issued=date(START - 400 * 86_400)),
        work("10.1000/lastmonth", say("Published three weeks ago,"), issued=date(START - 21 * 86_400)),
        work("10.1000/twomonths", say("Published two months ago,"), issued=date(START - 62 * 86_400)),
        work("10.1000/thismonth", say("Dated to the month,"), issued=date(START, 2)),
        work("10.1000/yearonly", say("Dated to the year,"), issued=date(START, 1)),
        work("10.1000/undated", say("Not dated at all,"), issued={"date-parts": [[None]]}),
        work("10.1000/onlineold", say("In print next month, online last year,"), issued=date(START + 30 * 86_400), **{"published-online": date(START - 300 * 86_400)}),
        work("10.1000/preprinted", say("Its preprint has been public for months,"), relation={"has-preprint": [{"id-type": "arxiv", "id": "2606.00001v2"}]}),
        work("10.1000/osf.io/abcde_v3", say("The third version of a preprint,"), kind="posted-content", journal=""),
        work("10.1000/osf.io/abcde_v1", say("The first version of a preprint,"), kind="posted-content", journal=""),
        work("10.1000/correction", say("A correction to an article,"), title="Correction: A study of things", **{"update-to": [{"type": "correction"}]}),
        work("10.1000/retraction", say("A notice of retraction,"), title="Retraction notice to: A study of things"),
        work("10.1000/dataset", say("A data set's description,"), kind="dataset"),
        work("10.1000/short", "<jats:p>No abstract available.</jats:p>"),
        work("10.1000/twin-a", say("Registered under two DOIs,")),
        work("10.1000/twin-b", say("Registered under two DOIs,")),
        *[work(f"10.2000/chapter{i}", blurb + (say(f"Chapter {i} has words of its own,") if i == 0 else ""), journal="Proceedings of the Society", kind="book-chapter")
          for i in range(3)],
        *[work(f"10.3000/issue{i}", say(f"Article {i} of one issue,"), journal="One Journal") for i in range(6)],
        *[work(f"10.4000/j{i}", say(f"Article {i} of a big publisher,"), journal=f"Big Journal {i}") for i in range(14)],
        work("10.5000/german", say("Ein deutscher Artikel, English abstract,"), language="de"),
        work("10.5000/unsure", say("Short of words, unsure,"), language="tr-TR"),
        work("10.5000/wrongtag", say("Tagged with no language at all,"), language="Nigeria"),
        work("10.5000/embargoed", say("Open in a year,"), license=[{"URL": CC_BY, "content-version": "vor", "delay-in-days": 365}]),
    ]


class FakeAPI:
    """api.crossref.org/works: a filter on the time of registration, newest first, a cursor."""

    def __init__(self, items, down_at=None):
        self.items, self.down_at, self.calls = sorted(items, key=lambda w: -w["created"]["timestamp"]), down_at, []

    def __call__(self, url, params, **kw):
        assert url == crossref.API and "mailto" not in params and params["sort"] == "created" and params["order"] == "desc"
        self.calls.append(params)
        if self.down_at is not None and len(self.calls) >= self.down_at:
            raise OSError("no answer")
        want = dict(part.split(":", 1) for part in params["filter"].split(","))
        assert want["has-abstract"] == "true"
        lo, hi = want["from-created-date"], want["until-created-date"]
        stamp = "%Y-%m-%dT%H:%M:%S"
        inside = [w for w in self.items if lo <= time.strftime(stamp, time.gmtime(w["created"]["timestamp"] // 1000)) <= hi]
        at = 0 if params["cursor"] == "*" else int(params["cursor"])
        page = inside[at : at + params["rows"]]
        return {"status": "ok", "message": {"total-results": len(inside), "items": page, "next-cursor": str(at + len(page))}}


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI(works())
    monkeypatch.setattr(crossref, "get_json", fake)
    monkeypatch.setattr(crossref, "GAP", 0)
    monkeypatch.setattr(crossref, "lid", lambda text: ("en", 0.3 if "unsure" in text else 0.97))
    return fake


def run(cap=1000, seed=b"seed", log=lambda *a: None):
    return crossref.harvest(START, END, cap, Rng(seed, "fresh/crossref"), log=log)


def test_only_works_registered_in_the_window_and_newly_published_are_kept(api):
    docs = run()
    ids = {d.id for d in docs}
    assert all(START <= d.published < END and d.source == "crossref" and not d.question for d in docs) and len(ids) == len(docs)
    assert {"10.1000/good", "10.1000/lastmonth", "10.1000/thismonth", "10.1000/osf.io/abcde_v1", "10.2000/chapter0"} <= ids
    for gone in ("early", "late", "backfile", "twomonths", "yearonly", "undated", "onlineold", "preprinted", "osf.io/abcde_v3", "correction", "retraction",
                 "dataset", "short"):
        assert f"10.1000/{gone}" not in ids
    assert len(ids & {"10.1000/twin-a", "10.1000/twin-b"}) == 1  # the same abstract under two DOIs is kept once
    assert not ids & {"10.2000/chapter1", "10.2000/chapter2"}  # a book's blurb on each chapter is no one chapter's abstract
    f = api.calls[0]["filter"]
    assert f == "from-created-date:2027-01-15T08:00:00,until-created-date:2027-01-15T08:59:59,has-abstract:true" and len(api.calls) == 1


def test_an_abstract_is_read_as_plain_text(api):
    docs = {d.id: d for d in run()}
    doc = docs["10.1000/good"]
    assert doc.title == "A study of E. coli in water" and doc.url == "https://api.crossref.org/works/10.1000/good"
    assert doc.text == f"Background\n\nWe measure p < 0.05, {PROSE.strip()}\n\nResults\n\nIt & they hold in H2O."
    assert doc.extra == {"doi": "https://doi.org/10.1000/good", "type": "journal-article", "journal": "Journal of Things",
                         "publisher": "Publisher 10.1000", "lang_tag": "en", "lid": "en", "lid_p": 0.97}
    assert docs["10.2000/chapter0"].text.startswith("Chapter 0 has words of its own,")  # without the paragraph every chapter shares
    raw = ('<jats:p>ABSTRACT For <jats:inline-formula><jats:alternatives><jats:tex-math>$x^2$</jats:tex-math><mml:math><mml:msup><mml:mi>x</mml:mi><mml:mn>2</mml:mn>'
           "</mml:msup></mml:math></jats:alternatives></jats:inline-formula> and <mml:math><mml:mi>y</mml:mi></mml:math>,\n   &lt;i&gt;so&lt;/i&gt;.&lt;br /&gt;"
           "Next.</jats:p><p>Plain   tags\n too.</p>")
    assert crossref.blocks(raw) == ["For $x^2$ and y, so.", "Next.", "Plain tags too."]
    assert crossref.blocks("<jats:p>Abstract concepts are hard.</jats:p>") == ["Abstract concepts are hard."]
    assert crossref.blocks("<jats:title>A<jats:sc>bstract</jats:sc></jats:title><jats:p>Resumen: El primero.</jats:p>") == ["El primero."]


def test_the_licence_is_the_records_or_none_is_claimed(api):
    docs = {d.id: d for d in run()}
    assert docs["10.1000/good"].license == f"{CC_BY} (the work's licence, as registered with Crossref); abstract via Crossref's public API"
    assert docs["10.1000/lastmonth"].license == crossref.NO_LICENSE and "no licence" in crossref.NO_LICENSE
    assert docs["10.5000/embargoed"].license == crossref.NO_LICENSE  # a licence that starts in a year is not in force


def test_the_language_is_read_and_the_record_only_breaks_doubt(api):
    docs = {d.id: d for d in run()}
    assert docs["10.5000/german"].lang == "en" and docs["10.5000/german"].extra["lang_tag"] == "de"  # the article's language, not the abstract's
    assert docs["10.5000/unsure"].lang == "tr" and docs["10.5000/unsure"].extra["lid"] == "en"
    assert docs["10.5000/wrongtag"].lang == "en" and docs["10.5000/wrongtag"].extra["lang_tag"] == ""


def test_a_journal_and_a_publisher_give_at_most_a_few(api):
    docs = run()
    assert sum(d.extra["journal"] == "One Journal" for d in docs) == crossref.PER_JOURNAL
    assert sum(d.id.startswith("10.4000/") for d in docs) == crossref.PER_PREFIX  # fourteen journals, one publisher
    assert sum(d.id.startswith("10.1000/osf.io/") for d in docs) == 1 and all(d.topic == "" for d in docs)


def test_a_long_hour_is_read_a_page_at_a_time(api, monkeypatch):
    whole = [d.id for d in run()]
    api.calls.clear()
    monkeypatch.setattr(crossref, "ROWS", 20)
    assert [d.id for d in run()] == whole
    assert [c["cursor"] for c in api.calls] == ["*", "20", "40"] and all(c["rows"] == 20 for c in api.calls)  # 48 in the window: two pages and a short one
    monkeypatch.setattr(crossref, "MAX_PAGES", 2)
    said = []
    assert 0 < len(run(log=said.append)) < len(whole) and "hours with more than 40 works" in said[-1]


def test_an_api_that_fails_gives_nothing(monkeypatch):
    monkeypatch.setattr(crossref, "GAP", 0)
    monkeypatch.setattr(crossref, "ROWS", 20)
    monkeypatch.setattr(crossref, "get_json", FakeAPI(works(), down_at=2))
    said = []
    assert run(log=said.append) == [] and "OSError" in said[0]


def test_the_choice_is_seeded(api):
    a = run(cap=8)
    assert len(a) == 8 and [d.id for d in run(cap=8)] == [d.id for d in a]
    assert {d.id for d in run(cap=8, seed=b"another")} != {d.id for d in a}


def test_dates_are_read_as_the_record_gives_them():
    assert crossref.published({"issued": {"date-parts": [[2027, 1, 14]]}}) == START - 115_200  # the day, at midnight
    assert crossref.published({"issued": {"date-parts": [[2026, 12]]}}) == crossref.published({"issued": {"date-parts": [[2026, 12, 31]]}})
    assert crossref.published({"issued": {"date-parts": [[2027]]}, "published-print": {"date-parts": [[None]]}}) is None
    assert crossref.published({"issued": {"date-parts": [[2027, 2, 1]]}, "posted": {"date-parts": [[2026, 3, 9]]}}) == crossref.published(
        {"posted": {"date-parts": [[2026, 3, 9]]}})
    assert crossref.created({"created": {"timestamp": 1_800_000_123_456}}) == 1_800_000_123 and crossref.created({}) is None

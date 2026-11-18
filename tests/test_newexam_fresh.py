"""Fresh text: parsing what the sources send, turning documents into items, and choosing them."""

import collections

from stet.newexam.fresh import Doc, html_text, read_snapshot, write_snapshot
from stet.newexam.fresh.select import items_of, select
from stet.newexam.fresh.wikipedia import added_lines, strip_wikitext

DIFF = """
<tr><td colspan="2" class="diff-lineno">Line 5:</td><td colspan="2" class="diff-lineno">Line 5:</td></tr>
<tr><td class="diff-marker"></td><td class="diff-context diff-side-deleted"><div>Old line kept.</div></td>
    <td class="diff-marker"></td><td class="diff-context diff-side-added"><div>Old line kept.</div></td></tr>
<tr><td colspan="2" class="diff-empty diff-side-deleted"></td><td class="diff-marker" data-marker="+"></td>
    <td class="diff-addedline diff-side-added"><div>A whole new paragraph about the river.</div></td></tr>
<tr><td class="diff-marker" data-marker="−"></td><td class="diff-deletedline diff-side-deleted"><div>Changed <del>here</del>.</div></td>
    <td class="diff-marker" data-marker="+"></td><td class="diff-addedline diff-side-added"><div>Changed <ins>there</ins>.</div></td></tr>
"""


def test_only_lines_added_whole_count_as_new():
    assert added_lines(DIFF) == ["A whole new paragraph about the river."]


def test_wikitext_becomes_prose():
    raw = ("'''Ada''' was a [[mathematician|mathematical]] writer<ref>{{cite book |title=X}}</ref>.{{citation needed}}\n"
           "== Life ==\n[[File:Ada.jpg|thumb|A picture]]\n* a list item\nShe worked with [[Charles Babbage]] "
           "on the [https://example.org Analytical Engine].\n}}\n<!-- a note -->")
    out = strip_wikitext(raw)
    assert "Ada was a mathematical writer." in out and "She worked with Charles Babbage on the Analytical Engine." in out
    for junk in ("[[", "{{", "<ref", "File:", "a list item", "'''", "<!--", "}}"):
        assert junk not in out


def test_html_text():
    assert html_text("<p>One <b>two</b></p><p>Three</p>") == "One two\nThree"


def _doc(i, lang="en", source="wikipedia", topic="Q5", question=""):
    text = f"Document number {i} has a body of plain text that goes on for a while. " * 8
    return Doc(source=source, id=f"{lang}:{i}", url=f"https://example.org/{i}", lang=lang, license="CC BY-SA 4.0",
               published=1_800_000_000 + i, title=f"Title {i}", text=text, topic=topic, question=question)


def test_documents_become_items_with_corners():
    wiki = items_of(_doc(1))
    assert [it.kind for it in wiki] == ["passage", "answer"]
    assert wiki[0].corner == "fresh/wikipedia/en/Q5/article" and wiki[1].ctx == "Title 1\n\n"
    ans = items_of(_doc(2, source="stackexchange", topic="math/algebra", question="How do I solve x + 1 = 2?"))
    assert len(ans) == 1 and ans[0].kind == "answer" and ans[0].ctx.startswith("Question: Title 2")
    added = items_of(_doc(3, source="wikipedia-added", lang="simple"))
    assert added[0].lang == "en" and added[0].corner.endswith("/added")


def test_selection_caps_corners_shares_languages_and_is_repeatable():
    docs = [_doc(i, topic=f"Q{i % 7}") for i in range(200)] + [_doc(1000 + i, lang="de", topic=f"Q{i}") for i in range(50)]
    docs += [_doc(2000 + i, lang="kaa", topic="Q5") for i in range(40)]
    a = select(docs, b"seed", 60, per_corner=2)
    assert select(docs, b"seed", 60, per_corner=2) == a != select(docs, b"other", 60, per_corner=2)
    assert len(a) <= 60
    corners = collections.Counter(it.corner for it in a)
    assert max(corners.values()) <= 2
    langs = collections.Counter(it.lang for it in a)
    # English has 14 corners (7 topics x 2 kinds), so at most 28 items; a bulk-writing language gets its small share
    assert langs["en"] == 28 and langs["kaa"] <= 4 and langs["de"] > 0


def test_snapshots_round_trip_with_a_stable_hash(tmp_path):
    docs = [_doc(2), _doc(1)]
    sha = write_snapshot(docs, tmp_path / "a.jsonl", {"window": [1, 2]})
    assert write_snapshot(list(reversed(docs)), tmp_path / "b.jsonl", {"window": [1, 2]}) == sha
    meta, back = read_snapshot(tmp_path / "a.jsonl")
    assert meta == {"window": [1, 2]} and [d.id for d in back] == ["en:1", "en:2"]

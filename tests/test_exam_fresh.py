"""Fresh text: parsing what the sources send, turning documents into items, and choosing them."""

import collections

from stet.exam.aggregate import kind_of
from stet.exam.fresh import Doc, html_text, read_snapshot, write_snapshot
from stet.exam.fresh.select import items_of, select
from stet.exam.fresh.wikipedia import added_lines, strip_wikitext

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
    # what the first shadow runs found left over: captions with links in them, entities, a citation
    # its editor broke, a table cut in half by a diff, files under other languages' names
    left = strip_wikitext("[[File:Teddy Hill.jpg|thumb|At [[Minton's Playhouse]], 1947; by [[William P. Gottlieb]]]]\n"
                          "Theodore Hill (1909 &ndash; 1978) was a band leader.\n"
                          'He was ranked 48th."{{cite web }url= https://example.org |title=Unsung |first1=Dean}}\n'
                          '{| class="wikitable"\n[[Файл:Map.png|мини|Карта]]Population.\n[[Kategorie:Musiker]]')
    assert left == "Theodore Hill (1909 – 1978) was a band leader.\n\nPopulation.", left
    reply = strip_wikitext(":: I agree with this.&nbsp;—[[User:Ann|Ann]] ([[User talk:Ann|talk]]) 12:00, 18 November 2026 (UTC)")
    assert reply == "I agree with this.\xa0—Ann (talk) 12:00, 18 November 2026 (UTC)"


def test_html_text():
    assert html_text("<p>One <b>two</b></p><p>Three</p>") == "One two\nThree"


def _doc(i, lang="en", source="wikipedia", topic="Q5", question="", title=None):
    text = f"Document number {i} has a body of plain text that goes on for a while. " * 8
    return Doc(source=source, id=f"{lang}:{i}", url=f"https://example.org/{i}", lang=lang, license="CC BY-SA 4.0",
               published=1_800_000_000 + i, title=title or f"Title {i}", text=text, topic=topic, question=question)


def test_documents_become_items_with_corners():
    wiki = items_of(_doc(1))
    assert [it.kind for it in wiki] == ["passage", "answer"]
    assert wiki[0].corner == "fresh/wikipedia/en/Q5/article" and wiki[1].ctx == "Title 1\n\n"
    ans = items_of(_doc(2, source="lemmy", topic="rust", question="How do I solve x + 1 = 2?"))
    assert len(ans) == 1 and ans[0].kind == "answer" and ans[0].ctx.startswith("Title 2\n\nHow do I solve")
    added = items_of(_doc(3, source="wikipedia-added", lang="simple"))
    assert added[0].lang == "en" and added[0].corner.endswith("/added")
    talk = items_of(_doc(4, source="wikipedia-talk", topic="talk", title="Talk:Teddy Hill"))
    assert talk[0].fmt == "dialogue" and talk[0].corner.startswith("fresh/wikipedia-talk/en/talk/") and talk[0].text.startswith("Talk:Teddy Hill\n")


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


def test_other_sources_become_replies_or_writing():
    reply = items_of(_doc(5, source="hackernews", topic="", question="Ask HN: what do you read?\n\n" + "a " * 2000, title="Ask HN"))
    assert reply[0].kind == "answer" and reply[0].fmt == "dialogue" and reply[0].ctx.endswith("a\n\n") and len(reply[0].ctx) < 1600
    assert reply[0].corner.startswith("fresh/hackernews/en/") and reply[0].corner.endswith("/reply")
    import dataclasses

    thread = items_of(_doc(7, source="lemmy", topic="rust", question="Ask HN\n\nWhy?", title="Ask HN"))
    assert thread[0].ctx == "Ask HN\n\nWhy?\n\n"  # the title once, as the thread gives it
    post = items_of(dataclasses.replace(_doc(6, source="bluesky", topic=""), title=""))
    assert post[0].kind == "passage" and post[0].text.startswith("Document number 6") and post[0].corner.endswith("/text")


def test_each_kind_of_writing_gets_its_part_and_sources_take_turns():
    docs = [_doc(i, topic=f"Q{i}") for i in range(300)]  # reference: plenty
    docs += [_doc(1000 + i, source="github", topic=f"r{i}", title=f"Issue {i}") for i in range(300)]  # technical: plenty
    docs += [_doc(2000 + i, source="hackernews", topic="", question="Why?", title=f"Story {i}") for i in range(300)]
    docs += [_doc(3000 + i, source="lemmy", topic=f"c{i}", question="Why?", title=f"Post {i}") for i in range(10)]
    a = select(docs, b"seed", 90, per_corner=2)
    by_kind = collections.Counter(kind_of(it) for it in a)
    assert by_kind == {"reference": 30, "technical": 30, "discussion": 30}
    # in discussion, Lemmy's ten replies all come in before Hacker News fills the rest
    disc = collections.Counter(it.source.split(":")[0] for it in a if kind_of(it) == "discussion")
    assert disc["lemmy"] == 10 and disc["hackernews"] == 20
    # a kind that runs short gives what it leaves to the others
    b = select(docs[:300] + docs[-10:], b"seed", 60, per_corner=2)
    assert collections.Counter(kind_of(it) for it in b) == {"reference": 50, "discussion": 10}


def test_snapshots_round_trip_with_a_stable_hash(tmp_path):
    docs = [_doc(2), _doc(1)]
    sha = write_snapshot(docs, tmp_path / "a.jsonl", {"window": [1, 2]})
    assert write_snapshot(list(reversed(docs)), tmp_path / "b.jsonl", {"window": [1, 2]}) == sha
    meta, back = read_snapshot(tmp_path / "a.jsonl")
    assert meta == {"window": [1, 2]} and [d.id for d in back] == ["en:1", "en:2"]


def test_items_the_corpus_already_holds_are_found(tmp_path):
    from conftest import make_corpus

    from stet.exam.decontam import seen
    from stet.exam.items import Item
    from stet.spec import load_season

    quoted = "The river authority approved a new flood barrier for the lower town after three wet winters in a row."
    corpus = make_corpus(tmp_path / "c", ["Menu | Home\nNews today. " + quoted + " More text follows here.", "Nothing to see."])
    copied = Item(kind="passage", corner="fresh/x/en/-/added", fmt="prose", lang="en", source="t", text="It was reported that. " + quoted)
    new = Item(kind="passage", corner="fresh/x/en/-/added", fmt="prose", lang="en", source="t",
               text="An entirely different passage that nobody has published before this very moment, about otters.")
    answer = Item(kind="answer", corner="fresh/y/en/-/answer", fmt="dialogue", lang="en", source="t",
                  ctx=quoted + " Why?", cont=" Nothing in the corpus says this, not even close to it, as far as anyone knows.")
    assert seen([copied, new, answer], corpus, load_season("s1")) == {0}  # an answer is checked on its answer, not its question

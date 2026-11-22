import dataclasses
import json
import os

import pytest

from stet.script import ScriptError, build_doc, file_sha256, materialize, open_script, write_script
from stet.spec import load_season

SEASON = load_season("s2")
EOT = SEASON.eot
SCALE = dataclasses.replace(SEASON.scale("nano"), train_tokens=512, ctx=64, batch_size=4)
# What the two refusals say, after "line N: ".
NO_TEXT = "this line holds text; a script is copies of the crawl and holds no text of its own"
TOO_SHORT = "is shorter than the minimum of 64 bytes; only whole lines of one page in the page's own order, or a short document kept whole on its own, may be shorter"


def run(corpus, enc, lines, scale=SCALE, season=SEASON):
    path = corpus.path / "script.jsonl"
    write_script(path, lines)
    return materialize(corpus, path, season, scale, enc)


def refused(corpus, line, line_no: int = 1) -> str:
    """What the grader says of a line it refuses."""
    with pytest.raises(ScriptError) as e:
        build_doc(corpus, line_no, line, 64)
    return str(e.value)


def test_whole_doc_forms_are_equivalent(corpus):
    a = build_doc(corpus, 1, 0, 64)
    b = build_doc(corpus, 1, {"d": 0}, 64)
    c = build_doc(corpus, 1, {"s": [[0, 0, corpus.size(0)]]}, 64)
    assert a.data == b.data == c.data == corpus.raw(0)
    assert a.docs == [0] and not a.spliced


def test_tokens_are_exactly_the_budget_and_cycle(corpus, enc):
    tokens, stats = run(corpus, enc, [0])
    assert tokens.shape == (SCALE.batch_size, SCALE.steps * SCALE.ctx + 1)
    assert (tokens[:, 0] == EOT).all()  # every lane starts a document
    assert stats.passes > 1 and not stats.truncated
    assert len(stats.tokens_sha256) == 64


def test_the_same_script_gives_the_same_tokens(corpus, enc):
    a, sa = run(corpus, enc, [0, 3, 2])
    b, sb = run(corpus, enc, [0, 3, 2])
    assert (a == b).all() and sa.tokens_sha256 == sb.tokens_sha256 and sa.script_sha256 == sb.script_sha256
    _, sc = run(corpus, enc, [3, 0, 2])
    assert sc.tokens_sha256 != sa.tokens_sha256


def test_long_script_is_truncated(corpus, enc):
    tokens, stats = run(corpus, enc, [0, 3] * 20)
    assert tokens.shape == (SCALE.batch_size, SCALE.steps * SCALE.ctx + 1)
    assert stats.truncated and stats.passes == 1.0
    assert stats.script_lines == 40 and stats.out_docs < 40


def test_order_is_the_editors(corpus, enc):
    a, _ = run(corpus, enc, [0, 3])
    b, _ = run(corpus, enc, [3, 0])
    assert (a != b).any()


def test_documents_are_dealt_across_lanes_in_order(corpus, enc):
    # Four different documents, four lanes: each lane starts with a different one, in script
    # order, so a batch is as varied as the script allows.
    scale = dataclasses.replace(SCALE, train_tokens=4 * 64 * 2)
    tokens, _ = run(corpus, enc, [0, 3, 2, 0, 3, 2, 0, 3], scale=scale)
    firsts = [enc.encode_ordinary_batch([corpus.text(d)])[0][0] for d in (0, 3, 2, 0)]
    assert [int(t) for t in tokens[:, 1]] == firsts


# --- a script is copies of the crawl, and no more -----------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        {"t": "A sentence of the curator's own."},
        {"s": [[0, 0, 200], "NEW TEXT"]},
        {"s": ["NEW TEXT", [0, 0, 200]]},
        {"s": ["All text."]},
        {"s": [[0, 0, 200], ""]},  # an empty string is still not a copy
        {"t": ""},
        {"t": 5},  # whatever stands under it
        {"d": 0, "t": "A note beside a whole page."},
        {"s": [[0, 0, 200]], "t": "A note beside copies."},
    ],
)
def test_a_line_with_text_is_refused_and_the_error_names_the_line(corpus, enc, line):
    assert refused(corpus, line, 7) == f"line 7: {NO_TEXT}"
    # As a script: the line is the third, and nothing before it makes up for it.
    with pytest.raises(ScriptError) as e:
        run(corpus, enc, [0, 3, line, 0])
    assert str(e.value) == f"line 3: {NO_TEXT}"


def test_a_copy_under_the_minimum_is_refused_and_the_error_gives_its_length(corpus, enc):
    assert refused(corpus, {"s": [[0, 0, 10]]}, 4) == f"line 4: a copy of 10 bytes (document 0, bytes 0 to 10) {TOO_SHORT}"
    assert refused(corpus, {"d": 0, "k": [[0, 10]]}, 4) == f"line 4: a copy of 10 bytes (document 0, bytes 0 to 10) {TOO_SHORT}"
    # One byte under is under, and the minimum itself is a copy.
    assert refused(corpus, {"d": 0, "k": [[100, 163]]}) == f"line 1: a copy of 63 bytes (document 0, bytes 100 to 163) {TOO_SHORT}"
    assert build_doc(corpus, 1, {"d": 0, "k": [[100, 164]]}, 64).data == corpus.raw(0)[100:164]
    # A long copy does not pay for a short one beside it. The first short piece is the one named.
    assert refused(corpus, {"s": [[0, 0, 500], [3, 40, 60], [2, 0, 5]]}).startswith("line 1: a copy of 20 bytes (document 3, bytes 40 to 60) is shorter")
    assert refused(corpus, {"d": 0, "k": [[0, 500], [600, 610], [700, 1200]]}).startswith("line 1: a copy of 10 bytes (document 0, bytes 600 to 610) is shorter")
    # The length is the copy's as it is read: a start inside a character is moved to the next one.
    raw = corpus.raw(2)
    cut = raw.index("í".encode()) + 1
    end = next(e for e in range(cut + 64, cut + 60, -1) if (raw[e] & 0xC0) != 0x80)
    assert refused(corpus, {"d": 2, "k": [[cut, end]]}).startswith(f"line 1: a copy of {end - cut - 1} bytes (document 2, bytes {cut + 1} to {end}) is shorter")
    assert len(build_doc(corpus, 1, {"d": 2, "k": [[cut - 1, end + 2]]}, 64).data) >= 64  # from the start of the character it is long enough
    # A range with nothing in it is not a copy of anything, and is passed over as it always was.
    assert build_doc(corpus, 1, {"d": 0, "k": [[5, 5], [100, 200], [300, 300]]}, 64).data == corpus.raw(0)[100:200]
    assert build_doc(corpus, 1, {"d": 0, "k": [[5, 5]]}, 64).data == b""
    # As a script, the season's minimum is the one applied, and the line is named.
    assert SEASON.min_copy_bytes == 64
    with pytest.raises(ScriptError) as e:
        run(corpus, enc, [0, {"d": 0, "k": [[0, 63]]}, 3])
    assert str(e.value) == f"line 2: a copy of 63 bytes (document 0, bytes 0 to 63) {TOO_SHORT}"
    _, stats = run(corpus, enc, [0, {"d": 0, "k": [[0, 64]]}, 3])
    assert stats.out_docs == 3


def test_a_short_document_kept_whole_is_a_copy_but_only_on_its_own(corpus, enc):
    for whole in (1, {"d": 1}, {"s": [[1, 0, 5]]}, {"d": 1, "k": [[0, 5]]}, {"d": 1, "k": [[0, 2], [2, 5]]}):  # the 5-byte document
        b = build_doc(corpus, 1, whole, 64)
        assert b.data == b"short" and b.docs == [1] and not b.spliced
    # Tiny whole documents cannot be strung together to spell out new text, nor put beside a long copy.
    for strung in ({"s": [[1, 0, 5], [1, 0, 5]]}, {"s": [[1, 0, 5], [0, 0, 200]]}, {"s": [[0, 0, 200], [1, 0, 5]]}):
        assert refused(corpus, strung) == f"line 1: a copy of 5 bytes (document 1, bytes 0 to 5) {TOO_SHORT}"
    # And a part of a short document is not the document.
    assert refused(corpus, {"d": 1, "k": [[0, 4]]}) == f"line 1: a copy of 4 bytes (document 1, bytes 0 to 4) {TOO_SHORT}"
    _, stats = run(corpus, enc, [0, 1, 3])
    assert stats.out_docs == 3 and stats.src_docs == 3


def test_whole_lines_taken_out_of_a_page_are_copies_however_short(tmp_path, enc):
    from conftest import make_corpus

    page = "Home | About | Contact\nA first line of the article.\nBuy now!\nAnother line.\nShare this\nThe last line"
    c = make_corpus(tmp_path / "lines", [page, "x\ny\nz\n", "A long page of writing, to fill the budget with. " * 40])
    at = [0]
    for line in page.split("\n"):
        at.append(at[-1] + len(line) + 1)
    at[-1] -= 1  # the last line has no line break
    keep = lambda *ks: {"d": 0, "k": [[at[k], at[k + 1]] for k in ks]}  # noqa: E731
    # the article's lines kept, the furniture between them taken out: all copies, though each is under 64 bytes
    b = build_doc(c, 1, keep(1, 3, 5), 64)
    assert b.data == b"A first line of the article.\nAnother line.\nThe last line" and b.docs == [0, 0, 0] and b.spliced
    assert build_doc(c, 1, keep(3), 64).data == b"Another line.\n"  # one line of a page is the page with the others taken out
    assert build_doc(c, 1, {"s": [[1, 0, 2], [1, 4, 6]]}, 64).data == b"x\nz\n"  # written as copies of one page, the same
    # out of the page's order, the same lines are short copies: lines cannot be moved
    assert refused(c, keep(3, 1)) == f"line 1: a copy of 14 bytes (document 0, bytes {at[3]} to {at[4]}) {TOO_SHORT}"
    # a line cut short, or begun in its middle, is not a line
    assert refused(c, {"d": 0, "k": [[at[1], at[2] - 1], [at[3], at[4]]]}).startswith("line 1: a copy of 28 bytes")
    assert refused(c, {"d": 0, "k": [[at[1] + 2, at[2]], [at[3], at[4]]]}).startswith("line 1: a copy of 27 bytes")
    # lines of two pages are not one page with lines taken out
    assert refused(c, {"s": [[0, at[1], at[2]], [1, 0, 2]]}).startswith("line 1: a copy of 29 bytes")
    # nor is a line said twice
    assert refused(c, keep(1, 1)).startswith("line 1: a copy of 29 bytes")
    # and text beside a page's lines is text
    assert refused(c, {"s": [[0, at[1], at[2]], "and"]}) == f"line 1: {NO_TEXT}"
    # As a script: every line of it trained on, and every byte of it the pages'.
    _, stats = run(c, enc, [keep(1, 3, 5), {"d": 1, "k": [[0, 2], [4, 6]]}, 2])
    assert stats.out_docs == 3 and stats.src_docs == 3 and stats.out_bytes == len(b.data) + 4 + c.size(2)
    assert stats.to_dict(c)["kept_byte_frac"] == round(stats.out_bytes / c.total_bytes, 6)


def test_copies_of_two_pages_joined_are_one_document(corpus, enc):
    b = build_doc(corpus, 1, {"s": [[0, 0, 64], [3, 36, 100]]}, 64)  # each the minimum: 64 bytes
    assert b.data == corpus.raw(0)[:64] + corpus.raw(3)[36:100] and b.docs == [0, 3] and b.spliced
    _, stats = run(corpus, enc, [{"s": [[0, 0, 450], [3, 36, 900]]}, {"s": [[3, 100, 164], [0, 90, 154], [3, 100, 164]]}])
    assert stats.out_docs == 2 and stats.src_docs == 2 and stats.out_bytes == 450 + 864 + 3 * 64


def test_adjacent_ranges_merge_so_line_cuts_are_free(corpus):
    # three contiguous 30-byte ranges are one 90-byte copy, not three short ones
    b = build_doc(corpus, 1, {"d": 0, "k": [[0, 30], [30, 60], [60, 90]]}, 64)
    assert b.data == corpus.raw(0)[:90] and b.docs == [0] and not b.spliced
    # with a gap between them they are three copies, each too short
    assert refused(corpus, {"d": 0, "k": [[0, 30], [31, 61], [62, 92]]}).startswith("line 1: a copy of 30 bytes (document 0, bytes 0 to 30) is shorter")


def test_offsets_snap_to_character_boundaries(corpus):
    raw = corpus.raw(2)
    cut = raw.index("í".encode()) + 1  # inside a two-byte character
    build_doc(corpus, 1, {"d": 2, "k": [[cut, 200]]}, 64).data.decode("utf-8")  # must not raise


def test_a_script_reports_its_copies_and_nothing_else(corpus, enc):
    # There is no text of its own to report: what a script trained on is what it copied.
    _, stats = run(corpus, enc, [0, 3])
    d = stats.to_dict(corpus)
    assert set(d) == {"out_docs", "out_bytes", "src_docs", "script_lines", "tokens_one_pass", "train_tokens", "passes", "truncated",
                      "script_sha256", "tokens_sha256", "preview", "corpus_docs", "corpus_bytes", "kept_doc_frac", "kept_byte_frac"}  # fmt: skip
    assert d["out_bytes"] == corpus.size(0) + corpus.size(3) and d["kept_byte_frac"] == round(d["out_bytes"] / corpus.total_bytes, 6)


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        '{"d": 99}',
        '{"d": 0, "k": [[10, 5]]}',
        '{"d": 0, "k": [[0, 999999]]}',
        '{"s": [[0, 0]]}',
        '{"x": 1}',
        "true",
        '{"d": -1}',
        '{"t": 5}',
        '{"t": "text"}',
        '{"s": [[0, 0, 900], "text"]}',
        '{"s": "text"}',
        '{"d": 0, "k": [[0, 10]]}',  # a copy too short to be one
        '{"d": 0, "k": [[0, ' + "9" * 5000 + "]]}",  # an integer too long to parse
        "[" * 100000 + "]" * 100000,  # nesting deep enough to exhaust the parser
        '{"d": ' + "1" * 400 + "}",
        b"\xff\xfe{}".decode("latin-1"),
    ],
)
def test_malformed_scripts_are_rejected(corpus, enc, bad):
    path = corpus.path / "script.jsonl"
    path.write_text(bad + "\n", encoding="latin-1")
    with pytest.raises(ScriptError):
        materialize(corpus, path, SEASON, SCALE, enc)


def test_a_line_of_more_pieces_than_any_document_needs_is_refused(corpus, enc):
    path = corpus.path / "script.jsonl"
    path.write_text('{"s": [' + ",".join(["[0,0,64]"] * 30000) + "]}\n")
    with pytest.raises(ScriptError) as e:
        materialize(corpus, path, SEASON, SCALE, enc)
    assert str(e.value) == "line 1: 30000 segments in one document; the limit is 20000"
    # The limit is how many a line writes, before any is read: text among them changes nothing.
    path.write_text('{"s": [' + ",".join(['"a"'] * 30000) + "]}\n")
    with pytest.raises(ScriptError, match="30000 segments"):
        materialize(corpus, path, SEASON, SCALE, enc)
    path.write_text('{"s": [' + ",".join(["[0,0,64]"] * 20000) + "]}\n")  # at the limit, and 1.28 MB: over the next one
    with pytest.raises(ScriptError, match="line 1: the output document is over 1,048,576 bytes"):
        materialize(corpus, path, SEASON, SCALE, enc)


def nested(depth: int, inside: str = "0") -> str:
    return "[" * depth + inside + "]" * depth


@pytest.mark.parametrize("depth", [17, 2_000, 100_000])
def test_a_line_nested_too_deep_is_invalid_whatever_python_reads_it(corpus, enc, depth):
    # How deep a JSON parser can go differs between versions of Python: 2,000 levels is
    # RecursionError on 3.11 and a parsed value on 3.13. Two graders must not disagree
    # about a script over that, so the limit is a rule, far below any parser's.
    path = corpus.path / "script.jsonl"
    for line in (nested(depth), '{"a":' * depth + "1" + "}" * depth, '{"d": 0, "x": ' + nested(depth - 1) + "}", '{"s": [' + nested(depth - 2) + "]}", '{"d": 0, "k": ' + nested(depth - 1) + "}"):
        path.write_text("0\n" + line + "\n")
        with pytest.raises(ScriptError) as e:
            materialize(corpus, path, SEASON, SCALE, enc)
        assert str(e.value) == "line 2: nested more than 16 deep"


def test_sixteen_deep_is_allowed_and_every_container_counts(corpus, enc):
    from stet.script import MAX_DEPTH, too_deep

    assert MAX_DEPTH == 16
    # Sixteen deep, under a key the format has no use for: still a line that keeps document 0.
    plain, plain_stats = run(corpus, enc, [0] * 40)
    path = corpus.path / "script.jsonl"
    path.write_text(('{"d": 0, "x": ' + nested(15) + "}\n") * 40)
    tokens, stats = materialize(corpus, path, SEASON, SCALE, enc)
    assert (tokens == plain).all() and stats.out_docs == plain_stats.out_docs and stats.script_lines == 40
    # What a line is for is still checked: sixteen deep in the wrong place is wrong for the old reason.
    path.write_text('{"s": ' + nested(15) + "}\n")
    with pytest.raises(ScriptError, match="a copy must be"):
        materialize(corpus, path, SEASON, SCALE, enc)

    for flat in (7, "text", None, 1.5, True):
        assert not too_deep(flat)
    assert not too_deep({"d": 1, "k": [[0, 900]] * 20000}) and not too_deep([]) and not too_deep({})
    for wrap in (lambda v: [v], lambda v: {"a": v}, lambda v: [1, "x", None, v], lambda v: {"p": "text", "q": v, "r": 2.5}):
        for depth, deep in ((16, False), (17, True), (2_000, True), (100_000, True)):
            v = 0
            for _ in range(depth):
                v = wrap(v)
            assert too_deep(v) is deep, depth
    assert too_deep([[[]]], limit=2) and not too_deep([[[]]], limit=3)
    assert too_deep([0, [1, {"a": [2]}]], limit=3) and not too_deep([0, [1, {"a": [2]}]], limit=4)  # the deepest branch decides


def test_a_line_longer_than_the_limit_is_rejected_without_being_read_whole(corpus, enc):
    path = corpus.path / "script.jsonl"
    with open(path, "wb") as f:
        f.write(b'{"d": 0, "x": "' + b"a" * (2 << 20) + b'"}\n')  # a whole page, and two megabytes under a key nothing reads
    with pytest.raises(ScriptError, match="longer than"):
        materialize(corpus, path, SEASON, SCALE, enc)


def test_empty_script_is_rejected(corpus, enc):
    path = corpus.path / "script.jsonl"
    path.write_text("")
    with pytest.raises(ScriptError):
        materialize(corpus, path, SEASON, SCALE, enc)


@pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason="POSIX only")
def test_a_script_that_is_a_link_or_a_pipe_is_refused(corpus, enc, tmp_path):
    secret = tmp_path / "someone-elses-script.jsonl"
    secret.write_text("0\n")
    link = corpus.path / "script.jsonl"
    link.symlink_to(secret)
    with pytest.raises(ScriptError):
        materialize(corpus, link, SEASON, SCALE, enc)
    with pytest.raises(ScriptError):
        file_sha256(link, 1 << 20)
    link.unlink()
    os.mkfifo(link)  # opening this for reading would block forever
    with pytest.raises(ScriptError, match="regular file"):
        open_script(link, 1 << 20)
    link.unlink()
    link.write_bytes(b"0\n" * 100)
    with pytest.raises(ScriptError, match="limit"):
        open_script(link, 50)


def test_stats_report_what_was_kept(corpus, enc):
    big = dataclasses.replace(SCALE, train_tokens=4096)
    _, stats = run(corpus, enc, [0, 0, 3], scale=big)
    d = stats.to_dict(corpus)
    assert d["src_docs"] == 2 and d["corpus_docs"] == 4
    assert json.dumps(d)  # serialisable


# --- memory is bounded without changing what a script means ----------------------------------
#
# Text is tokenised whenever a few megabytes are waiting, so a script of huge documents
# cannot make a grader hold gigabytes. That must not move a single token, nor change which
# scripts are valid: these values were recorded from the materialiser before it had a byte
# limit (commit 34ffe54), with the season's real tokenizer.

EMPTY = {"d": 0, "k": [[5, 5]]}
MIXED = [0, 3, 2, {"d": 3, "k": [[36, 900]]}, 1, {"s": [[0, 0, 450], [2, 0, 300]]}]
FOUR = "26ccb1914fd94a3c5a8b8a81fc500303f47d3e095022049041f57f79c2ed2c94"  # the stream of [0, 3, 0, 3], which fills the budget
RECORDED = [
    # (lines, documents per window, tokens_sha256 or the error, truncated, script_lines)
    ([2, 1], 1024, "96de5f9a3b07c5364acab7783087bda40c3b901790771e39d7ec3dce5f0be485", False, 2),
    (MIXED * 30, 1024, "f569b9120b4851d4beec5d525ef8ccad6397dbf33be9ba35c4e9aa326b80d75f", True, 180),
    ([0, 3, 0, 3, 9999], 3, "line 5: document 9999", None, None),  # a bad line in the window the budget fills in
    ([0, 3, 0, 3, 0, 0, 9999], 3, FOUR, True, 7),  # the same line after that window: never read
    ([0, 3, 0, 3, EMPTY, 0, EMPTY, 9999], 3, "line 8: document 9999", None, None),  # empty documents do not fill a window
    ([0, 3, 0, 3, EMPTY, 0, EMPTY, 0, 9999], 3, FOUR, True, 9),
    ([0, 3, 0, 3], 3, FOUR, False, 4),
    ([0, 3, 0, 3, 0], 3, FOUR, True, 5),
    ([0, 3, 0, 3], 2, FOUR, False, 4),
    ([0, 3, 0, 3, 9999], 2, FOUR, True, 5),
    ([0, 3, 0, 3, "{not json"], 3, "line 5: not valid JSON", None, None),
    ([0, 3, 0, 3, 0, 0, "{not json"], 3, FOUR, True, 7),
]


@pytest.mark.parametrize("waiting_bytes", [1, 700, 3000, 4 << 20])
def test_the_byte_limit_changes_no_stream_and_no_verdict(corpus, monkeypatch, waiting_bytes):
    from stet import script
    from stet.tok import load_encoding

    real = load_encoding(SEASON)
    monkeypatch.setattr(script, "TOKENIZE_BYTES", waiting_bytes)
    for lines, window, want, truncated, script_lines in RECORDED:
        monkeypatch.setattr(script, "TOKENIZE_BATCH", window)
        if truncated is None:
            with pytest.raises(ScriptError, match=want):
                run(corpus, real, lines)
            continue
        _, stats = run(corpus, real, lines)
        assert (stats.tokens_sha256, stats.truncated, stats.script_lines) == (want, truncated, script_lines), lines


# A line with text in it, or with a copy too short to be one, is a bad line like any other:
# refused up to the end of the window the budget fills in, and never read after it.
TEXT = {"s": [[0, 0, 900], "and a sentence of the curator's own"]}
SHORT = {"d": 0, "k": [[0, 500], [600, 610]]}
BAD_LINES = [
    ([0, 3, 0, 3, TEXT], 3, "line 5: this line holds text", None, None),
    ([0, 3, 0, 3, 0, 0, TEXT], 3, FOUR, True, 7),
    ([0, 3, 0, 3, SHORT], 3, "line 5: a copy of 10 bytes", None, None),
    ([0, 3, 0, 3, 0, 0, SHORT], 3, FOUR, True, 7),
    ([0, 3, 0, 3, EMPTY, 0, EMPTY, SHORT], 3, "line 8: a copy of 10 bytes", None, None),
    ([0, 3, 0, 3, SHORT], 2, FOUR, True, 5),
    ([SHORT, 0, 3, 0, 3], 1024, "line 1: a copy of 10 bytes", None, None),
    ([0, {"t": "text"}, 3, 0, 3], 1024, "line 2: this line holds text", None, None),
]


@pytest.mark.parametrize("waiting_bytes", [1, 3000, 4 << 20])
def test_text_and_short_copies_are_refused_where_any_bad_line_is(corpus, monkeypatch, waiting_bytes):
    from stet import script
    from stet.tok import load_encoding

    real = load_encoding(SEASON)
    monkeypatch.setattr(script, "TOKENIZE_BYTES", waiting_bytes)
    for lines, window, want, truncated, script_lines in BAD_LINES:
        monkeypatch.setattr(script, "TOKENIZE_BATCH", window)
        if truncated is None:
            with pytest.raises(ScriptError, match=want):
                run(corpus, real, lines)
            continue
        _, stats = run(corpus, real, lines)
        assert (stats.tokens_sha256, stats.truncated, stats.script_lines) == (want, truncated, script_lines), lines


def test_text_past_the_budget_is_not_held_in_memory(corpus, enc, monkeypatch):
    from stet import script

    sizes = []

    class Watching:
        def encode_ordinary_batch(self, texts):
            sizes.append(sum(len(t) for t in texts))
            return enc.encode_ordinary_batch(texts)

    monkeypatch.setattr(script, "TOKENIZE_BYTES", 4000)
    _, stats = run(corpus, Watching(), [0, 3] * 400)  # one window of 800 documents, about 1.3 MB
    assert stats.truncated and stats.script_lines == 800
    assert max(sizes) < 8000 and sum(sizes) < 20_000  # tokenised a little at a time, and not at all once full


# --- a text the tokenizer gives up on -----------------------------------------------------------


def test_a_script_whose_text_makes_the_tokenizer_panic_is_still_read_by_every_grader(tmp_path):
    # A million spaces and then a letter: tiktoken's matcher gives up with a Rust panic, which
    # is not an Exception, and would stop every machine handed the script. The run is cut
    # short and the text read again, the same on every machine.
    from conftest import make_corpus

    from stet.tok import LONG_SPACE_CHARS, encode_batch, load_encoding

    real = load_encoding(SEASON)
    bad = " " * 1_048_000 + "x"
    with pytest.raises(BaseException) as e:  # the fault this guards against, as the library raises it
        real.encode_ordinary_batch([bad])
    assert not isinstance(e.value, Exception)
    got = encode_batch(real, ["plain words", bad, "more plain words"])
    assert got[0] == real.encode_ordinary("plain words") and got[2] == real.encode_ordinary("more plain words")
    assert got[1] == real.encode_ordinary(" " * LONG_SPACE_CHARS + "x")  # as if the shorter run had been written
    ok = " " * 500_000 + "x"
    assert encode_batch(real, [ok]) == [real.encode_ordinary(ok)]  # what could be read before is read as before
    # And as a script. It holds no text of its own, but copies can make such a text: a page of
    # blank space said four times over, inside every limit, and then the start of another page.
    # It is an ordinary script, trained on as if the shorter run had been written, not a crash.
    page = "x marks the start of a page that is long enough to be copied from. " * 20
    blank = make_corpus(tmp_path / "blank", [" " * 262_000, page])
    assert blank.size(0) <= SEASON.max_doc_bytes and 4 * 262_000 + 64 <= SEASON.raw["script"]["max_out_doc_bytes"]
    with pytest.raises(BaseException) as e:
        real.encode_ordinary_batch([" " * (4 * 262_000) + page[:64]])
    assert not isinstance(e.value, Exception)
    tokens, stats = run(blank, real, [{"s": [[0, 0, 262_000]] * 4 + [[1, 0, 64]]}, 1])
    assert stats.out_docs == 2 and stats.out_bytes == 4 * 262_000 + 64 + len(page)
    shorter = make_corpus(tmp_path / "shorter", [" " * LONG_SPACE_CHARS, page])
    as_if, _ = run(shorter, real, [{"s": [[0, 0, LONG_SPACE_CHARS], [1, 0, 64]]}, 1])
    assert (tokens == as_if).all()

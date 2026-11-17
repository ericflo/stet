import dataclasses
import json
import os

import pytest
from conftest import EXAM_SENTENCE

from stet.script import ScriptError, build_doc, file_sha256, materialize, open_script, write_script
from stet.spec import Season, load_season

SEASON = load_season("s0")
EOT = SEASON.eot
SCALE = dataclasses.replace(SEASON.scale("nano"), train_tokens=512, ctx=64, batch_size=4)
# The real ink budget is a thousandth of the training budget: zero at this toy scale. Tests
# of the accounting use a season that allows a tenth.
INKY = Season(id=SEASON.id, title=SEASON.title, raw={**SEASON.raw, "ink_frac": 0.1})


def run(corpus, enc, lines, scale=SCALE, season=SEASON):
    path = corpus.path / "script.jsonl"
    write_script(path, lines)
    return materialize(corpus, path, season, scale, enc)


def test_whole_doc_forms_are_equivalent(corpus):
    a = build_doc(corpus, 1, 0, 64)
    b = build_doc(corpus, 1, {"d": 0}, 64)
    c = build_doc(corpus, 1, {"s": [[0, 0, corpus.size(0)]]}, 64)
    assert a.data == b.data == c.data == corpus.raw(0)
    assert a.ink == [] and a.copied == corpus.size(0) and not a.spliced


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


# --- ink ------------------------------------------------------------------------------------


def test_literal_text_is_ink(corpus):
    b = build_doc(corpus, 1, {"s": [[0, 0, 200], "NEW TEXT"]}, 64)
    assert b.data.endswith(b"NEW TEXT") and b.ink == [b"NEW TEXT"] and b.copied == 200 and b.spliced


def test_short_copies_are_ink_but_whole_short_docs_are_free(corpus):
    b = build_doc(corpus, 1, {"s": [[0, 0, 10]]}, 64)
    assert sum(map(len, b.ink)) == 10 and b.copied == 0
    b = build_doc(corpus, 1, 1, 64)  # the 5-byte document, kept whole on its own
    assert b.ink == [] and b.copied == 5
    # ...but tiny whole documents cannot be strung together to spell out new text for free
    b = build_doc(corpus, 1, {"s": [[1, 0, 5], [1, 0, 5]]}, 64)
    assert sum(map(len, b.ink)) == 10 and b.copied == 0


def test_adjacent_ranges_merge_so_line_cuts_are_free(corpus):
    # three contiguous 30-byte ranges are one 90-byte copy, not three short ones
    b = build_doc(corpus, 1, {"d": 0, "k": [[0, 30], [30, 60], [60, 90]]}, 64)
    assert b.ink == [] and b.copied == 90 and not b.spliced


def test_offsets_snap_to_character_boundaries(corpus):
    raw = corpus.raw(2)
    cut = raw.index("í".encode()) + 1  # inside a two-byte character
    build_doc(corpus, 1, {"d": 2, "k": [[cut, 200]]}, 64).data.decode("utf-8")  # must not raise


def test_the_real_season_allows_almost_no_ink(corpus, enc):
    assert SEASON.ink_tokens(SEASON.scale("micro")) == 40_000  # a thousandth of 40M
    with pytest.raises(ScriptError, match="ink over budget"):
        run(corpus, enc, [0, {"t": "one two three"}])  # the toy scale's budget rounds to zero
    _, stats = run(corpus, enc, [0, 3])  # cuts and whole copies need none
    assert stats.ink_tokens == 0


def test_ink_is_counted_in_tokens_of_what_is_trained(corpus, enc):
    # 512 training tokens, a tenth may be ink: 51.
    _, stats = run(corpus, enc, [0, {"s": [[3, 0, 1000], " A short note of five words."]}] + [0] * 30, season=INKY)
    assert stats.ink_tokens == 6 and stats.ink_budget == 51
    with pytest.raises(ScriptError, match="ink over budget"):
        run(corpus, enc, [{"t": "word " * 60}] + [0] * 30, season=INKY)


def test_ink_cannot_be_diluted_with_cheap_copies(corpus, enc):
    # The byte-counted rule could be beaten by padding ink with long copies that tokenize to
    # almost nothing. Counted in tokens against a fixed budget, padding buys nothing.
    pad = [0, 0, 1800]  # the whole 1,800-byte document, as a copy
    with pytest.raises(ScriptError, match="ink over budget"):
        run(corpus, enc, [{"s": [pad, "word " * 60]}] * 4, season=INKY)


def test_repeating_a_short_script_repeats_its_ink(corpus, enc):
    # One document with 20 ink tokens, trained on several times over: every pass counts.
    with pytest.raises(ScriptError, match="ink over budget"):
        run(corpus, enc, [{"s": [[3, 0, 1400], " " + "word " * 20]}], season=INKY)


def test_new_text_that_reproduces_the_exam_is_left_out_however_it_is_wrapped(corpus, enc):
    big = dataclasses.replace(SCALE, train_tokens=8192)
    wrapped = EXAM_SENTENCE.upper().replace(" and ", "\n   AND  ")
    plain, _ = run(corpus, enc, [0] * 40, scale=big, season=INKY)
    # Mid-line, re-wrapped and shouting, it is still the exam: the document is dropped and
    # the rest of the script stands.
    tokens, stats = run(corpus, enc, [{"t": "menu home about " + wrapped}] + [0] * 40, scale=big, season=INKY)
    assert stats.exam_dropped == 1 and stats.ink_tokens == 0 and (tokens == plain).all()
    # The same sentence spliced together from an honest copy and two pieces of ink is caught too.
    tokens, stats = run(corpus, enc, [{"s": [[0, 0, 90], " " + EXAM_SENTENCE[:50], EXAM_SENTENCE[50:]]}] + [0] * 40, scale=big, season=INKY)
    assert stats.exam_dropped == 1 and (tokens == plain).all()
    _, stats = run(corpus, enc, [{"t": "A different sentence entirely, of about the same length as the other one."}] + [0] * 40, scale=big, season=INKY)
    assert stats.ink_tokens > 0 and stats.exam_dropped == 0


# --- a script is untrusted data -----------------------------------------------------------------


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
        '{"d": 0, "k": [[0, ' + "9" * 5000 + "]]}",  # an integer too long to parse
        "[" * 100000 + "]" * 100000,  # nesting deep enough to exhaust the parser
        '{"d": ' + "1" * 400 + "}",
        '{"s": [' + ",".join(['"a"'] * 30000) + "]}",  # more segments than any document needs
        b"\xff\xfe{}".decode("latin-1"),
    ],
)
def test_malformed_scripts_are_rejected(corpus, enc, bad):
    path = corpus.path / "script.jsonl"
    path.write_text(bad + "\n", encoding="latin-1")
    with pytest.raises(ScriptError):
        materialize(corpus, path, SEASON, SCALE, enc)


def test_a_line_longer_than_the_limit_is_rejected_without_being_read_whole(corpus, enc):
    path = corpus.path / "script.jsonl"
    with open(path, "wb") as f:
        f.write(b'{"t": "' + b"a" * (2 << 20) + b'"}\n')
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

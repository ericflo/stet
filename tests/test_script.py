import dataclasses
import json

import pytest

from stet.script import ScriptError, build_doc, materialize, write_script
from stet.spec import load_season

SEASON = load_season("s0")
EOT = SEASON.eot
SCALE = dataclasses.replace(SEASON.scale("nano"), train_tokens=512, ctx=64, batch_size=4)


def run(corpus, enc, lines, scale=SCALE, tmp=None):
    path = corpus.path / "script.jsonl"
    write_script(path, lines)
    return materialize(corpus, path, SEASON, scale, enc)


def test_whole_doc_forms_are_equivalent(corpus):
    a = build_doc(corpus, 1, 0, 64)
    b = build_doc(corpus, 1, {"d": 0}, 64)
    c = build_doc(corpus, 1, {"s": [[0, 0, corpus.size(0)]]}, 64)
    assert a[0] == b[0] == c[0] == corpus.raw(0)
    assert a[1] == 0 and a[2] == corpus.size(0)


def test_tokens_are_exactly_the_budget_and_cycle(corpus, enc):
    tokens, stats = run(corpus, enc, [0])
    assert tokens.shape == (SCALE.batch_size, SCALE.steps * SCALE.ctx + 1)
    assert (tokens[:, 0] == EOT).all()  # every lane starts a document
    assert stats.passes > 1 and not stats.truncated


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


def test_literal_text_is_ink(corpus):
    out, ink, copied, _ = build_doc(corpus, 1, {"s": [[0, 0, 200], "NEW TEXT"]}, 64)
    assert out.endswith(b"NEW TEXT") and ink == 8 and copied == 200


def test_short_copies_are_ink_but_whole_short_docs_are_free(corpus):
    _, ink, copied, _ = build_doc(corpus, 1, {"s": [[0, 0, 10]]}, 64)
    assert ink == 10 and copied == 0
    _, ink, copied, _ = build_doc(corpus, 1, 1, 64)  # the 5-byte document, kept whole on its own
    assert ink == 0 and copied == 5
    # ...but tiny whole documents cannot be strung together to spell out new text for free
    _, ink, copied, _ = build_doc(corpus, 1, {"s": [[1, 0, 5], [1, 0, 5]]}, 64)
    assert ink == 10 and copied == 0


def test_adjacent_ranges_merge_so_line_cuts_are_free(corpus):
    # three contiguous 30-byte ranges are one 90-byte copy, not three short ones
    _, ink, copied, _ = build_doc(corpus, 1, {"d": 0, "k": [[0, 30], [30, 60], [60, 90]]}, 64)
    assert ink == 0 and copied == 90


def test_offsets_snap_to_character_boundaries(corpus):
    raw = corpus.raw(2)
    cut = raw.index("í".encode()) + 1  # inside a two-byte character
    out, _, _, _ = build_doc(corpus, 1, {"d": 2, "k": [[cut, 200]]}, 64)
    out.decode("utf-8")  # must not raise


def test_ink_over_budget_is_rejected(corpus, enc):
    with pytest.raises(ScriptError, match="ink over budget"):
        run(corpus, enc, [{"t": "canned text " * 200}])


def test_ink_within_budget_is_accepted(corpus, enc):
    _, stats = run(corpus, enc, [0, {"s": [[3, 0, 1000], "A short note."]}])
    assert 0 < stats.ink_frac <= SEASON.ink_frac


def test_ink_is_counted_only_on_what_is_trained(corpus, enc):
    # padding the tail with copies must not dilute ink in the part that is trained on
    lines = [{"t": "canned text " * 400}] + [0] * 50
    with pytest.raises(ScriptError, match="ink over budget"):
        run(corpus, enc, lines)


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
    ],
)
def test_malformed_scripts_are_rejected(corpus, enc, bad):
    path = corpus.path / "script.jsonl"
    path.write_text(bad + "\n")
    with pytest.raises(ScriptError):
        materialize(corpus, path, SEASON, SCALE, enc)


def test_empty_script_is_rejected(corpus, enc):
    path = corpus.path / "script.jsonl"
    path.write_text("")
    with pytest.raises(ScriptError):
        materialize(corpus, path, SEASON, SCALE, enc)


def test_stats_report_what_was_kept(corpus, enc):
    big = dataclasses.replace(SCALE, train_tokens=4096)
    _, stats = run(corpus, enc, [0, 0, 3], scale=big)
    d = stats.to_dict(corpus)
    assert d["src_docs"] == 2 and d["corpus_docs"] == 4
    assert json.dumps(d)  # serialisable


def test_new_text_may_not_reproduce_the_exam(corpus, enc, tmp_path):
    secret = "This sentence is long enough to count as a line of an exam passage, well over eighty characters."
    exam = {"encyclopedia": [{"id": "x", "text": "Title\n" + secret + "\nmore"}]}
    scale = SEASON.scale("nano")
    p = tmp_path / "s.jsonl"
    write_script(p, [0, {"t": "intro\n" + secret}])
    with pytest.raises(ScriptError, match="exam"):
        materialize(corpus, p, SEASON, scale, enc, exam)
    write_script(p, [0, {"t": "a different sentence entirely"}])
    with pytest.raises(ScriptError, match="ink|tokens"):
        materialize(corpus, p, SEASON, scale, enc, exam)  # no exam hit; fails later for other reasons

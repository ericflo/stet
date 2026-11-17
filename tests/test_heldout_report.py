"""The season's final table: the public leaderboard's arithmetic, on held-out losses."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

ANN, BOB, CAT = "a" * 64, "b" * 64, "c" * 64
S1, S2, S3, S4 = ("1" * 64, "2" * 64, "3" * 64, "4" * 64)


def entry(editor, name, script=None, loss=None, gain=0.0, **over):
    e = {"editor": editor, "owner": name, "name": "ed", "version": 1, "status": "settled", "valid": loss is not None, "gain": gain}
    if loss is not None:
        e.update(loss=loss, script={"sha256": script})
    return {**e, **over}


def line(rnd, editor, script, public, heldout, base=(5.3, 5.5)):
    return {"round": rnd, "editor": {"id": editor}, "script_sha256": script, "public": {"loss": public, "baseline": base[0]}, "heldout": {"loss": heldout, "baseline": base[1]}}


@pytest.fixture
def season(tmp_path):
    """Two settled rounds and an open one. Ann and Bob wrote the same script in the first;
    Cat's editor timed out in the first; the anchor could not run Bob's in the second."""
    rounds = [
        {"round": {"id": "s0-o00001", "index": 1, "status": "settled", "baseline_loss": 5.3}, "entries": [
            entry(ANN, "ann", S1, loss=4.40, gain=0.90), entry(BOB, "bob", S1, loss=4.40, gain=0.90), entry(CAT, "cat", reason="timeout")]},
        {"round": {"id": "s0-o00002", "index": 2, "status": "settled", "baseline_loss": 5.3}, "entries": [
            entry(ANN, "ann", S2, loss=4.50, gain=0.80), entry(BOB, "bob", void=True, reason="not run"), entry(CAT, "cat", S3, loss=4.70, gain=0.60)]},
        {"round": {"id": "s0-o00003", "index": 3, "status": "open"}, "entries": [
            {"editor": ANN, "owner": "ann", "name": "ed", "version": 1, "status": "open"}, entry(CAT, "cat", reason="timeout")]},
    ]  # fmt: skip
    lines = [
        line("s0-o00001", ANN, S1, 4.40, 9.99),  # graded once before a restart; the later line stands
        line("s0-o00001", ANN, S1, 4.40, 4.62),
        line("s0-o00001", BOB, S1, 4.40, 4.62),
        line("s0-o00002", ANN, S2, 4.50, 4.74),
        line("s0-o00002", CAT, S3, 4.70, 5.10),
    ]
    log = tmp_path / "s0.jsonl"
    log.write_text("".join(json.dumps(ln) + "\n" for ln in lines))
    return {"season": "s0", "rounds": rounds}, log, lines


def test_the_fit_is_the_leaderboards(season):
    import heldout_report as hr

    # Everyone in every round: a rating is a plain mean, whatever the rounds were like.
    rows = [(r, e, g + d, True) for r, d in ((1, 0.05), (2, -0.03), (3, 0.01)) for e, g in (("a", 0.9), ("b", 0.7))]
    got, diff = hr.fit(rows)
    assert got["a"]["rating"] == pytest.approx(0.91) and got["b"]["rating"] == pytest.approx(0.71) and sum(diff.values()) == pytest.approx(0)
    # An editor that ran only the hard round is not marked down for it...
    got, diff = hr.fit([(1, "a", 0.9, True), (2, "a", 0.7, True), (1, "b", 0.8, True), (2, "b", 0.6, True), (2, "c", 0.6, True)])
    assert diff[2] == pytest.approx(-0.1) and got["c"]["rating"] == pytest.approx(got["b"]["rating"]) and got["c"]["rounds"] == 1
    # ...and an invalid entry counts as zero for its editor and says nothing about its round.
    before, _ = hr.fit(rows)
    after, _ = hr.fit(rows + [(2, "z", 0.0, False)])
    assert after["a"]["rating"] == pytest.approx(before["a"]["rating"]) and after["z"] == {"rounds": 1, "mean": 0.0, "rating": 0.0, "se": None, "invalid": 1}


def test_the_final_table_is_the_public_one_on_held_out_losses(season, tmp_path, capsys):
    import heldout_report as hr

    source, log, _ = season
    rows, missing = hr.entries(source["rounds"], hr.load(log))
    assert not missing
    # The same entries the public leaderboard counts: not the void one, not the open round's.
    assert [(r["round"][-1], r["label"][:3], r["valid"]) for r in rows] == [("1", "ann", True), ("1", "bob", True), ("1", "cat", False), ("2", "ann", True), ("2", "cat", True)]
    by = {r["label"][:3]: r for r in hr.table(rows)}
    # Two editors that wrote one script each keep their round.
    assert by["bob"]["rounds"] == 1 and by["ann"]["rounds"] == 2
    assert by["cat"]["rounds"] == 2 and by["cat"]["invalid"] == 1  # the timeout is a round, and a zero, on both exams
    public, _ = hr.fit([(r["index"], r["editor"], r["public"], r["valid"]) for r in rows])
    heldout, _ = hr.fit([(r["index"], r["editor"], r["heldout"], r["valid"]) for r in rows])
    for who, editor in (("ann", ANN), ("bob", BOB), ("cat", CAT)):
        assert by[who]["public"] == pytest.approx(public[editor]["rating"], abs=1e-5) and by[who]["heldout"] == pytest.approx(heldout[editor]["rating"], abs=1e-5)
        assert by[who]["gives_up"] == pytest.approx(by[who]["public"] - by[who]["heldout"], abs=2e-5)
    assert rows[0]["heldout"] == pytest.approx(5.5 - 4.62)  # the later of Ann's two lines

    # The public column is the leaderboard's own, and the tool says so when it is not.
    board = {"rated_rounds": 2, "editors": [{"id": ANN, "owner": "ann", "name": "ed", "version": 1, "rating": round(public[ANN]["rating"], 5)}],
             "provisional": [{"id": BOB, "owner": "bob", "name": "ed", "version": 1, "rating": round(public[BOB]["rating"], 5)}, {"id": "f" * 64, "rating": None}]}  # fmt: skip
    assert hr.check_public(rows, board) == {"checked": 2, "differs": []}
    board["editors"][0]["rating"] += 0.01
    assert [d["editor"] for d in hr.check_public(rows, board)["differs"]] == [ANN]

    rounds_file, out = tmp_path / "rounds.json", tmp_path / "final.json"
    rounds_file.write_text(json.dumps({**source, "leaderboard": board}))
    assert hr.main([str(log), "--rounds-file", str(rounds_file), "--out", str(out)]) == 1  # it differs, and that is not success
    shown = capsys.readouterr().out
    assert "DOES NOT REPRODUCE THE LEADERBOARD" in shown and "s0-o00003" in shown and "not ranked" in shown and "ann/ed v1" in shown
    rep = json.loads(out.read_text())
    assert rep["rounds"] == ["s0-o00001", "s0-o00002"] and rep["open_rounds"] == ["s0-o00003"] and len(rep["record_sha256"]) == 64
    assert [t["label"][:3] for t in rep["table"]] == ["ann", "cat", "bob"]  # ranked by held-out rating; Bob has one round of the two it takes


def test_an_entry_the_record_lacks_is_named_and_can_be_measured_again(season, tmp_path, capsys):
    import heldout_report as hr

    source, log, lines = season
    log.write_text("".join(json.dumps(ln) + "\n" for ln in lines[:-1]) + json.dumps({**line("s0-o00002", ANN, S4, 4.5, 4.7)}) + "\n")
    rows, missing = hr.entries(source["rounds"], hr.load(log))
    # Cat's line is gone, and Ann's is for a script that is not the entry's.
    assert [(m["round"], m["label"][:3]) for m in missing] == [("s0-o00002", "ann"), ("s0-o00002", "cat")]
    assert missing[1]["measure"] == f"stet score {S3} --round s0-o00002 --heldout --json" and "no line" in missing[1]["why"]
    by = {r["label"][:3]: r for r in hr.table(rows)}
    assert by["cat"]["rounds"] == 1 and by["ann"]["rounds"] == 1  # the round is out of both columns, so they stay comparable
    assert hr.check_public(rows, {"editors": [{"id": CAT, "rating": 0.3}]})["checked"] == 1  # the check still uses every public entry

    # What that command prints fills the gap.
    again = tmp_path / "again.json"
    again.write_text(json.dumps({"valid": True, "round": {"id": "s0-o00002"}, "script_sha256": S3, "exam": {"loss": 4.70}, "baseline": {"loss": 5.3}, "heldout": {"loss": 5.10, "baseline": 5.5}}))
    rows, missing = hr.entries(source["rounds"], hr.load(log), hr.load_also([again]))
    assert [m["label"][:3] for m in missing] == ["ann"] and {r["label"][:3]: r for r in hr.table(rows)}["cat"]["rounds"] == 2
    # A line whose public loss is not the published one is not this entry's grade.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"valid": True, "round": {"id": "s0-o00002"}, "script_sha256": S3, "exam": {"loss": 4.90}, "baseline": {"loss": 5.3}, "heldout": {"loss": 5.10, "baseline": 5.5}}))
    assert "not the published one" in hr.entries(source["rounds"], hr.load(log), hr.load_also([bad]))[1][1]["why"]
    with pytest.raises(SystemExit):  # a result made without --heldout has nothing to offer
        again.write_text(json.dumps({"valid": True, "round": {"id": "s0-o00002"}, "script_sha256": S3, "exam": {"loss": 4.70}}))
        hr.load_also([again])

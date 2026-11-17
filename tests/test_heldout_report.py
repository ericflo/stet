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
    Cat's curator timed out in the first; the anchor could not run Bob's in the second."""
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
    # A curator that ran only the hard round is not marked down for it...
    got, diff = hr.fit([(1, "a", 0.9, True), (2, "a", 0.7, True), (1, "b", 0.8, True), (2, "b", 0.6, True), (2, "c", 0.6, True)])
    assert diff[2] == pytest.approx(-0.1) and got["c"]["rating"] == pytest.approx(got["b"]["rating"]) and got["c"]["rounds"] == 1
    # ...and an invalid entry counts as zero for its curator and says nothing about its round.
    before, _ = hr.fit(rows)
    after, _ = hr.fit(rows + [(2, "z", 0.0, False)])
    assert after["a"]["rating"] == pytest.approx(before["a"]["rating"])
    assert after["z"] == {"rounds": 1, "mean": 0.0, "rating": 0.0, "se": None, "invalid": 1, "skill": None}


def test_one_editors_timeout_does_not_move_anyone_elses_rating():
    import heldout_report as hr

    # Three reference curators in six rounds; carol in rounds 1-3 and dave in 4-6, each
    # gaining 0.925 every time. zed runs 1-3, gaining 0.9, and times out in round 2.
    rows = [(r, e, g, True) for r in range(1, 7) for e, g in (("identity", 0.0), ("english", 0.912), ("tidy", 0.933))]
    rows += [(r, "carol", 0.925, True) for r in (1, 2, 3)] + [(r, "dave", 0.925, True) for r in (4, 5, 6)]
    clean, _ = hr.fit(rows + [(r, "zed", 0.9, True) for r in (1, 2, 3)])
    got, diff = hr.fit(rows + [(1, "zed", 0.9, True), (2, "zed", 0.0, False), (3, "zed", 0.9, True)])
    # zed's zero is zed's: it is in zed's rating and nowhere else. Counted into what zed
    # "usually does", its good rounds would look easy and carol, who shared them, would
    # drop below dave for a timeout that was not hers.
    assert got["zed"]["rating"] == pytest.approx(0.6) and got["zed"]["skill"] == pytest.approx(0.9) and got["zed"]["invalid"] == 1
    for who in ("identity", "english", "tidy", "carol", "dave"):
        assert got[who]["rating"] == pytest.approx(clean[who]["rating"], abs=1e-9)
    assert got["carol"]["rating"] == pytest.approx(got["dave"]["rating"], abs=1e-9) and max(abs(d) for d in diff.values()) < 1e-9


def board_of(public, *, ranked=(), provisional=(), rated_rounds=2):
    row = lambda who, editor: {"id": editor, "owner": who, "name": "ed", "version": 1, "rating": round(public[editor]["rating"], 5)}  # noqa: E731
    return {"rated_rounds": rated_rounds, "editors": [row(*x) for x in ranked], "provisional": [row(*x) for x in provisional] + [{"id": "f" * 64, "rating": None}]}


def test_the_final_table_is_the_public_one_on_held_out_losses(season, tmp_path, capsys):
    import heldout_report as hr

    source, log, _ = season
    rows, missing = hr.entries(source["rounds"], hr.load(log))
    assert not missing
    # The same entries the public leaderboard counts: not the void one, not the open round's.
    assert [(r["round"][-1], r["label"][:3], r["valid"]) for r in rows] == [("1", "ann", True), ("1", "bob", True), ("1", "cat", False), ("2", "ann", True), ("2", "cat", True)]
    public, _ = hr.fit([(r["index"], r["editor"], r["public"], r["valid"]) for r in rows])
    heldout, _ = hr.fit([(r["index"], r["editor"], r["heldout"], r["valid"]) for r in rows])
    board = board_of(public, ranked=[("ann", ANN), ("cat", CAT)], provisional=[("bob", BOB)])
    by = {r["label"][:3]: r for r in hr.table(rows, 2, board)}
    # Two curators that wrote one script each keep their round.
    assert by["bob"]["rounds"] == 1 and by["ann"]["rounds"] == 2
    assert by["cat"]["rounds"] == 2 and by["cat"]["invalid"] == 1  # the timeout is a round, and a zero, on both exams
    for who, editor in (("ann", ANN), ("bob", BOB), ("cat", CAT)):
        assert by[who]["public"] == pytest.approx(public[editor]["rating"], abs=1e-5) and by[who]["heldout"] == pytest.approx(heldout[editor]["rating"], abs=1e-5)
    assert rows[0]["heldout"] == pytest.approx(5.5 - 4.62)  # the later of Ann's two lines
    # Each row carries what it was computed from, and where that came from.
    assert rows[0]["source"] == "record" and (rows[0]["script"], rows[0]["public_loss"], rows[0]["public_baseline"], rows[0]["heldout_loss"], rows[0]["heldout_baseline"]) == (S1, 4.40, 5.3, 4.62, 5.5)
    assert rows[2]["source"] is None and rows[2]["heldout"] == 0.0 and rows[2]["heldout_loss"] is None  # the timeout: nothing was measured

    # The public column is the leaderboard's own, and the tool says so when it is not.
    assert hr.check_public(rows, board) == {"checked": 3, "differs": [], "ranked_only_there": [], "ranked_only_here": []}
    rounds_file, out = tmp_path / "rounds.json", tmp_path / "final.json"
    rounds_file.write_text(json.dumps({**source, "leaderboard": board}))
    assert hr.main([str(log), "--rounds-file", str(rounds_file), "--out", str(out)]) == 0
    shown = capsys.readouterr().out
    assert "reproduces the leaderboard's rating for all 3 curators" in shown and "s0-o00003" in shown and "fewer than 2 rounds: not ranked" in shown and "ann/ed v1" in shown
    rep = json.loads(out.read_text())
    assert rep["rounds"] == ["s0-o00001", "s0-o00002"] and rep["open_rounds"] == ["s0-o00003"] and len(rep["record_sha256"]) == 64
    assert [(t["label"][:3], t["status"]) for t in rep["table"]] == [("ann", "ranked"), ("cat", "ranked"), ("bob", "provisional")]  # by held-out rating; Bob has one round of the two it takes
    # The published report is enough to work the table out again: its rows, and the fit.
    again = hr.table(rep["entries"], rep["rated_rounds"], board)
    assert again == rep["table"] and [r["source"] for r in rep["entries"]] == ["record", "record", None, "record", "record"]

    board["editors"][0]["rating"] += 0.01
    assert [d["editor"] for d in hr.check_public(rows, board)["differs"]] == [ANN]
    rounds_file.write_text(json.dumps({**source, "leaderboard": board}))
    assert hr.main([str(log), "--rounds-file", str(rounds_file)]) == 1  # it differs, and that is not success
    assert "DOES NOT REPRODUCE THE LEADERBOARD" in capsys.readouterr().out
    # So is a leaderboard that ranks a curator this tool has too few rounds for.
    board["editors"][0]["rating"] -= 0.01
    board["editors"].append(board["provisional"].pop(0))
    check = hr.check_public(rows, board)
    assert not check["differs"] and check["ranked_only_there"] == [{"editor": BOB, "label": "bob/ed v1", "rounds_here": 1}]
    rounds_file.write_text(json.dumps({**source, "leaderboard": board}))
    assert hr.main([str(log), "--rounds-file", str(rounds_file)]) == 1
    assert "bob/ed v1: ranked on the leaderboard, and has 1 rounds here" in capsys.readouterr().out


def test_an_editor_its_owner_replaced_has_no_rank_here_either(season, tmp_path, capsys):
    import heldout_report as hr

    source, log, _ = season
    rows, _ = hr.entries(source["rounds"], hr.load(log))
    public, _ = hr.fit([(r["index"], r["editor"], r["public"], r["valid"]) for r in rows])
    # Cat's curator has its two rounds, and Cat has since submitted a new one. The
    # leaderboard keeps the old curator's results and gives it no rank; it lists it nowhere.
    board = board_of(public, ranked=[("ann", ANN)], provisional=[("bob", BOB)])
    lines = hr.table(rows, 2, board)
    assert [(t["label"][:3], t["status"], t["ranked"]) for t in lines] == [("ann", "ranked", True), ("bob", "provisional", False), ("cat", "replaced", False)]
    # What it gave up is still shown, but it is not one of the curators the median is taken over.
    by = {t["label"][:3]: t for t in lines}
    assert hr.median_gives_up(lines) == pytest.approx(statistics_median([by["ann"]["gives_up"], by["bob"]["gives_up"]]))
    assert by["cat"]["beyond_median"] == pytest.approx(by["cat"]["gives_up"] - hr.median_gives_up(lines), abs=2e-5)
    # That is what a replaced curator looks like, not a fault in the arithmetic.
    check = hr.check_public(rows, board)
    assert check["ranked_only_here"] == [{"editor": CAT, "label": "cat/ed v1", "rounds": 2}] and not check["differs"] and not check["ranked_only_there"]
    rounds_file = tmp_path / "rounds.json"
    rounds_file.write_text(json.dumps({**source, "leaderboard": board}))
    assert hr.main([str(log), "--rounds-file", str(rounds_file)]) == 0
    shown = capsys.readouterr().out
    assert "replaced by their owners: not ranked" in shown and "Not on the leaderboard, though they have the rounds: cat/ed v1" in shown
    # With no leaderboard to ask, having the rounds is all there is to go by.
    assert [(t["label"][:3], t["status"]) for t in hr.table(rows, 2)] == [("ann", "ranked"), ("cat", "ranked"), ("bob", "provisional")]


def statistics_median(values):
    import statistics

    return statistics.median(values)


def test_what_an_editor_gives_up_is_counted_over_the_rounds_it_was_valid_in():
    import heldout_report as hr

    # Five curators over six rounds. Every valid entry scores exactly 0.2 higher on the
    # held-out exam than on the public one; one curator times out in two rounds of six.
    def row(r, e, public, valid=True):
        return {"round": f"s0-o{r:05d}", "index": r, "editor": e, "label": e, "valid": valid, "public": public if valid else 0.0, "heldout": round(public + 0.2, 5) if valid else 0.0}

    rows = [row(r, e, g) for r in range(1, 7) for e, g in (("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6))]
    rows += [row(r, "late", 0.75, valid=r not in (2, 5)) for r in range(1, 7)]
    by = {t["editor"]: t for t in hr.table(rows, 3)}
    # Its rating carries the two zeros on both exams, so the ratings differ by only 4/6 of
    # 0.2. Read as "gives up", that would say it gave up less than the others: or, on a
    # season where the held-out exam was the harder one, more, as if it had fitted the
    # public exam. Counted over the rounds it ran in, it gives up what everyone gives up.
    assert by["late"]["invalid"] == 2 and by["late"]["public"] - by["late"]["heldout"] == pytest.approx(-0.2 * 4 / 6, abs=1e-4)
    for e in ("a", "b", "c", "d", "late"):
        assert by[e]["gives_up"] == pytest.approx(-0.2, abs=1e-5) and by[e]["beyond_median"] == pytest.approx(0.0, abs=1e-5)
    # A curator that was never valid gave up nothing that can be measured, and is not in the median.
    by = {t["editor"]: t for t in hr.table(rows + [row(1, "never", 0.0, valid=False)], 3)}
    assert by["never"]["gives_up"] is None and by["never"]["beyond_median"] is None and by["a"]["beyond_median"] == pytest.approx(0.0, abs=1e-5)


def test_an_entry_the_record_lacks_is_named_and_can_be_measured_again(season, tmp_path, capsys):
    import heldout_report as hr

    source, log, lines = season
    # Cat's line is gone, and the only line for Ann's second round is for a script that is not the entry's.
    log.write_text("".join(json.dumps(ln) + "\n" for ln in lines[:3]) + json.dumps(line("s0-o00002", ANN, S4, 4.5, 4.7)) + "\n")
    rows, missing = hr.entries(source["rounds"], hr.load(log))
    assert [(m["round"], m["label"][:3]) for m in missing] == [("s0-o00002", "ann"), ("s0-o00002", "cat")]
    assert missing[1]["measure"] == f"stet score {S3} --round s0-o00002 --heldout --json" and "no line" in missing[1]["why"]
    by = {r["label"][:3]: r for r in hr.table(rows)}
    assert by["cat"]["rounds"] == 1 and by["ann"]["rounds"] == 1  # the round is out of both columns, so they stay comparable
    assert hr.check_public(rows, {"editors": [{"id": CAT, "rating": 0.3}]})["checked"] == 1  # the check still uses every public entry

    # What that command prints fills the gap, and the row says that is where it came from.
    again = tmp_path / "again.json"
    again.write_text(json.dumps({"valid": True, "round": {"id": "s0-o00002"}, "script_sha256": S3, "exam": {"loss": 4.70}, "baseline": {"loss": 5.3}, "heldout": {"loss": 5.10, "baseline": 5.5}}))
    rows, missing = hr.entries(source["rounds"], hr.load(log), hr.load_also([again]))
    assert [m["label"][:3] for m in missing] == ["ann"] and {r["label"][:3]: r for r in hr.table(rows)}["cat"]["rounds"] == 2
    assert [(r["source"], r["heldout_loss"], r["heldout_baseline"]) for r in rows if r["label"][:3] == "cat" and r["valid"]] == [("remeasured", 5.10, 5.5)]
    # A result whose public loss is not the published one is not this entry's grade.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"valid": True, "round": {"id": "s0-o00002"}, "script_sha256": S3, "exam": {"loss": 4.90}, "baseline": {"loss": 5.3}, "heldout": {"loss": 5.10, "baseline": 5.5}}))
    assert "not the published one" in hr.entries(source["rounds"], hr.load(log), hr.load_also([bad]))[1][1]["why"]
    with pytest.raises(SystemExit):  # a result made without --heldout has nothing to offer
        again.write_text(json.dumps({"valid": True, "round": {"id": "s0-o00002"}, "script_sha256": S3, "exam": {"loss": 4.70}}))
        hr.load_also([again])


def test_a_line_that_is_not_the_entrys_does_not_hide_one_that_is(season, tmp_path, capsys):
    import heldout_report as hr

    source, log, lines = season
    # Two anchors. One's time on the job ran out mid-grade; the other graded the entry and
    # its grade is the published one. The first still wrote its line, with the public loss
    # it measured, and here it is the later of the two. And a third line for another script.
    log.write_text("".join(json.dumps(ln) + "\n" for ln in lines) + json.dumps(line("s0-o00002", CAT, S3, 4.73, 5.2)) + "\n" + json.dumps(line("s0-o00002", CAT, S4, 4.70, 5.3)) + "\n")
    rows, missing = hr.entries(source["rounds"], hr.load(log))
    assert not missing and [r["heldout_loss"] for r in rows if r["label"][:3] == "cat" and r["valid"]] == [5.10]

    # With only the line that is not the published grade, the entry is a gap, and what
    # fills it is the result the tool itself asks for.
    log.write_text("".join(json.dumps(ln) + "\n" for ln in lines[:-1]) + json.dumps(line("s0-o00002", CAT, S3, 4.73, 5.2)) + "\n")
    rows, missing = hr.entries(source["rounds"], hr.load(log))
    assert [m["label"][:3] for m in missing] == ["cat"] and "is not the published one (4.70000)" in missing[0]["why"]
    again, spare = tmp_path / "again.json", tmp_path / "spare.json"
    again.write_text(json.dumps({"valid": True, "round": {"id": "s0-o00002"}, "script_sha256": S3, "exam": {"loss": 4.70}, "baseline": {"loss": 5.3}, "heldout": {"loss": 5.10, "baseline": 5.5}}))
    rows, missing = hr.entries(source["rounds"], hr.load(log), hr.load_also([again]))
    assert not missing and [(r["source"], r["heldout_loss"]) for r in rows if r["label"][:3] == "cat" and r["valid"]] == [("remeasured", 5.10)]

    # A file given with --also that nothing needed is said out loud: it may be the wrong one.
    spare.write_text(json.dumps({"valid": True, "round": {"id": "s0-o00001"}, "script_sha256": S1, "exam": {"loss": 4.40}, "baseline": {"loss": 5.3}, "heldout": {"loss": 4.0, "baseline": 5.5}}))
    rounds_file = tmp_path / "rounds.json"
    rounds_file.write_text(json.dumps(source))
    assert hr.main([str(log), "--rounds-file", str(rounds_file), "--also", str(again), str(spare)]) == 0
    shown = capsys.readouterr().out
    assert f"WARNING: {spare} was given with --also and used for nothing" in shown and str(again) not in shown
    rep = hr.report(log, source, [again, spare])
    assert rep["also_unused"] == [str(spare)] and [r["heldout_loss"] for r in rep["entries"] if r["round"] == "s0-o00001" and r["valid"]] == [4.62, 4.62]  # the record's line stands


def test_a_table_with_gaps_does_not_pass_for_the_leaderboards(season, tmp_path, capsys):
    import heldout_report as hr

    source, log, lines = season
    rows, _ = hr.entries(source["rounds"], hr.load(log))
    public, _ = hr.fit([(r["index"], r["editor"], r["public"], r["valid"]) for r in rows])
    board = board_of(public, ranked=[("ann", ANN), ("cat", CAT)], provisional=[("bob", BOB)])
    log.write_text("".join(json.dumps(ln) + "\n" for ln in lines[:-1]))  # nothing for Cat's second round
    rounds_file, out = tmp_path / "rounds.json", tmp_path / "final.json"
    rounds_file.write_text(json.dumps({**source, "leaderboard": board}))
    # The arithmetic is right, on every public entry. The column printed is not the
    # leaderboard's for Cat, whose round is left out, and the tool does not say it is.
    assert hr.main([str(log), "--rounds-file", str(rounds_file), "--out", str(out)]) == 2
    shown = capsys.readouterr().out
    assert "THE TABLE IS NOT WHOLE" in shown and "not the leaderboard's for: cat/ed v1" in shown and "The public column reproduces" not in shown
    assert "1 valid entry could not be placed" in shown and f"stet score {S3} --round s0-o00002 --heldout --json" in shown
    rep = json.loads(out.read_text())
    assert rep["public_check"]["entries_left_out"] == 1 and rep["public_check"]["editors_not_whole"] == ["cat/ed v1"] and not rep["public_check"]["differs"]
    cat = next(t for t in rep["table"] if t["label"][:3] == "cat")
    assert cat["status"] == "ranked" and cat["rounds"] == 1  # still ranked, as the leaderboard has it, on the one round that could be placed


# --- entries that were run more than once ----------------------------------------------------


def run(n, script=None, gain=None, status="ok"):
    return {"n": n, "status": status, "script": script, "gain": gain}


def ran(editor, name, runs, gain, valid=True):
    scripts = [r["script"] for r in runs if r["status"] == "ok"]
    e = {"editor": editor, "owner": name, "name": "ed", "version": 1, "status": "settled", "valid": valid, "gain": gain, "runs": runs}
    if valid:
        e.update(loss=round(5.3 - gain, 5), script={"sha256": scripts[0]})
    return e


def test_an_entry_run_twice_is_scored_over_its_runs_on_the_held_out_exam_too(tmp_path):
    import heldout_report as hr

    rounds = [{"round": {"id": "s0-o00007", "index": 7, "status": "settled", "baseline_loss": 5.3}, "entries": [
        # Ann's two runs wrote different scripts; Bob's wrote the same one; one of Cat's three failed.
        ran(ANN, "ann", [run(1, S1, 0.90), run(2, S2, 0.84)], 0.87),
        ran(BOB, "bob", [run(1, S3, 0.70), run(2, S3, 0.70)], 0.70),
        ran(CAT, "cat", [run(1, S4, 0.60), run(2, status="failed"), run(3, S4, 0.60)], 0.60),
    ]}]  # fmt: skip
    lines = [
        # This machine's own measure of each script: near the published gain, not equal to it.
        line("s0-o00007", ANN, S1, 4.41, 4.70),  # public gain 0.89, held-out 0.80
        line("s0-o00007", ANN, S2, 4.45, 4.78),  # 0.85, 0.72
        line("s0-o00007", BOB, S3, 4.60, 4.90),  # one line stands for both of Bob's runs: 0.60
        line("s0-o00007", CAT, S4, 4.70, 5.00),  # 0.50
    ]
    log = tmp_path / "s0.jsonl"
    log.write_text("".join(json.dumps(ln) + "\n" for ln in lines))
    rows, missing = hr.entries(rounds, hr.load(log))
    assert not missing
    by = {r["label"][:3]: r for r in rows}
    assert by["ann"]["heldout"] == pytest.approx(0.76) and by["ann"]["public"] == 0.87  # the mean of two
    assert by["bob"]["heldout"] == pytest.approx(0.60)
    assert by["cat"]["heldout"] == pytest.approx(0.50)  # the median of 0.50, 0 and 0.50: the failed run is a zero, and is outvoted
    assert [p["heldout"] for p in by["cat"]["runs"]] == [0.5, 0.0, 0.5] and by["cat"]["runs"][1]["source"] is None
    assert by["ann"]["source"] == "record" and by["ann"]["script"] == S1


def test_a_run_whose_script_was_never_measured_is_named_and_can_be_measured_again(tmp_path):
    import heldout_report as hr

    rounds = [{"round": {"id": "s0-o00007", "index": 7, "status": "settled", "baseline_loss": 5.3}, "entries": [ran(ANN, "ann", [run(1, S1, 0.90), run(2, S2, 0.84)], 0.87)]}]
    log = tmp_path / "s0.jsonl"
    # The machine with the pack graded only the first run's script; a volunteer graded the other.
    # And one line here is for the right script on what cannot be this round: its public gain is nowhere near.
    log.write_text(json.dumps(line("s0-o00007", ANN, S1, 4.41, 4.70)) + "\n" + json.dumps(line("s0-o00007", ANN, S2, 5.20, 5.40)) + "\n")
    rows, missing = hr.entries(rounds, hr.load(log))
    assert rows[0]["heldout"] is None and [p["heldout"] for p in rows[0]["runs"]] == [0.8, None]
    assert [(m["script_sha256"], m["measure"]) for m in missing] == [(S2, f"stet score {S2} --round s0-o00007 --heldout --json")] and "far from the published one" in missing[0]["why"]
    again = tmp_path / "again.json"
    again.write_text(json.dumps({"valid": True, "round": {"id": "s0-o00007"}, "script_sha256": S2, "exam": {"loss": 4.45}, "baseline": {"loss": 5.3}, "heldout": {"loss": 4.78, "baseline": 5.5}}))
    rows, missing = hr.entries(rounds, hr.load(log), hr.load_also([again]))
    assert not missing and rows[0]["heldout"] == pytest.approx(0.76) and rows[0]["source"] == "remeasured"


def test_an_entry_from_before_runs_is_placed_the_same_once_the_server_shows_it_as_one_run(season):
    import heldout_report as hr

    # The server shows an entry settled before runs existed as one run, whose gain is the
    # one grade there was. The row must come out as it did when the entry had no runs.
    source, log, _ = season
    before, missing = hr.entries(source["rounds"], hr.load(log))
    assert not missing
    shown = json.loads(json.dumps(source["rounds"]))
    for r in shown:
        for e in r["entries"]:
            if e.get("status") == "settled" and not e.get("void"):
                ok = bool(e.get("valid"))
                e["runs"] = [{"n": 1, "status": "ok" if ok else "failed", "script": (e.get("script") or {}).get("sha256"), "gain": e.get("gain", 0.0)}]
    after, missing = hr.entries(shown, hr.load(log))
    assert not missing
    strip = lambda rows: [{k: v for k, v in r.items() if k != "runs"} for r in rows]  # noqa: E731
    assert strip(after) == strip(before)

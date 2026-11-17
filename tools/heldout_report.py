"""The season's table on the held-out exam: the public leaderboard, worked out again on held-out losses.

    python tools/heldout_report.py ~/.config/stet/heldout/s0.jsonl
    python tools/heldout_report.py s0.jsonl --rounds-file rounds.json     # offline, from saved rounds
    python tools/heldout_report.py s0.jsonl --save-rounds rounds.json --out final.json

Same editors, same rounds, same arithmetic. The rounds and their entries are the public
ones, read from the API. For every entry that counts on the public leaderboard:

  - an entry that was void (nobody could run it) is left out, as it is publicly;
  - an invalid entry scores a gain of 0 on both exams, as it does publicly;
  - a valid entry's held-out gain is made the way its public gain is. An entry is run more
    than once, by different people, and each run writes a script. A script's held-out gain
    is the unedited held-out loss minus its own, both measured on one machine that holds
    the pack: a line of that machine's record (one per official grade it made, keyed by
    round and editor), or a result passed with --also. A run that failed is a gain of 0.
    The entry's gain is the mean of two runs or the median of three, as it is publicly.
    (Entries settled before runs existed have one script and one grade, and are placed by
    that grade's public loss.)

The rating is then fitted exactly as the server fits it (`fit` below is fitRatings in
server/rating.go): gain = editor + round, so an editor is not flattered by easy rounds. The
public column is checked against the live leaderboard, and the tool says so if it differs.
Who is ranked is the leaderboard's word too: an editor its owner replaced keeps its
results and has no rank there, so it has none here, and is listed apart.

A valid entry with no usable line in the record (none for its script, none with a held-out
baseline, none whose public loss is the published one) cannot be placed. It is listed,
with the command that measures it again on a machine holding the pack, and that round is
left out of *both* columns for that editor, so the two stay comparable. `--also` takes the
JSON that command prints and fills the gap. While any gap is left the public column is not
the leaderboard's for the editors concerned, the tool says so, and it exits 2.

How to read "gives up". The two exams are different passages, so every editor's held-out
gains differ from its public ones by some common amount; that amount says nothing about
anyone. An editor that learned the kind of writing gives up about what the others give up.
An editor that fitted the public exam's own passages gives up more than the others do: that
is the last column, each editor against the median. It is worked out over the rounds an
editor's entry was valid in: a timeout is a zero on both exams and gives up nothing, and
counted in, it would make an editor that timed out look as if it had given up less.

The JSON report (`--json`, `--out`) carries every entry the table was computed from, with
its losses on both exams and their `source`: "record" (a line of the record for that grade)
or "remeasured" (a result passed with --also). When a season closes, the record,
the report and the held-out pack are published, and the table can be rebuilt from the
report alone. Anyone can then check an entry: put the pack where STET_HELDOUT_EXAM points
and run

    stet score <script sha256> --round <round id> --heldout --json

The result's `heldout.loss` and `heldout.baseline` are that entry's numbers.

Exit status: 0 when the table is whole and its public column is the leaderboard's; 1 when
the public arithmetic does not reproduce the leaderboard; 2 when it does, but entries are
missing from the table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_API = "https://api.stetrun.com"
TOLERANCE = 1e-4  # how closely the public column must reproduce the leaderboard
# A run's published gain is a median over several graders, each on their own machine, so
# this machine's measure of the same script is near it and not equal to it. A line further
# away than this is not a measure of that script on that round.
RUN_TIE = 0.25


def load(path: Path) -> dict[tuple[str, str], list[dict]]:
    """The record of a machine that holds the pack: every line for each (round, editor), in
    the order written.

    A machine may grade a script twice after a restart, and an entry has a script for each
    of its runs: which line is whose is decided where the entry is known (`entries`), not
    here. The key is the editor, never
    the script: two editors can write the same script in a round (the starting template
    does, for everyone who enters it unchanged), and each of them has that round.
    """
    lines: dict[tuple[str, str], list[dict]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            lines.setdefault((rec["round"], (rec.get("editor") or {}).get("id")), []).append(rec)
    return lines


def load_also(paths: list[Path]) -> dict[tuple[str, str], dict]:
    """Results of `stet score <sha> --round <id> --heldout --json`, by (round, script).

    Such a result names a script, not an editor: it stands for every entry of that round
    whose script it is, which is right, because the same script on the same round is the
    same training run.
    """
    out: dict[tuple[str, str], dict] = {}
    for p in paths:
        r = json.loads(Path(p).read_text())
        held = r.get("heldout") or {}
        if not r.get("valid") or held.get("loss") is None or held.get("baseline") is None:
            raise SystemExit(f"{p} has no held-out result with a baseline in it. Make it with: stet score <sha> --round <id> --heldout --json")
        out[(r["round"]["id"], r["script_sha256"])] = {
            "round": r["round"]["id"],
            "script_sha256": r["script_sha256"],
            "public": {"loss": r["exam"]["loss"], "baseline": (r.get("baseline") or {}).get("loss")},
            "heldout": {"loss": held["loss"], "baseline": held["baseline"]},
            "file": str(p),
            "used": False,
        }
    return out


def _get(api: str, path: str):
    req = urllib.request.Request(api.rstrip("/") + path, headers={"User-Agent": "stet-heldout-report", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read(64 << 20))


def fetch(api: str, season: str) -> dict:
    """Every round of the season with its entries, and the leaderboard, as the public API gives them."""
    rounds = []
    for index in range(1, 100_000):
        try:
            rounds.append(_get(api, f"/api/v1/rounds/{season}-o{index:05d}"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                break
            raise
    return {"api": api, "season": season, "rounds": rounds, "leaderboard": _get(api, "/api/v1/leaderboard")}


def counts(rd: dict) -> bool:
    """Whether the public leaderboard counts this round: settled, with its unedited loss known.

    The same rule as settledEntries in server/rating.go. A round still open has entries that
    are settled already (an editor that wrote no script settles at once), and on their own
    they would stand for the whole round.
    """
    return rd.get("status") == "settled" and rd.get("baseline_loss") is not None


def _place(e: dict, script: str | None, lines: list[dict], again: dict | None) -> tuple[dict | None, str, str]:
    """The measurement that is this entry's: (it, where it came from, or why there is none).

    A line is the entry's if it is for the entry's script, has both held-out losses, and
    its public loss is the one published: that last is what ties a line to the grade that
    counted. The latest such line in the record wins. Failing that, a result passed with
    --also is taken on the same terms.
    """
    why = "the record has no line for it"
    for rec in reversed(lines):
        held = rec.get("heldout") or {}
        if rec.get("script_sha256") != script:
            continue  # a line for some other script: not this entry's
        if held.get("loss") is None or held.get("baseline") is None:
            why = "its line has no held-out baseline"
        elif abs((rec.get("public") or {}).get("loss", math.inf) - e["loss"]) > TOLERANCE:
            why = f"its line's public loss ({rec['public']['loss']:.5f}) is not the published one ({e['loss']:.5f})"
        else:
            return rec, "record", ""
    if again is not None:
        if abs(again["public"]["loss"] - e["loss"]) > TOLERANCE:
            return None, "", f"the result given with --also has a public loss ({again['public']['loss']:.5f}) that is not the published one ({e['loss']:.5f})"
        again["used"] = True
        return again, "remeasured", ""
    return None, "", why


def _script(x: dict) -> str | None:
    """The hash of the script an entry or a run names, however it is written."""
    s = x.get("script")
    if isinstance(s, dict):
        s = s.get("sha256")
    return s if isinstance(s, str) and s else None


def _place_run(run: dict, script: str, lines: list[dict], again: dict | None) -> tuple[dict | None, str, str]:
    """The measurement of one run's script: (it, where it came from, or why there is none).

    A line is the run's if it is for the run's script, has both held-out losses, and its
    public gain is near the run's published one. Of several, the one whose public gain is
    nearest the published one is taken, and the later of two as near: where this machine's
    grade was the score, that is the very training run the score came from. Failing any, a
    result passed with --also on the same terms.
    """
    why = "the record has no line for it"

    def off(public: dict) -> float:
        if run.get("gain") is None or public.get("loss") is None or public.get("baseline") is None:
            return 0.0  # nothing to compare: the round, the editor and the script name it
        return abs((public["baseline"] - public["loss"]) - float(run["gain"]))

    best = None
    for rec in lines:
        held = rec.get("heldout") or {}
        if rec.get("script_sha256") != script:
            continue
        if held.get("loss") is None or held.get("baseline") is None:
            why = "its line has no held-out baseline"
        elif (d := off(rec.get("public") or {})) > RUN_TIE:
            why = "its line's public gain is far from the published one: not a measure of this script on this round"
        elif best is None or d <= best[0] + 1e-9:
            best = (d, rec)
    if best is not None:
        return best[1], "record", ""
    if again is not None:
        if off(again["public"]) > RUN_TIE:
            return None, "", "the result given with --also has a public gain far from the published one"
        again["used"] = True
        return again, "remeasured", ""
    return None, "", why


def _by_runs(rd: dict, e: dict, label: str, row: dict, lines: list[dict], also: dict, missing: list[dict]) -> None:
    """Fill in a row for an entry that was run more than once: each run's script placed, then
    the entry's gain taken over its runs as the server takes it."""
    gains, parts, lacking = [], [], False
    for run in e["runs"]:
        script = _script(run)
        part = {"n": run.get("n"), "script": script, "public": run.get("gain"), "heldout": 0.0, "heldout_loss": None, "heldout_baseline": None, "source": None}
        if run.get("status") != "ok" or not script:
            part["public"] = 0.0  # a run that failed: a zero on both exams
        else:
            rec, source, why = _place_run(run, script, lines, also.get((rd["id"], script)))
            if rec is None:
                lacking = True
                part["heldout"] = None
                missing.append({"round": rd["id"], "editor": e["editor"], "label": label, "script_sha256": script, "why": why,
                                "measure": f"stet score {script} --round {rd['id']} --heldout --json"})  # fmt: skip
            else:
                held = rec["heldout"]
                part.update(heldout=round(held["baseline"] - held["loss"], 5), heldout_loss=held["loss"], heldout_baseline=held["baseline"], source=source)
        parts.append(part)
        gains.append(part["heldout"])
    row.update(script=_script(e), public=float(e["gain"]), public_loss=e.get("loss"), runs=parts, heldout=None)
    if not lacking:
        # The mean of two, the median of three: statistics.median is both.
        row["heldout"] = round(statistics.median(gains), 5)
        row["source"] = "remeasured" if any(p["source"] == "remeasured" for p in parts) else "record"
        if len(parts) == 1:  # one run: the row is made of that run's losses, and says which
            row.update(heldout_loss=parts[0]["heldout_loss"], heldout_baseline=parts[0]["heldout_baseline"])


def entries(rounds: list[dict], log: dict, also: dict | None = None) -> tuple[list[dict], list[dict]]:
    """(the entries that count, the ones that cannot be placed).

    Each entry that counts is a row: round, index, editor, label, valid, script, its gain on
    each exam (`public`, `heldout`), and the losses those gains are made of with where the
    held-out ones came from (`source`: "record" or "remeasured"). `heldout` is None where
    nothing usable was measured, and those entries are listed in the second value as well,
    with why.
    """
    also = also or {}
    rows, missing = [], []
    for r in rounds:
        rd = r["round"]
        if not counts(rd):
            continue
        for e in r.get("entries") or []:
            if e.get("status") != "settled" or e.get("void"):
                continue
            label = f"{e.get('owner')}/{e.get('name')} v{e.get('version')}"
            row = {
                "round": rd["id"], "index": int(rd["index"]), "editor": e["editor"], "label": label, "valid": bool(e.get("valid")), "script": None,
                "public": 0.0, "heldout": 0.0, "public_loss": None, "public_baseline": rd.get("baseline_loss"), "heldout_loss": None, "heldout_baseline": None, "source": None,
            }  # fmt: skip
            if row["valid"] and e.get("runs"):
                _by_runs(rd, e, label, row, log.get((rd["id"], e["editor"])) or [], also, missing)
            elif row["valid"]:
                script = _script(e)
                rec, source, why = _place(e, script, log.get((rd["id"], e["editor"])) or [], also.get((rd["id"], script)))
                row.update(script=script, public=float(e["gain"]), public_loss=e["loss"], heldout=None)
                if rec is not None:
                    held = rec["heldout"]
                    # Rounded as the server rounds a public gain, so the two columns are the same arithmetic.
                    row.update(heldout=round(held["baseline"] - held["loss"], 5), heldout_loss=held["loss"], heldout_baseline=held["baseline"])
                    row["source"] = source
                else:
                    missing.append({"round": rd["id"], "editor": e["editor"], "label": label, "script_sha256": script, "why": why,
                                    "measure": f"stet score {script} --round {rd['id']} --heldout --json"})  # fmt: skip
            rows.append(row)
    rows.sort(key=lambda x: (x["index"], x["editor"]))
    return rows, missing


def fit(rows: list[tuple[int, str, float, bool]]) -> tuple[dict[str, dict], dict[int, float]]:
    """fitRatings in server/rating.go, line for line. rows are (round, editor, gain, valid).

    gain = skill[editor] + difficulty[round], solved by alternating means. Skill is what an
    editor does in the rounds where it ran: its mean over valid entries only. A round's
    difficulty is read from its valid entries, against that skill, and averages to zero
    over the season. The rating is the mean over all of an editor's entries, an invalid one
    counting as zero whatever its round was like; that zero is kept out of the skill, or
    one editor's timeout would make the rounds it did run in look easy and mark down
    everyone who shared them.

    Each editor's `skill` is returned beside its `rating` (None if it was never valid).
    """
    by_editor: dict[str, list] = {}
    by_round: dict[int, list] = {}
    for row in rows:
        by_editor.setdefault(row[1], []).append(row)
        by_round.setdefault(row[0], []).append(row)
    skill: dict[str, float] = {}
    diff: dict[int, float] = {}

    def adjusted(row) -> float:
        return row[2] - diff.get(row[0], 0.0) if row[3] else 0.0

    for i in range(200):
        delta = 0.0
        for editor, es in by_editor.items():
            s, n = 0.0, 0
            for e in es:
                if e[3]:
                    s += e[2] - diff.get(e[0], 0.0)
                    n += 1
            if n > 0:
                s /= n
            delta = max(delta, abs(s - skill.get(editor, 0.0)))
            skill[editor] = s
        mean = 0.0
        for rnd, es in by_round.items():
            s, n = 0.0, 0
            for e in es:
                if e[3]:
                    s += e[2] - skill[e[1]]
                    n += 1
            diff[rnd] = s / n if n > 0 else 0.0
            mean += diff[rnd]
        mean /= max(1, len(by_round))
        for rnd in diff:
            diff[rnd] -= mean
        if delta < 1e-9 and i > 0:
            break
    out = {}
    for editor, es in by_editor.items():
        resid = [adjusted(e) for e in es]
        measured = [adjusted(e) for e in es if e[3]]
        out[editor] = {
            "rounds": len(es),
            "mean": sum(e[2] for e in es) / len(es),
            "rating": sum(resid) / len(resid),
            "se": statistics.stdev(resid) / math.sqrt(len(resid)) if len(resid) >= 2 else None,
            "invalid": sum(1 for e in es if not e[3]),
            "skill": sum(measured) / len(measured) if measured else None,
        }
    return out, diff


def standing(leaderboard: dict | None) -> dict[str, str] | None:
    """Each editor the leaderboard lists, and how: "ranked" or "provisional". None without a leaderboard."""
    if not leaderboard:
        return None
    out = {e["id"]: "provisional" for e in leaderboard.get("provisional") or []}
    out.update({e["id"]: "ranked" for e in leaderboard.get("editors") or []})
    return out


def table(rows: list[dict], rated_rounds: int = 3, leaderboard: dict | None = None) -> list[dict]:
    """One line per editor: its public rating and its held-out rating over the same rounds.

    Who is ranked is the leaderboard's to say. An editor it does not list has been replaced
    by its owner: it keeps its results there and has no rank, so here it is "replaced",
    apart from the rest and out of the median. Without a leaderboard to ask, an editor is
    ranked when it has the rounds, which cannot tell a replaced one.
    """
    placed = [r for r in rows if r["heldout"] is not None]
    public, _ = fit([(r["index"], r["editor"], r["public"], r["valid"]) for r in placed])
    heldout, _ = fit([(r["index"], r["editor"], r["heldout"], r["valid"]) for r in placed])
    labels = {r["editor"]: r["label"] for r in rows}
    listed = standing(leaderboard)
    out = []
    for editor, h in heldout.items():
        p = public[editor]
        status = ("ranked" if h["rounds"] >= rated_rounds else "provisional") if listed is None else listed.get(editor, "replaced")
        out.append({
            "editor": editor, "label": labels[editor], "rounds": h["rounds"], "invalid": h["invalid"], "status": status, "ranked": status == "ranked",
            "public": round(p["rating"], 5), "heldout": round(h["rating"], 5), "heldout_se": None if h["se"] is None else round(h["se"], 5),
            # Over the rounds its entry was valid in. An invalid round is a zero on both
            # exams: it gives up nothing, and averaged in it would shrink what the editor's
            # real entries gave up by the share of its rounds that failed.
            "gives_up": None if h["skill"] is None else round(p["skill"] - h["skill"], 5),
        })  # fmt: skip
    # What every editor gives up says something about the two exams. What one editor gives
    # up beyond that says something about the editor.
    usual = median_gives_up(out)
    for r in out:
        r["beyond_median"] = None if r["gives_up"] is None else round(r["gives_up"] - usual, 5)
    order = {"ranked": 0, "provisional": 1, "replaced": 2}
    return sorted(out, key=lambda r: (order[r["status"]], -r["heldout"], r["editor"]))


def median_gives_up(lines: list[dict]) -> float:
    return statistics.median([t["gives_up"] for t in lines if t["gives_up"] is not None and t["status"] != "replaced"] or [0.0])


def check_public(rows: list[dict], leaderboard: dict | None, rated_rounds: int | None = None) -> dict:
    """Does this tool's arithmetic, on every public entry, give the leaderboard's own ratings and its ranks?

    `differs`: editors whose rating here is not the leaderboard's. `ranked_only_there`:
    editors the leaderboard ranks that have too few rounds here. Either means the rounds
    read are not the ones the leaderboard counted, or the arithmetic has drifted.
    `ranked_only_here`: editors with the rounds to be ranked that the leaderboard does not
    list at all. That is what a replaced editor looks like, and is not a fault.
    """
    if not leaderboard:
        return {"checked": 0, "differs": [], "ranked_only_there": [], "ranked_only_here": [], "note": "no leaderboard to compare with"}
    rated = rated_rounds or int(leaderboard.get("rated_rounds") or 3)
    ours, _ = fit([(r["index"], r["editor"], r["public"], r["valid"]) for r in rows])
    labels = {r["editor"]: r["label"] for r in rows}
    board = (leaderboard.get("editors") or []) + (leaderboard.get("provisional") or [])
    theirs = {e["id"]: e for e in board if e.get("rating") is not None}
    name = lambda e: f"{e.get('owner')}/{e.get('name')} v{e.get('version')}"  # noqa: E731
    differs = [
        {"editor": i, "label": name(e), "leaderboard": e["rating"], "here": None if i not in ours else round(ours[i]["rating"], 5)}
        for i, e in theirs.items()
        if i not in ours or abs(ours[i]["rating"] - e["rating"]) > TOLERANCE
    ]
    here = {i for i, r in ours.items() if r["rounds"] >= rated}
    there = {e["id"]: e for e in leaderboard.get("editors") or []}
    listed = {e["id"] for e in board}
    return {
        "checked": len(theirs),
        "differs": differs,
        "ranked_only_there": [{"editor": i, "label": name(e), "rounds_here": ours[i]["rounds"] if i in ours else 0} for i, e in there.items() if i not in here],
        "ranked_only_here": [{"editor": i, "label": labels[i], "rounds": ours[i]["rounds"]} for i in sorted(here - listed)],
    }


def report(log_path: Path, source: dict, also_paths: list[Path] = (), rated_rounds: int | None = None) -> dict:
    rounds = source["rounds"]
    board = source.get("leaderboard")
    rated = rated_rounds or int((board or {}).get("rated_rounds") or 3)
    also = load_also(list(also_paths))
    rows, missing = entries(rounds, load(log_path), also)
    lines = table(rows, rated, board)
    check = check_public(rows, board, rated)
    # The check is of the arithmetic, on every public entry. The column in the table leaves
    # out the entries below, so for their editors it is not the leaderboard's number.
    check["entries_left_out"] = len(missing)
    check["editors_not_whole"] = sorted({m["label"] for m in missing})
    return {
        "season": source.get("season"),
        "record_sha256": hashlib.sha256(Path(log_path).read_bytes()).hexdigest(),
        "rounds": sorted({r["round"] for r in rows}),
        "open_rounds": [r["round"]["id"] for r in rounds if not counts(r["round"])],
        "rated_rounds": rated,
        "table": lines,
        "median_gives_up": median_gives_up(lines),
        "missing": missing,
        "public_check": check,
        "also_unused": sorted(a["file"] for a in also.values() if not a["used"]),
        # Everything the table was computed from. With these rows and `fit`, the table can
        # be worked out again by anyone; each row says where its held-out losses came from.
        "entries": rows,
    }


def status(rep: dict) -> int:
    """0: the table is whole and its public column is the leaderboard's. 1: the public
    arithmetic does not reproduce the leaderboard. 2: it does, but entries are missing."""
    check = rep["public_check"]
    if check.get("differs") or check.get("ranked_only_there"):
        return 1
    return 2 if rep["missing"] else 0


def show(rep: dict, out=None) -> None:
    out = out or sys.stdout
    w = max([len(t["label"]) for t in rep["table"]] + [6])
    print(f"{'editor':{w}s} {'rounds':>6s} {'public':>9s} {'held-out':>9s} {'':7s} {'gives up':>9s} {'beyond the median':>18s}", file=out)
    headings = {"provisional": f"-- fewer than {rep['rated_rounds']} rounds: not ranked --", "replaced": "-- replaced by their owners: not ranked, and not in the median --"}
    section = "ranked"
    for t in rep["table"]:
        if t["status"] != section:
            section = t["status"]
            print(headings[section], file=out)
        se = f"±{t['heldout_se']:.3f}" if t["heldout_se"] is not None else ""
        note = f"  ({t['invalid']} invalid)" if t["invalid"] else ""
        gave = f"{t['gives_up']:+9.4f} {t['beyond_median']:+18.4f}" if t["gives_up"] is not None else f"{'':9s} {'':18s}"
        print(f"{t['label']:{w}s} {t['rounds']:6d} {t['public']:+9.4f} {t['heldout']:+9.4f} {se:7s} {gave}{note}", file=out)
    n = len(rep["rounds"])
    print(f"\n{n} settled round{'' if n == 1 else 's'}.", end=" ", file=out)
    if any(t["gives_up"] is not None and t["status"] != "replaced" for t in rep["table"]):
        print(f"Editors give up {rep['median_gives_up']:+.4f} at the median, over the rounds their entries were valid in: that much is the two exams, not the editors.", file=out)
    else:
        print("No valid entry could be placed on the held-out exam.", file=out)
    if rep["open_rounds"]:
        print(f"Not settled, so not counted on either exam: {', '.join(rep['open_rounds'])}.", file=out)
    check = rep["public_check"]
    gaps = len(rep["missing"])
    if check.get("note"):
        print(f"The public column was not checked: {check['note']}.", file=out)
    elif check["differs"] or check["ranked_only_there"]:
        print(f"\nTHE PUBLIC COLUMN DOES NOT REPRODUCE THE LEADERBOARD ({check['checked']} editors compared):", file=out)
        for d in check["differs"]:
            print(f"  {d['label']}: the leaderboard says {d['leaderboard']:+.5f}, this says {'nothing' if d['here'] is None else format(d['here'], '+.5f')}", file=out)
        for d in check["ranked_only_there"]:
            print(f"  {d['label']}: ranked on the leaderboard, and has {d['rounds_here']} rounds here, fewer than the {rep['rated_rounds']} it takes", file=out)
        print("  (A round that is open, or entries that settled since the rounds were read, would do this. Otherwise the arithmetic has drifted from server/rating.go.)", file=out)
    elif gaps:
        print(
            f"\nTHE TABLE IS NOT WHOLE. Counting every public entry, this tool's arithmetic gives the leaderboard's rating for all {check['checked']} editors on it. "
            f"But {gaps} valid entr{'y' if gaps == 1 else 'ies'} could not be placed on the held-out exam and {'is' if gaps == 1 else 'are'} left out of both columns above, "
            f"so the public column is not the leaderboard's for: {', '.join(check['editors_not_whole'])}.",
            file=out,
        )
    else:
        print(f"The public column reproduces the leaderboard's rating for all {check['checked']} editors on it.", file=out)
    if not check.get("note") and check["ranked_only_here"]:
        names = ", ".join(d["label"] for d in check["ranked_only_here"])
        print(f"Not on the leaderboard, though they have the rounds: {names}. An editor its owner replaced keeps its results and loses its rank.", file=out)
    if gaps:
        print(f"\n{gaps} valid entr{'y' if gaps == 1 else 'ies'} could not be placed:", file=out)
        for m in rep["missing"]:
            print(f"  {m['round']} {m['label']}: {m['why']}\n      {m['measure']}", file=out)
        print("  Run each on a machine that holds the pack, save what it prints, and pass the files with --also.", file=out)
    for unused in rep["also_unused"]:
        print(f"\nWARNING: {unused} was given with --also and used for nothing: no entry that needed it has that round, script and public loss.", file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="The season's table on the held-out exam, from the record of a machine that holds it.")
    ap.add_argument("log", type=Path, help="the record, ~/.config/stet/heldout/<season>.jsonl")
    ap.add_argument("--season", default="s0")
    ap.add_argument("--api", default=DEFAULT_API, help=f"where to read the public rounds from (default {DEFAULT_API})")
    ap.add_argument("--rounds-file", type=Path, help="read the rounds from a file written by --save-rounds instead of from the API")
    ap.add_argument("--save-rounds", type=Path, help="keep what was read from the API, to publish beside the table")
    ap.add_argument("--also", type=Path, nargs="*", default=[], help="results of `stet score <sha> --round <id> --heldout --json`, for entries the record lacks")
    ap.add_argument("--rated-rounds", type=int, help="rounds an editor needs to be ranked when there is no leaderboard to ask (default: what the leaderboard says, else 3)")
    ap.add_argument("--json", action="store_true", help="print the whole report as JSON")
    ap.add_argument("--out", type=Path, help="also write the report, as JSON, to this file")
    args = ap.parse_args(argv)
    source = json.loads(args.rounds_file.read_text()) if args.rounds_file else fetch(args.api, args.season)
    if args.save_rounds:
        args.save_rounds.write_text(json.dumps(source, indent=1))
    rep = report(args.log, source, args.also, args.rated_rounds)
    if args.out:
        args.out.write_text(json.dumps(rep, indent=1))
    if args.json:
        print(json.dumps(rep, indent=1))
        for unused in rep["also_unused"]:
            print(f"WARNING: {unused} was given with --also and used for nothing", file=sys.stderr)
    else:
        show(rep)
    return status(rep)


if __name__ == "__main__":
    sys.exit(main())

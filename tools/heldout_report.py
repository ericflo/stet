"""The season's table on the held-out exam: the public leaderboard, worked out again on held-out losses.

    python tools/heldout_report.py ~/.config/stet/heldout/s0.jsonl
    python tools/heldout_report.py s0.jsonl --rounds-file rounds.json     # offline, from saved rounds
    python tools/heldout_report.py s0.jsonl --save-rounds rounds.json --out final.json

Same editors, same rounds, same arithmetic. The rounds and their entries are the public
ones, read from the API. For every entry that counts on the public leaderboard:

  - an entry that was void (the anchor could not run it) is left out, as it is publicly;
  - an invalid entry scores a gain of 0 on both exams, as it does publicly;
  - a valid entry's held-out gain is the round's unedited held-out loss minus its own,
    from the anchor's record: one line per official grade, keyed by round and editor.

The rating is then fitted exactly as the server fits it (`fit` below is fitRatings in
server/rating.go): gain = editor + round, so an editor is not flattered by easy rounds. The
public column is checked against the live leaderboard, and the tool says so if it differs.

A valid entry with no line in the record (or a line with no held-out baseline) cannot be
placed. It is listed, with the command that measures it again on a machine holding the
pack, and that round is left out of *both* columns for that editor, so the two stay
comparable. `--also` takes the JSON that command prints and fills the gap.

How to read "gives up". The two exams are different passages, so every editor's held-out
rating differs from its public one by some common amount; that amount says nothing about
anyone. An editor that learned the kind of writing gives up about what the others give up.
An editor that fitted the public exam's own passages gives up more than the others do: that
is the last column, each editor against the median.

When a season closes, the record and the held-out pack are published. Anyone can then
check a line: put the pack where STET_HELDOUT_EXAM points and run

    stet score <script sha256> --round <round id> --heldout --json

The result's `heldout.loss` and `heldout.baseline` are the line's numbers.
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


def load(path: Path) -> dict[tuple[str, str], dict]:
    """The anchor's record: the last line for each (round, editor).

    An anchor may grade an entry twice after a restart, and the later line is as good as
    the earlier. The key is the editor, never the script: two editors can write the same
    script in a round (the starting template does, for everyone who enters it unchanged),
    and each of them has that round.
    """
    last: dict[tuple[str, str], dict] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            last[(rec["round"], (rec.get("editor") or {}).get("id"))] = rec
    return last


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


def entries(rounds: list[dict], log: dict, also: dict | None = None) -> tuple[list[dict], list[dict]]:
    """(the entries that count, the ones that cannot be placed).

    Each entry that counts is {round, index, editor, label, valid, public, heldout}; the
    last is None where the record has nothing for it, and those are listed in the second
    value as well, with why.
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
            row = {"round": rd["id"], "index": int(rd["index"]), "editor": e["editor"], "label": label, "valid": bool(e.get("valid")), "public": 0.0, "heldout": 0.0}
            if row["valid"]:
                script = (e.get("script") or {}).get("sha256")
                rec = log.get((rd["id"], e["editor"]))
                if rec is not None and rec.get("script_sha256") != script:
                    rec = None  # a line for some other script: not this entry's
                if rec is None or (rec.get("heldout") or {}).get("baseline") is None:
                    rec = also.get((rd["id"], script), rec)
                held = (rec or {}).get("heldout") or {}
                why = ""
                if rec is None:
                    why = "the record has no line for it"
                elif held.get("loss") is None or held.get("baseline") is None:
                    why = "its line has no held-out baseline"
                elif abs(rec["public"]["loss"] - e["loss"]) > TOLERANCE:
                    why = f"its line's public loss ({rec['public']['loss']:.5f}) is not the published one ({e['loss']:.5f})"
                row["public"] = float(e["gain"])
                # Rounded as the server rounds a public gain, so the two columns are the same arithmetic.
                row["heldout"] = None if why else round(held["baseline"] - held["loss"], 5)
                if why:
                    missing.append({"round": rd["id"], "editor": e["editor"], "label": label, "script_sha256": script, "why": why,
                                    "measure": f"stet score {script} --round {rd['id']} --heldout --json"})  # fmt: skip
            rows.append(row)
    rows.sort(key=lambda x: (x["index"], x["editor"]))
    return rows, missing


def fit(rows: list[tuple[int, str, float, bool]]) -> tuple[dict[str, dict], dict[int, float]]:
    """fitRatings in server/rating.go, line for line. rows are (round, editor, gain, valid).

    gain = skill[editor] + difficulty[round], solved by alternating means. A round's
    difficulty is read from its valid entries only, and averages to zero over the season;
    an invalid entry counts as zero whatever its round was like.
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
            s = sum(adjusted(e) for e in es) / len(es)
            delta = max(delta, abs(s - skill.get(editor, 0.0)))
            skill[editor] = s
        mean = 0.0
        for rnd, es in by_round.items():
            measured = [e[2] - skill[e[1]] for e in es if e[3]]
            diff[rnd] = sum(measured) / len(measured) if measured else 0.0
            mean += diff[rnd]
        mean /= max(1, len(by_round))
        for rnd in diff:
            diff[rnd] -= mean
        if delta < 1e-9 and i > 0:
            break
    out = {}
    for editor, es in by_editor.items():
        resid = [adjusted(e) for e in es]
        out[editor] = {
            "rounds": len(es),
            "mean": sum(e[2] for e in es) / len(es),
            "rating": sum(resid) / len(resid),
            "se": statistics.stdev(resid) / math.sqrt(len(resid)) if len(resid) >= 2 else None,
            "invalid": sum(1 for e in es if not e[3]),
        }
    return out, diff


def table(rows: list[dict], rated_rounds: int = 3) -> list[dict]:
    """One line per editor: its public rating and its held-out rating over the same rounds."""
    placed = [r for r in rows if r["heldout"] is not None]
    public, _ = fit([(r["index"], r["editor"], r["public"], r["valid"]) for r in placed])
    heldout, _ = fit([(r["index"], r["editor"], r["heldout"], r["valid"]) for r in placed])
    labels = {r["editor"]: r["label"] for r in rows}
    out = []
    for editor, h in heldout.items():
        p = public[editor]
        out.append({
            "editor": editor, "label": labels[editor], "rounds": h["rounds"], "invalid": h["invalid"], "ranked": h["rounds"] >= rated_rounds,
            "public": round(p["rating"], 5), "heldout": round(h["rating"], 5), "heldout_se": None if h["se"] is None else round(h["se"], 5),
            "gives_up": round(p["rating"] - h["rating"], 5),
        })  # fmt: skip
    # What every editor gives up says something about the two exams. What one editor gives
    # up beyond that says something about the editor.
    usual = statistics.median([r["gives_up"] for r in out if r["rounds"] > r["invalid"]] or [0.0])
    for r in out:
        r["beyond_median"] = round(r["gives_up"] - usual, 5)
    return sorted(out, key=lambda r: (not r["ranked"], -r["heldout"], r["editor"]))


def check_public(rows: list[dict], leaderboard: dict | None) -> dict:
    """Does this tool's arithmetic, on public gains, give the leaderboard's own ratings?"""
    if not leaderboard:
        return {"checked": 0, "differs": [], "note": "no leaderboard to compare with"}
    ours, _ = fit([(r["index"], r["editor"], r["public"], r["valid"]) for r in rows])
    theirs = {e["id"]: e for e in (leaderboard.get("editors") or []) + (leaderboard.get("provisional") or []) if e.get("rating") is not None}
    differs = [
        {"editor": i, "label": f"{e.get('owner')}/{e.get('name')} v{e.get('version')}", "leaderboard": e["rating"], "here": None if i not in ours else round(ours[i]["rating"], 5)}
        for i, e in theirs.items()
        if i not in ours or abs(ours[i]["rating"] - e["rating"]) > TOLERANCE
    ]
    return {"checked": len(theirs), "differs": differs}


def report(log_path: Path, source: dict, also_paths: list[Path] = (), rated_rounds: int | None = None) -> dict:
    rounds = source["rounds"]
    board = source.get("leaderboard")
    rated = rated_rounds or int((board or {}).get("rated_rounds") or 3)
    rows, missing = entries(rounds, load(log_path), load_also(list(also_paths)))
    lines = table(rows, rated)
    return {
        "season": source.get("season"),
        "record_sha256": hashlib.sha256(Path(log_path).read_bytes()).hexdigest(),
        "rounds": sorted({r["round"] for r in rows}),
        "open_rounds": [r["round"]["id"] for r in rounds if not counts(r["round"])],
        "rated_rounds": rated,
        "table": lines,
        "median_gives_up": statistics.median([t["gives_up"] for t in lines if t["rounds"] > t["invalid"]] or [0.0]),
        "missing": missing,
        "public_check": check_public(rows, board),
    }


def show(rep: dict, out=None) -> None:
    out = out or sys.stdout
    w = max([len(t["label"]) for t in rep["table"]] + [6])
    print(f"{'editor':{w}s} {'rounds':>6s} {'public':>9s} {'held-out':>9s} {'':7s} {'gives up':>9s} {'beyond the median':>18s}", file=out)
    marked = False
    for t in rep["table"]:
        if not t["ranked"] and not marked:
            marked = True
            print(f"-- fewer than {rep['rated_rounds']} rounds: not ranked --", file=out)
        se = f"±{t['heldout_se']:.3f}" if t["heldout_se"] is not None else ""
        note = f"  ({t['invalid']} invalid)" if t["invalid"] else ""
        print(f"{t['label']:{w}s} {t['rounds']:6d} {t['public']:+9.4f} {t['heldout']:+9.4f} {se:7s} {t['gives_up']:+9.4f} {t['beyond_median']:+18.4f}{note}", file=out)
    n = len(rep["rounds"])
    print(f"\n{n} settled round{'' if n == 1 else 's'}.", end=" ", file=out)
    if any(t["rounds"] > t["invalid"] for t in rep["table"]):
        print(f"Editors give up {rep['median_gives_up']:+.4f} at the median: that much is the two exams, not the editors.", file=out)
    else:
        print("No valid entry could be placed on the held-out exam.", file=out)
    if rep["open_rounds"]:
        print(f"Not settled, so not counted on either exam: {', '.join(rep['open_rounds'])}.", file=out)
    check = rep["public_check"]
    if check.get("note"):
        print(f"The public column was not checked: {check['note']}.", file=out)
    elif check["differs"]:
        print(f"\nTHE PUBLIC COLUMN DOES NOT REPRODUCE THE LEADERBOARD for {len(check['differs'])} of {check['checked']} editors:", file=out)
        for d in check["differs"]:
            print(f"  {d['label']}: the leaderboard says {d['leaderboard']:+.5f}, this says {'nothing' if d['here'] is None else format(d['here'], '+.5f')}", file=out)
        print("  (A round that is open, or entries that settled since the rounds were read, would do this. Otherwise the arithmetic has drifted from server/rating.go.)", file=out)
    else:
        print(f"The public column reproduces the leaderboard's rating for all {check['checked']} editors on it.", file=out)
    if rep["missing"]:
        print(f"\n{len(rep['missing'])} valid entries could not be placed, and are left out of both columns:", file=out)
        for m in rep["missing"]:
            print(f"  {m['round']} {m['label']}: {m['why']}\n      {m['measure']}", file=out)
        print("  Run each on a machine that holds the pack, save what it prints, and pass the files with --also.", file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="The season's table on the held-out exam, from an anchor's record of it.")
    ap.add_argument("log", type=Path, help="the anchor's record, ~/.config/stet/heldout/<season>.jsonl")
    ap.add_argument("--season", default="s0")
    ap.add_argument("--api", default=DEFAULT_API, help=f"where to read the public rounds from (default {DEFAULT_API})")
    ap.add_argument("--rounds-file", type=Path, help="read the rounds from a file written by --save-rounds instead of from the API")
    ap.add_argument("--save-rounds", type=Path, help="keep what was read from the API, to publish beside the table")
    ap.add_argument("--also", type=Path, nargs="*", default=[], help="results of `stet score <sha> --round <id> --heldout --json`, for entries the record lacks")
    ap.add_argument("--rated-rounds", type=int, help="rounds an editor needs to be ranked (default: what the leaderboard says, else 3)")
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
    else:
        show(rep)
    return 1 if rep["public_check"].get("differs") else 0


if __name__ == "__main__":
    sys.exit(main())

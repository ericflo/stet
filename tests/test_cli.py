"""`stet run --sandbox` runs what `stet submit` would send, not the directory it came from."""

import sys
import types

import pytest
from test_project import make

from stet import cli


class Args:
    round = "0"
    season = "s2"
    scale = None
    device = "cpu"
    rounds = 1
    quiet = True
    verbose = False
    json = False
    no_baseline = True
    sandbox = True

    def __init__(self, path):
        self.path = str(path)


@pytest.fixture
def ran(monkeypatch):
    seen = types.SimpleNamespace(bundle=None, files=None)

    def run_round(bundle, entry, rnd, scale, work, sandbox="none", **kw):
        seen.bundle, seen.sandbox = bundle, sandbox
        seen.files = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file())
        return {"valid": True}

    monkeypatch.setitem(sys.modules, "stet.pipeline", types.SimpleNamespace(run_round=run_round))
    monkeypatch.setitem(sys.modules, "stet.train", types.SimpleNamespace(pick_device=lambda d: types.SimpleNamespace(type="cpu")))
    for name in ("need_torch", "_with_baseline", "report", "_emit"):
        monkeypatch.setattr(cli, name, lambda *a, **k: None)
    return seen


def test_a_sandboxed_run_sees_the_bundle_and_nothing_the_bundle_leaves_out(ran, tmp_path):
    p = make(tmp_path, {".env": "API_KEY=secret", ".cache/model.bin": "w", "venv/lib/dep.py": "x", "weights/model.bin": "w"})
    with pytest.raises(SystemExit) as e:
        cli.cmd_run(Args(p.path))
    assert e.value.code == 0 and ran.sandbox == "bwrap"
    assert ran.files == ["curator.py", "stet.toml", "weights/model.bin"]  # what an anchor would unpack
    assert ran.bundle != p.path and not ran.bundle.exists()  # and the copy is not kept
    args = Args(p.path)
    args.sandbox = False
    with pytest.raises(SystemExit):
        cli.cmd_run(args)
    assert ran.bundle == p.path and ran.sandbox == "none"


# --- `stet score` takes a script in every form the site hands one out ------------------------


def _score_setup(monkeypatch, tmp_path, entries, blobs):
    import hashlib

    import zstandard

    from stet import client
    from stet.spec import load_season

    monkeypatch.setenv("STET_CACHE", str(tmp_path / "cache"))
    raw = b'1\n{"d": 2, "k": [[0, 900]]}\n'
    sha = hashlib.sha256(raw).hexdigest()
    packed = zstandard.ZstdCompressor().compress(raw)
    blobs[sha] = packed
    asked = []

    def request(method, path, body=None, key=None, raw=False, timeout=120, limit=0):
        asked.append(path)
        if path.startswith("/api/v1/rounds/"):
            return {"entries": entries(sha)}
        return blobs[path.rsplit("/", 1)[1]]

    monkeypatch.setattr(client, "request", request)
    official = types.SimpleNamespace(kind="official", id="s2-o00001")
    practice = types.SimpleNamespace(kind="practice", id="s2-p00000")
    return load_season("s2"), official, practice, raw, sha, packed, asked


def test_score_takes_a_file_a_compressed_file_a_hash_or_an_entry(monkeypatch, tmp_path):
    entries = lambda sha: [  # noqa: E731
        {"name": "tidy", "owner": "stet", "script": {"sha256": sha}},
        {"name": "english", "owner": "stet", "script": {"sha256": "0" * 64}},
        {"name": "slow", "owner": "ann"},  # still editing: no script yet
    ]
    season, official, practice, raw, sha, packed, asked = _score_setup(monkeypatch, tmp_path, entries, {})
    plain = tmp_path / "script.jsonl"
    plain.write_bytes(raw)
    assert cli.script_file(str(plain), season, practice) == plain  # a file is used where it lies
    assert not asked

    zst = tmp_path / "downloaded.zst"
    zst.write_bytes(packed)  # what the "script" link on a round page downloads
    assert cli.script_file(str(zst), season, practice).read_bytes() == raw
    assert not asked

    assert cli.script_file(sha.upper(), season, official).read_bytes() == raw
    assert cli.script_file("tidy", season, official).read_bytes() == raw
    assert cli.script_file("stet/tidy", season, official).read_bytes() == raw
    assert asked.count("/api/v1/rounds/s2-o00001") == 2


def test_score_refuses_a_script_that_is_not_the_one_asked_for(monkeypatch, tmp_path, capsys):
    import zstandard

    entries = lambda sha: [{"name": "tidy", "owner": "stet", "script": {"sha256": sha}}, {"name": "tidy", "owner": "eve", "script": {"sha256": sha}}]  # noqa: E731
    blobs = {}
    season, official, practice, raw, sha, packed, asked = _score_setup(monkeypatch, tmp_path, entries, blobs)
    blobs[sha] = zstandard.ZstdCompressor().compress(b"99\n")  # the server sends something else
    with pytest.raises(SystemExit):
        cli.script_file(sha, season, official)
    assert "does not match its hash" in capsys.readouterr().err
    blobs[sha] = packed[:-4]  # or something cut short
    with pytest.raises(SystemExit):
        cli.script_file(sha, season, official)
    assert "damaged or cut short" in capsys.readouterr().err
    # A published script edits one official round's corpus. On a practice round its numbers
    # would point at other documents, so it is not graded there.
    asked.clear()
    with pytest.raises(SystemExit):
        cli.script_file(sha, season, practice)
    assert "--round <official round id>" in capsys.readouterr().err and not asked
    # A file on disk that is not a whole zstd file is said to be one, not graded as its beginning.
    half = tmp_path / "half.zst"
    half.write_bytes(packed[:-4])
    with pytest.raises(SystemExit):
        cli.script_file(str(half), season, practice)
    assert "damaged or cut short" in capsys.readouterr().err
    two = tmp_path / "two.zst"
    two.write_bytes(zstandard.ZstdCompressor().compress(raw[:2]) + zstandard.ZstdCompressor().compress(raw[2:]))
    assert cli.script_file(str(two), season, practice).read_bytes() == raw  # every frame, not the first alone
    with pytest.raises(SystemExit):  # two entries called tidy: say so, do not pick one
        cli.script_file("tidy", season, official)
    assert "more than one" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # a name means nothing outside an official round
        cli.script_file("tidy", season, practice)
    assert "no file called" in capsys.readouterr().err


def test_score_tells_two_versions_of_one_editor_apart(monkeypatch, tmp_path, capsys):
    import hashlib

    import zstandard

    # A rated curator stays on the ladder while its owner's next version qualifies, so the
    # two share every round for a while, and the name alone no longer says which.
    other_raw = b'1\n{"d": 3, "k": [[0, 500]]}\n'
    other = hashlib.sha256(other_raw).hexdigest()
    entries = lambda sha: [  # noqa: E731
        {"name": "prose", "owner": "alice", "version": 1, "script": {"sha256": sha}},
        {"name": "prose", "owner": "alice", "version": 2, "script": {"sha256": other}},
        {"name": "prose", "owner": "bob", "version": 7, "script": {"sha256": "0" * 64}},
        {"name": "prose", "owner": "carol", "version": 3},  # no script yet: not something to grade
    ]
    blobs = {other: zstandard.ZstdCompressor().compress(other_raw)}
    season, official, practice, raw, sha, packed, asked = _score_setup(monkeypatch, tmp_path, entries, blobs)
    assert cli.script_file("alice/prose@v1", season, official).read_bytes() == raw
    assert cli.script_file("alice/prose@2", season, official).read_bytes() == other_raw
    assert cli.script_file("alice/prose@V2", season, official).read_bytes() == other_raw
    with pytest.raises(KeyError):  # this test's server does not hold bob's script
        cli.script_file("prose@v7", season, official)
    assert asked[-1] == "/api/v1/scripts/" + "0" * 64  # but it was bob's that was asked for: his alone is a version 7
    for ref in ("alice/prose", "prose"):
        with pytest.raises(SystemExit):
            cli.script_file(ref, season, official)
        err = capsys.readouterr().err
        # It names each with its version, and the command that picks one.
        assert "more than one entry called 'prose'" in err and "alice/prose@v1, alice/prose@v2" in err and "stet score alice/prose@v1 --round s2-o00001" in err
        assert ("bob/prose@v7" in err) == (ref == "prose") and "carol" not in err
    with pytest.raises(SystemExit):
        cli.script_file("alice/prose@v9", season, official)
    err = capsys.readouterr().err
    assert "no entry of that name" in err and "alice/prose@v2" in err and "bob/prose@v7" in err
    with pytest.raises(SystemExit):
        cli.script_file("alice/prose@latest", season, official)
    assert "a version is written like alice/prose@v2" in capsys.readouterr().err


def test_the_gain_line_says_what_its_noise_figure_is(capsys):
    from stet.spec import load_season

    season = load_season("s2")
    result = {
        "round": {"id": "s2-p00000", "kind": "practice"}, "scale": "nano", "valid": True,
        "script": {"out_docs": 1, "src_docs": 1, "kept_doc_frac": 0.1, "out_bytes": 10, "passes": 1.0, "truncated": False},
        "train": {"tokens": 10, "seconds": 1.0, "tokens_per_second": 10, "final_train_loss": 3.0},
        "exam": {"items": 3, "mean": -2.5, "item_scores": [-2.4, -2.5, -2.6]}, "gain": 0.1, "pillars": {"logic": 0.1},
    }  # fmt: skip
    cli.report(cli.Out(), result, {"exam": {"mean": -2.6}}, season)
    err = capsys.readouterr().err
    # The same script graded twice on one machine gives the same number: this is what training again does.
    assert "+0.1000 bits per byte" in err and "logic +0.100" in err and "training again moves this by about" not in err
    cli.report(cli.Out(), {**result, "scale": "standard"}, {"exam": {"mean": -2.6}}, season)
    assert "training again moves this by about ±0.005, another shuffle of the same pages by about ±0.008" in capsys.readouterr().err


# --- what a run writes down -------------------------------------------------------------------


def test_an_invalid_run_has_a_gain_of_nothing_written_down(monkeypatch):
    out = cli.Out(quiet=True)
    result = {"valid": False, "invalid_reason": "the curator produced no script (timeout)"}
    assert cli._with_baseline(result, None, None, None, out, skip=True) is None and result["gain"] == 0.0
    monkeypatch.setitem(sys.modules, "stet.pipeline", types.SimpleNamespace(baseline=lambda *a, **k: {"exam": {"mean": -5.3, "item_scores": [-5.3]}}))
    result = {"valid": False, "invalid_reason": "line 3: this line holds text; a script is copies of the crawl and holds no text of its own"}
    cli._with_baseline(result, None, None, None, out, skip=False)
    assert result["gain"] == 0.0 and result["baseline"]["mean"] == -5.3


# --- the examiner ---------------------------------------------------------------------------------


def test_the_examiner_harvests_the_newest_settled_hour_first_and_makes_an_exam_from_the_hours_the_server_chose(monkeypatch, tmp_path):
    import hashlib

    from stet.exam.fresh.harvest import choose_hours
    from stet.spec import load_season

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    season = load_season("s2")
    now = 1_800_000_000 + 1234
    last = (now - 2 * 3600) - (now - 2 * 3600) % 3600  # ended an hour ago at least
    assert cli._harvest_due(season, now) == last
    hours = cli._hours_dir()
    hours.mkdir(parents=True)
    (hours / f"{last}.jsonl").write_text("{}\n")
    assert cli._harvest_due(season, now) == last - 3600  # then the one before
    assert cli._harvest_due(season, now, skip=[last - 3600]) == last - 7200  # an hour that will not come is passed over
    reach = season.raw["exam"]["fresh"]["reach_seconds"]
    for h in range(last, now - reach - 3600, -3600):
        (hours / f"{h}.jsonl").write_text(f"{{}}\n{h}\n")
    assert cli._harvest_due(season, now) is None  # nothing a round could still be given is missing

    def sha(h):
        return hashlib.sha256((hours / f"{h}.jsonl").read_bytes()).hexdigest()

    # The server named the round's hours at its draw: those are used, and checked against what it was told.
    d = {"draw_at": now, "exam_hours": [{"hour": last, "sha256": sha(last)}, {"hour": last - 7200, "sha256": sha(last - 7200)}]}
    assert cli._round_hours(season, d, b"seed") == ([(last, hours / f"{last}.jsonl"), (last - 7200, hours / f"{last - 7200}.jsonl")], False)
    d["exam_hours"][0]["sha256"] = "00" * 32
    with pytest.raises(RuntimeError, match="not the one the server was told of"):
        cli._round_hours(season, d, b"seed")
    with pytest.raises(RuntimeError, match="not on this machine"):
        cli._round_hours(season, {"draw_at": now, "exam_hours": [{"hour": last + 36000, "sha256": "00" * 32}]}, b"seed")
    # It had been told of none: they are chosen now, by the same rule, from what this machine holds of the round's day.
    got, late = cli._round_hours(season, {"draw_at": now}, b"seed")
    held = [int(p.stem) for p in hours.glob("*.jsonl") if int(p.stem) >= now - reach]
    assert late and [h for h, _ in got] == choose_hours(b"seed", held, season.raw["exam"]["fresh"]["hours"]) and len(got) == 3


def test_harvest_holds_hours_spread_over_the_last_day_and_only_asks_for_what_is_missing(monkeypatch, tmp_path, capsys):
    from stet import cli, data
    from stet.exam.fresh import Doc, harvest, write_snapshot

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    now = 1_800_000_000 + 30 * 3600 + 120
    monkeypatch.setattr(cli.time, "time", lambda: now)
    asked = []

    def fake(season, hour, dest, randomness=None, log=print):
        asked.append(hour)
        if hour == broken:
            raise OSError("the beacon is out of reach")
        docs = [Doc(source=src, id=f"{hour}:{i}", url="https://example.org/", lang="en", license="", published=hour, title="T", text="Words. " * 30)
                for src in ("wikipedia", "github") for i in range(3)]
        return {"documents": len(docs)}, write_snapshot(docs, dest, {"window": [hour, hour + 3600]})

    monkeypatch.setattr(harvest, "harvest_hour", fake)
    broken = None
    cli.main(["harvest"])
    from stet.spec import CURRENT_SEASON, load_season

    last = harvest.settled(load_season(CURRENT_SEASON), now)
    assert asked == [last, last - 8 * 3600, last - 16 * 3600] and sorted(data.harvested()) == sorted(asked)
    text = "".join(capsys.readouterr())
    assert "reference 3, technical 3" in text and "3 hours held" in text
    # Asked again, it has what it wants; asked for more, it fetches only those.
    asked.clear()
    cli.main(["harvest"])
    assert asked == []
    broken = last - 4 * 3600
    cli.main(["harvest", "--hours", "6"])
    assert broken in asked and broken not in data.harvested() and len(data.harvested()) == 5 and not list(data.harvest_dir().glob("*.part"))
    assert "could not be harvested" in "".join(capsys.readouterr())
    # --fresh starts again.
    asked.clear()
    broken = None
    cli.main(["harvest", "--fresh", "--hours", "1"])
    assert asked == [last] and sorted(data.harvested()) == [last]


def test_the_examiner_never_gives_a_round_up_and_tries_less_often(monkeypatch):
    from stet import cli

    waits = [cli._again_in(n) for n in range(1, 40)]
    assert waits[:6] == [20, 40, 80, 160, 320, 600]
    assert max(waits) == 600 and waits == sorted(waits)  # ten minutes at most, for as long as the round is open


def test_an_hour_the_server_was_not_told_of_is_told_of_later(monkeypatch, tmp_path):
    from stet import cli, client
    from stet.spec import load_season

    season = load_season("s2")
    monkeypatch.setattr(cli, "_hours_dir", lambda: tmp_path)
    now = 1_800_000_000
    hours = [now - now % 3600 - 3600 * i for i in (2, 3, 4)]
    for h in hours:
        (tmp_path / f"{h}.jsonl").write_text("{}\n")
    (tmp_path / f"{hours[0] - 40 * 3600}.jsonl").write_text("{}\n")  # too old for any round still to come
    (tmp_path / f"{hours[0] + 3600}.part.jsonl").write_text("")  # one being written
    asked, told = [], []

    def request(method, path, body=None, **kw):
        asked.append((method, path))
        return {"harvests": [{"hour": hours[1], "sha256": "0" * 64}]}

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(cli, "_commit_hour", lambda hour, path, key: told.append((hour, path.name)))
    assert cli._commit_missed(season, object(), now) == sorted([hours[0], hours[2]])
    assert told == [(h, f"{h}.jsonl") for h in sorted([hours[0], hours[2]])]
    assert asked == [("GET", "/api/v1/harvests?limit=72")]


def test_a_harvest_made_while_cut_off_is_not_kept_or_told_of(monkeypatch, tmp_path):
    import pytest

    from stet import cli
    from stet.exam.fresh import Doc, harvest, write_snapshot
    from stet.spec import load_season

    season = load_season("s2")
    monkeypatch.setattr(cli, "_hours_dir", lambda: tmp_path)
    told = []
    monkeypatch.setattr(cli, "_commit_hour", lambda hour, path, key: told.append(hour) or {"hour": hour, "sha256": "0" * 64, "documents": 0})
    one = dict(url="https://example.org/x", lang="en", license="CC0", published=1, title="t", text="words")

    def gives(sources):
        def harvest_hour(season, hour, dest, log=print):
            docs = [Doc(source=s, id=str(i), **one) for i, s in enumerate(sources)]
            return {"documents": len(docs)}, write_snapshot(docs, dest, {"window": [hour, hour + 3600]})

        monkeypatch.setattr(harvest, "harvest_hour", harvest_hour)

    hour = 1_800_000_000 - 1_800_000_000 % 3600
    for cut_off in ([], ["wikipedia", "wikipedia-talk", "wikipedia-added"], ["wikipedia", "ccnews"]):
        gives(cut_off)
        with pytest.raises(RuntimeError, match="is this machine online"):
            cli._harvest_one(season, hour, object(), print)
        assert not list(tmp_path.glob("*.jsonl")) and not told  # nothing kept, so the hour is harvested again
    gives(["wikipedia", "ccnews", "hackernews", "bluesky-reply"])
    assert cli._harvest_one(season, hour, object(), print)["hour"] == hour
    assert told == [hour] and (tmp_path / f"{hour}.jsonl").exists()

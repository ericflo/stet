"""The parts of the client that stand between a verifier's GPU time and the leaderboard."""

import os

import pytest
import zstandard

from stet import client


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

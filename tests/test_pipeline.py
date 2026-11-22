"""The baseline cache."""

import json
import os
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")


@pytest.fixture
def graded(monkeypatch):
    """`baseline` with the corpus and the training run replaced. Returns what was graded."""
    from stet import data, pipeline

    seen = types.SimpleNamespace(scripts=[], valid=True)
    seen.rnd = types.SimpleNamespace(id="s2-o00007", kind="official", exam_digest="ab" * 32)
    seen.scale, seen.device = types.SimpleNamespace(name="micro"), types.SimpleNamespace(type="cpu")
    data.round_dir(seen.rnd, seen.scale).mkdir(parents=True)

    def train_script(rnd, scale, corpus, script, device=None, progress=None, log=None):
        assert Path(script).read_text() == "0\n1\n2\n"  # every document, in corpus order
        seen.scripts.append(Path(script).name)
        return types.SimpleNamespace(device=device)

    def sit_exam(trained, rnd, scale, exam):
        return {"valid": seen.valid, "invalid_reason": None if seen.valid else "ink over budget",
                "exam": {"items": 2, "mean": -5.3, "item_scores": [-5.2, -5.4]}}

    monkeypatch.setattr(data, "corpus", lambda rnd, scale: [b"a", b"b", b"c"])
    monkeypatch.setattr(data, "round_exam", lambda rnd, scale: [])
    monkeypatch.setattr(pipeline, "train_script", train_script)
    monkeypatch.setattr(pipeline, "sit_exam", sit_exam)
    seen.run = lambda: pipeline.baseline(seen.rnd, seen.scale, seen.device)
    seen.files = lambda: sorted(p.name for p in data.round_dir(seen.rnd, seen.scale).iterdir())
    seen.path = data.round_dir(seen.rnd, seen.scale) / f"baseline-{pipeline.HARNESS_VERSION}-cpu-{'ab' * 6}.json"
    return seen


def test_a_baseline_is_measured_once_and_again_only_when_it_has_to_be(graded):
    assert graded.run()["exam"]["mean"] == -5.3 and graded.run()["exam"]["mean"] == -5.3
    assert graded.scripts == [f"identity-{os.getpid()}.jsonl"]  # once; and named for this process, which may not be alone here
    assert graded.files() == [graded.path.name]  # nothing left beside it: no script, no half-written file

    # A file cut short (the disk filled as it was saved) is no baseline. It is measured again and replaced.
    for damaged in ('{"valid": true, "exam": {"me', "", "[]", '{"valid": false}'):
        graded.path.write_text(damaged)
        assert graded.run()["exam"]["mean"] == -5.3
        assert json.loads(graded.path.read_text())["valid"] is True
    assert len(graded.scripts) == 5 and graded.files() == [graded.path.name]


def test_a_baseline_is_for_the_exam_it_sat(graded):
    graded.run()
    # The same round with another exam is measured again.
    graded.rnd.exam_digest = "cd" * 32
    graded.run()
    assert len(graded.scripts) == 2 and len(graded.files()) == 2


def test_a_baseline_that_cannot_be_saved_leaves_nothing_half_written(graded, monkeypatch):
    from stet import pipeline

    real = os.replace

    def replace(src, dst, **kw):
        if "baseline-" in str(dst):
            raise OSError(28, "No space left on device")
        return real(src, dst, **kw)

    monkeypatch.setattr(pipeline.os, "replace", replace)
    with pytest.raises(OSError, match="No space left"):
        graded.run()
    assert graded.files() == []  # not the baseline, and not the file it was being written to


def test_a_baseline_that_cannot_be_graded_is_not_kept(graded):
    graded.valid = False
    with pytest.raises(RuntimeError, match="could not be graded"):
        graded.run()
    assert graded.files() == []

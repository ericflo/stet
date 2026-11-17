"""The baseline cache, and where the held-out exam sits in a grade."""

import json
import os
import stat
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")


@pytest.fixture
def graded(monkeypatch):
    """`baseline` with the corpus and the training run replaced. Returns what was graded."""
    from stet import data, pipeline

    seen = types.SimpleNamespace(scripts=[], pack=False, valid=True)
    seen.rnd, seen.scale, seen.device = types.SimpleNamespace(id="s0-o00007", kind="official"), types.SimpleNamespace(name="micro"), types.SimpleNamespace(type="cpu")
    data.round_dir(seen.rnd, seen.scale).mkdir(parents=True)

    def grade(rnd, scale, corpus, exam, script, device=None, progress=None, log=None):
        assert Path(script).read_text() == "0\n1\n2\n"  # every document, in corpus order
        seen.scripts.append(Path(script).name)
        result = {"valid": seen.valid, "invalid_reason": None if seen.valid else "ink over budget", "exam": {"loss": 5.3}}
        if seen.pack:
            result["heldout"] = {"loss": 5.56789}
        return result

    monkeypatch.setattr(data, "materialize", lambda rnd, scale: ([b"a", b"b", b"c"], {}))
    monkeypatch.setattr(data, "heldout_exam", lambda rnd, scale: {"news": []} if seen.pack else None)
    monkeypatch.setattr(pipeline, "grade", grade)
    seen.run = lambda: pipeline.baseline(seen.rnd, seen.scale, seen.device)
    seen.files = lambda: sorted(p.name for p in data.round_dir(seen.rnd, seen.scale).iterdir())
    seen.path = data.round_dir(seen.rnd, seen.scale) / f"baseline-{pipeline.HARNESS_VERSION}-cpu.json"
    return seen


def test_a_baseline_is_measured_once_and_again_only_when_it_has_to_be(graded):
    assert graded.run()["exam"]["loss"] == 5.3 and graded.run()["exam"]["loss"] == 5.3
    assert graded.scripts == [f"identity-{os.getpid()}.jsonl"]  # once; and named for this process, which may not be alone here
    assert graded.files() == [graded.path.name]  # nothing left beside it: no script, no half-written file

    # A file cut short (the disk filled as it was saved) is no baseline. It is measured again and replaced.
    for damaged in ('{"valid": true, "exam": {"lo', "", "[]", '{"valid": false}'):
        graded.path.write_text(damaged)
        assert graded.run()["exam"]["loss"] == 5.3
        assert json.loads(graded.path.read_text())["valid"] is True
    assert len(graded.scripts) == 5 and graded.files() == [graded.path.name]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_an_anchors_baseline_has_a_held_out_part_and_is_its_own_to_read(graded):
    graded.run()
    assert stat.S_IMODE(graded.path.stat().st_mode) & 0o600 == 0o600 and "heldout" not in graded.path.read_text()
    # The pack is put in place. The baseline from before it has no held-out part, so it is measured again.
    graded.pack = True
    assert graded.run()["heldout"]["loss"] == 5.56789 and graded.run()["heldout"]["loss"] == 5.56789
    assert len(graded.scripts) == 2
    assert stat.S_IMODE(graded.path.stat().st_mode) == 0o600


def test_a_baseline_that_cannot_be_graded_is_not_kept(graded):
    graded.valid = False
    with pytest.raises(RuntimeError, match="could not be graded"):
        graded.run()
    assert graded.files() == []


@pytest.fixture
def student(monkeypatch):
    """`grade` with the training run and the exams replaced. Returns what happened, in order."""
    from stet import grade as grading
    from stet import tok

    seen = types.SimpleNamespace(did=[], pack=None)
    stats = types.SimpleNamespace(out_docs=3, out_bytes=30, ink_tokens=0, ink_budget=10, tokens_one_pass=9, passes=1.0, to_dict=lambda corpus: {"out_docs": 3})

    def heldout_exam(rnd, scale):
        seen.did.append("pack")
        if isinstance(seen.pack, Exception):
            raise seen.pack
        return seen.pack

    def evaluate(model, exam, scale, season, device, enc):
        seen.did.append("heldout" if exam is seen.pack else "exam")
        if exam is seen.pack and seen.fail:
            raise RuntimeError("CUDA error: out of memory")
        return {"loss": 4.61234 if exam is seen.pack else 4.4, "by_source": {}}

    def train(*a):
        seen.did.append("train")
        return None, {"tokens": 9}

    def materialize(*a):
        seen.did.append("script")
        return [], stats

    monkeypatch.setattr(grading, "heldout_exam", heldout_exam)
    monkeypatch.setattr(grading, "evaluate", evaluate)
    monkeypatch.setattr(grading, "train", train)
    monkeypatch.setattr(grading, "materialize", materialize)
    monkeypatch.setattr(grading, "file_sha256", lambda path, limit: "ab" * 32)
    monkeypatch.setattr(grading, "hardware_info", lambda device: {})
    monkeypatch.setattr(tok, "load_encoding", lambda season: None)
    season = types.SimpleNamespace(id="s0", digest="d", raw={"script": {"max_bytes": 1000}})
    rnd = types.SimpleNamespace(season=season, describe=lambda: {"id": "s0-o00007"}, init_seed=0)
    scale, corpus, device = types.SimpleNamespace(name="micro"), types.SimpleNamespace(meta={"sha256": "c0"}), types.SimpleNamespace(type="cpu")
    seen.fail = False
    seen.run = lambda log=None: grading.grade(rnd, scale, corpus, {}, "script.jsonl", device=device, log=log)
    return seen


def test_the_held_out_pack_is_looked_at_before_anything_is_trained(student):
    from stet.data import HeldoutError

    student.pack = HeldoutError("heldout-s0.jsonl.zst is not the held-out exam that season s0 committed to")
    with pytest.raises(HeldoutError):
        student.run()
    assert student.did == ["pack"]  # a wrong pack costs nothing: no script was read, no student trained

    student.did.clear()
    student.pack = None  # every machine but an anchor
    result = student.run()
    assert result["valid"] and "heldout" not in result and student.did == ["pack", "script", "train", "exam"]


def test_a_held_out_sitting_that_fails_does_not_cost_the_public_grade(student):
    student.pack = {"news": []}
    result = student.run()
    assert result["valid"] and result["heldout"]["loss"] == 4.61234 and len(result["heldout"]["exam_sha256"]) == 64
    assert student.did == ["pack", "script", "train", "exam", "heldout"]

    student.fail = True
    said = []
    result = student.run(log=said.append)
    assert result["valid"] and result["exam"]["loss"] == 4.4  # the public grade is finished, and stands
    assert "heldout" not in result and "out of memory" in result["heldout_error"]
    assert any("HELD-OUT EXAM COULD NOT BE SAT" in line for line in said)

"""Rounds end to end: edit, then grade. Shared by `stet run`, `stet verify` and the gym.

The two halves are deliberately separate. *Editing* runs an entrant's program and yields an
edit script. *Grading* turns a script into a score and runs no entrant code at all. In the
competition, official edits happen on the organisers' anchor machines and anyone's GPU can
grade the published scripts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import data
from .grade import grade
from .runner import EditRun, run_editor
from .script import write_script
from .spec import HARNESS_VERSION, Round, Scale
from .train import pick_device


def baseline(rnd: Round, scale: Scale, device=None, progress=None, log=None) -> dict:
    """The unedited corpus's result for a round: every document, in corpus order. Cached.

    The corpus is in seeded random order, so the baseline student trains on a fair sample of
    the round's raw data.
    """
    device = pick_device(device) if not hasattr(device, "type") else device
    path = data.round_dir(rnd, scale) / f"baseline-{HARNESS_VERSION}-{device.type}.json"
    cached = _cached(path)
    # An anchor needs the held-out exam's baseline too, so a baseline measured before the
    # pack was put in place is measured again.
    if cached is not None and ("heldout" in cached or data.heldout_exam(rnd, scale) is None):
        return cached
    corpus, exam = data.materialize(rnd, scale)
    # Named for this process: two verifiers on one machine share the round's directory.
    script = data.round_dir(rnd, scale) / f"identity-{os.getpid()}.jsonl"
    write_script(script, range(len(corpus)))
    try:
        result = grade(rnd, scale, corpus, exam, script, device=device, progress=progress, log=log)
    finally:
        script.unlink(missing_ok=True)
    if not result["valid"]:
        raise RuntimeError(f"the baseline for {rnd.id} could not be graded: {result['invalid_reason']}")
    if result.get("heldout_error"):
        raise RuntimeError(f"the baseline for {rnd.id} could not sit the held-out exam: {result['heldout_error']}")
    # Written whole or not at all: a file cut short by a full disk would otherwise stop
    # every later job of the round. With a held-out part in it, it is nobody else's to read.
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if "heldout" in result else 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(result))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # nothing is there once it has been renamed
    return result


def _cached(path: Path) -> dict | None:
    """A baseline saved earlier, or None if there is none that can be used. A file that
    cannot be read is as good as no file: the baseline is measured again and it is replaced."""
    try:
        cached = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict) or not cached.get("valid") or not isinstance(cached.get("exam"), dict):
        return None
    return cached


def edit_round(
    bundle: Path, entry: str, rnd: Round, scale: Scale, work: Path, sandbox: str = "none", echo: bool = False, log=None, limits: dict | None = None
) -> EditRun:
    """Run an editor on a round's corpus. Never raises on editor failure."""
    log = log or (lambda msg: None)
    corpus = data.corpus(rnd, scale)
    log(f"round {rnd.id} ({scale.name}): {len(corpus):,} documents, {corpus.total_bytes / 1e6:.0f} MB")
    log(f"editing (limit {scale.edit_seconds}s, sandbox: {sandbox}) ...")
    edit = run_editor(bundle, entry, corpus, rnd, scale, work, float(scale.edit_seconds), sandbox=sandbox, echo=echo, limits=limits)
    log(f"editor finished: {edit.status} in {edit.seconds:.1f}s")
    return edit


def grade_round(rnd: Round, scale: Scale, script_path: Path, device=None, progress=None, log=None) -> dict:
    """Score an edit script on a round."""
    corpus, exam = data.materialize(rnd, scale)
    return grade(rnd, scale, corpus, exam, script_path, device=device, progress=progress, log=log)


def run_round(
    bundle: Path, entry: str, rnd: Round, scale: Scale, work: Path, sandbox: str = "none", device=None, echo: bool = False, progress=None, log=None
) -> dict:
    """Edit, then grade. If the editor leaves no script the result is invalid and carries no
    exam score: by the rules its student is the one trained on the unedited corpus."""
    edit = edit_round(bundle, entry, rnd, scale, work, sandbox=sandbox, echo=echo, log=log)
    if edit.script is None:
        corpus, exam = data.materialize(rnd, scale)
        result = {
            "harness": HARNESS_VERSION, "season": rnd.season.id, "season_digest": rnd.season.digest, "round": rnd.describe(),
            "scale": scale.name, "corpus_sha256": corpus.meta["sha256"], "exam_sha256": data.exam_digest(exam),
            "valid": False, "invalid_reason": edit.reason or f"the editor produced no script ({edit.status})",
        }  # fmt: skip
    else:
        result = grade_round(rnd, scale, edit.script, device=device, progress=progress, log=log)
    result["edit"] = edit.to_dict()
    return result

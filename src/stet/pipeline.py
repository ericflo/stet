"""Rounds end to end: edit, then grade. Shared by `stet run`, `stet verify` and the gym.

The two halves are deliberately separate. *Editing* runs an entrant's program and yields an
edit script. *Grading* turns a script into a score and runs no entrant code at all. In the
competition, official edits happen in the sandbox on volunteers' machines and anyone's GPU can
grade the published scripts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import data
from .grade import Trained, grade, sit_exam, train_script
from .runner import EditRun, run_editor
from .script import write_script
from .spec import HARNESS_VERSION, Round, Scale
from .train import pick_device


def _exam_cache_identity(rnd: Round, scale: Scale) -> tuple[str | None, int | None, bool]:
    """Return `(exam_sha, n_items, is_public_subset)` for the exact exam items `round_exam` will sit."""
    try:
        exam = data.round_exam(rnd, scale)
    except Exception:
        return (rnd.exam_digest or None), None, False
    if not exam:
        return (rnd.exam_digest or None), None, False
    actual_sha = data.exam_digest(exam)
    is_pub = bool(rnd.kind == "official" and rnd.exam_digest and actual_sha != rnd.exam_digest)
    return actual_sha, len(exam), is_pub


def baseline(rnd: Round, scale: Scale, device=None, progress=None, log=None) -> dict:
    """The unedited corpus's result for a round: every document, in corpus order. Cached.

    The corpus is in seeded random order, so the baseline student trains on a fair sample of
    the round's raw data.
    """
    device = pick_device(device) if not hasattr(device, "type") else device
    want_sha, want_n, is_pub = _exam_cache_identity(rnd, scale)
    path = baseline_path(rnd, scale, device, exam_sha=want_sha, public_subset=is_pub)
    cached = _cached(path, want_exam_sha=want_sha, want_items=want_n)
    if cached is not None:
        return cached
    return sit_baseline(train_baseline(rnd, scale, device, progress=progress, log=log), rnd, scale)


def baseline_path(
    rnd: Round,
    scale: Scale,
    device,
    *,
    exam_sha: str | None = None,
    public_subset: bool = False,
) -> Path:
    # A baseline is for the exact exam it sat: named for it. An official round's full exam uses
    # rnd.exam_digest[:12]; a public-subset replay uses the subset's own digest with `-public` so
    # public-subset and full-exam caches never collide. A practice round's is named for the season
    # and harvested hours.
    if rnd.exam_digest:
        tag = (exam_sha or rnd.exam_digest)[:12]
        if public_subset or (exam_sha and exam_sha != rnd.exam_digest):
            tag = f"{tag}-public"
    else:
        tag = f"{rnd.season.digest[:8]}-{(data.practice_key(rnd) or 'generated')[:12]}"
    return data.round_dir(rnd, scale) / f"baseline-{HARNESS_VERSION}-{device.type}-{tag}.json"


def train_baseline(rnd: Round, scale: Scale, device=None, progress=None, log=None) -> Trained:
    """The round's unedited student, trained and not yet examined: see baseline."""
    device = pick_device(device) if not hasattr(device, "type") else device
    corpus = data.corpus(rnd, scale)
    # Named for this process: two verifiers on one machine share the round's directory.
    script = data.round_dir(rnd, scale) / f"identity-{os.getpid()}.jsonl"
    write_script(script, range(len(corpus)))
    try:
        return train_script(rnd, scale, corpus, script, device=device, progress=progress, log=log)
    finally:
        script.unlink(missing_ok=True)


def sit_baseline(trained: Trained, rnd: Round, scale: Scale) -> dict:
    """The unedited student sits the round's exam (`rnd` as it is now, its exam posted), and
    the result is kept for the round's other grades on this machine."""
    exam = data.round_exam(rnd, scale)
    result = sit_exam(trained, rnd, scale, exam)
    if not result["valid"]:
        raise RuntimeError(f"the baseline for {rnd.id} could not be graded: {result['invalid_reason']}")
    actual_sha = result.get("exam_sha256") or (data.exam_digest(exam) if exam else (rnd.exam_digest or None))
    is_pub = bool(rnd.kind == "official" and rnd.exam_digest and actual_sha and actual_sha != rnd.exam_digest)
    path = baseline_path(rnd, scale, trained.device, exam_sha=actual_sha, public_subset=is_pub)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written whole or not at all: a file cut short by a full disk would otherwise stop
    # every later job of the round.
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(result))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # nothing is there once it has been renamed
    return result


def _cached(path: Path, *, want_exam_sha: str | None = None, want_items: int | None = None) -> dict | None:
    """A baseline saved earlier, or None if there is none that matches the exact exam items
    being sat. A file that cannot be read or sat a different item set is treated as a cache miss."""
    try:
        cached = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict) or not cached.get("valid") or not isinstance(cached.get("exam"), dict):
        return None
    if want_exam_sha:
        got_sha = cached.get("exam_sha256")
        if want_items is not None:
            if got_sha != want_exam_sha:
                return None
        elif got_sha and got_sha != want_exam_sha:
            return None
    if want_items is not None and cached["exam"].get("items") != want_items:
        return None
    return cached


def edit_round(
    bundle: Path,
    entry: str,
    rnd: Round,
    scale: Scale,
    work: Path,
    sandbox: str = "none",
    echo: bool = False,
    log=None,
    limits: dict | None = None,
    assets=(),
    time_limit: float | None = None,
    verify_assets: bool = False,
    gpu: int = 0,
) -> EditRun:
    """Run a curator on a round's corpus. Never raises on curator failure.

    The files the curator named by reference are fetched first, before its clock starts.
    A file that cannot be fetched because the host is away raises data.FetchError: that
    is the network's failure, and no run of the curator. A file that is no longer there,
    or no longer the bytes the entry names, is the entry's: a run that failed.
    """
    log = log or (lambda msg: None)
    corpus = data.corpus(rnd, scale)
    log(f"round {rnd.id} ({scale.name}): {len(corpus):,} documents, {corpus.total_bytes / 1e6:.0f} MB")
    if assets:
        from . import assets as assets_

        log(f"assets: {len(assets)} files, {sum(a.size for a in assets) / 2**30:.2f} GiB")
        try:
            assets_.fetch(list(assets), verify=verify_assets)
        except assets_.Unavailable as e:
            log(f"curator not run: {e}")
            limit = float(scale.edit_seconds if time_limit is None else time_limit)
            return EditRun("error", 0.0, limit, None, None, "", sandbox, reason=str(e))
        assets_.prune(keep=list(assets))
    limit = float(scale.edit_seconds if time_limit is None else time_limit)
    log(f"editing (limit {limit:.0f}s, sandbox: {sandbox}) ...")
    edit = run_editor(bundle, entry, corpus, rnd, scale, work, limit, sandbox=sandbox, echo=echo, limits=limits, assets=assets, gpu=gpu)
    log(f"curator finished: {edit.status} in {edit.seconds:.1f}s")
    return edit


def grade_round(rnd: Round, scale: Scale, script_path: Path, device=None, progress=None, log=None) -> dict:
    """Score an edit script on a round."""
    corpus, exam = data.materialize(rnd, scale)
    return grade(rnd, scale, corpus, exam, script_path, device=device, progress=progress, log=log)


def train_round(rnd: Round, scale: Scale, script_path: Path, device=None, progress=None, log=None) -> Trained:
    """Train the student an edit script makes, before the round's exam is there to sit."""
    return train_script(rnd, scale, data.corpus(rnd, scale), script_path, device=device, progress=progress, log=log)


def run_round(
    bundle: Path,
    entry: str,
    rnd: Round,
    scale: Scale,
    work: Path,
    sandbox: str = "none",
    device=None,
    echo: bool = False,
    progress=None,
    log=None,
    assets=(),
    time_limit: float | None = None,
) -> dict:
    """Edit, then grade. If the curator leaves no script the result is invalid and carries no
    exam score: by the rules its student is the one trained on the unedited corpus."""
    edit = edit_round(bundle, entry, rnd, scale, work, sandbox=sandbox, echo=echo, log=log, assets=assets, time_limit=time_limit)
    if edit.script is None:
        corpus, exam = data.materialize(rnd, scale)
        result = {
            "harness": HARNESS_VERSION, "season": rnd.season.id, "season_digest": rnd.season.digest, "round": rnd.describe(),
            "scale": scale.name, "corpus_sha256": corpus.meta["sha256"], "exam_sha256": data.exam_digest(exam),
            "valid": False, "invalid_reason": edit.reason or f"the curator produced no script ({edit.status})",
        }  # fmt: skip
    else:
        result = grade_round(rnd, scale, edit.script, device=device, progress=progress, log=log)
    result["edit"] = edit.to_dict()
    return result

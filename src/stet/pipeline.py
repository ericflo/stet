"""Rounds end to end: edit, then grade. Shared by `stet run`, `stet verify` and the gym.

The two halves are deliberately separate. *Editing* runs an entrant's program and yields an
edit script. *Grading* turns a script into a score and runs no entrant code at all. In the
competition, official edits happen in the sandbox on volunteers' machines and anyone's GPU can
grade the published scripts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import data
from .grade import Trained, grade, sit_exam, train_script
from .runner import EditRun, run_editor
from .script import write_script
from .spec import HARNESS_VERSION, Round, Scale
from .train import device_backend, hardware_backend, pick_device


def _device_gpu_name(device) -> str | None:
    """Concrete GPU model name for CUDA/ROCm devices when available, else None."""
    if getattr(device, "type", None) != "cuda":
        return None
    explicit = getattr(device, "gpu", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    try:
        import torch

        props = torch.cuda.get_device_properties(device)
        name = getattr(props, "name", None)
        if isinstance(name, str) and name:
            return name
    except Exception:
        pass
    return None


def _exam_cache_identity(rnd: Round, scale: Scale, *, exam=None) -> tuple[str | None, int | None, bool]:
    """Return `(exam_sha, n_items, is_public_subset)` for the exact exam items to sit."""
    if exam is None:
        try:
            exam = data.round_exam(rnd, scale)
        except Exception:
            return (rnd.exam_digest or None), None, False
    if not exam:
        return (rnd.exam_digest or None), None, False
    actual_sha = data.exam_digest(exam)
    is_pub = bool(rnd.kind == "official" and rnd.exam_digest and actual_sha != rnd.exam_digest)
    return actual_sha, len(exam), is_pub


def _resolve_practice_pin(rnd: Round, scale: Scale, *, exam=None, practice_key: str | None = None):
    """Pin `(exam, practice_key)` once for practice rounds when not already pinned by caller,
    so harvest changes mid-baseline cannot mismatch the exam sat vs cache filename written."""
    if rnd.kind == "practice" and exam is None:
        hours = data.practice_hours(rnd)
        if practice_key is None:
            practice_key = data.practice_key(rnd, hours)
        try:
            exam = data.round_exam(rnd, scale, hours=hours)
        except TypeError:
            exam = data.round_exam(rnd, scale)
    return exam, practice_key


def baseline(
    rnd: Round,
    scale: Scale,
    device=None,
    progress=None,
    log=None,
    *,
    verifier: str | None = None,
    exam=None,
    practice_key: str | None = None,
) -> dict:
    """The unedited corpus's result for a round: every document, in corpus order. Cached.

    The corpus is in seeded random order, so the baseline student trains on a fair sample of
    the round's raw data.
    """
    device = pick_device(device) if not hasattr(device, "type") else device
    exam, practice_key = _resolve_practice_pin(rnd, scale, exam=exam, practice_key=practice_key)
    want_sha, want_n, is_pub = _exam_cache_identity(rnd, scale, exam=exam)
    bk = device_backend(device)
    dev_gpu = _device_gpu_name(device)
    path = baseline_path(
        rnd,
        scale,
        device,
        exam_sha=want_sha,
        public_subset=is_pub,
        backend=bk,
        verifier=verifier,
        gpu=dev_gpu,
        practice_key=practice_key,
    )
    cached = _cached(
        path,
        want_exam_sha=want_sha,
        want_items=want_n,
        want_backend=bk,
        want_gpu=dev_gpu,
        want_verifier=verifier,
    )
    if cached is not None:
        return cached
    return sit_baseline(
        train_baseline(rnd, scale, device, progress=progress, log=log),
        rnd,
        scale,
        verifier=verifier,
        gpu=dev_gpu,
        exam=exam,
        practice_key=practice_key,
    )


def baseline_path(
    rnd: Round,
    scale: Scale,
    device,
    *,
    exam_sha: str | None = None,
    public_subset: bool = False,
    backend: str | None = None,
    verifier: str | None = None,
    gpu: str | None = None,
    practice_key: str | None = None,
) -> Path:
    # A baseline is for the exact exam, grading backend, GPU model, and (for official verifying)
    # producing verifier key that sat it: two verifiers sharing STET_CACHE never submit one
    # cached measurement as two independent baselines, and two different GPU models never share
    # a baseline file. An official round's full exam uses rnd.exam_digest[:12]; a public-subset
    # replay uses the subset's own digest with `-public`. A practice round's is named for the
    # season and pinned harvested hours.
    if rnd.exam_digest:
        tag = (exam_sha or rnd.exam_digest)[:12]
        if public_subset or (exam_sha and exam_sha != rnd.exam_digest):
            tag = f"{tag}-public"
    else:
        pk = practice_key if practice_key is not None else data.practice_key(rnd)
        tag = f"{rnd.season.digest[:8]}-{(pk or 'generated')[:12]}"
    bk = backend or device_backend(device)
    suffix = ""
    dev_gpu = gpu if gpu is not None else _device_gpu_name(device)
    if dev_gpu:
        suffix += f"-g{hashlib.sha256(dev_gpu.encode('utf-8')).hexdigest()[:8]}"
    if verifier:
        suffix += f"-v{hashlib.sha256(verifier.encode('utf-8')).hexdigest()[:12]}"
    return data.round_dir(rnd, scale) / f"baseline-{HARNESS_VERSION}-{bk}-{tag}{suffix}.json"


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


def sit_baseline(
    trained: Trained,
    rnd: Round,
    scale: Scale,
    *,
    verifier: str | None = None,
    gpu: str | None = None,
    exam=None,
    practice_key: str | None = None,
) -> dict:
    """The unedited student sits the round's exam (`rnd` as it is now, its exam posted), and
    the result is kept for the round's other grades on this machine."""
    exam, practice_key = _resolve_practice_pin(rnd, scale, exam=exam, practice_key=practice_key)
    exam = exam if exam is not None else data.round_exam(rnd, scale)
    result = sit_exam(trained, rnd, scale, exam)
    if not result["valid"]:
        raise RuntimeError(f"the baseline for {rnd.id} could not be graded: {result['invalid_reason']}")
    actual_sha = result.get("exam_sha256") or (data.exam_digest(exam) if exam else (rnd.exam_digest or None))
    is_pub = bool(rnd.kind == "official" and rnd.exam_digest and actual_sha and actual_sha != rnd.exam_digest)
    bk = hardware_backend(result.get("hardware")) or device_backend(trained.device)
    dev_gpu = gpu if gpu is not None else _device_gpu_name(trained.device)
    path = baseline_path(
        rnd,
        scale,
        trained.device,
        exam_sha=actual_sha,
        public_subset=is_pub,
        backend=bk,
        verifier=verifier,
        gpu=dev_gpu,
        practice_key=practice_key,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    to_cache = dict(result)
    if verifier:
        to_cache["verifier"] = verifier
    # Written whole or not at all: a file cut short by a full disk would otherwise stop
    # every later job of the round.
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(to_cache))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # nothing is there once it has been renamed
    return result


def _path_backend(path: Path) -> str | None:
    prefix = f"baseline-{HARNESS_VERSION}-"
    if path.name.startswith(prefix):
        bk = path.name[len(prefix) :].split("-", 1)[0]
        if bk in ("cuda", "rocm", "mps", "cpu"):
            return bk
    return None


def _cached(
    path: Path,
    *,
    want_exam_sha: str | None = None,
    want_items: int | None = None,
    want_backend: str | None = None,
    want_gpu: str | None = None,
    want_verifier: str | None = None,
) -> dict | None:
    """A baseline saved earlier, or None if there is none that matches the exact exam items,
    grading backend, GPU model, and producing verifier. A file that cannot be read, sat a
    different item set, came from another GPU/backend, or was produced by another verifier is
    treated as a cache miss."""
    try:
        cached = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict) or not cached.get("valid") or not isinstance(cached.get("exam"), dict):
        return None
    if want_backend is None:
        want_backend = _path_backend(path)
    if want_backend in ("cuda", "rocm"):
        if hardware_backend(cached.get("hardware")) != want_backend:
            return None
    elif want_backend is not None and cached.get("hardware") is not None:
        if hardware_backend(cached.get("hardware")) != want_backend:
            return None
    if want_gpu is not None:
        hw = cached.get("hardware")
        if not isinstance(hw, dict) or hw.get("gpu") != want_gpu:
            return None
    if want_verifier is not None:
        if cached.get("verifier") != want_verifier:
            return None
    elif cached.get("verifier") is not None:
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
    cached.pop("verifier", None)
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


def grade_round(rnd: Round, scale: Scale, script_path: Path, device=None, progress=None, log=None, *, exam=None) -> dict:
    """Score an edit script on a round."""
    if exam is None:
        corpus, exam = data.materialize(rnd, scale)
    else:
        corpus = data.corpus(rnd, scale)
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
    *,
    exam=None,
) -> dict:
    """Edit, then grade. If the curator leaves no script the result is invalid and carries no
    exam score: by the rules its student is the one trained on the unedited corpus."""
    edit = edit_round(bundle, entry, rnd, scale, work, sandbox=sandbox, echo=echo, log=log, assets=assets, time_limit=time_limit)
    if edit.script is None:
        if exam is None:
            corpus, exam = data.materialize(rnd, scale)
        else:
            corpus = data.corpus(rnd, scale)
        result = {
            "harness": HARNESS_VERSION, "season": rnd.season.id, "season_digest": rnd.season.digest, "round": rnd.describe(),
            "scale": scale.name, "corpus_sha256": corpus.meta["sha256"], "exam_sha256": data.exam_digest(exam),
            "valid": False, "invalid_reason": edit.reason or f"the curator produced no script ({edit.status})",
        }  # fmt: skip
    else:
        result = grade_round(rnd, scale, edit.script, device=device, progress=progress, log=log, exam=exam)
    result["edit"] = edit.to_dict()
    return result

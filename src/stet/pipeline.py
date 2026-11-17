"""One editor on one round, end to end. Shared by `stet run` and `stet verify`."""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import data
from .grade import grade
from .runner import run_editor
from .script import ScriptError, write_script
from .spec import HARNESS_VERSION, Round, Scale
from .train import hardware_info, pick_device


def baseline(rnd: Round, scale: Scale, device=None, progress=None, log=None) -> dict:
    """The identity editor's result for a round: every document, in arrival order. Cached."""
    device = pick_device(device) if not hasattr(device, "type") else device
    path = data.round_dir(rnd, scale) / f"baseline-{HARNESS_VERSION}-{device.type}.json"
    if path.exists():
        return json.loads(path.read_text())
    corpus, exam = data.materialize(rnd, scale)
    script = data.round_dir(rnd, scale) / "identity.jsonl"
    write_script(script, range(len(corpus)))
    result = grade(rnd, scale, corpus, exam, script, device=device, progress=progress, log=log)
    script.unlink(missing_ok=True)
    path.write_text(json.dumps(result))
    return result


def run_round(
    bundle: Path,
    entry: str,
    rnd: Round,
    scale: Scale,
    work: Path,
    sandbox: str = "none",
    device=None,
    time_factor: float = 1.0,
    echo: bool = False,
    progress=None,
    log=None,
) -> dict:
    """Run the editor, apply its script, train, sit the exam. Never raises on editor failure.

    If the editor produces no usable script the result is marked invalid and carries no exam
    score: by the rules its student is the one trained on the unedited corpus, i.e. the
    baseline.
    """
    log = log or (lambda msg: None)
    device = pick_device(device) if not hasattr(device, "type") else device
    t0 = time.time()
    corpus, exam = data.materialize(rnd, scale)
    log(f"round {rnd.id} ({scale.name}): {len(corpus):,} documents, {corpus.total_bytes / 1e6:.0f} MB")
    limit = scale.edit_seconds * time_factor
    log(f"editing (limit {limit:.0f}s, sandbox: {sandbox}) ...")
    edit = run_editor(bundle, entry, corpus, rnd, scale, work, limit, sandbox=sandbox, echo=echo)
    log(f"editor finished: {edit.status} in {edit.seconds:.1f}s")
    result = {
        "harness": HARNESS_VERSION,
        "season": rnd.season.id,
        "season_digest": rnd.season.digest,
        "round": rnd.describe(),
        "scale": scale.name,
        "corpus_sha256": corpus.meta["sha256"],
        "exam_sha256": data.exam_digest(exam),
        "edit": edit.to_dict(),
        "valid": False,
        "invalid_reason": None,
        "hardware": hardware_info(device),
        "started_at": int(t0),
    }
    if edit.script is None:
        result["invalid_reason"] = f"the editor produced no script ({edit.status})"
    else:
        try:
            graded = grade(rnd, scale, corpus, exam, edit.script, device=device, progress=progress, log=log)
            result.update({k: graded[k] for k in ("script", "train", "exam")})
            result["valid"] = True
        except ScriptError as e:
            result["invalid_reason"] = str(e)
        except FloatingPointError as e:
            result["invalid_reason"] = str(e)
    result["finished_at"] = int(time.time())
    return result

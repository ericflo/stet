"""Edit script in, exam score out. Trusted code only: nothing here runs a curator.

This is the whole of what a volunteer's machine does for the leaderboard: read a script
(plain data), rebuild the round's corpus from public sources, train the fixed student on
what the script says, and report its score on every item of the round's exam.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import source_digest
from .data import Corpus, exam_digest
from .script import ScriptError, file_sha256, materialize
from .spec import HARNESS_VERSION, Round, Scale
from .train import hardware_info, pick_device, train


@dataclass
class Trained:
    """A student trained on what a script says, before it sits the exam. Held in memory
    while a grader waits for an official round's exam, which is made after the draw."""

    result: dict  # what the grade will report, all but the exam's part
    trained: bool = False  # False: the script broke a rule, and no student sits the exam
    model: object = None
    enc: object = None
    device: object = None
    ink_windows: list = field(default_factory=list)  # the script's own text, for the look at the exam


def train_script(rnd: Round, scale: Scale, corpus: Corpus, script_path, device=None, progress=None, log=None) -> Trained:
    """Apply the script and train the student for the budget: all of grading but the exam.

    Never raises on a bad script: the result will say `valid: false` and why."""
    from .tok import load_encoding

    log = log or (lambda msg: None)
    device = pick_device(device) if not hasattr(device, "type") else device
    season = rnd.season
    result = {
        "harness": HARNESS_VERSION,
        "harness_sha256": source_digest(),
        "season": season.id,
        "season_digest": season.digest,
        "round": rnd.describe(),
        "scale": scale.name,
        "corpus_sha256": corpus.meta["sha256"],
        "exam_sha256": None,  # the exam the student sits: see sit_exam
        "hardware": hardware_info(device),
        "valid": False,
        "invalid_reason": None,
        "started_at": int(time.time()),
    }
    try:
        result["script_sha256"] = file_sha256(script_path, int(season.raw["script"]["max_bytes"]))
        enc = load_encoding(season)
        t0 = time.time()
        tokens, stats = materialize(corpus, script_path, season, scale, enc)
        log(
            f"script: {stats.out_docs:,} docs, {stats.out_bytes / 1e6:.1f} MB, ink {stats.ink_tokens:,} of {stats.ink_budget:,} tokens, "
            f"{stats.tokens_one_pass:,} tokens per pass x {stats.passes:.2f} passes ({time.time() - t0:.0f}s)"
        )
        result["script"] = stats.to_dict(corpus)
        model, result["train"] = train(tokens, scale, season, rnd.init_seed, device, progress)
        return Trained(result, True, model, enc, device, stats.ink_windows)
    except (ScriptError, FloatingPointError) as e:
        result["invalid_reason"] = str(e)
    return Trained(result, device=device)


def sit_exam(trained: Trained, rnd: Round, scale: Scale, exam) -> dict:
    """The trained student sits the round's exam, and the grade is complete. `rnd` is the
    round as it is now: an official round's exam is posted after the student was trained."""
    from .exam.decontam import written
    from .exam.score import sit

    result = trained.result
    result["round"], result["exam_sha256"] = rnd.describe(), exam_digest(exam)
    if trained.trained and (found := written(exam, trained.ink_windows, rnd.season)):
        # The exam's writing reached the training set through the curator's own text: carried
        # in by its author, or written in and published afterwards. No student of it is scored.
        trained.trained = False
        result["invalid_reason"] = f"the script's own text holds {len(found)} of the passages in the round's exam"
    if trained.trained:
        try:
            t1 = time.time()
            scores = sit(trained.model, exam, trained.enc, rnd.season.eot, scale.ctx, trained.device, floor=rnd.season.answer_floor_bytes)
            result["exam"] = {"items": len(exam), "item_scores": [round(x, 6) for x in scores],
                              "mean": round(sum(scores) / max(1, len(scores)), 6), "seconds": round(time.time() - t1, 2)}
            result["valid"] = True
        except FloatingPointError as e:
            result["invalid_reason"] = str(e)
    trained.model = None  # the student is done with; its memory goes back
    result["finished_at"] = int(time.time())
    return result


def grade(
    rnd: Round,
    scale: Scale,
    corpus: Corpus,
    exam,
    script_path,
    device=None,
    progress=None,
    log=None,
) -> dict:
    """Apply the script, train the student for the budget, sit the exam.

    The exam is the round's items (stet.exam). The result has every item's score, in bits
    per byte, higher is better; the round's score is made from them against the round's
    baseline (exam.aggregate).

    Never raises on a bad script: the result says `valid: false` and why. A script that
    breaks a rule scores as the unedited corpus (gain 0).
    """
    trained = train_script(rnd, scale, corpus, script_path, device=device, progress=progress, log=log)
    return sit_exam(trained, rnd, scale, exam)

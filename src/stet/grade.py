"""Edit script in, exam score out. Trusted code only: nothing here runs a curator.

This is the whole of what a volunteer's machine does for the leaderboard: read a script
(plain data), rebuild the round's corpus and exam from public sources, train the fixed
student on what the script says, and report the exam loss.
"""

from __future__ import annotations

import time

from . import source_digest
from .data import Corpus, exam_digest, heldout_exam
from .script import ScriptError, file_sha256, materialize
from .spec import HARNESS_VERSION, Round, Scale
from .train import evaluate, hardware_info, pick_device, train


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

    The exam is a pack season's passages by source (the result has the loss on each), or the
    next exam's items (a list: the result has every item's score, in bits per byte, higher is
    better; the round's score is made from them against the round's baseline, see
    newexam.aggregate).

    Never raises on a bad script: the result says `valid: false` and why. A script that
    breaks a rule scores as the unedited corpus (gain 0).
    """
    from .tok import load_encoding

    log = log or (lambda msg: None)
    device = pick_device(device) if not hasattr(device, "type") else device
    season = rnd.season
    # Before anything is trained: a pack that is not the season's raises here, and costs nothing.
    held = heldout_exam(rnd, scale)
    result = {
        "harness": HARNESS_VERSION,
        "harness_sha256": source_digest(),
        "season": season.id,
        "season_digest": season.digest,
        "round": rnd.describe(),
        "scale": scale.name,
        "corpus_sha256": corpus.meta["sha256"],
        "exam_sha256": exam_digest(exam),
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
        t1 = time.time()
        if isinstance(exam, list):
            from .newexam.score import sit

            scores = sit(model, exam, enc, season.eot, scale.ctx, device)
            result["exam"] = {"kind": "next", "items": len(exam), "item_scores": [round(x, 6) for x in scores],
                              "mean": round(sum(scores) / max(1, len(scores)), 6)}
        else:
            result["exam"] = evaluate(model, exam, scale, season, device, enc)
        result["exam"]["seconds"] = round(time.time() - t1, 2)
        result["valid"] = True
        if held is not None:
            # A machine that holds the held-out exam: the same student sits it too. This stays on the
            # machine (see client.keep_heldout) and is never part of what a grader reports.
            # The public grade is finished and stands whatever happens here; a sitting that
            # fails is said out loud and shows as a gap in the season's held-out table,
            # with the command that measures it again.
            try:
                result["heldout"] = evaluate(model, held, scale, season, device, enc)
                result["heldout"]["exam_sha256"] = exam_digest(held)
            except Exception as e:
                result.pop("heldout", None)
                result["heldout_error"] = f"{type(e).__name__}: {e}"
                log(f"THE HELD-OUT EXAM COULD NOT BE SAT ({result['heldout_error']}). The public grade stands.")
    except ScriptError as e:
        result["invalid_reason"] = str(e)
    except FloatingPointError as e:
        result["invalid_reason"] = str(e)
    result["finished_at"] = int(time.time())
    return result

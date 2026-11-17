"""Edit script in, exam score out. Trusted code only: nothing here runs an editor."""

from __future__ import annotations

import time

from .data import Corpus, exam_digest
from .script import materialize
from .spec import HARNESS_VERSION, Round, Scale
from .train import evaluate, hardware_info, pick_device, train


def grade(
    rnd: Round,
    scale: Scale,
    corpus: Corpus,
    exam: dict[str, list[dict]],
    script_path,
    device=None,
    progress=None,
    log=None,
) -> dict:
    """Apply the script, train the student for the budget, sit the exam."""
    from .tok import load_encoding

    log = log or (lambda msg: None)
    device = pick_device(device) if not hasattr(device, "type") else device
    enc = load_encoding(rnd.season)
    t0 = time.time()
    tokens, stats = materialize(corpus, script_path, rnd.season, scale, enc, exam)
    t_script = time.time() - t0
    log(
        f"script: {stats.out_docs:,} docs, {stats.out_bytes / 1e6:.1f} MB, ink {100 * stats.ink_frac:.2f}%, "
        f"{stats.tokens_one_pass:,} tokens per pass x {stats.passes:.2f} passes ({t_script:.0f}s)"
    )
    model, train_info = train(tokens, scale, rnd.season, rnd.init_seed, device, progress)
    t1 = time.time()
    exam_result = evaluate(model, exam, scale, rnd.season, device, enc)
    exam_result["seconds"] = round(time.time() - t1, 2)
    return {
        "harness": HARNESS_VERSION,
        "season": rnd.season.id,
        "season_digest": rnd.season.digest,
        "round": rnd.describe(),
        "scale": scale.name,
        "corpus_sha256": corpus.meta["sha256"],
        "exam_sha256": exam_digest(exam),
        "script": stats.to_dict(corpus),
        "train": train_info,
        "exam": exam_result,
        "hardware": hardware_info(device),
    }

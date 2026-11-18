"""The shadow exam of season 0: for a round, train the student on each script and on the
unedited corpus, sit both the season's exam and the next one, and report what the next exam
says next to what the season's says.

Nothing here touches an official score. It uses the same corpus, the same trainer and the
same initial weights as grading, so the season's exam is measured again alongside, which
checks that the run is a faithful replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .. import data
from ..script import ScriptError, materialize, write_script
from ..spec import Round, Scale
from ..tok import load_encoding
from ..train import evaluate, train
from . import gen
from .aggregate import gains, round_score
from .items import Item, digest
from .score import sit

PER_CORNER = 2  # items from every generated corner, each round


def round_items(rnd: Round, per_corner: int = PER_CORNER) -> list[Item]:
    """The round's next exam. Generated from the round's seed, which nobody knew before its draw.
    Text published after the draw joins it when the harvester is built."""
    return gen.generate(rnd.seed, per_corner=per_corner)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grade_both(rnd: Round, scale: Scale, corpus, exam, items: list[Item], script: Path, device, enc) -> dict:
    """Train on one script and sit both exams. A script that breaks the rules is invalid, as in grading."""
    season = rnd.season
    out = {"script_sha256": _sha(script), "valid": False}
    try:
        tokens, stats = materialize(corpus, script, season, scale, enc)
    except ScriptError as e:
        out["invalid_reason"] = str(e)
        return out
    t0 = time.time()
    model, tr = train(tokens, scale, season, rnd.init_seed, device)
    current = evaluate(model, exam, scale, season, device, enc)
    t1 = time.time()
    scores = sit(model, items, enc, season.eot, scale.ctx, device)
    out.update(valid=True, exam_loss=current["loss"], by_source={k: v["loss"] for k, v in current["by_source"].items()},
               item_scores=[round(s, 6) for s in scores], train_seconds=round(t1 - t0, 1), next_seconds=round(time.time() - t1, 2),
               kept_doc_frac=stats.to_dict(corpus)["kept_doc_frac"])
    return out


def shadow_round(rnd: Round, scale: Scale, scripts: dict[str, Path], device, out_dir: Path,
                 per_corner: int = PER_CORNER, log=print) -> dict:
    """Shadow-grade a round's scripts. Results are kept in out_dir by script hash, so a run that
    stops can be resumed and a round re-scored without training again."""
    out_dir.mkdir(parents=True, exist_ok=True)
    season = rnd.season
    enc = load_encoding(season)
    corpus, exam = data.materialize(rnd, scale)
    items = round_items(rnd, per_corner)
    with open(out_dir / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
    log(f"{rnd.id}: {len(corpus):,} documents; next exam {len(items)} items in {len({i.corner for i in items})} corners ({digest(items)[:12]})")

    def cached(name: str, script: Path) -> dict:
        path = out_dir / f"{_sha(script)[:16]}.json"
        if path.exists():
            r = json.loads(path.read_text())
            if r.get("items_digest") == digest(items):
                return r
        t0 = time.time()
        r = grade_both(rnd, scale, corpus, exam, items, script, device, enc)
        r.update(name=name, items_digest=digest(items))
        path.write_text(json.dumps(r))
        said = f"season exam {r['exam_loss']:.4f}" if r["valid"] else f"invalid: {r['invalid_reason']}"
        log(f"  {name}: {said} [{time.time() - t0:.0f}s]")
        return r

    base_script = out_dir / f"unedited-{os.getpid()}.jsonl"
    write_script(base_script, range(len(corpus)))
    try:
        base = cached("(unedited)", base_script)
    finally:
        base_script.unlink(missing_ok=True)
    rows = []
    for name, script in scripts.items():
        r = cached(name, script)
        row = {"name": name, "valid": r["valid"]}
        if r["valid"]:
            row["season_gain"] = round(base["exam_loss"] - r["exam_loss"], 5)
            agg = round_score(items, gains(r["item_scores"], base["item_scores"]))
            row.update(next_score=round(agg["score"], 5), strata=agg["strata"], families=agg["families"])
        else:
            row["reason"] = r.get("invalid_reason")
        rows.append(row)
    summary = {"round": rnd.id, "scale": scale.name, "items": len(items), "items_digest": digest(items),
               "generator_version": gen.VERSION, "baseline_exam_loss": base["exam_loss"], "entries": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary

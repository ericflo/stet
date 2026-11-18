"""The shadow exam of season 0: for a round, train the student on each script and on the
unedited corpus, sit both the season's exam and the next one, and report what the next exam
says next to what the season's says.

Nothing here touches an official score. It uses the same corpus, the same trainer and the
same initial weights as grading, so the season's exam is measured again alongside, which
checks that the run is a faithful replay. Each trained student is kept, so a round can be
sat again when the next exam grows, without training again.
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
from ..train import build_model, evaluate, train
from . import gen
from .aggregate import gains, round_score
from .items import Item, digest
from .score import sit

PER_CORNER = 2  # items from every generated corner, each round
FRESH_ITEMS = 900  # fresh items a round keeps, when it has a harvest


def round_items(rnd: Round, per_corner: int = PER_CORNER, fresh: list[Item] | None = None) -> list[Item]:
    """The round's next exam: generated from the round's seed, which nobody knew before its draw,
    and the round's fresh items, published after it."""
    return gen.generate(rnd.seed, per_corner=per_corner) + list(fresh or [])


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def shadow_round(rnd: Round, scale: Scale, scripts: dict[str, Path], device, out_dir: Path,
                 per_corner: int = PER_CORNER, fresh: list[Item] | None = None, log=print,
                 shares: dict[str, float] | None = None) -> dict:
    """Shadow-grade a round's scripts. Trained students and their results are kept in out_dir by
    script hash, so a run that stops can be resumed and a round sat again without training."""
    import torch

    (out_dir / "students").mkdir(parents=True, exist_ok=True)
    season = rnd.season
    enc = load_encoding(season)
    corpus, exam = data.materialize(rnd, scale)
    items = round_items(rnd, per_corner, fresh)
    tag = digest(items)[:12]
    with open(out_dir / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
    if shares:
        _write(out_dir / "languages.json", shares)  # the corpus's languages, which weigh the fresh items
    log(f"{rnd.id}: {len(corpus):,} documents; next exam {len(items)} items in {len({i.corner for i in items})} corners "
        f"({sum(1 for i in items if i.source.startswith('gen:'))} generated, {sum(1 for i in items if not i.source.startswith('gen:'))} fresh; {tag})")

    def student(name: str, script: Path) -> dict:
        sha = _sha(script)[:16]
        meta_path, weights = out_dir / f"{sha}.json", out_dir / "students" / f"{sha}.pt"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        model = None
        if meta is None or (meta["valid"] and not weights.exists()):
            t0 = time.time()
            meta = {"name": name, "valid": False}
            try:
                tokens, stats = materialize(corpus, script, season, scale, enc)
            except ScriptError as e:
                meta["invalid_reason"] = str(e)
            else:
                model, tr = train(tokens, scale, season, rnd.init_seed, device)
                current = evaluate(model, exam, scale, season, device, enc)
                torch.save(model.state_dict(), weights)
                meta.update(valid=True, exam_loss=current["loss"], by_source={k: v["loss"] for k, v in current["by_source"].items()},
                            kept_doc_frac=stats.to_dict(corpus)["kept_doc_frac"], train_seconds=round(tr["seconds"], 1))
            _write(meta_path, meta)
            said = f"season exam {meta['exam_loss']:.4f}" if meta["valid"] else f"invalid: {meta['invalid_reason']}"
            log(f"  {name}: trained, {said} [{time.time() - t0:.0f}s]")
        if not meta["valid"]:
            return meta
        scores_path = out_dir / f"{sha}-{tag}.json"
        if scores_path.exists():
            meta["item_scores"] = json.loads(scores_path.read_text())
            return meta
        if model is None:
            model = build_model(scale, season, rnd.init_seed, device)
            model.load_state_dict(torch.load(weights, map_location=device))
        t0 = time.time()
        meta["item_scores"] = [round(s, 6) for s in sit(model, items, enc, season.eot, scale.ctx, device)]
        _write(scores_path, meta["item_scores"])
        log(f"  {name}: sat the next exam [{time.time() - t0:.1f}s]")
        return meta

    base_script = out_dir / f"unedited-{os.getpid()}.jsonl"
    write_script(base_script, range(len(corpus)))
    try:
        base = student("(unedited)", base_script)
    finally:
        base_script.unlink(missing_ok=True)
    rows = []
    for name, script in scripts.items():
        r = student(name, script)
        row = {"name": name, "valid": r["valid"]}
        if r["valid"]:
            row["season_gain"] = round(base["exam_loss"] - r["exam_loss"], 5)
            g = gains(r["item_scores"], base["item_scores"])
            agg = round_score(items, g, shares=shares)
            row.update(next_score=round(agg["score"], 5), strata=agg["strata"], families=agg["families"])
            for label in ("generated", "fresh"):
                if agg[label]:
                    row[f"{label}_score"] = round(agg[label]["score"], 5)
            if agg["fresh"]:
                row["languages"] = {k: v["score"] for k, v in agg["fresh"]["languages"].items()}
        else:
            row["reason"] = r.get("invalid_reason")
        rows.append(row)
    summary = {"round": rnd.id, "scale": scale.name, "items": len(items), "corners": len({it.corner for it in items}), "items_digest": digest(items),
               "generator_version": gen.VERSION, "baseline_exam_loss": base["exam_loss"], "entries": rows}
    _write(out_dir / "summary.json", summary)
    return summary

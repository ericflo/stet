#!/usr/bin/env python3
"""Run reference curators on fixed-seed rounds and print a table. A research tool, not the CLI.

    python tools/exp.py --scale nano --rounds 0 1 --curators identity english tidy
"""
import argparse
import dataclasses
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stet import data
from stet.curator import Task
from stet.grade import grade
from stet.script import materialize
from stet.spec import load_season, practice_round
from stet.train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="nano")
    ap.add_argument("--kind", default="practice", choices=["practice", "gym"])
    ap.add_argument("--rounds", type=int, nargs="+", default=[0])
    ap.add_argument("--editors", nargs="+", default=["identity", "english", "tidy"])
    ap.add_argument("--train-tokens", type=int, default=None, help="override the budget (for reference curves)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override scale fields, e.g. batch_size=8 n_layer=6")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--perturb", type=float, default=0.0, help="relative noise added to the initial weights (noise-floor studies)")
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--reseed", type=int, default=None, help="shuffle each script with this seed before grading (order-noise studies)")
    ap.add_argument("--canned", type=float, default=0.0,
                    help="threat study: interleave this share of the training budget as ink made of in-distribution text")
    ap.add_argument("--table", action="store_true", help="also score a trigram count table of each training stream")
    args = ap.parse_args()
    if args.perturb:
        import torch

        from stet import train as _train

        _build = _train.build_model

        def build_perturbed(scale, season, init_seed, device):
            model = _build(scale, season, init_seed, torch.device("cpu"))
            g = torch.Generator().manual_seed(args.perturb_seed)
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(1 + args.perturb * torch.randn(p.shape, generator=g))
            return model.to(device)

        _train.build_model = build_perturbed

    season = load_season("s0")
    if args.canned:
        # This study measures what the ink limit is protecting against, so lift the limit.
        season.raw["ink_frac"] = 1.0
        _mat = materialize

        def lenient(corpus, script_path, season, scale, enc=None, check_exam=True):
            return _mat(corpus, script_path, season, scale, enc, check_exam=False)

        import stet.grade

        stet.grade.materialize = lenient
    scale = season.scale(args.scale)
    over = {k: type(getattr(scale, k))(v) for k, v in (kv.split("=") for kv in args.set)}
    if args.train_tokens:
        over["train_tokens"] = args.train_tokens
    if args.lr:
        over["lr"] = args.lr
    data_scale = scale  # the corpus and exam always come from the named scale
    scale = dataclasses.replace(scale, **over)
    device = pick_device(args.device)
    rows = []
    for r in args.rounds:
        rnd = practice_round(season, r, args.kind)
        t0 = time.time()
        corpus, exam = data.materialize(rnd, data_scale)
        print(f"== {rnd.id} {scale.name}: {len(corpus):,} docs, {corpus.total_bytes / 1e6:.0f} MB "
              f"(decontaminated {corpus.meta['decontaminated_docs']}), exam {sum(len(v) for v in exam.values())} items "
              f"[{time.time() - t0:.0f}s]", flush=True)
        for name in args.editors:
            out = data.round_dir(rnd, data_scale) / "exp" / name
            out.mkdir(parents=True, exist_ok=True)
            info = {"train_tokens": scale.train_tokens, "ink_frac": season.ink_frac, "ink_tokens": season.ink_tokens(scale),
                    "min_copy_bytes": season.min_copy_bytes, "time_limit": scale.edit_seconds}
            task = Task(info, str(corpus.path), str(out), str(out), time.time() + 3600)
            t0 = time.time()
            mod = importlib.import_module(f"stet.editors.{name}")
            lines = list(mod.edit(task))
            if args.reseed is not None:
                import random

                random.Random(args.reseed).shuffle(lines)
                name = f"{name}~{args.reseed}"
            if args.canned:
                import random

                # Stand-in for text an entrant could ship: passages from the same six sources
                # as the exam that this round is not examined on.
                other = data.load_exam_pool(season, "official" if rnd.split == "practice" else "practice")
                rng = random.Random(0)
                canned = [it["text"] for items in other.values() for it in items]
                rng.shuffle(canned)
                want = args.canned * scale.train_tokens * 3.3  # bytes of ink for roughly this share of the stream
                picked, got = [], 0
                for text in canned:
                    if got >= want:
                        break
                    picked.append({"t": text})
                    got += len(text.encode())
                every = max(1, len(lines) // max(1, len(picked)))
                merged = []
                for i, ln in enumerate(lines):
                    if i % every == 0 and picked:
                        merged.append(picked.pop())
                    merged.append(ln)
                lines = merged
                name = f"{name}+canned{args.canned:g}"
            task.write(lines)
            t_edit = time.time() - t0
            res = grade(rnd, scale, corpus, exam, out / "script.jsonl", device=device)
            if not res["valid"]:
                print(f"  {name:10s} INVALID: {res['invalid_reason']}", flush=True)
                continue
            row = {
                "round": rnd.id, "editor": name, "edit_s": round(t_edit, 1),
                "loss": res["exam"]["loss"], "bpb": res["exam"]["bits_per_byte"],
                "by_source": {k: v["loss"] for k, v in res["exam"]["by_source"].items()},
                "train_loss": res["train"]["final_train_loss"], "train_s": res["train"]["seconds"],
                "tps": res["train"]["tokens_per_second"], "kept_docs": res["script"]["kept_doc_frac"],
                "kept_bytes": res["script"]["kept_byte_frac"], "passes": round(res["script"]["passes"], 2),
                "ink_frac": res["script"]["ink_frac"], "tokens": scale.train_tokens, "lr": scale.lr, "over": over,
                "perturb": [args.perturb, args.perturb_seed] if args.perturb else None,
            }
            extra = ""
            if args.table:
                from count_table import exam_loss, fit

                from stet.tok import load_encoding

                enc = load_encoding(season)
                tokens, _ = materialize(corpus, out / "script.jsonl", season, scale, enc, check_exam=False)
                row["table_loss"] = round(exam_loss(fit(tokens, season.vocab_size), exam, enc, season.eot), 4)
                extra = f"  trigram table {row['table_loss']:.4f}"
            rows.append(row)
            src = " ".join(f"{k[:4]}={v:.3f}" for k, v in row["by_source"].items())
            print(f"  {name:10s} loss {row['loss']:.4f}{extra}  [{src}]  train {row['train_loss']:.3f} "
                  f"kept {100 * row['kept_docs']:.0f}% docs {100 * row['kept_bytes']:.0f}% bytes x{row['passes']} ink {100 * row['ink_frac']:.2f}% "
                  f"edit {row['edit_s']}s train {row['train_s']:.0f}s ({row['tps']:,} tok/s)", flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

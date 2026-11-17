#!/usr/bin/env python3
"""Run reference editors on practice rounds and print a table. A research tool, not the CLI.

    python tools/exp.py --scale nano --rounds 0 1 --editors identity shuffle english tidy
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stet import data
from stet.editor import Task
from stet.grade import grade
from stet.spec import load_season, official_round, practice_round
from stet.train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="nano")
    ap.add_argument("--rounds", type=int, nargs="+", default=[0])
    ap.add_argument("--editors", nargs="+", default=["identity", "shuffle", "english", "tidy"])
    ap.add_argument("--train-tokens", type=int, default=None, help="override the budget (for reference curves)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override scale fields, e.g. batch_size=8 n_layer=6")
    ap.add_argument("--repeat", type=int, default=1, help="repeat each run to measure noise")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wet", default=None, help="draw official-style rounds from a crawl: ID:PATHS_SHA256:N_SHARDS")
    ap.add_argument("--perturb", type=float, default=0.0, help="relative noise added to the initial weights (noise-floor studies)")
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--canned", type=float, default=0.0,
                    help="threat study: interleave this fraction of in-distribution text (exam passages from the OTHER split) as ink")
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
        season.raw["ink_frac"] = 1.0  # this study measures what the ink limit is protecting against
    scale = season.scale(args.scale)
    import dataclasses

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
        if args.wet:
            import hashlib

            crawl, sha, n = args.wet.split(":")
            beacon = {"round": 1, "randomness": hashlib.sha256(f"exp|{r}".encode()).hexdigest()}
            rnd = official_round(season, 900000 + r, beacon, {"crawl": crawl, "paths_sha256": sha, "n_shards": int(n)})
        else:
            rnd = practice_round(season, r)
        t0 = time.time()
        corpus, exam = data.materialize(rnd, data_scale)
        print(f"== {rnd.id} {scale.name}: {len(corpus):,} docs, {corpus.total_bytes / 1e6:.0f} MB "
              f"(decontaminated {corpus.meta['decontaminated_docs']}), exam {sum(len(v) for v in exam.values())} items "
              f"[{time.time() - t0:.0f}s]", flush=True)
        for name in args.editors:
            out = data.round_dir(rnd, data_scale) / "exp" / name
            out.mkdir(parents=True, exist_ok=True)
            task = Task(
                {"train_tokens": scale.train_tokens, "ink_frac": season.ink_frac, "time_limit": scale.edit_seconds},
                str(corpus.path), str(out), str(out), time.time() + 3600,
            )
            t0 = time.time()
            mod = importlib.import_module(f"stet.editors.{name}")
            lines = list(mod.edit(task))
            if args.canned:
                import random

                other = data.load_exam_pool(season, "official" if rnd.kind == "practice" else "practice")
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
            for rep in range(args.repeat):
                res = grade(rnd, scale, corpus, exam, out / "script.jsonl", device=device)
                row = {
                    "round": rnd.id, "editor": name, "rep": rep, "edit_s": round(t_edit, 1),
                    "loss": res["exam"]["loss"], "bpb": res["exam"]["bits_per_byte"],
                    "by_source": {k: v["loss"] for k, v in res["exam"]["by_source"].items()},
                    "train_loss": res["train"]["final_train_loss"], "train_s": res["train"]["seconds"],
                    "tps": res["train"]["tokens_per_second"], "kept_docs": res["script"]["kept_doc_frac"],
                    "kept_bytes": res["script"]["kept_byte_frac"], "passes": round(res["script"]["passes"], 2),
                    "tokens": scale.train_tokens, "lr": scale.lr, "over": over,
                }
                rows.append(row)
                src = " ".join(f"{k[:4]}={v:.3f}" for k, v in row["by_source"].items())
                print(f"  {name:10s} loss {row['loss']:.4f}  [{src}]  train {row['train_loss']:.3f} "
                      f"kept {100 * row['kept_docs']:.0f}% docs {100 * row['kept_bytes']:.0f}% bytes x{row['passes']} "
                      f"edit {row['edit_s']}s train {row['train_s']:.0f}s ({row['tps']:,} tok/s)", flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

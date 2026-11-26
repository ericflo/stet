# Stet

**The model is fixed. The trainer is fixed. The only thing you can touch is the data.**

Stet is an open competition to make the best training data. You enter a *curator*:
a program that is handed raw web pages it has never seen and twenty minutes on one GPU. It
decides what to keep, what to cut, and in what order. How is up to you: a few rules, a small
model you fine-tuned and run over every chunk, an agent harness driving a model of your own.
A small language model, the student, is trained on the result and sits an exam made when
the round is drawn. The bigger its gain over a student trained on the raw pages, the better.

[www.stetrun.com](https://www.stetrun.com) · *stet*: the proofreader's mark for "let it stand".

## Enter

```sh
curl -fsSL https://www.stetrun.com/install.sh | sh
stet init my-curator && cd my-curator
stet harvest              # once: recent human writing for practice exams, no account needed
stet run                  # edit, train, grade a practice round (12 GB+ GPU; ~1–4 min nano/micro, ~20–30 min standard)
stet login --github       # entering needs a GitHub account at least 30 days old
stet submit
```

`stet run` has no limit and uploads nothing: try as many ideas as you like, then submit
the ones worth volunteers' GPU time. Out of the box, practice rounds sit a bundled
openly-licensed CC-BY-SA Wikipedia practice pack (both twin-context comprehension and
reasoning); run `stet harvest` for fresh local hours. Using a coding agent? Tell it: *Read
https://www.stetrun.com/AGENTS.md and enter Stet.* · [How to enter](https://www.stetrun.com/enter)

## Lend a GPU

Scores are made on volunteers' machines: each curator is run by two people, each script
graded by someone else, all of it published.

```sh
stet login
stet verify               # grade edit scripts: plain data, nobody else's code
stet verify --edits       # also run entrants' curators, in a sandbox (Linux)
stet service install      # leave it running: starts with the machine, upgrades itself
```

[Verify scores](https://www.stetrun.com/verify) · [Run curators](https://www.stetrun.com/run)

## How it stays honest

- **Draw after commit.** A public randomness beacon picks each round's corpus from the
  newest Common Crawl after entries are locked.
- **An exam nobody can know.** Each round's exam is **100% fresh** and made at its draw,
  with zero static benchmarks, zero synthetic templates, and zero raw web perplexity. The
  official score is **100% fresh harvest-derived reasoning** (`explain`, `infer`, `synthesis`,
  one-third each, written once at the draw at `temp=0` by pinned `thinkingmachines/Inkling-2-Mini`
  with `reasoning_effort=low` from three hours of the day before, picked by the draw, weighted by
  every language's share of the crawl, with varied wrappers and an anti-copying filter, and
  published in full at settlement). Four fresh human-written twin-context comprehension kinds from
  those same three hours are sat and shown beside the score at `0%` weight as published
  diagnostics until a human-written measure passes DataDecide. Every student (`ERNIE-4.5-0.3B`,
  bottom half frozen, 65.5M tokens, ≤10 GiB VRAM so 12 GB+ NVIDIA GPUs can grade) sits the exam
  after a fixed 1-epoch commercial-permissive multilingual SFT probe. A curator runs with no
  network and can add no text of its own, so the exam cannot reach the student through it.
- **Only what tells curators apart.** The scored exam is validated before the season for
  multi-seed stability, resistance to QA-spam and over-filtering, and agreement with published
  1B pretraining rankings (DataDecide 1B with `Inkling-2-Mini`: **`58/66` (`87.9%`), `31/31`
  (`100%`) on `2pt+` pairs, Spearman `ρ = +0.860`, exact `#1` recipe `dclm-baseline-qc-20p`
  (`0.00pp` shortfall)**; earlier 2-draw `Qwen3-8B` baseline: `49/66` and `46/66`); parts
  that did not clear DataDecide (the four human-written twin-context kinds) carry `0%` weight
  and are shown only as diagnostics. See [Experiments](docs/EXPERIMENTS.md) for all validation
  numbers, including the head-to-head teacher comparison and perplexity-pruning baselines.
- **Copies only.** Everything the student reads is copied byte for byte from the round's
  corpus. A script holds no text of its own.
- **Checked by other people.** Two volunteers run each curator, and
  someone else grades each script (official Season 3 verifying runs on NVIDIA CUDA GPUs,
  cross-verified across RTX 5090, 4090 and 3090; AMD ROCm and Apple Silicon remain supported
  for local practice). A score that only one machine stands behind says so, and no score needs the
  organisers' machines to run or grade anything.
- **The same machine for every curator.** Six cores, 20 GB of RAM, 22 GB of GPU memory,
  up to 10 GB of pinned model weights, and a fixed list of libraries (`flash-linear-attention`,
  `transformers`, PyTorch, plus `vLLM` and `liger-kernel` on NVIDIA Linux), whoever runs it.
- **Paired scores.** Gain is measured against the unedited corpus on the same round, and
  the record needs a margin over shared rounds.
- **Finals decide.** The season's top eight, and one more drawn by lot, are trained again
  at a larger size, on two rounds of a crawl published after the board is frozen.
- **Everything else is public.** Curators, scripts, seeds, every settled round's exam and
  every score on every item of it, the harness.

## Train a policy on it

```python
from stet.gym import Env

env = Env(scale="nano")
obs = env.reset(0)                    # a raw corpus and the task
reward, info = env.step(edit_script)  # the gain, in bits per byte
```

## Layout

| Path | What |
|---|---|
| `src/stet/` | The harness and CLI (Python); `src/stet/exam/` makes and scores the exam |
| `server/` | The API and website (Go) |
| `examples/` | Reference curators, entered by the organisers: `identity`, `english`, `tidy`, `reader` (a 0.6B multilingual vLLM quality rater), and `dclm`, `fineweb-edu`, `heuristic`, `fineweb2` (published recipes, rebuilt); plus `mimic`, a regex/URL selection yardstick (~74–79% of `fineweb2` on Season 3's exam). Pattern, not entered: `agent` (an agent harness and a model served in the sandbox) |
| `cloud/` | `stet-cloud`, for the organisers: rents GPUs from Vast.ai, RunPod, Lium, Clore.ai and Salad to run curators and grade while volunteers are few (Go, built on its own) |
| `docs/` | [Rules](docs/RULES.md), [design](docs/DESIGN.md), [verifying](docs/VERIFYING.md), [experiments](docs/EXPERIMENTS.md), [API](docs/API.md), [FAQ](docs/FAQ.md), [operating](docs/OPERATING.md), and [the reasoning behind the exam](docs/research/exam-first-principles.md) |

## Develop

```sh
uv venv --python 3.13 && uv pip install -e '.[dev,run]'   # with --index for an AMD build of PyTorch
pytest && ruff check src tests tools examples
(cd server && go test ./...)
tools/dev.sh                            # the whole thing locally, on a real durable-db
```

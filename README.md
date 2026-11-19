# Stet

**The model is fixed. The trainer is fixed. The only thing you can touch is the data.**

Stet is an open competition to make the best training data. You enter a *curator*:
a program that is handed raw web pages it has never seen and twenty minutes on one GPU. It
decides what to keep, what to cut, and in what order. How is up to you: a few rules, a small
model you fine-tuned and run over every chunk, an agent harness driving a model of your own.
A small language model, the student, is trained on the result and sits an exam made after
the round is drawn. The bigger its gain over a student trained on the raw pages, the better.

[www.stetrun.com](https://www.stetrun.com) · *stet*: the proofreader's mark for "let it stand".

## Enter

```sh
curl -fsSL https://www.stetrun.com/install.sh | sh
stet init my-curator && cd my-curator
stet run                  # edit, train, grade a practice round: a minute or two on a GPU
stet login --github       # entering needs a GitHub account at least 30 days old
stet submit
```

`stet run` has no limit and uploads nothing: try as many ideas as you like, then submit
the ones worth volunteers' GPU time. Using a coding agent? Tell it: *Read
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
- **An exam nobody can know.** Each round's exam is made after its draw: human writing
  published in the two hours after it, and questions generated from its seed. Seven
  capabilities, each counting the same: knowledge, reading, reasoning, code, agents,
  writing and conversation.
- **Almost no ink.** 99.9% of what the student reads must be copied byte for byte from the
  round's corpus.
- **Checked by other people.** Two volunteers run each curator, on NVIDIA and AMD, and
  someone else grades each script. A score that only one machine stands behind says so,
  and no score needs the organisers' machines to run or grade anything.
- **The same machine for every curator.** Six cores, 12 GB, 8 GB of GPU memory, a fixed
  list of libraries, whoever runs it.
- **Paired scores.** Gain is measured against the unedited corpus on the same round, and
  the record needs a margin over shared rounds.
- **Finals decide.** The season's top eight are trained again at a larger size, on a fresh
  round.
- **Everything else is public.** Curators, scripts, seeds, every settled round's exam, the
  harness.

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
| `examples/` | Reference curators: `identity`, `english`, `tidy`; `reader` (a small language model reads the pages); `agent` (an agent harness and a model served in the sandbox); `dclm`, `fineweb-edu`, `heuristic`, `fineweb2` (published recipes, rebuilt) |
| `runpod/` | `stet-runpod`, for the organisers: rents GPUs from RunPod to grade while volunteers are few (Go, built on its own) |
| `docs/` | [Rules](docs/RULES.md), [design](docs/DESIGN.md), [verifying](docs/VERIFYING.md), [experiments](docs/EXPERIMENTS.md), [API](docs/API.md), [FAQ](docs/FAQ.md), [operating](docs/OPERATING.md) |

## Develop

```sh
uv venv --python 3.13 && uv pip install -e '.[dev,run]'   # with --index for an AMD build of PyTorch
pytest && ruff check src tests tools examples
(cd server && go test ./...)
tools/dev.sh                            # the whole thing locally, on a real durable-db
```

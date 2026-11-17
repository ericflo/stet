# Stet

**The model is fixed. The trainer is fixed. The only thing you can touch is the data.**

Stet is an open competition for programs that edit pretraining data. Your *editor* is
handed raw web pages it has never seen and a few minutes on one GPU. It decides what to
keep, what to cut, and in what order. A small language model is trained on the result and
sits an exam. Lowest loss wins.

[www.stetrun.com](https://www.stetrun.com) · *stet*: the proofreader's mark for "let it stand".

## Enter

```sh
curl -fsSL https://www.stetrun.com/install.sh | sh
stet init my-editor && cd my-editor
stet run                  # edit, train, grade a practice round: about a minute on a GPU
stet login --github       # entering needs a GitHub account at least 30 days old
stet submit
```

Using a coding agent? Tell it: *Read https://www.stetrun.com/AGENTS.md and enter Stet.*

## Lend a GPU

Every score is graded on an organiser's machine, then replicated by volunteers, and the
replications are published. Grading runs none of the entrants' code.

```sh
stet login && stet verify
```

## How it stays honest

- **Draw after commit.** A public randomness beacon picks each round's corpus from the
  newest Common Crawl after entries are locked.
- **Almost no ink.** 99.9% of what the student reads must be copied byte for byte from the
  round's corpus.
- **Edit here, grade anywhere.** Editors run on the organisers' machines and publish an
  edit script, which is plain data. Anyone's GPU can re-grade it.
- **Paired scores.** Gain is measured against the unedited corpus on the same round, and
  the record needs a margin over shared rounds.
- **A held-out exam.** A second exam, drawn in secret from the same collections and
  committed to by hash, decides the season's final table.
- **Everything else is public.** Editors, scripts, seeds, the exam, the harness.

## Train a policy on it

```python
from stet.gym import Env

env = Env(scale="nano")
obs = env.reset(0)                    # a raw corpus and the task
reward, info = env.step(edit_script)  # the gain, in nats per token
```

## Layout

| Path | What |
|---|---|
| `src/stet/` | The harness and CLI (Python) |
| `server/` | The API and website (Go) |
| `examples/` | Reference editors: `identity`, `english`, `tidy` |
| `docs/` | [Rules](docs/RULES.md), [design](docs/DESIGN.md), [verifying](docs/VERIFYING.md), [experiments](docs/EXPERIMENTS.md), [API](docs/API.md), [FAQ](docs/FAQ.md) |

## Develop

```sh
uv venv && uv pip install -e '.[dev]'   # plus a PyTorch build for your machine
pytest && ruff check src tests tools examples
(cd server && go test ./...)
tools/dev.sh                            # the whole thing locally, on a real durable-db
```

# Stet

**The model is fixed. The trainer is fixed. The only thing you can touch is the data.**

Stet is an open competition for programs that edit pretraining data. Your *editor* is
handed raw web pages it has never seen and ten minutes on one GPU. It decides what to
keep, what to cut, and in what order. How is up to you: a few rules, a small model you
fine-tuned and run over every chunk, an agent harness driving a model of your own. A
small language model, the student, is trained on the result and sits an exam. Lowest
loss wins.

[www.stetrun.com](https://www.stetrun.com) · *stet*: the proofreader's mark for "let it stand".

## Enter

```sh
curl -fsSL https://www.stetrun.com/install.sh | sh
stet init my-editor && cd my-editor
stet run                  # edit, train, grade a practice round: a minute or two on a GPU
stet login --github       # entering needs a GitHub account at least 30 days old
stet submit
```

Using a coding agent? Tell it: *Read https://www.stetrun.com/AGENTS.md and enter Stet.*

## Lend a GPU

Scores are made on volunteers' machines: each editor is run by two people, each script
graded by someone else, all of it published.

```sh
stet login
stet verify            # grade edit scripts: plain data, nobody else's code
stet verify --edits    # also run editors, in a sandbox (Linux)
```

## How it stays honest

- **Draw after commit.** A public randomness beacon picks each round's corpus from the
  newest Common Crawl after entries are locked.
- **Almost no ink.** 99.9% of what the student reads must be copied byte for byte from the
  round's corpus.
- **Checked by other people.** Two volunteers run each editor, on NVIDIA and AMD, and
  someone else grades each script. A score that only one machine stands behind says so,
  and no score needs the organisers' hardware.
- **The same machine for every editor.** Six cores, 24 GB, 8 GB of GPU memory, a fixed
  list of libraries, whoever runs it.
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
| `examples/` | Reference editors: `identity`, `english`, `tidy`; `reader` (a small language model reads the pages); `agent` (an agent harness and a model served in the sandbox) |
| `docs/` | [Rules](docs/RULES.md), [design](docs/DESIGN.md), [verifying](docs/VERIFYING.md), [experiments](docs/EXPERIMENTS.md), [API](docs/API.md), [FAQ](docs/FAQ.md) |

## Develop

```sh
uv venv --python 3.13 && uv pip install -e '.[dev,run]'   # with --index for an AMD build of PyTorch
pytest && ruff check src tests tools examples
(cd server && go test ./...)
tools/dev.sh                            # the whole thing locally, on a real durable-db
```

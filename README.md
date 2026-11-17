# Stet

**The model is fixed. The trainer is fixed. The only thing you can touch is the data.**

Stet is an open competition for programs that edit pretraining data. Your *editor* is
handed raw web text it has never seen and a few minutes on one GPU. It decides what to
keep, what to cut and in what order. A small language model is trained on the result and
sits an exam. Lower exam loss wins.

*stet*: the proofreader's mark for "let it stand".

## Try it

```sh
curl -fsSL https://www.stetrun.com/install.sh | sh
stet init my-editor && cd my-editor
stet run            # edit, train, grade a practice round (about a minute on a GPU)
stet login && stet submit
```

Working with a coding agent? `stet init` drops an `AGENTS.md` with the whole brief.

## Lend a GPU

Scores come from volunteers, not a central cluster.

```sh
stet login && stet verify
```

Editors run in a sandbox with no network. Validated work is credited on the verifier
leaderboard, and your own entries are run first.

## How it works

- **Commit, then draw.** Editors are uploaded before a public randomness beacon picks the
  round's corpus and exam. Nobody can tune to a round, the organisers included.
- **Fresh data.** Official rounds use the newest Common Crawl crawl. Practice rounds use a
  different, pinned dataset.
- **Paired scoring.** Gain = unedited corpus's exam loss minus yours, same round, same
  initial weights. The headline number is how many times more raw data the model would
  have needed to do as well.
- **Replicated verification.** Results are sealed until independent verifiers agree.
- **Everything is public.** Editors, seeds, exam and harness. Any score can be re-run by
  anyone, and `stet fork` starts you from any editor on the board.

## Layout

| Path | What |
|---|---|
| `src/stet/` | The harness and CLI (Python) |
| `server/` | The API and website (Go) |
| `docs/DESIGN.md` | The design and the reasoning behind it |
| `docs/EXPERIMENTS.md` | Measurements the rules are based on |
| `docs/research/` | Prior art and mechanics research |

## Develop

```sh
uv venv && uv pip install -e '.[dev]'   # plus a torch build for your machine
pytest
cd server && go test ./...
```

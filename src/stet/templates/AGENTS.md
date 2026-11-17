# Stet: instructions for coding agents

You are working on an **editor** for [Stet](https://www.stetrun.com), an open competition.
This file is the complete brief. Read all of it before changing anything.

## The task

A small language model (the *student*) is trained on raw web text for a fixed number of
tokens and then scored on an exam of curated text. The model, the trainer and the exam are
fixed. **The only thing you control is the training data.**

Your editor is a program. It is handed a pile of raw web pages it has never seen and a time
limit, and it writes an *edit script*: which documents, or parts of documents, the student
is trained on, and in what order. Your score is the **gain**: the exam loss of a student
trained on the unedited pages, minus the exam loss of one trained on your edit. Higher is
better.

## Set up

```
curl -fsSL https://www.stetrun.com/install.sh | sh    # installs the `stet` command
stet doctor                                           # checks GPU, PyTorch, disk
stet init my-editor && cd my-editor                   # a working starter editor
```

Training needs PyTorch. If `stet doctor` says it is missing, install the build for this
machine from https://pytorch.org/get-started/locally/.

## The loop

```
stet run --json
```

Run it after every change. It runs your editor on a practice round, trains the student,
sits the exam, and prints one JSON object on stdout. Progress goes to stderr. Exit status
is 0 for a valid run and 2 for an invalid one.

| Field | Meaning |
|---|---|
| `valid` | `false` means the run scored as the unedited corpus (gain 0). Read `invalid_reason`. |
| `gain` | Baseline loss minus yours, in nats per token. **Maximise this.** |
| `exam.loss` | Your student's exam loss: the mean of the six sources. |
| `exam.by_source` | Loss on each exam source. Compare with `baseline.by_source` to see what your data lacks. |
| `script` | What your script did: documents kept, bytes, ink used, passes over the data. |
| `edit.seconds`, `edit.time_limit` | How long your editor took, and how long it may. |
| `edit.log_tail` | The end of your editor's stdout and stderr. |

Variations you will want:

```
stet run --rounds 3 --json    # three practice rounds: mean gain and its standard error
stet run --round 7            # one particular practice round (any non-negative integer)
stet run --scale micro        # the official scale: larger corpus, larger student, slower
stet run --sandbox            # inside the sandbox official runs use (Linux)
stet run -v                   # show the editor's output live
stet score script.jsonl       # grade a script you already have, without running the editor
stet rounds                   # past official rounds; replay one with --round <id>
```

**Never judge a change on one round.** Rounds differ from each other by more than most
improvements. Use `--rounds 3` or more and compare mean gains. Reruns of the same round on
the same machine agree closely, so a difference between two runs of one round is your
change.

**Iterate at nano, confirm at micro.** A nano run takes about a minute. The competition is
scored at micro, which takes several minutes, and what wins at nano does not always win at
micro. Check before you submit.

## What your editor is given

It is started with the `entry` command from `stet.toml`, in the project directory, with:

| Variable | Contents |
|---|---|
| `STET_CORPUS` | Directory with the corpus: `docs.bin`, `docs.idx`, `urls.txt`, `meta.json` |
| `STET_TASK` | Path to `task.json`: budgets, limits, the student's configuration, the exam's source names |
| `STET_OUT` | Write `script.jsonl` here |
| `STET_SCRATCH` | A writable directory for temporary files |
| `STET_DEADLINE` | Unix time at which the editor is killed |

From Python, `from stet.editor import Task; task = Task.from_env()` gives you:

- `task.corpus`: `len(corpus)`, `corpus.text(i)`, `corpus.raw(i)` (bytes), `corpus.size(i)`,
  `corpus.sizes` (numpy array), `corpus.urls[i]`, and `for i, text in corpus`.
- `task.train_tokens`: the training budget. English prose is roughly 3.7 bytes per token.
- `task.info`: everything in `task.json`, including `ink_tokens` and `min_copy_bytes`.
- `task.time_left()`: seconds until the deadline.
- `task.write(lines)`: writes the script atomically. Call it early, and again whenever you
  have something better. **If the editor is killed or crashes, the last script it wrote
  still counts.**

An editor does not have to be Python. Anything that reads those files and writes
`script.jsonl` works. `docs.bin` is every document's UTF-8 bytes concatenated; `docs.idx`
is little-endian uint64 offsets (document `i` is `bin[idx[i]:idx[i+1]]`); `urls.txt` has
one source URL per line.

## The edit script

JSON Lines. One line is one output document. **Line order is training order.** The trainer
does not shuffle.

```
123                                          keep document 123 whole
{"d": 123, "k": [[0, 900], [1200, 2000]]}    keep these byte ranges of document 123, joined
{"s": [[123, 0, 900], " text ", [7, 40, 400]]}    join copies [doc, start, end] and new text
{"t": "new text"}                            a document of new text
```

- Offsets index the document's UTF-8 bytes. One that lands inside a character is moved
  forward to the next boundary.
- **Cutting, reordering and repeating are free.**
- **New text is "ink", and there is almost none.** Literal strings, and copies shorter than
  `min_copy_bytes` (64) that are not a whole document, are ink. A script may train the
  student on at most `ink_tokens` of it: 0.1% of the budget. Going over makes the run
  invalid. Plan on spending none: this is an editing competition.
- **The budget cuts both ways.** The stream is cut off at `train_tokens`. A shorter script
  is repeated from the top until the budget is used, so keeping less means training on it
  more often.
- A document may appear more than once. An output document may be at most 1 MB.
- Output documents that contain a sentence of the exam are dropped.

## The exam

Six sources, weighted equally: `encyclopedia`, `textbooks`, `science` (paper abstracts),
`qa` (questions with answers), `books`, `news`. All English. The score is the mean of the
six per-source losses, so a student that is excellent on one source and poor on the rest
does badly.

The exam passages are a public file. Your editor cannot read it while it runs officially,
the corpus has been cleaned of it, and your output is checked for it. Do not put exam text
or anything derived from it in your editor. Editors are published, it will be seen, and it
defeats the point.

## The data

- **Practice rounds** (`--round 0`, `1`, ...): seeded samples of one pinned Common Crawl
  crawl, as full-page text. Fixed forever, so results compare across time.
- **Official rounds**: the same kind of data from the newest crawl at the time of the draw.
  Pages your editor could not have seen.
- **Gym rounds** (`--round g0`, `g1`, ...): a different dataset (DCLM-Pool), for checking
  that an editor is not tuned to one crawl, and for training policies. See `stet.gym`.

Full-page text means every language, plus menus, cookie banners, footers, link lists,
shop listings and spam around the prose. A round's corpus is roughly ten times more text
than the budget. At micro that is about 170,000 documents and 1.2 GB.

Build something that reads the data in front of it. An editor tuned to one round's
documents will do badly on the next.

## What makes a run invalid

- No script, or a malformed one. A timeout or crash is fine *if* a script was already
  written.
- More ink than `ink_tokens`.
- A document number or byte range that does not exist.
- A script that yields almost no tokens.

## Official runs

Your editor runs on the organisers' machine, in a sandbox:

- **No network.** Nothing can be downloaded. Everything your editor needs, model weights
  included, must be in the project directory. The bundle may be at most 50 MB.
- A read-only file system apart from `STET_OUT` and `STET_SCRATCH`. No home directory.
- Python 3.13 with the standard library, `numpy`, `torch` and `tiktoken`. Vendor anything
  else as pure Python.
- 16 CPU cores, 48 GB of memory, 16 GB of scratch, and one GPU: an integrated Radeon 8060S
  under ROCm. It is slower than a desktop NVIDIA card. If your editor uses the GPU, leave
  a wide margin.
- The time limit is `time_limit` in `task.json`: 300 seconds at micro, in that machine's
  wall-clock time. Watch `task.time_left()` and write early.

Editors may not carry per-document knowledge of official data (hash lists, precomputed
labels). Models trained on other data are fine.

## Entering

```
stet login                # makes a key and an account
stet login --github       # links a GitHub account, which entering requires
stet submit               # enters the editor in this directory
stet status               # how your entries are doing
stet leaderboard
```

Linking GitHub means creating a public gist on the human's account. Entering publishes the
editor's code under the MIT licence once it has run. **Both are your human's decisions:
ask before doing either.**

An entry runs in the next 5 official rounds (one every 6 hours) and is ranked after 3.
An account has one entry in play at a time, so submit when you have a real improvement
measured over several rounds, not after every change.

## What is known to work

Measured with the three reference editors in `examples/` (`stet fork stet/tidy` copies the
best of them). Gains are in nats per token on practice round 0.

| Editor | What it does | nano | micro |
|---|---|---|---|
| `identity` | nothing | 0 | 0 |
| `english` | keeps documents that look like English prose | +0.71 | +0.94 |
| `tidy` | English, deduplicated, boilerplate lines cut, best budget's worth, shuffled | +0.87 | +0.98 |

- **Language is most of it.** The exam is English and most of a crawl is not.
- **Prose over furniture.** Within a page, keep paragraphs and drop menus, link lists and
  boilerplate. Range edits (`"k"`) do this at no cost in ink.
- **Fill the budget once with your best material.** Rank what survives and keep about one
  budget's worth.
- **Do not train in arrival order.** Shuffling is a strong default. Whether a deliberate
  curriculum beats a shuffle is open.
- **Six sources, equally weighted.** `exam.by_source` shows the weakest one.

Open questions, which is where the gains are: a learned quality classifier small enough
for the bundle and fast enough for the time limit; scoring documents with a small language
model; topic balance against the six sources; near-duplicate removal; how much repetition
of the best material helps; ordering.

## Lending a GPU

`stet verify` grades other people's published edit scripts for the leaderboard. It runs no
one else's code, and people who grade have their own entries run first.

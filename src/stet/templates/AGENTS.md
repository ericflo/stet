# Stet: instructions for coding agents

You are working on an **editor** for [Stet](https://stet.maxint.co), an open competition.
Read this file fully before changing anything. It is the complete brief.

## The task in one paragraph

A small language model (the *student*) is trained on raw web text for a fixed number of
tokens and then scored on an exam of curated text. The model, the trainer and the exam are
fixed. **The only thing you control is the training data.** Your editor is a program that
is handed a pile of raw web documents it has never seen and a time limit. It writes an
*edit script*: which documents (or parts of documents) to train on, and in what order.
Lower exam loss wins. Your score is the **gain**: the unedited corpus's exam loss minus
yours, on the same round.

## The loop

```
stet run --json          # run the editor on a practice round, train, grade; JSON on stdout
```

Run that after every change. It prints one JSON object. The fields that matter:

| Field | Meaning |
|---|---|
| `valid` | `false` means the run scored as the unedited corpus (gain 0). Read `invalid_reason`. |
| `gain` | Baseline loss minus yours, in nats per token. **Maximise this.** |
| `exam.loss` | Your student's exam loss (mean of the six sources). |
| `exam.by_source` | Loss per exam source. Look here for what your data is missing. |
| `script` | What your script did: documents kept, bytes, ink used, passes over the data. |
| `edit.seconds`, `edit.time_limit` | How long your editor ran, and how long it may. |
| `edit.log_tail` | The end of your editor's stdout and stderr. |

Human-readable progress goes to stderr. Exit status is 0 for a valid run, 2 for an invalid
one.

Useful variations:

```
stet run --round 3            # a different practice round (any non-negative integer)
stet run --scale micro        # the official scale: bigger corpus, bigger student, slower
stet run --sandbox            # run inside the sandbox verifiers use (Linux)
stet run -v                   # show the editor's output live
stet rounds                   # list past official rounds; replay one with --round <id>
stet doctor                   # check GPU, sandbox and disk
```

**Never judge a change on one round.** Round-to-round differences are larger than most
improvements. Compare on at least three rounds (`--round 0`, `1`, `2`, ...) and look at
the mean gain. Results on one machine are repeatable, so any difference between two runs
of the same round is caused by your change.

## What your editor is given

It is started as the `entry` command in `stet.toml`, with its working directory set to the
project, and these environment variables:

| Variable | Contents |
|---|---|
| `STET_CORPUS` | Directory with the corpus: `docs.bin`, `docs.idx`, `urls.txt`, `meta.json` |
| `STET_TASK` | Path to `task.json`: budgets, limits, the student's configuration, exam source names |
| `STET_OUT` | Write `script.jsonl` here |
| `STET_SCRATCH` | A writable directory for temporary files |
| `STET_DEADLINE` | Unix time at which the editor is killed |

From Python, `from stet.editor import Task; task = Task.from_env()` gives you:

- `task.corpus`: `len(corpus)`, `corpus.text(i)`, `corpus.raw(i)` (bytes), `corpus.size(i)`,
  `corpus.sizes` (numpy array), `corpus.urls[i]`, and `for i, text in corpus`.
- `task.train_tokens`: the training budget. Roughly 3.3 bytes of English prose per token.
- `task.info`: everything in `task.json`.
- `task.time_left()`: seconds until the deadline.
- `task.write(lines)`: writes the script atomically. Call it early and again whenever you
  have something better. **If the editor is killed or crashes, the last script it wrote
  still counts.**

Editors do not have to be Python. Any program that reads those files and writes
`script.jsonl` works. The corpus format: `docs.bin` is every document's UTF-8 bytes
concatenated; `docs.idx` is little-endian uint64 offsets (document `i` is
`bin[idx[i]:idx[i+1]]`); `urls.txt` has one source URL per line.

## The edit script

JSON Lines. One line is one output document. **Line order is training order**: the trainer
does not shuffle.

```
123                                 keep document 123 whole
{"d": 123, "k": [[0, 900], [1200, 2000]]}   keep these byte ranges of document 123, joined
{"s": [[123, 0, 900], " new text ", [7, 40, 400]]}   mix copies [doc, start, end] and new text
{"t": "new text"}                   a document that is entirely new text
```

- Byte offsets index the document's UTF-8 bytes; they are moved forward to a character
  boundary if they land inside one.
- **Cutting, reordering and repeating are free.**
- **New text is "ink" and is rationed.** Literal strings, and any copy shorter than
  `min_copy_bytes`, count as ink. Ink may be at most `ink_frac` of the bytes trained on
  (both are in `task.json`). Going over makes the run invalid.
- **The budget cuts both ways.** The stream is cut off at `train_tokens`. If your script is
  shorter than that, it is repeated from the top until the budget is used. Keeping less
  therefore means training on it more times.
- A document may appear more than once. An output document may be at most 1 MB.

## The exam

Six sources, weighted equally: `encyclopedia` (Wikipedia), `textbooks` (open textbooks),
`science` (arXiv abstracts), `qa` (Stack Exchange questions and answers), `books` (Project
Gutenberg), `news`. All English. The score is the mean of the six per-source losses, so a
student that is excellent on one source and poor on the rest does badly.

The exam passages are public. Your editor cannot read them while it runs, and training
data that reproduces exam text is removed. Do not try to smuggle exam text in; it is
checked, editors are public, and it defeats the point.

## The data

Practice rounds and official rounds use **different datasets on purpose**:

- **Practice** (`--round 0`, `1`, ...): DCLM-Pool, raw Common Crawl up to 2022 with
  main-content extraction. About a third is not English. Fixed forever, so results are
  comparable across time.
- **Official**: the newest Common Crawl crawl at the time of the draw, as full-page text.
  Every language, and much more page furniture: menus, cookie banners, footers, link lists.
  Your editor is judged on pages that were crawled after you could have seen them.

An editor that only works on the practice data will not do well. Replay past official
rounds (`stet rounds`, then `stet run --round <id>`) to see what official data looks like.
Build something that reads the data in front of it rather than something tuned to one
round.

## Rules that will invalidate a run

- No script written, or a malformed one. (Timeouts and crashes are fine *if* a script was
  already written.)
- Ink over the limit.
- A document id or byte range that does not exist.
- A script that yields almost no tokens.

## Rules of the competition

- In official runs the editor has **no network**, a read-only filesystem apart from
  `STET_OUT` and `STET_SCRATCH`, and cannot see the home directory. Do not depend on
  downloading anything at run time.
- The time limit is in `task.json` (`time_limit`, seconds) and is enforced by killing the
  process. It is stated for a reference machine and scaled on others.
- One GPU. Use it if you like (`torch` is available if the machine has it). Do not assume
  more than 16 GB of GPU memory or 24 GB of RAM.
- Third-party Python packages beyond the standard library, `numpy` and `torch` are not
  guaranteed to exist where your editor is verified. Vendor pure-Python code into the
  project if you need it.
- Model weights and other large files go in `[[assets]]` in `stet.toml` (URL plus sha256),
  not in the project directory. The bundle of project files must stay under 50 MB.
- An editor may not contain per-document knowledge of the official data (lists of document
  hashes, precomputed labels). Models trained on other data are fine.
- Editors become public, under the MIT licence, once they have run in an official round.
  That is how everyone learns from each other. `stet fork <editor>` copies any of them.

## What tends to work

Measured on the reference editors that ship with Stet (`stet.editors`):

- **Language**: the exam is English. Dropping documents that are not English is the single
  largest gain.
- **Prose over furniture**: within a page, keep paragraphs and drop menus, link lists,
  boilerplate and tables of numbers. Range edits (`"k"`) do this.
- **Deduplicate**: exact and near-duplicate documents waste budget.
- **Fill the budget with your best material**: rank what survives and keep roughly the
  budget's worth, rather than keeping everything or keeping too little.
- **Order**: the trainer follows your order exactly, with a warm-up and then a decaying
  learning rate. Arrival order is bad (documents from one site arrive together). Shuffling
  is a strong default. Whether a deliberate curriculum beats a shuffle is an open question.
- **Balance**: six equally weighted sources. Check `exam.by_source` for the weakest one.

Ideas nobody has settled: learned quality classifiers, scoring documents with a small
language model trained during the edit phase, using the URL, curriculum by difficulty,
repeating high-value documents, repairing text with ink, picking a different amount of data
for different student sizes.

## How to work

1. `stet run --json` first, unchanged, to get a starting number.
2. Look at the data before theorising: print a few dozen random documents from
   `~/.cache/stet/rounds/<round>/<scale>/corpus` (or `STET_CORPUS` inside the editor).
3. Change one thing. Run three rounds. Keep it only if the mean gain went up.
4. Keep the editor well inside the time limit; leave a margin, and write a script early.
5. Record what you tried and what it did in `NOTES.md` so the next session does not repeat
   it.
6. Before submitting, run `stet run --scale micro --sandbox` once.

## Submitting

```
stet login               # once: creates a key and an account
stet submit              # uploads this project as an entry
stet status              # how it is doing
stet leaderboard
```

Ask the person you are working for before submitting: entries are public and carry their
name.

# Stet: instructions for coding agents

You are working on a **curator** for [Stet](https://www.stetrun.com), an open competition.
This file is the complete brief. Read all of it before changing anything.

## The task

A small language model (the *student*) is trained on raw web text for a fixed number of
tokens and then scored on an exam of curated text. The model, the trainer and the exam are
fixed. **The only thing you control is the training data.**

Your curator is a program. It is handed a pile of raw web pages it has never seen and a time
limit, and it writes an *edit script*: which documents, or parts of documents, the student
is trained on, and in what order. Your score is the **gain**: the exam loss of a student
trained on the unedited pages, minus the exam loss of one trained on your edit. Higher is
better.

## Set up

```
curl -fsSL https://www.stetrun.com/install.sh | sh    # installs the `stet` command
stet doctor                                           # checks GPU, libraries, disk
stet init my-curator && cd my-curator                 # a working starter curator
```

The installer brings its own Python 3.13, PyTorch in the build for this machine's GPU
(NVIDIA or AMD on Linux, Metal on a Mac), and every library a curator may import, at the
versions official runs use. `stet doctor` says if anything differs.

Words used below: the *student* is the small model that gets trained. A *nat* is the unit
of its loss. *Ink* is text in the training data that was not copied from the corpus. A
*verifier* is a volunteer's machine that runs curators or grades their scripts.

## The loop

```
stet run --json
```

Run it after every change. It runs your curator on a practice round, trains the student,
sits the exam, and prints one JSON object on stdout. Progress goes to stderr. Exit status
is 0 for a valid run and 2 for an invalid one.

| Field | Meaning |
|---|---|
| `valid` | `false` means the run scored as the unedited corpus. Read `invalid_reason`. |
| `gain` | Baseline loss minus yours, in nats per token; 0 for an invalid run. **Maximise this.** |
| `exam.loss` | Your student's exam loss: the mean of the six sources. |
| `exam.by_source` | Each exam source: `exam.by_source.<source>.loss`. Compare with `baseline.by_source.<source>` (a plain number) to see what your data lacks. |
| `script` | What your script did: documents kept, bytes, ink used, passes over the data. |
| `edit.seconds`, `edit.time_limit` | How long your curator took, and how long it may. |
| `edit.log_tail` | The end of your curator's stdout and stderr. |

With `--rounds N` the JSON is a summary instead: `rounds` (ids), `gains`, `mean_gain` and
`se` (its standard error). Every run's full result is also saved under
`.stet/runs/<time>-<round>-<scale>/result.json`.

Variations you will want:

```
stet run --rounds 3 --json    # three practice rounds: mean gain and its standard error
stet run --round 7            # one particular practice round (any non-negative integer)
stet run --scale micro        # the official scale: larger corpus, larger student, slower
stet run --sandbox            # inside the sandbox official runs use (Linux)
stet run --time-limit 300     # with a shorter time limit than the scale's
stet run -v                   # show the curator's output live
stet score script.jsonl       # grade a script you already have, without running the curator
stet rounds                   # past official rounds; replay one with --round <id>
stet score stet/tidy --round s0-o00001    # re-grade a published entry of an official round
                                          # (stet/tidy@v2 when a round holds two versions)
stet clean                    # delete downloaded rounds and model files (fetched again when needed;
                              # unused ones are deleted after 14 days anyway)
```

**Never judge a change on one round.** Rounds differ from each other by more than most
improvements. Use `--rounds 3` or more and compare mean gains. The same script on the
same round and machine gives the same number every time. But a change that only shuffles
the same documents moves the loss by about 0.01, so on one round a difference under about
0.03 is noise. Confirm anything smaller over `--rounds 5`.

**Iterate at nano, confirm at micro.** A nano run takes a minute or two. The competition is
scored at micro, which takes several minutes, and what wins at nano does not always win at
micro. Check before you submit.

## What your curator is given

It is started with the `entry` command from `stet.toml`, in the project directory, with:

| Variable | Contents |
|---|---|
| `STET_CORPUS` | Directory with the corpus: `docs.bin`, `docs.idx`, `urls.txt`, `meta.json` |
| `STET_TASK` | Path to `task.json`: budgets, limits, the student's configuration, the exam's source names |
| `STET_OUT` | Write `script.jsonl` here |
| `STET_SCRATCH` | A writable directory for temporary files |
| `STET_ASSETS` | Directory with the model files `stet.toml` names, read-only |
| `STET_DEADLINE` | Unix time at which the curator is killed |

From Python, `from stet.curator import Task; task = Task.from_env()` gives you:

- `task.corpus`: `len(corpus)`, `corpus.text(i)`, `corpus.raw(i)` (bytes), `corpus.size(i)`,
  `corpus.sizes` (numpy array), `corpus.urls[i]`, and `for i, text in corpus`.
- `task.train_tokens`: the training budget. English prose is roughly 3.7 bytes per token.
- `task.info`: everything in `task.json`, including `ink_tokens` and `min_copy_bytes`.
- `task.time_left()`: seconds until the deadline.
- `task.asset("name")`: the path of a model file or directory named in `stet.toml`.
- `task.device()`: `"cuda"` when there is a GPU PyTorch can use (NVIDIA or AMD), else `"cpu"`.
- `task.cores`, `task.memory_gb`, `task.gpu_memory_gb`: what this run has been given.
- `task.write(lines)`: writes the script atomically. Call it early, and again whenever you
  have something better. **If the curator is killed or crashes, the last script it wrote
  still counts.**

A curator does not have to be Python. Anything that reads those files and writes
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
- **New text is "ink", and there is almost none.** Literal strings and copies shorter than
  `min_copy_bytes` (64) are ink. The one exception is a whole short document kept on a
  line of its own; joined to anything else in an `s` line, it is ink too. A script may train the
  student on at most `ink_tokens` of it: 0.1% of the budget. Going over makes the run
  invalid. Plan on spending none: this is an editing competition.
- **The budget cuts both ways.** The stream is cut off at `train_tokens`. A shorter script
  is repeated from the top until the budget is used, so keeping less means training on it
  more often.
- A document may appear more than once. An output document may be at most 1 MB.
- Output documents that contain the opening 64 bytes of a long exam sentence are dropped.

## The exam

Six sources, weighted equally: `encyclopedia`, `textbooks`, `science` (paper abstracts),
`qa` (questions with answers), `books`, `news`. All English. The score is the mean of the
six per-source losses, so a student that is excellent on one source and poor on the rest
does badly.

The season's final table is worked out from a second, held-out exam that only the
organisers hold: the same six collections, other documents. Fit the kind of writing, not
the passages.

The exam passages are a public file. Your curator cannot read it while it runs officially,
the corpus has been cleaned of it, and your output is checked for it. Do not put exam text
or anything derived from it in your curator. Curators are published, it will be seen, and it
defeats the point.

## The data

- **Practice rounds** (`--round 0`, `1`, ...): seeded samples of one pinned Common Crawl
  crawl, as full-page text. Fixed forever, so results compare across time.
- **Official rounds**: the same kind of data from the newest crawl at the time of the draw.
  Which of the crawl's files a round gets is decided by a public randomness beacon after
  entries close, so nobody can know the sample in advance. The crawl itself is public,
  and practice rounds may come from the same one.
- **Gym rounds** (`--round g0`, `g1`, ...): a different dataset (DCLM-Pool), for checking
  that a curator is not tuned to one crawl, and for training policies. See `stet.gym`.

Full-page text means every language, plus menus, cookie banners, footers, link lists,
shop listings and spam around the prose. A round's corpus is roughly ten times more text
than the budget. At micro that is about 170,000 documents and 1.2 GB.

Build something that reads the data in front of it. A curator tuned to one round's
documents will do badly on the next.

## What makes a run invalid

- No script, or a malformed one. A timeout or crash is fine *if* a script was already
  written. `task.write` replaces the file in one step; if you write it yourself, write
  to a temporary file and rename it, or a kill can leave half a script.
- More ink than `ink_tokens`. A copied range shorter than 64 bytes counts as ink.
- A document number or byte range that does not exist.
- A script that yields almost no tokens.
- A script over 64 MB, a line over 1 MB, or a line nested more than 16 brackets deep.

More than 10,000 files in `STET_OUT` is an error too, though a script already written
still counts.

## Official runs

Your curator is run by volunteers, on their machines, twice, by two different people: on
an NVIDIA GPU and on an AMD one when both are around. Your gain for a round is the mean
of the two runs. A run that fails scores 0, so **it has to work on both**.

Every run gets the same machine, whatever the volunteer has:

| | |
|---|---|
| CPU | 6 cores |
| Memory | 12 GB |
| GPU | one, NVIDIA (CUDA) or AMD (ROCm), with 8 GB of its memory |
| Scratch | `STET_SCRATCH`, 8 GB, held in memory: what you write there comes out of the 12 GB |
| Time | `time_limit` in `task.json`: 600 seconds at micro, wall-clock |
| Network | none |

- **Write for the slowest machine.** Machines that run curators are at least as fast as a
  floor, and `stet run` prints how this machine compares with it. If yours is twice as
  fast, your curator has to finish in half the limit here. Better: make it *anytime*.
  Write a script early, improve it while `task.time_left()` allows, and stop with a
  margin. A killed curator keeps the last script it wrote.
- **GPU work goes through PyTorch.** `device = task.device()`, or `"cuda"`: that name is
  right on AMD too. Do not import anything that exists for one maker only (CUDA-only
  kernels, `bitsandbytes`, TensorRT, flash-attention builds). Half precision
  (`torch.float16`) works on both. Results differ in the last decimal places between
  makes, so round scores before you threshold or sort on them, and break ties by
  document number.
- **A read-only file system.** Writable: `STET_OUT`, `STET_SCRATCH` and `/tmp` (2 GB,
  where `TMPDIR` points). No home directory.
- **Two sandboxes.** Volunteers run curators in bubblewrap; the organisers' rented machines,
  which are containers, in a sandbox made for containers. Write for both: on loopback use
  TCP, on a port from 1024 up; do not count on scratch coming out of the memory limit (in
  the container sandbox it is on disk) or on being the only process listed in `/proc`.
  The rules list every difference.
- **Not everything in the directory is sent.** Hidden files (any name starting with a
  dot), `venv/`, `node_modules/`, `__pycache__/`, `runs/`, compiled Python and symbolic
  links stay at home. The bundle may be at most 50 MB packed, 400 MB unpacked and 2,000
  files: it is for code and small tables. `stet submit --dry-run` lists exactly what
  would be sent, and `stet run --sandbox` runs exactly that.
- Staying inside these limits is a rule as well as a mechanism: a curator that lifts its
  GPU memory limit or its restriction to six cores is removed.

### Libraries

Python 3.13 with the standard library and exactly these, on every machine. Import them
freely; do not bundle them. Anything else goes in your project: pure Python as it is,
JavaScript as an archive (see *Your own harness*).

```
accelerate 1.16.0                networkx 3.7           regex 2026.11.15
bm25s 0.3.16                     nltk 3.10.4            safetensors 0.8.0
datasketch 2.0.0                 nodejs-wheel 24.19.0   scikit-learn 1.9.1
einops 0.8.2                     numba 0.68.1           scipy 1.18.1
emoji 2.16.0                     numpy 2.5.5            sentence-transformers 6.3.0
faiss-cpu 1.16.0                 onnxruntime 1.32.0     sentencepiece 0.2.3
fastapi 0.163.2                  openai 3.53.0          tiktoken 0.14.0
fasttext-numpy2 0.10.4           orjson 3.13.0          tokenizers 0.23.3
fasttext-predict 0.9.2.4         pandas 3.1.1           torch 2.15.0
ftfy 6.3.1                       peft 0.21.3            tqdm 4.70.1
gensim 4.4.0                     polars 2.1.0           transformers 5.23.0
jinja2 3.1.6                     protobuf 7.36.2        unidecode 1.4.0
joblib 1.6.0                     psutil 7.2.2           uvicorn 0.54.0
lightgbm 4.7.0                   py3langid 0.4.0        xgboost 3.4.1
lingua-language-detector 2.2.0   pyahocorasick 2.3.1    xxhash 4.1.0
mmh3 5.3.1                       pyarrow 26.0.1         zstandard 0.25.0
model2vec 0.10.0                 rapidfuzz 3.14.6
```

`faiss` and `onnxruntime` are the CPU builds. `nltk` comes without its data files; name
any you need as assets. `import fasttext` works on Linux; `fasttext-predict` is the
loader that is there on every platform. `nodejs-wheel` is Node.js 24: `node` is on the
`PATH`. `fastapi`, `uvicorn` and `openai` are what serving a model to a harness needs.

### Model weights

Weights do not go in the bundle. Name them, and they are in place when your curator
starts:

```
stet asset add HuggingFaceTB/SmolLM2-135M     # writes [[assets]] tables into stet.toml
stet asset list
stet asset fetch                              # downloads them here, for `stet run`
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = task.asset("smollm2-135m")             # a directory of the files you named
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(task.device()).eval()
```

- Up to 32 files and 4 GiB in all. Each is a public file on Hugging Face, pinned to a
  commit and named by its sha256; the server checks every one when you submit. A gated or
  private repository will not do: volunteers have no login to give.
- `--include "*.safetensors" --include "*.json"` picks files; `--revision` picks a commit.
- There is no network at run time, so load from the path. `from_pretrained("owner/name")`
  fails.
- Your own fine-tuned model is welcome: upload it to a public repository and add it.
  Train it on anything except the two exams.
- 8 GB of GPU memory holds a model of a billion or two parameters at half precision with
  room to run it. Ten minutes is the tighter limit: at micro the corpus is about 170,000
  documents, so decide cheaply which of them, and how much of each, the model reads.
  `examples/reader` is a complete curator built this way.

Curators may not carry per-document knowledge of official data: hash lists, precomputed
labels, or tables keyed by host or URL built from the crawl. Models trained on other data
are fine.

### Your own harness

How the curator works is yours to decide. `entry` in `stet.toml` is one command line, and
what it starts may start more. Three shapes people build, all inside the rules:

- **A loop.** Code that goes over every document, or every chunk of one, and decides:
  rules, a classifier, a model you fine-tuned for the job. `examples/tidy` and
  `examples/reader`.
- **An agent.** A harness (pi, or one you wrote) that drives a model with tools: it looks
  at the round, runs commands, and writes the script or the plan for one.
  `examples/agent` runs pi against a 0.6B model, complete.
- **Anything else.** Several models, a search over recipes, a small student trained on
  samples inside the time limit. If it writes a script in time, it is a curator.

What the machine gives a harness:

- **Processes.** Start as many as you like: they share the 6 cores, the 12 GB and the
  clock, and all of them are stopped when the curator's time is up.
- **Loopback.** There is no network, but `127.0.0.1` works. Serve a model there and point
  the harness at it:
  `python -m transformers.cli.transformers serve /assets/<model> --host 127.0.0.1 --port <port>`
  speaks the OpenAI chat API, tool calls included, and the `openai` client is installed.
  Or load the model in the same process and skip the server.
- **Node.** `node` (24) is on the `PATH`. Ship a JavaScript harness as one archive in the
  bundle and unpack it to `STET_SCRATCH` when the curator starts: `node_modules/` is never
  sent as it is, and a bundle holds 2,000 files. `examples/agent/pack.sh` does this. Use
  a harness that needs no install step and no network when it runs.
- **Your model.** Fine-tune it at home, on any hardware, on anything except the two
  exams; publish the weights on Hugging Face; name them in `[[assets]]`. See *Model
  weights*.

What decides the design is time. At micro the corpus is about 490 million tokens and the
limit is 600 seconds on a machine that may be no faster than the floor. On that machine,
roughly: a 20M-parameter encoder reads 670,000 tokens a second, a 150M-parameter one
76,000, a 135M-parameter language model 65,000, a 0.6B one 16,000. So a small classifier
can read the opening of every document; a 0.6B model can read about 2% of the corpus:
say, the opening of the few tens of thousands of documents that cheaper code liked best.
Generating is far slower than reading: an agent's turn takes seconds. An agent can decide
how a round should be handled; it cannot visit every page.

An agent is not deterministic, and each entry is run twice, by two people. Keep a script
on disk before the agent starts, give every step a time limit from `task.time_left()`,
close its standard input, and check what it writes before you use it.

## Entering

```
stet login --handle <name>     # makes a key and an account; ask your human for the name
stet login --github            # prints one line to put in a public gist on their GitHub account
stet login --gist <gist url>   # links that account, which entering requires
stet submit --dry-run          # lists every file that would be published
stet submit --yes              # enters the curator in this directory
stet status                    # how your entries are doing
stet leaderboard
```

Before you submit, set `name`, `description` and `technique` in `stet.toml`: they are what
the leaderboard shows. Field names are lowercase, `stet.toml` must be a plain file, not a
link, and its numbers must fit in 64 bits.

The GitHub account must be at least 30 days old. Linking it means creating a public gist
on your human's account. Entering publishes every file `--dry-run` lists, under the MIT
licence, from the moment the curator is first drawn into an official round; its name is
public at once. **Both are your human's decisions: ask before doing either.**

An entry runs in 3 official rounds and is ranked when it has them. A round is drawn when
an entry is waiting and the one before has settled (and every 6 hours regardless), and
takes up to 3 new entries, so there can be a queue. The eight best curators, one per
person, stay on the *ladder* and keep running in every round. You may have 3 entries in
play at once and submit 6 times a day. Submitting an entry again under the same name
replaces the version that is still waiting or qualifying, and the new one starts its 3
rounds from nothing; another name is another entry; a round already drawn cannot be
changed or left. Local runs (`stet run`) have no limit and are where to try things. So
submit when you have a real improvement measured over several rounds, not after every
change.

## What is known to work

Measured with the three reference curators in `examples/` (`stet fork stet/tidy` copies the
best of them). Gains are in nats per token: at nano the mean of practice rounds 0 to 4 and
its standard error, at micro the mean of practice rounds 0 to 2 and the first official round.

| Curator | What it does | nano | micro |
|---|---|---|---|
| `identity` | nothing | 0 | 0 |
| `english` | keeps documents that look like English prose | +0.70 ± 0.04 | +0.93 ± 0.01 |
| `tidy` | English, deduplicated, boilerplate lines cut, best budget's worth, shuffled | +0.83 ± 0.03 | +0.94 ± 0.02 |

The gap between the two is 0.14 at nano and about 0.02 at micro, where `english` is ahead
on one round in four. A bigger student is less hurt by boilerplate, so the easy wins
shrink at the scale that counts, and `tidy`'s extra work buys almost nothing there. What
beats both at micro is not known yet.

- **Language is most of it.** The exam is English and most of a crawl is not.
- **Prose over furniture.** Within a page, keep paragraphs and drop menus, link lists and
  boilerplate. Range edits (`"k"`) do this at no cost in ink.
- **Fill the budget once with your best material.** Rank what survives and keep about one
  budget's worth.
- **Do not train in rank order.** The corpus arrives already shuffled. If you rank
  documents, shuffle what you keep. Whether a deliberate curriculum beats a shuffle is
  open.
- **Six sources, equally weighted.** `exam.by_source` shows the weakest one.

Open questions, which is where the gains are: a small language model, or a classifier
fine-tuned for the job, that picks better pages than rules do in ten minutes; topic
balance against the six sources; near-duplicate removal; how much repetition of the best
material helps; ordering.

One thing already tried: asking a 0.6-billion-parameter chat model, as it comes, "is
this good training text, yes or no" about every page `tidy` would keep did no better
than `tidy` alone. A model used as it comes is not obviously enough; one trained for the
question may be.

## Lending a GPU

`stet verify` grades other people's edit scripts, and `stet verify --edits` also runs
other people's curators in a sandbox. That is where every score on the leaderboard comes
from. People who do it have their own entries run first. Running other people's programs
on your human's machine is their decision: ask first.

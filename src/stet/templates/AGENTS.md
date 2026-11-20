# Stet: instructions for coding agents

You are working on a **curator** for [Stet](https://www.stetrun.com), an open competition.
This file is the complete brief. Read all of it before changing anything.

## The task

A small language model (the *student*) is trained on raw web text for a fixed number of
tokens and then sits an exam that is made after the round is drawn. The model and the
trainer are fixed, and so is how the exam is made. **The only thing you control is the
training data.**

Your curator is a program. It is handed a pile of raw web pages it has never seen and a time
limit, and it writes an *edit script*: which documents, or parts of documents, the student
is trained on, and in what order. Your score is the **gain**: how much better a student
trained on your edit does on the exam than one trained on the unedited pages, in bits per
byte. Higher is better.

## Set up

```
curl -fsSL https://www.stetrun.com/install.sh | sh    # installs the `stet` command
stet doctor                                           # checks GPU, libraries, disk
stet init my-curator && cd my-curator                 # a working starter curator
```

The installer brings its own Python 3.13, PyTorch in the build for this machine's GPU
(NVIDIA or AMD on Linux, Metal on a Mac), and every library a curator may import, at the
versions official runs use. `stet doctor` says if anything differs.

Words used below: the *student* is the small model that gets trained. *Bits per byte*
measure how likely the student finds a text: the fewer bits it needs for each byte, the
better it knows it. *Ink* is text in the training data that was not copied from the corpus.
A *verifier* is a volunteer's machine that runs curators or grades their scripts.

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
| `gain` | Your student's exam score less the unedited corpus's, in bits per byte: the mean of the pillars. 0 for an invalid run. **Maximise this.** |
| `pillars` | The gain in each pillar: `pillars.knowledge`, `pillars.code` and so on. The weakest shows what your data lacks. |
| `exam.items`, `exam.mean` | How many items the student sat, and its mean score on them (minus the bits per byte it needed, so nearer 0 is better). `baseline.mean` is the unedited student's. |
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
stet run --scale standard     # the official scale: larger corpus, larger student, slower
stet run --sandbox            # inside the sandbox official runs use (Linux)
stet run --time-limit 300     # with a shorter time limit than the scale's
stet run -v                   # show the curator's output live
stet score script.jsonl       # grade a script you already have, without running the curator
stet rounds                   # past official rounds; replay one with --round <id>
stet score stet/tidy --round s1-o00001    # re-grade a published entry of a settled official round
                                          # (stet/tidy@v2 when a round holds two versions)
stet clean                    # delete downloaded rounds and model files (fetched again when needed;
                              # unused ones are deleted after 14 days anyway)
```

**Never judge a change on one round.** Rounds differ from each other by more than most
improvements. Use `--rounds 3` or more and compare mean gains. Training the same script
again moves its gain by about 0.025 at micro and 0.013 at standard, so on one round a
difference smaller than twice that is noise. Confirm anything smaller over `--rounds 5`.

**Iterate small, confirm at standard.** A nano run takes a minute or two and micro several
minutes. The competition is scored at standard: your curator's twenty minutes, then about
18 minutes of training on an RTX 3090. Small students rank curators differently: micro
rewards narrow English data that loses at standard. Check at standard before you submit.

## What your curator is given

It is started with the `entry` command from `stet.toml`, in the project directory, with:

| Variable | Contents |
|---|---|
| `STET_CORPUS` | Directory with the corpus: `docs.bin`, `docs.idx`, `urls.txt`, `meta.json` |
| `STET_TASK` | Path to `task.json`: budgets, limits, the student's configuration, the exam's pillars (`exam_sources`) |
| `STET_OUT` | Write `script.jsonl` here |
| `STET_SCRATCH` | A writable directory for temporary files |
| `STET_ASSETS` | Directory with the model files `stet.toml` names, read-only |
| `STET_DEADLINE` | Unix time at which the curator is killed |

From Python, `from stet.curator import Task; task = Task.from_env()` gives you:

- `task.corpus`: `len(corpus)`, `corpus.text(i)`, `corpus.raw(i)` (bytes), `corpus.size(i)`,
  `corpus.sizes` (numpy array), `corpus.urls[i]`, and `for i, text in corpus`.
- `task.train_tokens`: the training budget. English prose is roughly 3 bytes per token, and
  the raw crawl as a whole about 2.7.
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
  student on at most `ink_tokens` of it: 0.1% of the budget, 160,000 tokens at standard.
  Going over makes the run invalid. Plan on spending none: this is an editing competition.
- **The budget cuts both ways.** The stream is cut off at `train_tokens`. A shorter script
  is repeated from the top until the budget is used, so keeping less means training on it
  more often.
- A document may appear more than once. An output document may be at most 1 MB.

## The exam

An official round's exam is made after its draw, so nobody can know it in advance, the
organisers included. It asks for seven things a pretrained model is for, and each counts the
same:

| Pillar | What the student is asked |
|---|---|
| `knowledge` | Facts the web knows, in ten languages |
| `reading` | An answer, or a new rule, taken from the context |
| `reasoning` | Mathematics and logic, in thirteen languages |
| `code` | What programs print; reading structured data |
| `agents` | What a tool says back, in a session at a shell |
| `writing` | Human writing published after the draw, of every kind and in every language; and pairs of a right and a wrong sentence |
| `conversation` | Replies in discussions published after the draw |

- **Every item is scored by likelihood**: how likely the student finds the right text, in
  bits per byte. A passage, an answer given its question, or the right sentence of a pair.
  Never by picking a choice.
- **Every language counts by its share of the round's corpus**, inside every pillar. Most
  of a crawl is not English. A curator that keeps only English gives up knowledge, reading
  and writing in every other language.
- **No one area can carry the rest.** Inside a pillar each family of questions, or kind of
  writing, counts the same, and gains on items are clipped. `pillars` shows the weakest.
- **Fresh text** is gathered by kind of writing, each an equal part: reference (Wikipedia),
  news (Common Crawl's news crawl), technical (GitHub), social (Bluesky, Mastodon) and
  discussion (Hacker News, Lemmy, GitHub, Wikipedia's talk pages, replies on social sites).
- **Practice rounds sit generated items only**: no fresh text was written for them. So
  `stet run` on a practice round shows six pillars, without `conversation`, and its
  `writing` is only the sentence pairs. Official rounds add the fresh writing. Keep the
  kinds of text people write and reply to, in every language, even though a practice run
  cannot show you what they are worth.
- A settled official round can be replayed: `stet run --round <id>` sits what the round
  published, its generated items and its Wikipedia text.

Do not put published exam text in your curator, or train a model on it. Curators are
published, it will be seen, and it defeats the point.

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
shop listings and spam around the prose. At standard a round's corpus is about 335,000
documents and 2.5 GB, some six times more text than the budget.

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
| Time | `time_limit` in `task.json`: 1200 seconds at standard, the official scale (600 seconds at micro), wall-clock |
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
  Train it on anything except published exams.
- 8 GB of GPU memory holds a model of a billion or two parameters at half precision with
  room to run it. Twenty minutes is the tighter limit: at standard the corpus is about
  335,000 documents, so decide cheaply which of them, and how much of each, the model reads.
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
- **Your model.** Fine-tune it at home, on any hardware, on anything except published
  exams; publish the weights on Hugging Face; name them in `[[assets]]`. See *Model
  weights*.

What decides the design is time. At standard the corpus is about 2.5 GB and the limit is
1,200 seconds on a machine that may be no faster than the floor. On that machine,
roughly: a 20M-parameter encoder reads 670,000 tokens a second, a 150M-parameter one
76,000, a 135M-parameter language model 65,000, a 0.6B one 16,000. So a small classifier
can read the opening of every document; a 0.6B model can read about 2% of the corpus:
say, the opening of the tens of thousands of documents that cheaper code liked best.
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
takes up to 3 new entries, so there can be a queue. A round takes about two and a half
hours: its exam is made two hours after the draw, once the fresh text is in, and its
scripts are graded while it is. The eight best
curators, one per person, stay on the *ladder* and keep running in every round. You may
have 3 entries in play at once and submit 6 times a day. Submitting an entry again under
the same name replaces the version that is still waiting or qualifying, and the new one
starts its 3 rounds from nothing; another name is another entry; a round already drawn
cannot be changed or left. Local runs (`stet run`) have no limit and are where to try
things. So submit when you have a real improvement measured over several rounds, not
after every change.

When the season closes, the eight best-rated curators go to the finals, which decide the
season: a fresh round, with a larger student (10 layers by 640, 640M tokens) trained twice
on what each writes.

## What is known

Stet's repository holds the reference curators in `examples/`, all seven entered
(`stet fork stet/tidy` copies one): `identity`, `english` and `tidy`, and four that rebuild
published recipes: `dclm`, `fineweb-edu` and `heuristic` (English filters), and `fineweb2`
(every language, cleaned). Their ratings on the board are where known methods land. What the organisers' measurements on this exam say so far:

- **Languages decide a lot.** At standard, the recipe that keeps every language and cleans
  each (`fineweb2`) ranks above the English-only filters. At micro the order flips: a small
  student learns most from narrow English data. That is why rounds are scored at standard.
- **Quality filters trade capabilities.** They help English prose and mathematics. They cost
  code and tool logs (rules that delete lines with braces delete programs), and everything
  not in English.
- **Keep the forms answers come in.** Much of `code`, `agents` and `reasoning` is whether
  the student has seen logs, program output, tables and numbers at all.
- **Prose over furniture.** Within a page, keep the text and drop menus, link lists and
  boilerplate. Range edits (`"k"`) do this at no cost in ink.
- **Repeating costs more as the student grows.** Keeping a small best slice and training on
  it many times pays most for a small student.
- **Do not train in rank order.** The corpus arrives already shuffled. If you rank
  documents, shuffle what you keep. Whether a deliberate curriculum beats a shuffle is open.

Open questions, which is where the gains are: a classifier or small model trained for the
job that cleans every language in twenty minutes; balance across the seven pillars;
near-duplicate removal; how much repetition helps; ordering.

## Lending a GPU

`stet verify` grades other people's edit scripts, and `stet verify --edits` also runs
other people's curators in a sandbox. That is where every score on the leaderboard comes
from. People who do it have their own entries run first. Running other people's programs
on your human's machine is their decision: ask first.

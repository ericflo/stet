# Stet: instructions for coding agents

You are working on a **curator** for [Stet](https://www.stetrun.com), an open competition.
This file is the complete brief. Read all of it before changing anything.

## The task

A fixed 0.3B language model (the *student*: `baidu/ERNIE-4.5-0.3B-Base-PT`, 18 × 1024, with
its embeddings and lower half of layers frozen and its upper half reset from the round's
seed) is trained on raw web text for a fixed number of tokens (65.5M at official `standard`
scale), run through a short fixed multilingual fine-tuning probe, and then sits a 100%-fresh
exam made when the round is drawn: half fresh twin-context comprehension from three hours of
the day before, half fresh harvest-derived multi-sentence reasoning from those same three
hours. The model, the trainer, the probe and how the exam is made are fixed. **The only thing
you control is the training data.**

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
(NVIDIA or AMD on Linux, Metal on a Mac — note `faiss-cpu` prebuilt wheels require macOS 15+
on Apple Silicon), and every library a curator may import, at the
versions official runs use. `stet doctor` says if anything differs.

Words used below: the *student* is the small model that gets trained. *Bits per byte*
measure how likely the student finds a text: the fewer bits it needs for each byte, the
better it knows it. A *verifier* is a volunteer's machine that runs curators or grades their
scripts.

## The loop

```
stet harvest       # once, before the first run
stet run --json
```

**Run `stet harvest` once first.** It fetches three hours of recent human writing from the
sources official exams are made from: no account, no key, some minutes for each hour. From
then on every practice round's exam holds fresh twin-context comprehension items from the
hours you hold. Without it a practice round uses the bundled openly-licensed CC-BY-SA
Wikipedia practice pack (both twin-context comprehension and reasoning).

Run `stet run --json` after every change. It runs your curator on a practice round, trains
the student, sits the exam, and prints one JSON object on stdout. Progress goes to stderr.
Exit status is 0 for a valid run and 2 for an invalid one.

| Field | Meaning |
|---|---|
| `valid` | `false` means the run scored as the unedited corpus. Read `invalid_reason`. |
| `gain` | Your student's exam score less the unedited corpus's, in bits per byte: comprehension half, reasoning half. 0 for an invalid run. **Maximise this.** |
| `pillars` | The gain in each pillar the exam has: `pillars.reference`, `pillars.explain` and so on. The weakest shows what your data lacks. |
| `writing` | Whether the exam held fresh harvested writing (or the bundled CC-BY-SA practice pack). |
| `exam.items`, `exam.mean` | How many items the student sat, and its mean score on them (minus the bits per byte it needed, so nearer 0 is better). `baseline.mean` is the unedited student's. |
| `script` | What your script did: documents kept, bytes, passes over the data. |
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
stet score stet/tidy --round s3-o00001    # re-grade a published entry of a settled official round
                                          # (stet/tidy@v2 when a round holds two versions)
stet harvest --hours 6        # hold six hours of writing: more variety between practice rounds
stet harvest --fresh          # replace what is held with newly written hours
stet clean                    # delete downloaded rounds and model files (fetched again when needed;
                              # unused ones are deleted after 14 days anyway)
```

**Never judge a change on one round.** Rounds differ from each other by more than most
improvements. Use `--rounds 3` or more and compare mean gains. Training the same script
again moves its gain by about 0.02 at micro and 0.004 at standard (provisional; repeated
deterministic standard runs of the unedited corpus on `s3-p00000` had sd `0.0004` on the full
exam), so on one round a difference smaller than twice that is noise. Confirm anything smaller
over `--rounds 5`. The order of a script matters too: the trainer does not shuffle, and the
same pages in another order can move a standard gain.

**Compare runs made against the same harvest.** A practice round's corpus never changes,
but its exam's fresh comprehension items come from three of the hours you hold, chosen by the
round's seed. Harvesting more hours, or new ones, changes it, and the round's exam and
baseline are made again the first time it is run. Many runs against the same hours can fit
them: `stet harvest --fresh` before you submit is the check.

**Iterate small, confirm at standard.** A nano run takes under a minute and micro a few
minutes on any 12 GB+ GPU (peak VRAM ≤10 GiB). The competition is scored at standard: your
curator's twenty minutes on a 24 GB+ GPU, then about 20 minutes of training on an RTX 5090
(~29 minutes on an RTX 4090, ~52 minutes on an RTX 3090) for 65.5M tokens plus the 32-step
SFT probe. Check at standard before you submit.

## What your curator is given

It is started with the `entry` command from `stet.toml`, in the project directory, with:

| Variable | Contents |
|---|---|
| `STET_CORPUS` | Directory with the corpus: `docs.bin`, `docs.idx`, `urls.txt`, `meta.json` |
| `STET_TASK` | Path to `task.json`: budgets, limits, the student's configuration, the exam's pillars (`exam_pillars`) |
| `STET_OUT` | Write `script.jsonl` here |
| `STET_SCRATCH` | A writable directory for temporary files |
| `STET_ASSETS` | Directory with the model files `stet.toml` names, read-only |
| `STET_DEADLINE` | Unix time at which the curator is killed |

From Python, `from stet.curator import Task; task = Task.from_env()` gives you:

- `task.corpus`: `len(corpus)`, `corpus.text(i)`, `corpus.raw(i)` (bytes), `corpus.size(i)`,
  `corpus.sizes` (numpy array), `corpus.urls[i]`, and `for i, text in corpus`.
- `task.train_tokens`: the training budget (65,536,000 at `standard`). Under ERNIE-4.5's
  103,424-token multilingual tokenizer, English prose is roughly 3.8 bytes per token, and the
  raw crawl as a whole about 3.4.
- `task.info`: everything in `task.json`, including `min_copy_bytes`.
- `task.time_left()`: seconds until the deadline.
- `task.asset("name")`: the path of a model file or directory named in `stet.toml`.
- `task.device()`: `"cuda"` when there is a GPU PyTorch can use (NVIDIA or AMD), else `"cpu"`.
- `task.cores`, `task.memory_gb`, `task.gpu_memory_gb`: what this run has been given.
- `task.map(fn)`: `fn(i, text)` for every page, on every core your curator is given, with
  `(i, result)` back in page order.
- `task.write(lines)`: writes the script atomically. Call it early, and again whenever you
  have something better. **If the curator is killed or crashes, the last script it wrote
  still counts.**

A curator does not have to be Python. Anything that reads those files and writes
`script.jsonl` works. `docs.bin` is every document's UTF-8 bytes concatenated; `docs.idx`
is little-endian uint64 offsets (document `i` is `bin[idx[i]:idx[i+1]]`); `urls.txt` has
one source URL per line.

**Use every core.** A loop over the corpus runs on one of the cores your curator is given,
and a standard round is about 500,000 pages. `task.map` runs the page-by-page work on all
of them:

```python
def keep(i, text):  # in a worker process, once for every page
    return looks_written(text)

task.write([i for i, ok in task.map(keep) if ok])
```

Results come back in page order whichever worker finished first, so anything that depends
on order (which of two copies of a page is kept) is decided in your loop exactly as a plain
loop decides it, and the script is the same on any number of cores. The workers are forked
when the loop starts. A model loaded before then is shared by all of them and counted once
against `task.memory_gb`; one a worker loads for itself is held once per worker. `fn` runs
in another process, so return what you need instead of changing a global, and return plain
values (they are pickled). Do not use the GPU in `fn`, and map before PyTorch is loaded
where you can. `task.map(fn, docs=[...])` goes through only those pages, in that order. If
`fn` raises, the same exception is raised in your loop; if a worker is killed (the kernel
does that when the curator is over its memory), `stet.curator.WorkerError` is raised. Keep
what you hold per page small: it grows with the pool.

## The edit script

JSON Lines. One line is one output document. **Line order is training order.** The trainer
does not shuffle.

```
123                                          keep document 123 whole
{"d": 123, "k": [[0, 900], [1200, 2000]]}    keep these byte ranges of document 123, joined
{"s": [[123, 0, 900], [7, 40, 400]]}         join copies [doc, start, end] of any documents
```

- **A script is copies of the corpus and no more.** It holds no text of your own. A string
  in an `s` line, or a line with a `t` key (`{"t": "text"}`), makes the run invalid. This
  is an editing competition: you choose what the student reads, and write none of it.
- Offsets index the document's UTF-8 bytes. One that lands inside a character is moved
  forward to the next boundary.
- **Cutting, reordering and repeating are free.**
- **Deleting lines is always allowed.** An output document made only of whole lines of one
  page, in the page's order, is a copy however short the lines are: take out the menu, the
  cookie banner and the footer and keep the article. Each range starts at the start of a
  line and ends just after a line break, or at the end of the page. Lines cannot be moved,
  cut short, repeated or mixed with another page's that way, and the lines kept of a page
  hold at least 64 bytes between them.
- **Any other copy is at least `min_copy_bytes` (64) long.** A shorter one makes the run
  invalid: with enough tiny copies a script could spell out any text it liked. Ranges of
  one document that touch are one copy, so three 30-byte ranges end to end are a 90-byte
  copy. A copy's length is what is left once its offsets have been moved to character
  boundaries. The one exception is a whole short document kept on a line of its own;
  joined to anything else in an `s` line, it is too short.
- **The budget cuts both ways.** The stream is cut off at `train_tokens`. A shorter script
  is repeated from the top until the budget is used, so keeping less means training on it
  more often.
- A document may appear more than once. An output document may be at most 1 MB.

## The exam

An official round's exam is **100% fresh** and made at its draw, so nobody can know it in
advance, the organisers included. There are zero static benchmarks, zero synthetic templates,
and zero raw web perplexity. **Half of the score is fresh twin-context comprehension from
writing people published. Half is fresh harvest-derived multi-sentence reasoning from those
exact same three hours.** Every part is measured in unified **bits per byte**. Inside each
group, every pillar counts the same:

| Pillar | Group | What the student is asked |
|---|---|---|
| `reference` | comprehension | Same continuation under real vs wrong openings of new Wikipedia articles (same source & language) |
| `news` | comprehension | Same continuation under real vs wrong openings of fresh news reports (same source & language) |
| `technical` | comprehension | Same continuation under real vs wrong openings of GitHub issues, PRs, release notes and patches |
| `conversation` | comprehension | Same continuation under real vs wrong openings of Hacker News, Lemmy, forums, Bluesky, Mastodon, talk pages |
| `explain` | reasoning | Multi-sentence causal or mechanism explanation derived from a fresh harvested excerpt (in its own language) |
| `infer` | reasoning | Multi-sentence analytical inference of implications or trade-offs derived from a fresh harvested excerpt |
| `synthesis` | reasoning | Declarative prose synthesis of a fresh harvested excerpt's main point and rationale (zero Q/A punctuation) |

- **100% newer than the crawl.** Both halves come from what people published in three hours
  of the day before the draw, picked by the draw. The round's corpus was crawled before any
  of it existed, and your curator has no network when it runs and can add no text of its own,
  so there is nothing to look up or to fit in advance.
- **Every language counts by its share of the round's corpus**, across BOTH comprehension and
  reasoning. More than half of a crawl is not English, so a curator that keeps only English
  (or scores non-English pages lower on an English-biased rubric) gives up roughly half the
  exam.
- **Harvest-derived reasoning is deterministic and anti-gaming.** At the draw, the examiner
  derives multi-sentence `explain`, `infer`, and `synthesis` items from the same three fresh
  hours using pinned greedy (`temperature=0`) decoding (`Qwen/Qwen3-8B`), six varied natural
  prompt wrappers (`synthesis` uses zero `?` or `Q:/A:` syntax so QA-spam cannot game it),
  and a `<50%` 5-gram verbatim-copying filter. Both the source excerpt and the response are
  decontaminated against the round's training corpus.
- **Scored by twin-context comprehension or answer likelihood, never raw web perplexity.**
  Fresh comprehension items (`reference`, `news`, `technical`, `conversation`) split each
  document cleanly on a sentence or paragraph boundary and score the exact same continuation
  under its real opening minus the mean of up to three length-matched wrong openings from the
  exact same source, language, and boundary class, in bits per byte:
  `(mean_w nll(cont | wrong_w) − nll(cont | ctx)) / (ln 2 · max(16, bytes(cont)))`.
  Because the exact same continuation is scored under both, unconditional passage ease and
  surface style mimicry cancel identically; only genuine contextual understanding moves the
  score. Reasoning items (`explain`, `infer`, `synthesis`) score multi-sentence response
  likelihood in bits per byte (floored at 16 bytes) after the fixed 1-epoch multilingual SFT
  probe.
- **No one area can carry the rest.** A pillar missing from an exam gives its share to the
  others of its group. Gains on items are clipped, and a language or a format with fewer
  than eight items counts in proportion. `pillars` shows the weakest.
- **Everything on the exam counts.** The season names every part of the exam: the pillars
  above, and nothing else. Each part is validated before the season for multi-seed stability,
  DataDecide 1B pretraining ranking agreement, and adversarial resistance (QA-spam, splice-QA,
  surface mimicry, and over-filtering). Nothing is sat that the score leaves out.
- **Practice rounds work out of the box.** Without harvested hours, `stet run` sits a bundled
  openly-licensed CC-BY-SA Wikipedia practice pack (both twin-context comprehension and
  reasoning); once you run `stet harvest`, practice rounds sit fresh twin-context writing
  from the hours you hold.
- A settled official round can be replayed: `stet run --round <id>` sits what the round
  published (openly licensed Wikipedia items whole).

Do not put published exam text in your curator, or train a model on it. Curators are
published, it will be seen, and it defeats the point.

## The data

- **Practice rounds** (`--round 0`, `1`, ...): seeded samples of one pinned Common Crawl
  crawl, as full-page text. The corpus is fixed forever, so results compare across time
  while you hold the same harvest.
- **Official rounds**: the same kind of data from the newest crawl at the time of the draw.
  Which of the crawl's files a round gets is decided by a public randomness beacon after
  entries close, so nobody can know the sample in advance. The crawl itself is public,
  and practice rounds may come from the same one.
- **Gym rounds** (`--round g0`, `g1`, ...): a different dataset (DCLM-Pool), for checking
  that a curator is not tuned to one crawl, and for training policies. See `stet.gym`.

Full-page text means every language, plus menus, cookie banners, footers, link lists,
shop listings and spam around the prose. At standard a round's corpus is 12 units (24
Common Crawl WET shards, about 500,000 documents and 3.8 GB, roughly 1.1B tokens — ~17×
the 65.5M-token student budget). That ~17× pool lets CPU-speed filters (rules, fastText,
small encoders) select a 6–15% slice without repetition, while neural curators (`vLLM` with
0.6B–0.8B models) can prefilter and then rate tens to hundreds of thousands of candidate
excerpts inside twenty minutes (`examples/reader` rates all ~145,000 prefiltered excerpts
on `s3-p00000`). You do not need to read every page: `task.stream()` lets you stream
lazily and stop once you have filled the budget.

Build something that reads the data in front of it. A curator tuned to one round's
documents will do badly on the next.

## What makes a run invalid

- No script, or a malformed one. A timeout or crash is fine *if* a script was already
  written. `task.write` replaces the file in one step; if you write it yourself, write
  to a temporary file and rename it, or a kill can leave half a script.
- Text of your own anywhere in the script: a string in an `s` line, or a `t` key.
- A copied range shorter than 64 bytes, unless the line is whole lines of one page in
  order (64 bytes of them or more), or one short document kept whole.
- A document number or byte range that does not exist.
- A script that yields almost no tokens.
- A script over 64 MB, a line over 1 MB, or a line nested more than 16 brackets deep.

More than 10,000 files in `STET_OUT` is an error too, though a script already written
still counts.

## Official runs

Your curator is run by volunteers, on their machines, twice, by two different people: on
an NVIDIA GPU and on an AMD one when both are around. Your gain for a round is the mean
of the two runs. Failing is never better than finishing: beside a run that finished, a
failed one counts 0, or as your worst finished run when that is below 0, and a round in
which most runs failed counts as the lowest gain any valid entry got in it, or 0 if that
is lower. So **it has to work on both**.

Every run gets the same machine, whatever the volunteer has:

| | |
|---|---|
| CPU | 6 cores |
| Memory | 20 GB |
| GPU | one, NVIDIA (CUDA) or AMD (ROCm), with 22 GB of its memory |
| Scratch | `STET_SCRATCH`, 12 GB, held in memory: what you write there comes out of the 20 GB |
| Time | `time_limit` in `task.json`: 1200 seconds at standard, the official scale (300 seconds at micro), wall-clock |
| Network | none |

- **Write for the slowest machine.** Machines that run curators are at least as fast as a
  floor, and `stet run` prints how this machine compares with it. If yours is twice as
  fast, your curator has to finish in half the limit here. Better: make it *anytime*.
  Write a script early, improve it while `task.time_left()` allows, and stop with a
  margin. A killed curator keeps the last script it wrote.
- **GPU work goes through PyTorch, vLLM, or flash-linear-attention.** `device = task.device()`,
  or `"cuda"`: that name is right on AMD too. `vllm` and `liger-kernel` are preinstalled on
  NVIDIA Linux runners (all cloud runners and NVIDIA volunteers); if you import `vllm` or
  `liger-kernel`, include a clean PyTorch/`transformers` fallback (see `examples/reader`) so
  your curator also runs on AMD ROCm machines. When using `vLLM`, always cap
  `gpu_memory_utilization` below `task.gpu_memory_gb` (for example
  `min(0.85, (task.gpu_memory_gb - 2.0) / total_vram_gb)`, as `examples/reader` does), because
  `vLLM` runs its engine in a subprocess outside PyTorch's in-process memory cap and may OOM
  or exceed the 22 GB sandbox budget otherwise. `flash-linear-attention` (`fla`) and
  bfloat16/float16 work on both. Results differ in the last decimal places between makes, so
  round scores before you threshold or sort on them, and break ties by document number.
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
accelerate 1.16.0                model2vec 0.10.0       regex 2026.11.15
bm25s 0.3.16                     networkx 3.7           safetensors 0.8.0
datasketch 2.0.0                 nltk 3.10.4            scikit-learn 1.9.1
einops 0.8.2                     nodejs-wheel 24.19.0   scipy 1.18.1
emoji 2.16.0                     numba 0.65.0           sentence-transformers 6.3.0
faiss-cpu 1.16.0                 numpy 2.4.6            sentencepiece 0.2.3
fastapi 0.136.3                  onnxruntime 1.32.0     tiktoken 0.14.0
fasttext-numpy2 0.10.4           openai 3.58.0          tokenizers 0.23.3
fasttext-predict 0.9.2.4         orjson 3.13.0          torch 2.14.0
flash-linear-attention 0.6.0     pandas 3.1.1           tqdm 4.70.1
ftfy 6.3.1                       peft 0.21.3            transformers 5.22.2
gensim 4.4.0                     polars 2.1.0           unidecode 1.4.0
jinja2 3.1.6                     protobuf 7.36.2        uvicorn 0.54.0
joblib 1.6.0                     psutil 7.2.2           vllm 0.34.0
liger-kernel 0.8.5               py3langid 0.4.0        xgboost 3.4.1
lightgbm 4.7.0                   pyahocorasick 2.3.1    xxhash 4.1.0
lingua-language-detector 2.2.0   pyarrow 26.0.1         zstandard 0.25.0
mmh3 5.3.1                       rapidfuzz 3.14.6
```

`faiss` and `onnxruntime` are the CPU builds. `nltk` comes without its data files; name
any you need as assets. `import fasttext` works on Linux; `fasttext-predict` is the
loader that is there on every platform. `vllm` and `liger-kernel` are installed on NVIDIA
Linux machines (`cuda` extra); provide a `transformers` fallback for AMD ROCm runners (see
`examples/reader`). `flash-linear-attention` (`fla`) provides fast Triton kernels for
linear-attention/DeltaNet/hybrid models (`Qwen3.5`, `LFM2.5`). `nodejs-wheel` is Node.js 24:
`node` is on the `PATH`. `fastapi`, `uvicorn` and `openai` are what serving a model to a
harness needs.

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

- Up to 32 files and 10 GiB in all. Each is a public file on Hugging Face, pinned to a
  commit and named by its sha256; the server checks every one when you submit. A gated or
  private repository will not do: volunteers have no login to give.
- `--include "*.safetensors" --include "*.json"` picks files; `--revision` picks a commit.
- There is no network at run time, so load from the path. `from_pretrained("owner/name")`
  fails.
- Your own fine-tuned model is welcome: upload it to a public repository and add it.
  Train it on anything except published exams.
- 22 GB of GPU memory comfortably holds 0.6B–4B models at bf16/fp16 with `vLLM` prefix
  caching and KV cache headroom. With `vLLM`, a 0.6B–0.8B model (like `Qwen3-0.6B`,
  `Qwen3.5-0.8B`, or `LFM2.5-350M`) can score hundreds of thousands of candidate excerpts
  inside the twenty-minute limit! `examples/reader` is a complete `vLLM` curator built this
  way.

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

- **Processes.** Start as many as you like: they share the 6 cores, the 20 GB of memory
  and the clock, and all of them are stopped when the curator's time is up.
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

What decides the design is time. At standard the corpus is about 3.8 GB (~500,000 pages)
and the limit is 1,200 seconds on a 24 GB+ GPU at or above the floor. A fastText classifier
or 20M-parameter encoder can scan every document; with `vLLM` prefix caching on a 0.6B–0.8B
model (like `Qwen3-0.6B`), batched scoring handles tens to hundreds of thousands of candidate
excerpts inside twenty minutes (see `examples/reader`). Generating multi-turn completions is
much slower than single-token scoring: an agent can decide how a round should be handled,
while a batched scorer visits the candidate pool.

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

**An entry runs in the next round drawn.** A new version is a new submission.

An entry runs in 3 official rounds and is ranked when it has them. Rounds are drawn **on
demand** whenever an entry is waiting (or still qualifying) and the previous round has
settled — there are no idle clock rounds between submissions. A round's exam is made at its
draw, and the round takes as long as its runs and grades do. A curator's rating is its mean
gain, adjusted for how hard its rounds were; its ± is wide while its rounds are few. The
eight best curators, one per person, stay on the *ladder* and keep
running in every round: the best by rating less 1.28 standard errors, so a few lucky
rounds do not take a place from many steady ones. You may
have 3 entries in play at once, and submit as often as you like. Submitting an entry again under
the same name replaces the version that is still waiting or qualifying, and the new one
starts its 3 rounds from nothing; another name is another entry; a round already drawn
cannot be changed or left. Local runs (`stet run`) have no limit and are where to try
things. So submit when you have a real improvement measured over several rounds, not
after every change.

When the season closes, the eight best-rated curators, and one more drawn by lot from the
next eight, go to the finals, which decide the season: two fresh rounds, with the larger
budget (`large`: 131M tokens) trained twice on what each writes.

## What is known

Stet's repository holds the organisers' reference curators in `examples/` (`stet fork
stet/reader` or `stet fork stet/fineweb2` copies one): `identity`, `english`, `tidy`,
`heuristic`, `dclm`, `fineweb-edu`, `fineweb2`, `mimic`, and `reader` (a 0.6B `vLLM`
multilingual quality rater). `fineweb2` and `reader` run in every round, and each of the
others in one round in eight.

What earlier seasons and the Season 3 validation experiments showed (gains over the unedited
corpus, in bits per byte):

- **Surface mimicry and QA-spam no longer buy the exam.** In Season 2's raw-perplexity exam,
  `mimic` beat `fineweb2` (+0.10 vs +0.05) by matching surface style. Under Season 3's
  100%-fresh twin-context comprehension + harvest-derived reasoning exam (`ERNIE-4.5-0.3B`
  frozen bottom half, 65.5M tokens + 1-epoch SFT probe), `fineweb2` (`+0.0255`, SNR `64×`
  against unedited sd `0.0004`) decisively beats `mimic` (`+0.0217`), QA-spam (`+0.0044`),
  English-only (`-0.0508`), spliced Q&A (`-0.0558`), and extreme over-filtering (`-0.5011`).
- **Agreement with 1B pretraining.** Across all 12 published DataDecide recipes (Dolma,
  DCLM, FineWeb-Edu, C4, etc.), the Season 3 student + 100%-fresh exam cleared 81.8%
  pairwise agreement with 1B-parameter 100B-token downstream accuracy (`93.5%` on pairs ≥2pt
  apart, Spearman `ρ = +0.832`, exact Top-4 match).
- **Balance every language — watch out for English-rubric bias.** Keeping every language and
  cleaning each (`fineweb2`, `reader`) beats English-only corpora by >0.07 bpb. When using an
  English-prompted LLM judge (like `examples/reader`), normalize scores per language/script
  quantile before selecting and ordering documents — raw 1–5 scores from English rubrics
  underrate Korean, Russian, and Arabic and starve them during selection and LR cooldown!
- **Quality filters trade capabilities.** They help prose. They can cost code and technical
  text (rules that delete lines with braces delete programs), which `technical` asks for, and
  everything not in English.
- **Cut lines with care.** `tidy` used to keep only the lines of a page that read as
  sentences. The same pages copied whole scored 0.03 bits per byte higher on the writing,
  in four orders out of four: the lines it cut held the code and the lists that
  `technical` asks for. Whole lines taken out with a range edit (`"k"`) are allowed,
  however short the lines that are kept; make sure they were not worth keeping.
- **Repeating costs more as the student grows.** Keeping a tiny slice and repeating it dozens
  of times collapses generalization (`adv_overfilter`: `-0.5011` bpb at 65.5M tokens).
- **Do not train in raw rank order.** The corpus arrives already shuffled. If you rank
  documents, shuffle inside quality bands (a gentle curriculum) or shuffle what you keep.

Open questions, which is where the gains are: a classifier or small model trained for the
job that cleans every language fairly in twenty minutes; balancing prose, technical/code,
and reasoning text; near-duplicate removal; how much repetition helps; curriculum ordering.

## Lending a GPU

`stet verify` grades other people's edit scripts, and `stet verify --edits` also runs
other people's curators in a sandbox. That is where every score on the leaderboard comes
from. People who do it have their own entries run first. Running other people's programs
on your human's machine is their decision: ask first.

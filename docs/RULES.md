# The rules

Season 3 (rules version 7). These are the rules as the server and the harness enforce them. Where a number
is still being measured it says *provisional*, and [Experiments](EXPERIMENTS.md) says how it
will be set.

## The idea

A fixed 0.3B language model, the **student** (`baidu/ERNIE-4.5-0.3B-Base-PT`, 18 × 1024, with
its embeddings and lower half of layers frozen and its upper half reset from the round's
seed), is trained on raw web text for a fixed number of tokens (65.5M at official `standard`
scale), run through a short fixed multilingual fine-tuning probe, and then sits a 100%-fresh
**exam**. The exam is made when the round is drawn: half of it is fresh twin-context
comprehension from three hours of the day before, which the draw picks (the same continuation
scored under its real opening minus wrong openings from the exact same source and language),
and half is fresh deterministic harvest-derived reasoning (`explain`, `infer`, `synthesis`)
from those same three hours. There are zero static benchmarks, zero synthetic templates, and
zero raw web perplexity. The model, the trainer, and the probe never change. You change one
thing: the data. You write a **curator**, a program that is handed raw web pages it has never
seen and twenty minutes on one 24 GB+ GPU. It answers with an **edit script**: what to keep,
what to cut, and in what order to train on it. The better the student does on the exam than
one trained on the raw pages, the better the curator.

## What you enter

A curator is a directory with a `stet.toml` and any program. `stet init` makes one.

- **The harness is yours.** `stet.toml` names one command, and how it gets from the corpus
  to a script is up to you: a loop that runs a classifier or a model you fine-tuned over
  every chunk, an agent harness driving a small model with tools, or something nobody has
  tried. It may start other processes and talk to them over loopback; Node.js is there
  beside Python. The limits are the machine and the clock, not the method.

- **Code goes in the bundle.** The bundle is the files in the directory, compressed: at
  most 50 MB, 400 MB unpacked and 2,000 files. Hidden files, `venv/`, `node_modules/`,
  `__pycache__/`, `runs/`, compiled Python and symbolic links are never packed;
  `stet submit --dry-run` lists exactly what is.
- **Model weights go by reference.** A curator may use a model of its own: up to 32 files
  and 10 GiB in all, named in `stet.toml` as `[[assets]]`. Each is a public file on Hugging
  Face, pinned to a commit and named by its sha256. `stet asset add <owner>/<repository>`
  writes the tables. The files are fetched before the clock starts and are in place,
  read-only, when the curator runs. Nothing can be downloaded while it runs.
- **Libraries are provided.** PyTorch, transformers, flash-linear-attention, scikit-learn and
  nearly fifty others, plus Node.js, at exact versions on every machine (`vLLM` and
  `liger-kernel` are also preinstalled on NVIDIA Linux runners): see
  [the curator's machine](#the-curators-machine). You bundle your own code and nothing else.
- **It has to run on both makes of GPU.** Curators are run by volunteers, on NVIDIA and on
  AMD. Use the GPU through PyTorch (`device="cuda"` is right on both) and it will. A run
  that fails is never better than one that finishes: see [the score](#the-score).
- **Try as much as you like on your own machine.** `stet run` is the leaderboard's own
  harness: nothing is uploaded and nothing is counted. Practice rounds never change, and
  every official round can be replayed once it has settled (`stet run --round <id>`).
  A practice round's exam uses an openly licensed CC-BY-SA Wikipedia practice pack out of the
  box (both twin-context comprehension and reasoning) and, once you have run `stet harvest`,
  the same kinds of recent writing an official exam has. A replay sits what its round published (see [the exam](#the-exam)), so its score
  differs from the round's.
- Officially you may have 3 entries in play at once, waiting or qualifying. How often you
  submit is not limited. A new version of an entry (the same name) takes the old version's
  place; another name is another entry, and past three the oldest makes way.
- Entering needs a GitHub account at least 30 days old, linked with `stet login --github`.
  Every entry costs volunteers GPU time, so one person is one entrant: the limits above
  are counted per GitHub identity, and linking yours to a new key withdraws the old key's
  entries.
- **Curators are published.** From the moment yours is first drawn into an official round,
  its code is public under the MIT licence, and so is every edit script it writes once
  its round settles. `stet fork` copies any of them. Its name and yours are public as
  soon as you submit.
- Not allowed: per-document knowledge of official data (lists of hashes, precomputed
  labels, tables by host or URL built from the crawl), exam text in any form, models
  trained on published exams, lifting the limits the sandbox sets, and anything that tries
  to leave it. Models trained on other data are fine. Curators that break these are removed,
  and since every curator is public, anyone can check.

## A round

Rounds are drawn **on demand** whenever a curator is waiting (or still qualifying) and a
runner is around, with no idle clock rounds between submissions. Never two at once: a round
starts only once the one before has settled, and the fleet scales down to zero whenever
nothing is waiting.

1. **Announce.** The server names, in advance, the drand round whose randomness will seed
   the draw and the Common Crawl crawl it will draw from. Verifiers keep a log of these
   announcements and refuse a round that contradicts what they were told. No draw is
   announced less than six minutes ahead, and a draw that is moved is announced again the
   same way, for a time no round was ever announced for before.
2. **Draw.** The seed is `sha256(season, round number, drand round, drand randomness,
   crawl)`. It picks the corpus files, the order of the documents, which three fresh hours
   make the exam, and the student's initial weights. Nobody can know or steer any of that beforehand,
   organisers included. Draws happen only at fixed ten-minute marks, so the organisers
   cannot pick a favourable drand round either.
3. **Edit.** Two volunteers each run every curator, in a sandbox, with a time limit. Each
   run writes an edit script.
4. **Exam.** The round's exam is made at the draw. The organisers' examiner harvests what
   people publish, every hour of the clock, and tells the server each harvest's hash as it is
   made. The draw's seed picks three of the day's hours, and the exam is made from exactly
   those and posted with its hash (see [the exam](#the-exam)). No round waits for its exam.
5. **Grade.** Other volunteers train the student on what each script says, as soon as it
   is written, and have it sit the exam. Grading runs no entrant code: a
   script is plain data.
6. **Settle.** A round settles as a whole. Until then every script and every grade is
   sealed. Then each score is published, with every run and every grade behind it,
   whoever made them and whatever they said.

[Running and grading](#running-and-grading) says who those volunteers are and how their
work becomes a score.

Who runs in a round: the ladder (see below), entries still qualifying, and every new entry
waiting at its draw. A round turns nobody away, and nobody waits behind anybody else. What
is limited is what one person may have waiting or qualifying at once: three entries, so at
most three of one person's in a round. Rounds do not overlap, so a round with a great many
entries takes longer, and the next is drawn when it has settled.

Rounds never overlap. If one is still being graded when the next is due, the next is
drawn at the first ten-minute mark that is at least six minutes after it settles. A round
takes as long as its curators and its trainings do: typically 1.5 to 2.5 hours from draw
to settlement on a 6-GPU fleet (each standard training takes ~20 minutes on an RTX 5090,
~29 minutes on an RTX 4090, or ~52 minutes on an RTX 3090, and baseline graders train two
passes). Season 1's rounds took two and a half hours, two of them waiting for the exam.

## The corpus

The newest published Common Crawl crawl at the time of the draw, as full-page text: every
language, with menus, cookie banners, footers, shop listings, spam and duplicates
alongside real prose. A round's corpus is a seeded sample of it: at the official scale
about 500,000 pages and 3.8 GB (~1.1B tokens), roughly 17× more text than the 65.5M-token
student budget. Micro and large rounds keep a similar excess of raw text over the student
budget. Your curator has to choose.

Nothing is removed from the corpus for the exam's sake: the exam did not exist yet. Fresh
text that the corpus happens to hold already is dropped from the exam instead.

## The edit script

One line per output document. Line order is training order: the trainer does not shuffle.

```
123                                   keep document 123 whole
{"d": 123, "k": [[0, 900]]}           keep byte ranges of document 123
{"s": [[123, 0, 900], [7, 40, 400]]}  join copies [doc, start, end] of any documents
```

- **A script is copies of the crawl and no more.** It holds no text of its own: a line
  with text in it (a string among the copies, or a `"t"` key) makes the script
  invalid. That is what makes this an editing competition: with text of its own, the
  winning move is to ignore the corpus and ship a dataset. A little was allowed at first,
  160,000 tokens at the official scale, and a tenth of it was measured: written in the
  exam's forms, it bought 0.004 bits per byte, 1.3 standard errors. That is too small and
  uncertain to earn a rule, so there is none.
- **Cutting, reordering and repeating are free.**
- **Deleting lines is always allowed.** Whole lines of one page, kept in the page's order,
  are copies however short they are: take out the menu, the cookie banner and the footer
  and keep the article. Lines cannot be moved, cut short, repeated or mixed with another
  page's that way, and what is kept of a page is at least 64 bytes in all: single short
  lines cannot be strung together to spell something.
- **Any other copy is at least 64 bytes long.** With enough tiny copies a script could
  spell out any text it liked from the corpus's letters and words, so a shorter copy makes
  the script invalid. Ranges that touch are one copy. The one exception is a whole short
  document kept on a line of its own.
- **The budget cuts both ways.** The stream is cut off at the training budget. A script
  that is shorter is repeated from the top, so keeping less means training on it more
  often.
- Limits: 64 MB per script, 1 MB per line and per output document, 20,000 pieces joined
  in one document, and no line nested more than 16 brackets deep (the lines above need 3).

## The exam

**100% fresh. Half of the score is fresh twin-context comprehension from writing people
published. Half is fresh harvest-derived reasoning from those exact same three hours.** There
are zero static benchmarks, zero synthetic templates, and zero raw web perplexity. Every part
is measured in unified **bits per byte**. Inside each group, every row below counts the same.

| | | What it asks | Where it comes from |
|---|---|---|---|
| Comprehension (50%) | Reference | Same continuation under real vs wrong openings of encyclopedia articles (same source & language) | New and extended Wikipedia articles |
| | News | Same continuation under real vs wrong openings of fresh news reports (same source & language) | Common Crawl's news crawl |
| | Technical | Same continuation under real vs wrong openings of issues, PRs, release notes and patches | GitHub, the kernel's mailing lists |
| | Conversation | Same continuation under real vs wrong openings of posts and replies | Hacker News, Lemmy, forums, Bluesky, Mastodon, Wikipedia's talk pages |
| Reasoning (50%) | Explain | Multi-sentence causal and mechanism explanation in the excerpt's own language | Deterministic (`temp=0`, pinned `Qwen3-8B`) synthesis from the same 3 fresh hours |
| | Infer | Multi-sentence analytical implications and trade-offs in the excerpt's own language | Deterministic synthesis from the same 3 fresh hours |
| | Synthesis | Declarative prose synthesis of the excerpt's main point and rationale (zero Q/A syntax) | Deterministic synthesis from the same 3 fresh hours |

**The writing** is what people published in three hours of the day before the draw. The
examiner harvests every hour of the clock an hour after it ends, keeping a sample chosen
with a public random number emitted as the hour ends, and tells the server the harvest's
hash at once. At the draw, the round's seed picks three of the hours harvested in the 26
before it. So:

- nobody knows which hours a round will be given until its draw, the organisers included;
- nobody writing in an hour can tell what the harvest will keep of it;
- and the organisers cannot make a harvest again to suit a round: its hash was public first.

Inside a kind of writing, every language gets its share of the round's corpus, and the
sources take turns. No source needs a key or an account. Writing the round's corpus already
holds is dropped from the exam.

**The reasoning items** (`explain`, `infer`, `synthesis`) are derived deterministically from
a stratified sample of those exact same three fresh hours at the draw using pinned greedy
decoding (`Qwen/Qwen3-8B`, `temperature=0`), six varied natural wrappers (`synthesis` has no
`?` or `Q:/A:` markers), and a `<50%` 5-gram verbatim-copying filter. Any item whose excerpt
or response matches the round's training corpus is dropped.

**Everything on the exam counts.** The season names every part of the exam: those in the
table, and no other. Each part is validated before the season for multi-seed stability,
resistance to surface style mimicry and QA-spam, and agreement with 1B pretraining rankings
(DataDecide). A part is in the score or it is not on the exam at all: nothing is sat that
the score leaves out, and nothing on the exam counts for nothing.

**The score of an item** is measured in **bits per byte** everywhere:
- on fresh comprehension (`kind="context"`), each article is split cleanly on a sentence or
  paragraph boundary into an opening and a continuation, and the exact same continuation is
  scored under its real opening minus the mean of up to three length-matched wrong openings from
  the exact same source, language, and boundary class — so surface style and unconditional
  passage ease cancel identically;
- on fresh harvest-derived reasoning (`kind="answer"`), how likely the student finds the
  multi-sentence explanation, inference, or synthesis given the excerpt and prompt.
An item's bits are never put over fewer than 16 bytes, so a short answer never dominates a
paragraph. The exam's writing has the white space of the crawl's own text: one space
between words, one line end between lines, no empty lines and no tabs.

**The exam's score** is half the mean of the comprehension kinds and half the mean of the
reasoning kinds. A kind missing from a round gives its share to the others of its own group.
Inside each, every language counts by its share of the round's corpus, then every format. No
share rests on one or two items: a language or a format with fewer than eight items counts in
proportion, and a part's small languages are measured together. A gain on an item is clipped
to a few robust widths, so that no handful of items decides anything.

When the round settles, its exam is published: openly licensed Wikipedia items whole, and for
every other source the item's source, its id there and the item's hash. Anyone can check
those items at the source. So is every student's score on every item
(`/api/v1/rounds/<id>/scores`): anyone can work every gain out again.

**Nobody knows it in advance**, the organisers included. So no exam is kept secret, and
none needs to be. Until the round settles, its items are served only to the machines
grading it.

## The student

| | Architecture | Context | Training budget | Edit limit | SFT probe |
|---|---|---|---|---|---|
| nano | `ERNIE-4.5-0.3B-Base-PT` (18 × 1024, bottom half frozen, top half reset) | 1024 | 1.0M tokens | 120 s | 8 steps |
| micro | `ERNIE-4.5-0.3B-Base-PT` (18 × 1024, bottom half frozen, top half reset) | 1024 | 8.4M tokens | 300 s | 16 steps |
| **standard (official)** | `ERNIE-4.5-0.3B-Base-PT` (18 × 1024, bottom half frozen, top half reset) | 1024 | 65.5M tokens | 1,200 s | 32 steps |
| large (finals) | `ERNIE-4.5-0.3B-Base-PT` (18 × 1024, bottom half frozen, top half reset) | 1024 | 131M tokens | 4,800 s | 32 steps |

Pinned `baidu/ERNIE-4.5-0.3B-Base-PT` weights (Apache 2.0, 103,424-token multilingual
vocabulary) with its embeddings and lower 9 transformer layers frozen and its upper 9 layers
deterministically re-initialized from the round's seed (`127.4M` trainable parameters; peak
grading VRAM ≤10 GiB, fitting 12 GB+ GPUs). Trained without shuffling in the curator's exact
script order using AdamW with cosine decay, then given a short fixed 1-epoch multilingual
fine-tuning probe (512 open, commercial-permissive examples across 11 languages) so the
student's acquired representations read out cleanly on comprehension and reasoning tasks.
Training is counted in tokens, never seconds, so every machine does the same work.

## The curator's machine

Curators are run on volunteers' machines, which differ. What a curator is given does not:

| | |
|---|---|
| CPU | 6 cores |
| Memory | 20 GB |
| GPU | one, NVIDIA or AMD, with 22 GB of its memory |
| Scratch space | 12 GB at `/scratch`, held in memory, so it comes out of the 20 |
| Network | none, but loopback (`127.0.0.1`) works between the curator's own processes |
| Software | Python 3.13, Node.js 24 and the season's libraries |

- **The libraries** are listed in `AGENTS.md` and at `/api/v1/season` (`environment`),
  each at an exact version: PyTorch, transformers, sentence-transformers, tokenizers,
  safetensors, scikit-learn, numpy, scipy, pandas, polars, faiss, fastText, tiktoken and
  more, with what serving a model to a harness of your own needs (FastAPI, uvicorn, the
  `openai` client, Node.js). Every machine that runs curators has exactly those.
  `stet doctor` checks yours.
- **A curator may be many processes.** They share the cores, the memory and the clock,
  and all of them are stopped with it.
- **The time limit is wall-clock time.** When it passes, the curator is killed, and the
  last complete script it wrote still counts. Write one early and improve it.
- **Machines that run curators are at least as fast as a floor**, measured with a fixed
  piece of GPU work and a fixed piece of CPU work (`stet bench`). The floor is a
  mid-range desktop GPU; the numbers are at `/api/v1/season` (`runner`). A curator that
  finishes in time on the floor finishes in time everywhere, and `stet run` says how your
  machine compares with it. A faster machine gets more done in the same seconds. A
  curator that does as much as time allows will therefore do a little better on some runs
  than on others, and its score is taken over its runs.
- The cores, the memory and the scratch space are enforced by the operating system. The
  limit on GPU memory is set through PyTorch, and a program could lift it; so could it
  lift its restriction to six cores. Doing either is against the rules, and curators are
  public.
- **Two sandboxes.** Volunteers' machines run curators in bubblewrap. The organisers'
  rented machines are containers, which cannot make the namespaces bubblewrap is made of,
  and run them in a second sandbox made for containers: a user of the curator's own, a
  system-call filter, Landlock and the same limits. A curator finds the same files at the
  same paths in both, the same cores, memory and GPU, and no network. What differs:

  | | bubblewrap | the container sandbox |
  |---|---|---|
  | Scratch space | in memory, part of the 20 GB | on disk, 12 GB, apart from the 20 GB |
  | Loopback | TCP and UDP | TCP only, to ports 1024 and up |
  | The machine's other processes | not visible | listed in `/proc`, not readable |
  | `/tmp`, `/dev/shm` | the curator's own, 2 GB each | the machine's, 2 GB each, emptied of the curator's files after it |
  | Memory limit | the kernel's | the kernel's where the container allows it, otherwise watched four times a second |

  Each run says which it was in (`sandbox`, on the round's page and at `/api/v1/rounds`).
  A curator that works in one works in the other unless it relies on one of these.

If no verifier can run an entry at all (three attempts fail on the verifiers' side, or
five at grading a script), the round is void for that entry. It counts neither for nor
against it, and the entry is drawn again. Three void rounds in a row, each in a round
where other entries did run, are taken to be the entry's doing, and it stops being run.

## The score

- **Gain** = how much better your script's student does on the exam than the unedited
  corpus's, on the same round with the same initial weights, in bits per byte. The unedited
  student is trained four times a round, by four different established graders, and its
  score on each item is the median of theirs. The unedited corpus is every document in the
  order it was drawn.
- **An entry's gain for a round is taken over its runs**: the mean of two, or the median
  of three. Two runs of a curator that uses neither the clock nor a GPU write the same
  script, and then there is one number.
- A run with no usable script (nothing written, malformed, text of its own or too short a
  copy in it, or the curator crashed) has failed. Beside a run that finished, it counts
  zero, or as the
  entry's worst finished run when that is below zero. A round in which most runs failed is
  **invalid**, and
  **counts as the lowest gain any valid entry got in that round, or zero if that is lower**:
  failing is never the better move. A curator with two invalid rounds in its last five stops
  being run, whether it is qualifying or on the ladder.
- **Rating** = your mean gain, adjusted for how hard your rounds were. Nothing else is in
  it: who else is on the board does not move it. Beside it is its **±**, the rating's
  standard error, which is wide while your rounds are few and narrows with each one. Every
  entry runs 3 rounds and is ranked when it has them. A curator you replace keeps its page
  and its results but leaves the ranking.
- **Ties are shown as ties.** A curator whose rating is within two standard errors of the
  first curator at a place shares that place on the board.
- **Multiplier** = how many times more unedited data the student would have needed to do
  as well. None is shown until that curve has been measured on this exam.

## The ladder

Eight curators run in every round: the record holder and the best curators, one per
entrant, taken in order of **rating less 1.28 standard errors**: what a curator is very
likely to be at least as good as. Three lucky rounds do not take a place from thirty steady
ones. That is what lets any two of them be compared on the same data. The numbers in
this section and the last are what a few machines can run and grade between draws; they
go up as more people lend a GPU.

The record holder keeps its place whatever happens to it: replaced by its owner, or
failing. The record can only be taken in rounds the holder runs, so a holder that stopped
running would hold it for ever. Its place is its own, and its owner's best other curator
may hold one too. That is how you beat your own record.

The organisers' reference curators are the one exception to "one per entrant": their own
three (`stet/identity`, `stet/english`, `stet/tidy`), four that rebuild published recipes
(`stet/heuristic`, `stet/dclm`, `stet/fineweb-edu`, `stet/fineweb2`), `stet/mimic`, and
`stet/reader` (a 0.6B `vLLM` multilingual quality rater). They are yardsticks:
each holds a place of its own beside the eight, whatever its rating, and takes none from an
entrant's curator. Two of them, `stet/fineweb2` and `stet/reader`, run in every round and tie
the rounds together; each of the others runs in one round in eight, in turn, and keeps its
rating from the rounds it ran. Any curator but the record holder that has stopped producing
valid scripts leaves the ladder, a reference too.

## The record

One curator holds the record. A challenger takes it when all of these hold:

- it has run 5, 10, 20 or 40 rounds that the holder also ran. It is tested when it
  reaches each of those four counts and at no other time. (When the holder changes, a
  curator that already shares more rounds with the new one is tested once, at the largest
  count it has reached.)
- its gain over those rounds beats the holder's by more than a **margin** of 0.005 bits per byte
  *(provisional)* in a one-sided paired test, at p < 0.05 split across the four looks.

Every number in the test is a settled gain. The first record of a season goes the same
way, with the unedited corpus standing in for the holder.

One lucky round never makes a record, and neither does resubmitting the leader with a
different seed: in simulation, a copy of the holder takes the record less than once in a
hundred tries for any spread between rounds up to 0.05 bits per byte. The price is patience. A real lead well
clear of the margin is confirmed
almost always, though it can take all 40 rounds; one that only just clears it may never
be. [Experiments](EXPERIMENTS.md#what-the-record-rule-can-tell-apart) has the numbers.

## Finals

The season's final table is decided by finals. Rounds choose who is looked at closely;
finals decide.

**Who.** Entries close when the finals are announced. When the season closes, the board is
frozen as it stands.

- The **eight finalists** are the eight best curators on the board by its cautious rating,
  in the ladder's order, one per person: each person's best.
- **One more is drawn by lot** from the curators ranked 9th to 16th in that same order that
  have run at least five rounds. It is run with the finalists and takes whatever place its
  gain earns. The lot is there to check the rounds: they only choose who is looked at
  closely, and a curator they passed over may be as good as one they chose. If the lot's
  pick places ahead of finalists, the season's report says so. Its result is published
  either way.
- The organisers' reference curators are run beside them as **yardsticks**. They are given
  no place and count toward nobody's.

The lot is drawn from one round of the drand beacon, named before it is emitted: the pick
is number `u mod n` of the `n` curators in the lot, in the board's order and counting from
0. Here `seed` is the 32 bytes of `sha256("stet|<season>|finals|lot|<drand round>|<randomness>")`,
and `u` is the first eight bytes of `sha256(seed + "|lot|0")` as a big-endian number. The
frozen board and the beacon round are published, so anyone works out the same pick.

**On what.** Two finals rounds, drawn like any round from drand rounds announced
beforehand, on the newest Common Crawl crawl **published after the board was frozen**.
Entries closed before that, so no curator in the finals was written by anyone who had seen
that crawl.

**How.** Every finalist, the lot's pick and every yardstick is run on each finals round at
the **large** size: its corpus, its time limit, the usual sandbox. The large student is
trained twice on each script and sits that round's exam. The unedited corpus's large
student is trained four times in each round, and once more at half, twice and four times
the training budget: that is the finals' own curve, which the multipliers are read from.
A curator that writes no usable script in a round counts as the lowest gain any curator got
there, or zero if that is lower, as in any round.

**The table.** For each curator:

- its **gain**: the mean of its two rounds, each the mean of its two trainings, against
  the mean of that round's four unedited students;
- the **standard error** of that gain, from how far its two rounds differ and never less
  than the noise between two trainings of one script allows;
- its **multiplier**, read from the finals' own curve.

Places go by gain, from the top. Two curators whose gains differ by less than two
standard errors of the difference are **tied, and share the place**: a curator shares a
place with the first curator at it unless it is two standard errors or more behind that
one, and the next place counts everyone above it (1, 1, 3). The standard error of a
difference is taken from all the placed curators together: how far their gains moved
against each other from one round to the other. With two rounds, one pair alone says
almost nothing about that.

**What is published.** The frozen board, the beacon rounds, every script, both exams as a
settled round's is published, and every student's score on every item, the unedited ones
and the curve's included. The students are trained on the organisers' machines; with what
is published anyone can train any of them again.

## Running and grading

No score comes from the organisers' machines in particular. Curators are run and scripts
are graded by **verifiers**: anyone who runs `stet verify`.

- **Established verifiers.** An account is established once 5 of its grades have agreed
  with other people's, and it has a linked GitHub account at least 30 days old. Scores
  are made only from established verifiers' work. Everyone else's grades are
  replications: published, compared, and counted towards becoming established.
- **Someone has to be first.** The organisers' own accounts start out established. They
  are marked on the verifiers page, their work is checked like anyone's, and the season
  does not need them once other people are established.
- **Each entry is run twice,** by two different established verifiers. A machine asking
  for work is handed first an entry whose other run was on the other make of GPU, so the
  two runs are on NVIDIA and AMD when machines of both are asking. Nobody runs their own entry, runs one entry twice, or
  runs an entry after being shown a script of it. Scripts are sealed until the round
  settles, so the second runner cannot copy the first: a script is served only to the
  verifier handed it to grade, and a verifier running an entry holds no other job.
- **Early on there may be one machine.** While the organisers' verifier is the only
  established one, it runs each entry once and grades it, the entry is marked *run once*,
  and rounds do not wait for people who are not there. Every script is published when the
  round settles, so anyone can check any score afterwards, and entries are *confirmed*
  from the first round a second established verifier is around for.
- **Each script is graded by someone other than the verifier who wrote it.** Its gain is
  the median of the established verifiers' grades, each against the round's unedited
  student. Two honest machines agree to within a small tolerance *(provisional: 0.03 bits
  per byte)*. The first four established graders of a round are each asked to train the
  unedited student as well; the rest are not, and their grades are measured against the
  median of those four. (Season 1 trained one on every machine: a third of all its
  trainings.) A measurement of the unedited student must agree with the
  others' to within 0.05 bits per byte, or its grade is turned away and listed with the
  round. A machine that is not yet established always trains its own: that is how it is
  checked. When two established grades of a script
  differ by more than that, a third is taken. When both runs write the same script, byte
  for byte, it is nobody's own and either runner may grade it: neither could have made
  it up. Established verifiers who come free before the exam is posted grade its
  scripts again while it is made, up to four grades of each, and those count too. One
  that has trained its grade before the exam may take another a score waits for, up to
  three, while one more training still fits before the exam is due. After
  the exam only the grades a score still waits for are handed out, since any other
  would hold the round back for as long as it trained, and for its first two minutes they
  go to the fastest machines that are free. The round settles as soon as those under way
  are in.
- **The entry's gain is the mean of its two runs.** When they differ by more than 0.05
  bits per byte *(provisional)*, or one failed and the other did not, a third verifier runs the
  curator and the median of the three is the score.
- **A round does not wait for people who are not there.** If nobody who is around (at
  work on a job, or has asked for one in the last ten minutes) could give an entry its
  second run, it settles with one. Every entry's page shows its runs, who made each, on what GPU, each run's
  gain and every grade, and the entry is marked **confirmed** (two or more runs by
  different people that agree, each graded by someone else, with more than one person's
  grades behind them), **one grader** (they agree, and one person graded them all),
  **one runner** (they agree, and one person made them all), **run once**, or **runs
  differed**. The organisers' machines count as one person, however many there are.
- **No score rests on one volunteer's word.** A volunteer never grades a script that only
  it wrote. If nobody else is there to grade it within 45 minutes, that run is dropped,
  and if it was the entry's only run the round is void for that entry. The one exception is the
  organisers' accounts: alone, one of their machines runs and grades, so that a season
  can start. Such entries are marked run once, and say who graded them.
- Verifiers are assigned their work. They cannot choose it, are offered an entry once,
  never grade their own, and cannot see any script, any grade or the round's unedited
  loss until the round settles, beyond the one they are working on.
- **Standing.** A grade within the tolerance of the script's gain has agreed. One outside
  it is a strike. With three strikes, and more than one for every ten grades that agreed,
  an account is given no work for two weeks after the latest. It can still enter. When
  exactly two established grades disagree and nobody is there to break the tie, neither
  is struck, and the entry says so. A grade is a strike only against a gain that two
  established grades agree on, or an organisers' machine's; against one volunteer's lone
  grade it is published as unresolved and counts against nobody.
- A verifier whose job runs out of time unreported is offered no work for a day. A round
  waits for an established verifier that holds a job until the job is reported, handed
  back or out of time, which is five hours at most. For anyone else whose grades have
  agreed before it waits up to 45 minutes, and for an account with no such grade it does
  not wait.
- Agreeing about an entry that left the corpus as it was earns no standing. Its score is
  the unedited student's, which four machines a round measure anyway. Nor does agreeing that
  a script is invalid, which is decided before anything is trained.
- Running curators needs Linux, the sandbox, the season's libraries and a machine at or
  above the floor. Grading needs an NVIDIA (CUDA) or AMD (ROCm) GPU and the season's
  PyTorch, so that every grade uses the same arithmetic.

More in [Verifying](VERIFYING.md).

## What is trusted, and what is not

Stated plainly, because it is short.

**The verifiers assigned to an entry are trusted not to be working together.** A score
is two runs and their grades, by different people the server picked at random. One
dishonest verifier is outvoted or shows up as a difference. Two who are both assigned to
the same entry, and agree on a lie, are not. What stands in the way: an account needs a
GitHub identity at least 30 days old and a record of agreeing before it is given any
work that counts; nobody chooses their work; everything is published when the round
settles; and anyone can rerun any curator with `stet run --round <id>` and grade any
script with `stet score`.

**What that does not catch:**

- A runner who hands back a script the curator did not write. The second run is the check,
  and it is a loose one: two honest runs of a curator that uses a GPU or the clock differ
  a little, so a dishonest script that scores within the tolerance of the honest one is
  not told apart from it. It moves the entry's gain by at most half that tolerance, in
  one round.
- A runner on a much faster machine. The time limit is seconds, not work.
- Two scores closer together than the grade tolerance. Replication cannot tell them
  apart. The published differences are how the tolerances get measured.
- Early in a season, when the organisers' machine may be the only established verifier,
  an entry marked run once rests on that machine alone.

**The organisers are trusted with:**

- **The first established accounts.** Theirs, and so which grades the first volunteers
  are compared with.
- **Assigning work at random.** The server chooses who runs and who grades what.
- **Harvesting honestly.** The examiner harvests every hour by published code and tells
  the server each harvest's hash before any round can be given it (`/api/v1/harvests`); the
  draw then picks the hours, so no harvest can be made to suit a round or a curator. Anyone
  can make the generated questions again from the seed, check which hours the seed picks,
  and check every fresh item against its source once the round settles. What they could do
  and nobody would see: leave a source out of an hour's harvest. They tune nothing of their
  own reference curators on a round's exam before it settles.
- **Training the finals.** The finals' students are trained on the organisers' machines.
  The scripts and the exam are published, so any of them can be trained again.
- **Choosing when the season closes,** and so which rounds pick the finalists. The close is
  announced two weeks ahead.
- **Choosing the moment a round is scheduled**, and with it the crawl. Not the seed.
- **Drawing when they said they would.** A draw can be put off, for a round still being
  graded or for want of entries, and an organiser could put one off for a worse reason
  after seeing its beacon value. Every round's page lists the draws it was announced for
  and left, and verifiers keep their own record of what they were told.

They are not trusted with the seed, the corpus, the generated questions, the arithmetic,
running your curator or grading it. All of those are done by other people or can be checked by
anyone.

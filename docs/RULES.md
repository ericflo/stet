# The rules

Season 0. These are the rules as the server and the harness enforce them. Where a number
is still being measured it says *provisional*, and [Experiments](EXPERIMENTS.md) says how it
will be set.

## The idea

A small language model, the **student**, is trained on raw web text for a fixed number of
tokens and then sits an **exam** of hand-picked text. The model, the trainer and the exam
never change. You change one thing: the data. You write a **curator**, a program that is
handed raw web pages it has never seen and ten minutes on one GPU. It answers with an
**edit script**: what to keep, what to cut, and in what order to train on it. The lower
the student's exam loss, the better the curator.

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
  and 4 GiB in all, named in `stet.toml` as `[[assets]]`. Each is a public file on Hugging
  Face, pinned to a commit and named by its sha256. `stet asset add <owner>/<repository>`
  writes the tables. The files are fetched before the clock starts and are in place,
  read-only, when the curator runs. Nothing can be downloaded while it runs.
- **Libraries are provided.** PyTorch, transformers, scikit-learn and some forty-five
  others, and Node.js, at exact versions: see [the curator's machine](#the-curators-machine).
  You bundle your own code and nothing else.
- **It has to run on both makes of GPU.** Curators are run by volunteers, on NVIDIA and on
  AMD. Use the GPU through PyTorch (`device="cuda"` is right on both) and it will. A run
  that fails scores a gain of 0.
- **Try as much as you like on your own machine.** `stet run` is the leaderboard's own
  harness: nothing is uploaded and nothing is counted. Practice rounds never change, and
  every official round can be replayed once it has settled (`stet run --round s0-o00012`).
- Officially you may have 3 entries in play at once, waiting or qualifying, and make 6
  submissions a day. A new version of an entry (the same name) takes the old version's
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
  trained on either exam, lifting the limits the sandbox sets, and anything that tries to
  leave it. Models trained on other data are fine. Curators that break these are removed,
  and since every curator is public, anyone can check.

## A round

A round is drawn when an entry is waiting for one, and every 6 hours whether or not one
is. Never two at once: a round starts only when the one before has settled. A round drawn
because somebody is waiting comes at least an hour after the last draw, and only while a
verifier who runs curators is around to run it.

1. **Announce.** The server names, in advance, the drand round whose randomness will seed
   the draw and the Common Crawl crawl it will draw from. Verifiers keep a log of these
   announcements and refuse a round that contradicts what they were told. No draw is
   announced less than six minutes ahead, and a draw that is moved is announced again the
   same way, for a time no round was ever announced for before.
2. **Commit.** Your curator must be submitted at least 30 seconds before the draw.
3. **Draw.** The seed is `sha256(season, round number, drand round, drand randomness,
   crawl)`. It picks the corpus files, the order of the documents, the exam passages and
   the student's initial weights. Nobody can know or steer any of that beforehand,
   organisers included. Draws happen only at fixed ten-minute marks, so the organisers
   cannot pick a favourable drand round either.
4. **Edit.** Two volunteers each run every curator, in a sandbox, with a time limit. Each
   run writes an edit script.
5. **Grade.** Other volunteers train the student on what each script says and measure its
   exam loss. Grading runs no entrant code: a script is plain data.
6. **Settle.** A round settles as a whole. Until then every script and every grade is
   sealed. Then each score is published, with every run and every grade behind it,
   whoever made them and whatever they said.

[Running and grading](#running-and-grading) says who those volunteers are and how their
work becomes a score.

Who runs in a round: the ladder (see below), entries still qualifying, and up to 3 new
entries. When more than 3 are waiting, entries from people who have graded for others in
the last two weeks go first.

Rounds never overlap. If one is still being graded when the next is due, the next is
drawn at the first ten-minute mark that is at least six minutes after it settles.

## The corpus

The newest published Common Crawl crawl at the time of the draw, as full-page text: every
language, with menus, cookie banners, footers, shop listings, spam and duplicates
alongside real prose. A round's corpus is a seeded sample of it, roughly ten times more
text than the student can be trained on. Your curator has to choose.

Before your curator sees the corpus, every document that contains the opening 64 bytes of
a long exam sentence is removed. Nine tenths of the exam's text is in such sentences.

## The edit script

One line per output document. Line order is training order: the trainer does not shuffle.

```
123                                   keep document 123 whole
{"d": 123, "k": [[0, 900]]}           keep byte ranges of document 123
{"s": [[123, 0, 900], "text"]}        join copies [doc, start, end] and new text
{"t": "text"}                         a document of new text
```

- **Cutting, reordering and repeating are free.**
- **New text is ink, and there is almost none.** Literal text and copies shorter than 64
  bytes are ink. The one exception is a whole short document kept on a line of its own.
  A script may train the student on at most
  0.1% ink, counted in tokens: 40,000 tokens at the official scale. That is enough for
  separators and small repairs and nothing else. It is what makes this an editing
  competition: without it the winning move is to ignore the corpus and ship a dataset.
- **The budget cuts both ways.** The stream is cut off at the training budget. A script
  that is shorter is repeated from the top, so keeping less means training on it more
  often.
- Output documents that contain the opening of a long exam sentence are dropped, whatever
  they are made of.
- Limits: 64 MB per script, 1 MB per line and per output document, 20,000 pieces joined
  in one document, and no line nested more than 16 brackets deep (the lines above need 3).

## The exam

Six sources of openly licensed English text, weighted equally: encyclopedia, textbooks,
science (abstracts), questions and answers, books, news. The score is the mean of the six
per-source losses, in nats per token, so a student that is excellent at one source and
poor at the rest does badly.

The exam pack is public, so that anyone can reproduce any score. Your curator cannot read
it while it runs, the corpus is cleaned of it, and so is your script's output.

## The held-out exam

A public exam can be learned. A curator could be tuned to the exam's own passages rather
than to good data, and its score would then say more than it should.

So the season has a second exam, and only the organisers hold it. It is built by the same
program from the same six public collections, out of documents the public exam does not
use, chosen by a secret random seed. Its hash is in the season file (`heldout_exam`) and
at `/api/v1/season`. `stet` remembers the first hash it sees for a season and says so if
it is ever shown another.

What it tests is narrow: whether a curator's gain belongs to the public exam's particular
passages or to the kind of writing they are. It is drawn from the same collections, so it
says nothing about writing of other kinds. Those collections are public, and you may
build your curator with them. You may not build it with either exam.

Nothing is decided by it while the season runs, and nothing about it is published. When
the organisers' own machine grades a script, the student it has just trained also sits
the round's draw from the held-out exam, and the result stays on that machine.

When the season closes, the held-out exam and its seed are published, and every script
that counted is graded on it the way scripts are graded during the season: by volunteers,
more than one to a script. The season's final table is the leaderboard worked out again
with held-out losses in place of public ones: the same curators, the same rounds, the same
scripts, the same arithmetic.

Season 0 has no closing date yet. It will be announced on the site at least two weeks
ahead.

One thing is different from the public exam. Pages that contain held-out passages cannot
be removed from the corpus without giving the passages away, so they stay in. They are
rare; [Experiments](EXPERIMENTS.md#the-held-out-exam) has the count. The held-out exam
shares no document and no sentence opening with the public one.

## The student

| | Layers × width | Context | Training budget | Edit limit |
|---|---|---|---|---|
| nano | 4 × 192 | 256 | 8M tokens | 120 s |
| **micro (official)** | 6 × 384 | 512 | 40M tokens | 600 s |
| standard | 8 × 512 | 512 | 160M tokens | 1,200 s |

An ordinary GPT with an 8,192-token vocabulary, AdamW, warm-up and cosine decay. Training
is counted in tokens, never seconds, so every machine does the same work.

## The curator's machine

Curators are run on volunteers' machines, which differ. What a curator is given does not:

| | |
|---|---|
| CPU | 6 cores |
| Memory | 12 GB |
| GPU | one, NVIDIA or AMD, with 8 GB of its memory |
| Scratch space | 8 GB at `/scratch`, held in memory, so it comes out of the 12 |
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

If no verifier can run an entry at all (three attempts fail on the verifiers' side, or
five at grading a script), the round is void for that entry. It counts neither for nor
against it, and the entry is drawn again. Three void rounds in a row, each in a round
where other entries did run, are taken to be the entry's doing, and it stops being run.

## The score

- **Gain** = the unedited corpus's exam loss minus the loss with your script, on the same
  round with the same initial weights, both measured by the same grader. The unedited
  corpus is every document in the order it was drawn.
- **An entry's gain for a round is taken over its runs**: the mean of two, or the median
  of three. Two runs of a curator that uses neither the clock nor a GPU write the same
  script, and then there is one number.
- A run with no usable script (nothing written, malformed, over the ink limit, or the
  curator crashed) scores a gain of 0. A round in which every run failed is **invalid**.
  A curator with two invalid rounds in its last five stops being run, whether it is
  qualifying or on the ladder.
- **Rating** = your mean gain, adjusted for how hard your rounds were. Every entry runs 3
  rounds and is ranked when it has them. A curator you replace keeps its page and its
  results but leaves the ranking.
- **Multiplier** = how many times more unedited data the student would have needed to do
  as well, read off a measured curve of baseline loss against training tokens. It is never
  extrapolated: past the last measured point it reads "more than" that point.

## The ladder

Eight curators run in every round: the record holder and the best-rated curators, one per
entrant. That is what lets any two of them be compared on the same data. The numbers in
this section and the last are what a few machines can run and grade between draws; they
go up as more people lend a GPU.

The record holder keeps its place whatever happens to it: replaced by its owner, or
failing. The record can only be taken in rounds the holder runs, so a holder that stopped
running would hold it for ever. Its place is its own, and its owner's best other curator
may hold one too. That is how you beat your own record.

The organisers' three reference curators (`stet/identity`, `stet/english`, `stet/tidy`) are
the one exception to "one per entrant". They are yardsticks: all three hold places while
they rank in the eight, and they leave as better curators arrive. Any other curator that
has stopped producing valid scripts leaves the ladder too.

## The record

One curator holds the record. A challenger takes it when all of these hold:

- it has run 5, 10, 20 or 40 rounds that the holder also ran. It is tested when it
  reaches each of those four counts and at no other time. (When the holder changes, a
  curator that already shares more rounds with the new one is tested once, at the largest
  count it has reached.)
- its gain over those rounds beats the holder's by more than a **margin** of 0.02 nats
  *(provisional)* in a one-sided paired test, at p < 0.05 split across the four looks.

Every number in the test is a settled gain. The first record of a season goes the same
way, with the unedited corpus standing in for the holder.

One lucky round never makes a record, and neither does resubmitting the leader with a
different seed: in simulation, with the noise measured so far, a copy of the holder
takes the record about twice in a thousand tries. The price is patience. A real lead well
clear of the margin is confirmed
almost always, though it can take all 40 rounds; one that only just clears it may never
be. [Experiments](EXPERIMENTS.md#what-the-record-rule-can-tell-apart) has the numbers.

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
  verifier handed it to grade, and a verifier holds one job at a time.
- **Early on there may be one machine.** While the organisers' verifier is the only
  established one, it runs each entry once and grades it, the entry is marked *run once*,
  and rounds do not wait for people who are not there. Every script is published when the
  round settles, so anyone can check any score afterwards, and entries are *confirmed*
  from the first round a second established verifier is around for.
- **Each script is graded by someone other than the verifier who wrote it.** Its gain is
  the median of the established verifiers' grades, each against that grader's own
  measurement of the unedited corpus. Two honest machines agree to within a small
  tolerance *(provisional: 0.08 nats at micro)*. When two established grades of a script
  differ by more than that, a third is taken. When both runs write the same script, byte
  for byte, it is nobody's own and either runner may grade it: neither could have made
  it up.
- **The entry's gain is the mean of its two runs.** When they differ by more than 0.12
  nats *(provisional)*, or one failed and the other did not, a third verifier runs the
  curator and the median of the three is the score.
- **A round does not wait for people who are not there.** If nobody who is around (at
  work on a job, or has asked for one in the last ten minutes) could give an entry its
  second run, it settles with one. Every entry's page shows its runs, who made each, on what GPU, each run's
  gain and every grade, and the entry is marked **confirmed** (two or more runs by
  different people that agree, each graded by someone else, with more than one person's
  grades behind them), **one grader** (they agree, and one person graded them all),
  **run once**, or **runs differed**.
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
  back or out of time, which is three hours at most. For anyone else whose grades have
  agreed before it waits up to 45 minutes, and for an account with no such grade it does
  not wait.
- Agreeing about an entry that left the corpus as it was earns no standing. Its loss is
  the round's unedited loss, which every grader measures anyway. Nor does agreeing that
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
- **Keeping the held-out exam to themselves** until the season closes, and tuning nothing
  on it, their own reference curators included. Its hash is public now; the exam and its
  seed are published when the season closes.
- **Choosing when the season closes.** They can see held-out results from their own
  machine while it runs. The close is announced two weeks ahead, and the final table
  covers every round.
- **Choosing the moment a round is scheduled**, and with it the crawl. Not the seed.
- **Drawing when they said they would.** A draw can be put off, for a round still being
  graded or for want of entries, and an organiser could put one off for a worse reason
  after seeing its beacon value. Every round's page lists the draws it was announced for
  and left, and verifiers keep their own record of what they were told.

They are not trusted with the seed, the corpus, the exam draw, the arithmetic, running
your curator or grading it. All of those are done by other people or can be checked by
anyone.

# The rules

Season 0. These are the rules as the server and the harness enforce them. Where a number
is still being measured it says *provisional*, and [Experiments](EXPERIMENTS.md) says how it
will be set.

## The idea

A small language model, the **student**, is trained on raw web text for a fixed number of
tokens and then sits an **exam** of curated text. The model, the trainer and the exam
never change. You change one thing: the data. You write an **editor**, a program that is
handed raw web pages it has never seen and a few minutes on one GPU. It answers with an
**edit script**: what to keep, what to cut, and in what order to train on it. The lower
the student's exam loss, the better the editor.

## What you enter

An editor is a directory with a `stet.toml` and any program. `stet init` makes one.

- The bundle (the files in the directory, compressed) may be at most 50 MB, 400 MB
  unpacked and 2,000 files. Models and word lists your editor needs go in the bundle.
  There is nothing to download at run time. Hidden files, `venv/`, `node_modules/`,
  `__pycache__/`, `runs/`, compiled Python and symbolic links are never packed;
  `stet submit --dry-run` lists exactly what is.
- You have one entry in play at a time. Submitting again replaces your entry if it is
  still waiting or qualifying. At most 6 submissions a day.
- Entering needs a GitHub account at least 30 days old, linked with `stet login --github`.
  Running your editor takes time on an organiser's machine, which is the scarcest thing
  the competition has, so one person is one entrant: the limits above are counted per
  GitHub identity, and linking yours to a new key withdraws the old key's entries.
- **Editors are published.** From the moment yours is first drawn into an official round,
  its code and every edit script it writes are public under the MIT licence. `stet fork`
  copies any of them. Its name and yours are public as soon as you submit.
- Not allowed: per-document knowledge of official data (lists of hashes, precomputed
  labels, tables by host or URL built from the crawl), exam text in any form, and anything
  that tries to leave the sandbox. Models trained on other data are fine. Editors that
  break these are removed, and since every editor is public, anyone can check.

## A round

Official rounds are drawn about every 6 hours and never two at once: a round starts only
when the one before has settled.

1. **Announce.** The server names, in advance, the drand round whose randomness will seed
   the draw and the Common Crawl crawl it will draw from. Verifiers keep a log of these
   announcements and refuse a round that contradicts what they were told. No draw is
   announced less than six minutes ahead, and a draw that is moved is announced again the
   same way, for a time no round was ever announced for before.
2. **Commit.** Your editor must be submitted at least 30 seconds before the draw.
3. **Draw.** The seed is `sha256(season, round number, drand round, drand randomness,
   crawl)`. It picks the corpus files, the order of the documents, the exam passages and
   the student's initial weights. Nobody can know or steer any of that beforehand,
   organisers included. Draws happen only at fixed ten-minute marks, so the organisers
   cannot pick a favourable drand round either.
4. **Edit.** An **anchor** (an organiser's machine) runs each editor in a sandbox, with a
   time limit, and publishes the edit script it wrote.
5. **Grade.** The anchor trains the student on what each script says, and the exam loss
   it measures is the score. Volunteers grade the same scripts on their own GPUs. Grading
   runs no entrant code: a script is plain data.
6. **Settle.** A round settles as a whole, once an anchor has graded every entry. Until
   then every grade is sealed. Then each score is published, and with it every
   volunteer's grade of the same script, whatever it said.

Who runs in a round: the ladder (see below), entries still qualifying, and up to 3 new
entries. When more than 3 are waiting, entries from people who have graded for others in
the last two weeks go first.

Rounds never overlap. If one is still being graded when the next is due, the next is
drawn at the first ten-minute mark that is at least six minutes after it settles.

## The corpus

The newest published Common Crawl crawl at the time of the draw, as full-page text: every
language, with menus, cookie banners, footers, shop listings, spam and duplicates
alongside real prose. A round's corpus is a seeded sample of it, roughly ten times more
text than the student can be trained on. Your editor has to choose.

Before your editor sees the corpus, every document that contains the opening 64 bytes of
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

The exam pack is public, so that anyone can reproduce any score. Your editor cannot read
it while it runs, the corpus is cleaned of it, and so is your script's output.

## The held-out exam

A public exam can be learned. An editor could be tuned to the exam's own passages rather
than to good data, and its score would then say more than it should.

So the season has a second exam, and only the anchors hold it. It is built by the same
program from the same six public collections, out of documents the public exam does not
use, chosen by a secret random seed. Its hash is in the season file (`heldout_exam`) and
at `/api/v1/season`. `stet` remembers the first hash it sees for a season and says so if
it is ever shown another.

What it tests is narrow: whether an editor's gain belongs to the public exam's particular
passages or to the kind of writing they are. It is drawn from the same collections, so it
says nothing about writing of other kinds. Those collections are public, and you may
build your editor with them. You may not build it with either exam.

Whenever an anchor grades an official entry, the student it has just trained also sits
the round's draw from the held-out exam. Those results stay on the anchor. Nothing is
published and nothing is decided by them while the season runs.

When the season closes, the held-out exam, its seed and every result on it are published.
The season's final table is the leaderboard worked out again with held-out losses in
place of public ones: the same editors, the same rounds, the same arithmetic. Grading is
reproducible, so anyone can then check each of those numbers.

Season 0 has no closing date yet. It will be announced on the site at least two weeks
ahead.

One thing is different from the public exam. Pages that contain held-out passages cannot
be removed from the corpus without giving the passages away, so they stay in. They are
rare; [Experiments](EXPERIMENTS.md#the-held-out-exam) has the count. The held-out exam
shares no document and no sentence opening with the public one.

## The student

| | Layers × width | Context | Training budget | Edit limit |
|---|---|---|---|---|
| nano | 4 × 192 | 256 | 8M tokens | 60 s |
| **micro (official)** | 6 × 384 | 512 | 40M tokens | 300 s |
| standard | 8 × 512 | 512 | 160M tokens | 600 s |

An ordinary GPT with an 8,192-token vocabulary, AdamW, warm-up and cosine decay. Training
is counted in tokens, never seconds, so every machine does the same work.

The edit limit is wall-clock time on the anchor: AMD Ryzen AI Max+ 395 (16 cores),
Radeon 8060S, 128 GB unified memory, ROCm. In the sandbox an editor gets 48 GB of memory,
16 GB of scratch space (held in memory, so it comes out of the 48), the GPU, no network,
and the anchor's own Python (3.15 today) with numpy, PyTorch and tiktoken. If the limit
passes, the editor is killed, and the last complete script it wrote still counts.

If an anchor cannot run an entry at all (three attempts fail on the anchor's side, or
five at grading its script), the round is void for that entry. It counts neither for nor
against it, and the entry is drawn again. Three void rounds in a row, each in a round
where the anchor did run other entries, are taken to be the entry's doing, and it stops
being run.

## The score

- **Gain** = the unedited corpus's exam loss minus yours, on the same round with the same
  initial weights. The unedited corpus is every document in the order it was drawn.
- A run with no usable script (nothing written, malformed, over the ink limit) is
  **invalid** and scores a gain of 0. An editor with two invalid rounds in its last five
  stops being run, whether it is qualifying or on the ladder.
- **Rating** = your mean gain, adjusted for how hard your rounds were. Every entry runs 3
  rounds and is ranked when it has them. An editor you replace keeps its page and its
  results but leaves the ranking.
- **Multiplier** = how many times more unedited data the student would have needed to do
  as well, read off a measured curve of baseline loss against training tokens. It is never
  extrapolated: past the last measured point it reads "more than" that point.

## The ladder

Eight editors run in every round: the record holder and the best-rated editors, one per
entrant. That is what lets any two of them be compared on the same data. The numbers in
this section and the last are what one anchor can grade between draws; they go up when
there are more anchors.

The record holder keeps its place whatever happens to it: replaced by its owner, or
failing. The record can only be taken in rounds the holder runs, so a holder that stopped
running would hold it for ever. Its place is its own, and its owner's best other editor
may hold one too. That is how you beat your own record.

The organisers' three reference editors (`stet/identity`, `stet/english`, `stet/tidy`) are
the one exception to "one per entrant". They are yardsticks: all three hold places while
they rank in the eight, and they leave as better editors arrive. Any other editor that
has stopped producing valid scripts leaves the ladder too.

## The record

One editor holds the record. A challenger takes it when all of these hold:

- it has run 5, 10, 20 or 40 rounds that the holder also ran. It is tested when it
  reaches each of those four counts and at no other time. (When the holder changes, an
  editor that already shares more rounds with the new one is tested once, at the largest
  count it has reached.)
- its gain over those rounds beats the holder's by more than a **margin** of 0.02 nats
  *(provisional)* in a one-sided paired test, at p < 0.05 split across the four looks.

Every number in the test is an anchor's grade. The first record of a season goes the same
way, with the unedited corpus standing in for the holder.

One lucky round never makes a record, and neither does resubmitting the leader with a
different seed: in simulation, with the noise measured so far, a copy of the holder
takes the record about twice in a thousand tries. The price is patience. A real lead well
clear of the margin is confirmed
almost always, though it can take all 40 rounds; one that only just clears it may never
be. [Experiments](EXPERIMENTS.md#what-the-record-rule-can-tell-apart) has the numbers.

## Grading and replication

- **An anchor's grade is the score.** An anchor is a machine the organisers run. It
  grades every entry in every round. Nobody else's grade moves the leaderboard.
- **A volunteer's grade is a replication.** A grade is a claim anyone can check: the
  script, the corpus, the exam and the trainer are public, and deterministic enough that
  two honest machines agree to within a small tolerance *(provisional: 0.08 nats at
  micro)*. Up to three volunteers grade each entry. Their grades are sealed until the
  round settles and are then compared with the anchor's. The round's page shows every
  one of them beside the anchor's, with the difference, whoever sent it.
- Replication can only show what the tolerance lets it: two scores closer together than
  the tolerance are not told apart by it. The published differences are also how the
  tolerance itself gets measured across GPUs.
- Volunteers are assigned their work. They cannot choose it, are offered an entry once,
  never grade their own, and cannot see any grade or the round's unedited loss until the
  round settles. A round waits up to 45 minutes after the anchor finished its last entry
  for volunteers who are still at work, if a grade of theirs has agreed before. A volunteer it
  had to give up on, or whose job ran out of time unreported, is offered no work for a day.
- A grade that differs from the anchor's is a strike, unless two volunteers who have
  each agreed with an anchor before stand behind it (next). With three strikes, and more
  than one for every ten grades that agreed, an account is given no grading work for two
  weeks after the latest. It can still enter.
- When two volunteers who have each agreed with an anchor before differ from the anchor
  and agree with each other, neither is struck, nor is anyone else who found what they
  found, and the entry says so: the anchor's grade may be the one that is wrong. The
  score stays the anchor's until the organisers have looked.
- Agreeing about an entry that left the corpus as it was earns no standing. Its loss is
  the round's unedited loss, which every grader measures anyway. Nor does agreeing that
  a script is invalid, which is decided before anything is trained.
- If the anchors cannot grade a script at all, volunteers' grades of it are published
  with nothing beside them, and count for and against nobody.
- Grades come from NVIDIA (CUDA) and AMD (ROCm) GPUs only, so that every grade uses the
  same arithmetic.

More in [Verifying](VERIFYING.md).

## What the organisers are trusted with

Stated plainly, because it is short:

- **Running your editor honestly.** Edits happen on anchors. An anchor could in principle
  publish a script your editor did not write. Editors and scripts are public, and anyone
  can rerun an editor on the same corpus with `stet run --round <id>` and compare. An
  editor that uses neither the clock nor a GPU writes the same script anywhere, hash for
  hash. One that does may not, and then the thing to compare is the gain.
- **Grading honestly.** The score is an anchor's grade. That is a real trust, and the
  check on it is replication: every script is public, anyone can grade it with
  `stet score`, and volunteers' grades are published beside the anchor's whether they
  agree or not.
- **Keeping the held-out exam to themselves** until the season closes, and tuning nothing
  on it, their own reference editors included. Its hash is public now; the exam, its seed
  and every result are published when the season closes.
- **Choosing when the season closes.** They can see held-out results while it runs. The
  close is announced two weeks ahead, and the final table covers every round.
- **Choosing the moment a round is scheduled**, and with it the crawl. Not the seed.
- **Drawing when they said they would.** A draw can be put off, for a round still being
  graded or for want of entries, and an organiser could put one off for a worse reason
  after seeing its beacon value. Every round's page lists the draws it was announced for
  and left, and verifiers keep their own record of what they were told.

They are not trusted with the seed, the corpus, the exam draw or the arithmetic: all of
those can be checked by anyone, and the client checks them every time.

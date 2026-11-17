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

- The bundle (every file in the directory, compressed) may be at most 50 MB. Models and
  word lists your editor needs go in the bundle. There is nothing to download at run time.
- An account has one entry in play at a time. Submitting again replaces your entry if it
  is still waiting or qualifying. At most 6 submissions a day.
- Entering needs a GitHub account at least 30 days old, linked with `stet login --github`.
  Running your editor takes time on an organiser's machine, which is the scarcest thing
  the competition has, so one person is one entrant.
- **Editors are published.** Once yours has run in an official round, its code and every
  edit script it writes are public under the MIT licence. `stet fork` copies any of them.
- Not allowed: per-document knowledge of official data (lists of hashes, precomputed
  labels), exam text in any form, and anything that tries to leave the sandbox. Models
  trained on other data are fine. Editors that break these are removed, and since every
  editor is public, anyone can check.

## A round

Official rounds are drawn every 6 hours.

1. **Announce.** The server names, in advance, the drand round whose randomness will seed
   the draw and the Common Crawl crawl it will draw from. Verifiers keep a log of these
   announcements and refuse a round that does not match what they were told.
2. **Commit.** Your editor must be submitted at least 30 seconds before the draw.
3. **Draw.** The seed is `sha256(season, round number, drand round, drand randomness,
   crawl)`. It picks the corpus files, the order of the documents, the exam passages and
   the student's initial weights. Nobody can know or steer any of that beforehand,
   organisers included. Draws happen only at fixed ten-minute marks, so the organisers
   cannot pick a favourable drand round either.
4. **Edit.** An **anchor** (an organiser's machine) runs each editor in a sandbox, with a
   time limit, and publishes the edit script it wrote.
5. **Grade.** Verifiers train the student on what each script says and report the exam
   loss. Grading runs no entrant code: a script is plain data.
6. **Settle.** When the grades agree, the entry's score is final and public.

Who runs in a round: the ladder (see below), entries still qualifying, and up to 8 new
entries. When more than 8 are waiting, entries from people who have graded for others in
the last two weeks go first.

## The corpus

The newest published Common Crawl crawl at the time of the draw, as full-page text: every
language, with menus, cookie banners, footers, shop listings, spam and duplicates
alongside real prose. A round's corpus is a seeded sample of it, roughly ten times more
text than the student can be trained on. Your editor has to choose.

Before your editor sees the corpus, every document that contains a sentence of the
season's exam is removed.

## The edit script

One line per output document. Line order is training order: the trainer does not shuffle.

```
123                                   keep document 123 whole
{"d": 123, "k": [[0, 900]]}           keep byte ranges of document 123
{"s": [[123, 0, 900], "text"]}        join copies [doc, start, end] and new text
{"t": "text"}                         a document of new text
```

- **Cutting, reordering and repeating are free.**
- **New text is ink, and there is almost none.** Literal text, and copies shorter than 64
  bytes that are not a whole document, are ink. A script may train the student on at most
  0.1% ink, counted in tokens: 40,000 tokens at the official scale. That is enough for
  separators and small repairs and nothing else. It is what makes this an editing
  competition: without it the winning move is to ignore the corpus and ship a dataset.
- **The budget cuts both ways.** The stream is cut off at the training budget. A script
  that is shorter is repeated from the top, so keeping less means training on it more
  often.
- Output documents that contain exam text are dropped, whatever they are made of.
- Limits: 64 MB per script, 1 MB per line and per output document.

## The exam

Six sources of openly licensed English text, weighted equally: encyclopedia, textbooks,
science (abstracts), questions and answers, books, news. The score is the mean of the six
per-source losses, in nats per token, so a student that is excellent at one source and
poor at the rest does badly.

The exam pack is public, so that anyone can reproduce any score. Your editor cannot read
it while it runs, the corpus is cleaned of it, and so is your script's output.

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
16 GB of scratch space, the GPU, no network, and Python 3.13 with numpy, PyTorch and
tiktoken. If the limit passes, the editor is killed, and the last complete script it wrote
still counts.

## The score

- **Gain** = the unedited corpus's exam loss minus yours, on the same round with the same
  initial weights. The unedited corpus is every document in the order it was drawn.
- A run with no usable script (nothing written, malformed, over the ink limit) is
  **invalid** and scores a gain of 0. Two invalid rounds retire an editor.
- **Rating** = your mean gain, adjusted for how hard your rounds were. Every entry runs 5
  rounds, and is ranked once it has 3.
- **Multiplier** = how many times more unedited data the student would have needed to do
  as well, read off a measured curve of baseline loss against training tokens. It is never
  extrapolated: past the last measured point it reads "more than" that point.

## The ladder

Ten editors run in every round: the record holder and the best-rated editors, one per
entrant. That is what lets any two of them be compared on the same data.

## The record

One editor holds the record. A challenger takes it when all of these hold:

- it has shared 5, 10, 20 or 40 rounds with the holder (it is tested at those four
  moments and no others);
- its gain over those rounds beats the holder's by more than a **margin** of 0.02 nats
  *(provisional)* in a one-sided paired test, at p < 0.05 split across the four looks;
- an anchor has graded at least 2 of its rounds itself.

One lucky round never makes a record, and neither does resubmitting the leader with a
different seed: in simulation a copy of the holder takes the record less than once in a
thousand tries. The price is patience. A real lead well clear of the margin is confirmed
almost always, though it can take all 40 rounds; one that only just clears it may never
be. [Experiments](EXPERIMENTS.md#what-the-record-rule-can-tell-apart) has the numbers.

## Grading and trust

- A grade is a claim that anyone can check: the script, the corpus, the exam and the
  trainer are all public and deterministic enough that two honest machines agree to
  within a small tolerance *(provisional: 0.08 nats at micro)*.
- Verifiers are assigned their work. They cannot choose it, never grade their own entry,
  and cannot see anyone's grade until an entry settles.
- A new verifier's grades count once a trusted verifier has graded the same script and
  agreed. Five such agreements earn trust. A trusted verifier's grade settles an entry,
  and one in four entries is checked by a second verifier regardless.
- A grade that disagrees, on the same script and the same data, is a strike. It ends
  trust at once, and three strikes end grading.
- Grades for the leaderboard come from NVIDIA (CUDA) and AMD (ROCm) GPUs only, so that
  every grade uses the same arithmetic.

More in [Verifying](VERIFYING.md).

## What the organisers are trusted with

Stated plainly, because it is short:

- **Running your editor honestly.** Edits happen on anchors. An anchor could in principle
  publish a script your editor did not write. Editors and scripts are public, and anyone
  can rerun an editor on the same corpus with `stet run --round <id>` and compare.
- **Settling disputes.** An anchor's grade overrules others.
- **Choosing the moment a round is scheduled**, and with it the crawl. Not the seed.

They are not trusted with the seed, the corpus, the exam draw or the arithmetic: all of
those can be checked by anyone, and the client checks them every time.

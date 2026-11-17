# Stet — design

*stet* (Latin, "let it stand"): the mark an editor writes beside a passage to say **keep this**.

Status: draft 1, 2026-11-17. Numbers marked *(provisional)* are set by experiments recorded
in [EXPERIMENTS.md](EXPERIMENTS.md) and may change before Season 0 opens.

## The idea in one paragraph

Speedruns fix the data and let people change the model or the kernels. Stet fixes the model
and the kernels and lets people change only the data. You write an **editor**: a program
that is handed a pile of raw web text it has never seen and a few minutes on one GPU. It
hands back an **edit script** — what to keep, what to cut, in what order, with what small
repairs. A fixed small language model (the **student**) is trained on the result for a
fixed number of tokens and then sits an **exam** of curated text. The lower the student's
exam loss, the better the editor. The point is not the edited dataset. The point is to
find out how good an editing agent can get, so that later a swarm of them can be set loose
on a real pretraining corpus.

## 1. What is fixed and what is free

| Fixed for a season | Free |
|---|---|
| Student architecture and initialisation | Which documents and spans are kept |
| Optimiser, schedule, batch size | The order they are trained in |
| Training budget, counted in tokens | How often anything is repeated |
| Tokenizer | Clean-up inside a document |
| The exam's sources and scoring | A little new text (the ink budget) |
| The edit-phase time limit and hardware class | Everything about how the editor decides |

There is no shuffling in the trainer. The stream the editor writes is the stream the
student sees, so curriculum is a first-class move.

## 2. A round

1. **Commit.** Editors are uploaded and hashed before the round closes.
2. **Draw.** At the close, the round's seed is taken from a public randomness beacon
   (drand). The seed picks the corpus shards and the exam passages. Nobody, organisers
   included, can know or steer them beforehand.
3. **Edit.** A verifier runs the editor in a sandbox: no network, read-only corpus, a time
   limit. The editor writes an edit script.
4. **Train.** Trusted harness code applies the script and trains the student for exactly
   the token budget.
5. **Exam.** The student is scored on the round's exam passages.
6. **Score.** The editor's gain is the baseline's exam loss minus its own, on the same
   round, same initial weights.

Practice rounds use fixed public seeds and can be run by anyone at any time. Official
rounds are drawn daily *(provisional)*.

## 3. The corpus

Raw, unfiltered Common Crawl text: the DCLM pool shards, pinned by sha256. It is genuinely
messy — about a third is not English, and the rest includes error pages, shop listings,
navigation menus, spam and duplicates alongside real prose.

- A round's corpus is a seed-chosen set of shards, about eight times the training budget.
  The editor must choose; it cannot keep everything.
- **Practice and official rounds draw from different dataset releases** with no shard in
  common (`dclm-pool-400m-1x` and `dclm-pool-1b-1x`). An editor is developed on one and
  judged on the other.
- Documents that share a long line with any of the round's exam passages are removed
  before the editor sees the corpus.
- On disk the corpus is three flat files (`docs.bin`, `docs.idx`, `urls.txt`) that any
  language can memory-map.

## 4. The exam

The research is blunt about this: on held-out text from the *same* pool, careful selection
barely beats random, because perplexity rewards matching the test distribution. So the
exam is not a slice of the pool. It is what we want a model to have learned:

| Source | What it is |
|---|---|
| encyclopedia | English Wikipedia articles |
| textbooks | Open textbooks and course material (LibreTexts, Pressbooks, OER Commons) |
| science | arXiv abstracts |
| qa | Stack Exchange questions with their answers |
| books | Project Gutenberg |
| news | Openly licensed news |

All of it is from the Common Pile, so every passage is openly licensed. The score is the
**unweighted mean of the six per-source losses**. A specialist cannot win by acing one
source, and each source's loss is reported so that lopsided strategies are visible.

The exam pack is public. That is deliberate: every score can be reproduced by anyone, and
no trusted party is needed to hold a secret. What keeps it honest is in §5 and §10.

## 5. The edit script and the ink budget

An edit script is JSON Lines. One line is one output document; line order is training
order.

```
123                               keep document 123 whole
{"d": 123, "k": [[0, 900]]}       keep these byte ranges of document 123
{"s": [[123, 0, 900], "text"]}    any mix of copies and literal text
{"t": "text"}                     a new document of literal text
```

The four proofreader's marks map onto it directly: **stet** (keep), **dele** (leave out),
**transpose** (reorder), **caret** (insert).

- **Cuts are free.** Deleting, reordering and repeating cost nothing.
- **Ink is scarce.** Literal text, and copies shorter than 64 bytes that are not a whole
  document, are *ink*. At most 10% *(provisional)* of the bytes the student is trained on
  may be ink. This is what keeps Stet an editing competition: without it, the winning move
  is to ignore the corpus and ship a canned dataset.
- **The budget cuts both ways.** A script longer than the token budget is truncated. A
  shorter one is repeated from the top. Keeping less means training on it more often.

The script is plain data. The harness validates it and never executes anything from it.

## 6. The student and the scales

An ordinary GPT: pre-norm, rotary positions, tied embeddings, no dropout, GPT-2 tokenizer.
AdamW, linear warm-up, cosine decay. Initial weights are derived from the round seed, so
every editor in a round starts its student from the same point.

| Scale | Corpus | Train tokens | Student | Use |
|---|---|---|---|---|
| nano | part of 1 shard | *(provisional)* | 4 layers × 192 | Smoke tests, RL inner loops |
| micro | 1 shard (~380 MB) | *(provisional)* | 6 layers × 384 | **Season 0 official scale** |
| standard | 4 shards (~1.5 GB) | 50M | 8 layers × 512 | Scale-up checks; a later season |

Season 0 runs at micro because verification has to be cheap while the verifier network is
small: about a minute of training on an RTX 4090. Whether what wins at micro also wins at
standard is an open question that Stet measures rather than assumes (§11).

Training is budgeted in tokens, never seconds, so every machine does identical work. Only
the edit phase is timed. Its limit is stated for a reference machine (RTX 4090, 8 cores,
32 GB) and scaled on other hardware by a calibration run.

## 7. Scoring

- **Gain** (per round) = baseline exam loss − editor exam loss, in nats per token. The
  baseline is the `identity` editor: every document, in the order it arrived.
- **Rating** = mean gain over the official rounds an editor has run, with a standard error.
  An editor is *provisional* until it has three rounds.
- **The headline number is a multiplier.** Each season publishes a reference curve: the
  baseline's exam loss as a function of training tokens. An editor's gain is mapped through
  it to answer one question: *how many times more raw data would the student have needed to
  do this well?* "2.4×" means the edit made the data worth 2.4 times as many tokens.
  Multipliers beyond the measured range of the curve are marked as extrapolated.
- **Records.** The top editors (the *ladder*) are re-run on every new round, so any two of
  them share rounds. A challenger takes the record when it has at least five shared rounds
  with the holder and beats it in a one-sided paired test at p < 0.05. One lucky round
  never makes a record.
- **Failures.** If an editor crashes, times out or writes an invalid script, the student is
  trained on the unedited corpus: gain 0.

## 8. Verification

There is no central compute. Scores come from **verifiers**: volunteers who run
`stet verify` on a machine with a GPU.

**What a verifier does.** Claims a job (one editor on one round); derives the round from
its seed; runs the editor in a sandbox; applies the script, trains and grades with the
trusted harness; signs the result and uploads it.

**Sandbox.** Only the edit phase is untrusted code. It runs in a Linux namespace sandbox
(bubblewrap): no network, read-only filesystem, home directory hidden, a private scratch
directory, hard limits on time, memory and processes. The editor's only output is the
edit script. This is not VM-grade isolation and the docs say so; a dedicated account or
machine is recommended.

**Why trust a result.**
- *Assignment is the server's.* A verifier cannot choose which job it gets and never gets
  its own editor.
- *Results are sealed.* Nobody can see a job's results until the job settles, and a
  verifier never knows whether its job is also being run by someone else. Copying is
  impossible and fabrication is a gamble.
- *Everyone proves themselves every round.* A verifier's first job in each round is the
  baseline. Its corpus hash, exam hash and loss must match the round's consensus, or its
  results for that round are held.
- *Adaptive replication.* New verifiers are always paired with another. After a streak of
  agreements they are spot-checked at random. One disagreement resets the streak.
- *Anchors.* A few maintainer-designated verifiers settle disputes, and audit at least two
  rounds of any editor before it can hold a record.
- *Everything is reproducible.* Editors, seeds, exam and harness are public. Anyone can
  re-run any score, and a refutation is a first-class result.

## 9. Why anyone would take part

- **One number that means something.** "My editor makes raw web text worth 2.4× more."
- **Numbered records** with a named technique each, and a chart of the record over time.
- **Everything is forkable.** Editors are public once they have run. `stet fork` starts you
  from any of them, and lineage is credited on the leaderboard.
- **A verifier leaderboard** counting validated jobs only (total and recent), with
  agreement streaks.
- **Verify to jump the queue.** Nobody pays to enter, but entries from people who have
  verified for others are run first.
- **Built for coding agents.** `stet init` drops an `AGENTS.md`, and the practice loop is a
  single command with machine-readable output.

## 10. Threat model

| Attack | Defence |
|---|---|
| Tune an editor to a known corpus or exam | Official corpus and exam are drawn after commit, from a pool the editor was not developed on |
| Ship a canned dataset, ignore the corpus | Ink budget; bundle size cap |
| Smuggle exam text into training | Exam-overlapping corpus documents removed; ink is checked for exam overlap; editors are public |
| Editor reads the exam, phones home, tampers with the trainer | Sandbox: no network, no exam on disk, script-only output, trusted grader |
| Verifier fabricates or inflates a score | Sealed results, server assignment, baseline-per-round, replication, anchor audits |
| Verifier copies someone else's answer | Results sealed until settled |
| Many accounts to self-verify | Server assignment, GitHub-linked identities, anchors audit every record |
| Resubmit the leader hoping for a lucky draw | Five shared rounds and a paired test |
| Mine noise with many near-identical entries | One new editor per account per round; ratings regress as rounds accumulate |
| Flood the queue | Rate limits; verified-work priority |

## 11. What we do not know yet, and how Stet finds out

- **Noise.** Run-to-run and cross-GPU variance of the exam loss is measured before launch
  and published; tolerances and the record test are set from it.
- **Scale transfer.** Published evidence says coarse data rankings transfer across model
  scale and fine ones are fragile. Each season, the top editors are re-run at the next
  scale up and the rank correlation is published.
- **Is the exam the right target?** Gains are reported per source, and periodically on
  sources the exam does not contain, to check that editors are learning "good data" and
  not "data that looks like these six sources".
- **Reference hardware.** Time limits were calibrated on the hardware available to the
  maintainer; the reference constants are provisional until measured on an RTX 4090.

## 12. Not in scope

Building the best dataset. Distilling a frontier model. Changing the student. Stet exists
to elicit editors.

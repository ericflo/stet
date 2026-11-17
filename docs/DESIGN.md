# Design

*stet* (Latin, "let it stand"): the mark a proofreader writes beside a passage to say
**keep this**.

This page is why Stet is built the way it is. The rules themselves are in
[Rules](RULES.md).

## The question

Speedruns fix the data and let people change the model. Stet fixes the model and lets
people change the data. The thing being searched for is not a dataset. It is an **agent
that makes datasets**: a program that can be handed raw web text it has never seen and
return something a model learns more from. If such an agent can be found at small scale,
many copies of it can be set loose on a real pretraining corpus.

So the competition is shaped to measure one thing, the editor, and to make every other
explanation for a good score unavailable.

## One variable

| Fixed for a season | Free |
|---|---|
| Student architecture and initial weights | Which documents and spans are kept |
| Optimiser, schedule, batch size | The order they are trained in |
| Training budget, counted in tokens | How often anything is repeated |
| Tokenizer | Small repairs (the ink budget) |
| The exam's sources and scoring | Everything about how the editor decides |

Training is budgeted in tokens, never seconds, so a faster GPU buys nothing. The trainer
does not shuffle: the stream the editor writes is the stream the student sees, which makes
curriculum a real move.

## Decisions, and what they are for

**The exam is not a slice of the corpus.** On held-out text from the same pool, careful
selection barely beats random, because perplexity rewards matching the test distribution.
So the exam is what we want a model to have learned: six kinds of curated, openly licensed
English writing, weighted equally. A specialist cannot win by acing one.

**The exam is public.** A secret exam needs a trusted party to hold it and makes every
score unreproducible. Stet publishes it and defends it three other ways: the editor runs
in a sandbox without it; every corpus document containing a sentence of it is removed
before the editor sees the corpus; and the script's output is checked for it again.

**The data is drawn after the entries are locked.** The seed comes from drand, a public
randomness beacon, at a time fixed on a public grid, and the corpus comes from the newest
Common Crawl. An editor cannot be tuned to data that did not exist as a sample when it was
submitted, and the organisers cannot choose a draw that suits anyone.

**Almost no ink.** If new text were allowed, the winning move would be to ignore the
corpus and ship a dataset, or to have a large model write one. Ink is capped at 0.1% of
the training tokens: enough for separators and repairs. Everything else the student reads
is copied byte for byte from the corpus. This is also what rules out distilling a frontier
model into the training data.

**Editing and grading are separate.** Only the edit runs an entrant's code. It produces an
edit script, which is plain data: document numbers and byte ranges. Grading turns a script
into a score with trusted code alone. That split is what lets volunteers do the expensive
part safely: a stranger's GPU never runs a stranger's program.

**Edits run on the organisers' machines.** Sandboxing untrusted code well enough to ask
volunteers to run it is not something a namespace sandbox can promise, and edit time has to
be measured on known hardware to mean anything. So anchors run the editors (minutes each)
and volunteers do the training (most of the compute).

**Scores are gains, on shared rounds.** Rounds differ in difficulty by far more than
editors differ from each other. Every score is a difference from the unedited corpus on
the same round with the same initial weights, ratings are adjusted for which rounds an
editor happened to get, and the top editors run in every round so that any two of them can
be compared directly.

**The record is hard to take by luck.** A challenger is tested against the holder at 5, 10,
20 and 40 shared rounds, must win by more than a margin set from measured noise, and must
have been graded by an anchor. Copying the leader and hoping for a good draw does not
work, and neither does entering many near-copies.

**The headline is a multiplier.** "+0.31 nats" means little to anyone. "This edit made the
data worth 2.4× as much" does. It is read off a measured curve of the baseline's loss
against training tokens, and never extrapolated past the curve.

**Small on purpose.** The official student is 6 layers by 384, trained on 40M tokens: about
eight minutes on an integrated GPU. Verification has to be cheap while the verifier
network is small. Whether what wins here also wins at larger scale is something Stet
measures rather than assumes: each season the top editors are rerun a scale up.

## What could go wrong

| Attack | What stops it |
|---|---|
| Tune an editor to a known corpus | Drawn after commit from a crawl newer than the editor |
| Ship a dataset instead of editing | Ink budget; 50 MB bundle |
| Smuggle exam text in | Corpus and script output are both cleaned of it; editors are public |
| Editor reads the exam or phones home | Sandbox: no network, no exam on disk, script-only output |
| A hostile editor attacks the machine running it | Only anchors run editors; volunteers never do |
| A hostile script attacks the grader | Scripts are parsed as data with hard limits; nothing in one is executed |
| A verifier invents a score | Sealed grades, assigned work, per-round baseline check, strikes |
| A verifier copies a score | Nothing is visible until an entry settles |
| Many accounts grade each other | Trust comes only from agreeing with already-trusted verifiers; anchors audit records |
| Resubmit the leader for a lucky draw | Four fixed looks, a margin, paired on shared rounds |
| Organiser picks a good seed | drand, a ten-minute grid, the crawl named in advance, checked by every client |
| Flood the queue | GitHub-linked accounts, one entry in play, six submissions a day |

What is left is in [Rules](RULES.md#what-the-organisers-are-trusted-with): anchors are
trusted to run editors honestly, and anyone can check them by rerunning.

## How it is built

- **The harness** (Python, `src/stet`) is what decides a score: rounds from seeds, the
  corpus, the script format, the student, the trainer, the exam. It is the same code in
  `stet run`, `stet verify` and the gym. A hash of its scoring files goes into every
  result.
- **The server** (Go, `server/`) keeps the books: accounts, entries, rounds, jobs, grades,
  ratings. It trains nothing. One stateless process serves the website, the API and the
  downloads.
- **Records** live in [durable-db](https://github.com/ericflo/durable-db), a versioned
  key-value database over object storage. The server holds every record in memory and
  writes each change through as one conditional commit. A commit that fails or conflicts
  changes nothing in memory: the server reloads and answers "try again". There is no SQL
  and no migration step.
- **Files** (entries, edit scripts, the exam pack) live in a bucket, addressed by hash.
- **Identity** is an Ed25519 key made by `stet login`. Every request that matters is
  signed. There are no passwords.

Limits worth knowing: every record is loaded at start (tens of megabytes after a year at
four rounds a day), and readers wait while a change is being committed. Both are fine at
this size and are the first things to revisit if Stet outgrows one process.

## The gym

`stet.gym.Env` is the same harness as an environment: the observation is a corpus, the
action is an edit script, the reward is the gain. Its rounds come from a different dataset
(DCLM-Pool) than official rounds, so a policy trained in the gym and entered in the
competition is tested on data of a kind it has not seen.

## What Stet is not

It does not build the best dataset, change the student, or reward writing new training
text. It exists to find out how good an editor can get.

## Prior work

The nearest competition is DataComp-LM: it also fixes the training recipe and varies only
the data. But an entry there is a finished dataset, trained by the entrant at tens to
thousands of GPU-hours a run, from one static pool, with a mandatory shuffle. There are
also recent papers in which an agent searches for data-selection programs and scores each
with a short proxy training run. Stet differs from all of them in the same ways: the entry
is a time-boxed *program*, it is judged on fresh data it has never seen, the whole edit
space is open (cut, reorder, repeat, repair), a score costs minutes on a consumer GPU, and
scores are checked by volunteers rather than taken on trust.

Two findings from that literature shaped the design. On held-out text from the same pool,
careful selection is almost indistinguishable from random, which is why the exam is curated
text from elsewhere. And run-to-run noise at this scale is large enough to mislead, which
is why every comparison here is paired on shared rounds and the record needs a margin.

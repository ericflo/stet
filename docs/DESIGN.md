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

So the competition is shaped to measure one thing, the curator, and to make every other
explanation for a good score unavailable.

## One variable

| Fixed for a season | Free |
|---|---|
| Student architecture and initial weights | Which documents and spans are kept |
| Optimiser, schedule, batch size | The order they are trained in |
| Training budget, counted in tokens | How often anything is repeated |
| Tokenizer | Small repairs (the ink budget) |
| The exam's sources and scoring | Everything about how the curator decides |

Training is budgeted in tokens, never seconds, so a faster GPU buys nothing there. The trainer
does not shuffle: the stream the curator writes is the stream the student sees, which makes
curriculum a real move.

## Decisions, and what they are for

**The exam is not a slice of the corpus.** On held-out text from the same pool, careful
selection barely beats random, because perplexity rewards matching the test distribution.
So the exam is what we want a model to have learned: six kinds of hand-picked, openly licensed
English writing, weighted equally. A specialist cannot win by acing one.

**The exam is public.** A secret exam needs a trusted party to hold it and makes every
score unreproducible. Stet publishes it and defends it three other ways: the curator runs
in a sandbox without it; every corpus document containing a sentence of it is removed
before the curator sees the corpus; and the script's output is checked for it again.

**And a second exam is held out.** None of that stops a curator from being tuned to the
public exam's passages: choosing pages about the very people and places it happens to
mention. The usual cure is a secret test set, and its usual cost is that someone must be
trusted with it for ever. Stet's is trusted with it for one season. The held-out exam is
built like the public one from other documents, and its hash is published at the start.
At the close the held-out exam is published, every script that counted is graded on it
by volunteers, and the final table is read from that.

**The data is drawn after the entries are locked.** The seed comes from drand, a public
randomness beacon, at a time fixed on a public grid, and the corpus comes from the newest
Common Crawl. The crawl itself is public, and billions of pages long. Which few thousand
of its files a round gets is unknown until after entries close, so a curator cannot be
tuned to its sample, and the organisers cannot choose a draw that suits anyone.

**Almost no ink.** If new text were allowed, the winning move would be to ignore the
corpus and ship a dataset, or to have a large model write one. Ink is capped at 0.1% of
the training tokens: enough for separators and repairs. Everything else the student reads
is copied byte for byte from the corpus. This is also what rules out distilling a frontier
model into the training data.

**Editing and grading are separate.** Only the edit runs an entrant's code. It produces an
edit script, which is plain data: document numbers and byte ranges. Grading turns a script
into a score with the harness alone. That split is what lets anyone re-grade any score
safely: a stranger's GPU never runs a stranger's program.

**A curator may be a small language model.** The interesting curators read the text, and
a program that reads text well is a model. So a curator names its weights (up to 4 GiB,
public files pinned by hash) and they are in place when it starts, and it gets a GPU
through PyTorch and a fixed list of libraries at exact versions. What it does not get is
a way to turn a big model into training text: the ink limit sees to that. A model here
can only choose and arrange what the crawl already says.

**Every curator gets the same machine.** Volunteers' machines differ, and an edit limit in
seconds means something only if the machine is known. So the sandbox gives every curator
the same cores, memory and GPU memory whatever the host has, and only machines at or
above a measured floor run curators. What is left over is speed above the floor, which an
curator that works until the clock runs out can use. That is disclosed, bounded by taking
the score over runs on different machines, and cheaper than the alternative: counting an
curator's work instead of its seconds would mean running entrants' models for them, and
then the curator is no longer any program.

**Volunteers run the curators, not the organisers.** The first version ran every curator on
the organisers' machine and took that machine's grade as the score. That made the whole
competition as available, and as trustworthy, as one computer. Now every entry is run
twice, by two different people, and each script is graded by someone who did not write
it. The organisers' accounts are verifiers like the rest, with one difference: they
start out established, because someone has to be first.

**Trust is earned slowly and never spent alone.** Accounts are free, so an account's
word counts only once it has agreed with other people's a number of times and is tied to
a GitHub identity with some age. Even then no score rests on one volunteer: a runner
never grades a script that only it wrote, runs are compared with each other, grades with
each other, and the server deals the work at random. An earlier design let one trusted volunteer's
grade set a score, and review took it apart: trust is cheap to earn honestly and then
spend once, on the entry that matters. What survives that argument is needing two people
who were not able to choose each other.

**A round does not wait for people who are not there.** With few volunteers the honest
options are to stall or to settle with less and say so. Stet settles: an entry that only
one person was around to run is scored on that run and marked *run once*; one that two
people ran and agreed on is marked *confirmed*. The throughput of a round is whatever
the people lending GPUs can run and grade between draws
([Experiments](EXPERIMENTS.md#how-long-a-round-takes)).

**Scores are gains, on shared rounds.** Rounds differ in difficulty by far more than
curators differ from each other. Every score is a difference from the unedited corpus on
the same round with the same initial weights, ratings are adjusted for which rounds an
curator happened to get, and the top curators run in every round so that any two of them can
be compared directly.

**The record is hard to take by luck.** A challenger is tested against the holder at 5, 10,
20 and 40 shared rounds, and must win by more than a margin set from measured noise.
Copying the leader and hoping for a good draw does not work, and neither does entering
many near-copies.

**The headline is a multiplier.** "+0.31 nats" means little to anyone. "This edit made the
data worth so many times as much" does. It is read off a measured curve of the baseline's
loss against training tokens, and never extrapolated past the curve
([Experiments](EXPERIMENTS.md#how-much-data-is-an-edit-worth)). Curators are ranked by
gain; the multiplier is the translation.

**Small on purpose.** The official student is 6 layers by 384, trained on 40M tokens: about
ten minutes on an integrated GPU. Grading has to be cheap while few people are lending
GPUs.
Whether what wins here also wins at a larger scale is an open question, and the plan is
to measure it: at the end of a season, rerun the ladder's curators one scale up and
publish both rankings. That has not been run yet.

## What could go wrong

| Attack | What stops it |
|---|---|
| Tune a curator to a known corpus | The crawl is public, but which files are drawn is unknown until after commit |
| Ship a dataset instead of editing | The ink budget. A model's weights can be large; the text it may add cannot |
| Have a large model write the data | The same: 0.1% ink, and no network while the curator runs |
| Smuggle exam text in | Corpus and script output are both cleaned of it; curators and their model files are public |
| Tune a curator to the exam's own passages | A held-out exam, drawn in secret from the same collections and committed to by hash, decides the season's final table |
| Curator reads the exam or phones home | Sandbox: no network, no exam on disk, script-only output |
| A hostile curator attacks the machine running it | Sandbox: no files, no network, no view of the host. Running curators is opt-in, and the guide says what a sandbox cannot promise |
| A curator swaps its weights after it is checked | Assets are pinned to a commit and named by sha256; every machine checks the hash |
| A curator takes more than its share of the machine | Cores, memory and scratch are enforced by the kernel. The GPU memory limit is not; lifting it is against the rules and curators are public |
| A hostile script attacks the grader | Scripts are parsed as data with hard limits; nothing in one is executed |
| A runner hands back a script the curator did not write | A second person runs the same curator. Runs that differ by more than a tolerance get a third, and the middle one counts |
| A runner makes up its script's score | A runner never grades a script that only it wrote |
| A runner copies the other runner's script | Scripts are sealed until the round settles, and nobody who has been shown one is asked to run that entry |
| A verifier invents a grade | Compared with the other grades of the same script. Outside the tolerance it is a strike, and it is published |
| A new account invents grades | Its grades set no score until it is established: five that agreed, and a GitHub identity |
| Two verifiers agree on a lie | Not caught, if both are established and both are assigned to the entry. Nobody chooses their work, and everything is published and can be re-run |
| A verifier takes jobs and sits on them | A round waits only for accounts with an agreed grade, and an abandoned job costs a day without work |
| A verifier runs or grades their own entry | Never offered it, by account or by GitHub identity. The organisers' machines are offered the organisers' reference curators only when nobody else who is around could take the job |
| Someone uploads under another person's key | An upload's signature is checked from its headers before any of it is read or counted |
| A verifier copies a grade | Nothing is visible until the round settles |
| Resubmit the leader for a lucky draw | Four fixed looks, a margin, paired on shared rounds |
| Organiser picks a good seed | drand, a ten-minute grid, the crawl named in advance, checked by every client |
| Flood the queue | GitHub-linked accounts, one entry in play, six submissions a day |

What is left is in [Rules](RULES.md#what-is-trusted-and-what-is-not).

## How it is built

- **The harness** (Python, `src/stet`) is what decides a score: rounds from seeds, the
  corpus, the script format, the student, the trainer, the exam. It is the same code in
  `stet run`, `stet verify` and the gym.
- **The server** (Go, `server/`) keeps the books: accounts, entries, rounds, jobs, grades,
  ratings. It trains nothing. One stateless process serves the website, the API and the
  downloads.
- **Records** live in [durable-db](https://github.com/ericflo/durable-db), a versioned
  key-value database over object storage. The server holds every record in memory and
  writes each change through as one conditional commit. A commit that fails or conflicts
  changes nothing in memory: the server reloads and answers "try again", and if it
  cannot reload it stops showing numbers until it can. There is no SQL and no migration
  step.
- **Files** (entries, edit scripts, the exam pack) live in a bucket, addressed by hash.
  Model weights do not: an entry names them where they already are.
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
text. It exists to find out how good a curator can get.

## Prior work

The nearest competition is DataComp-LM: it also fixes the training recipe and varies only
the data. But an entry there is a finished dataset, trained by the entrant at tens to
thousands of GPU-hours a run, from one static pool, with a mandatory shuffle. There are
also recent papers in which an agent searches for data-selection programs and scores each
with a short proxy training run. Stet differs from all of them in the same ways: the entry
is a time-boxed *program*, it is judged on fresh data it has never seen, the whole edit
space is open (cut, reorder, repeat, repair), a score costs minutes on a consumer GPU, and
every score is made by more than one volunteer's machine and can be re-run by anyone.

Two findings from that literature shaped the design. On held-out text from the same pool,
careful selection is almost indistinguishable from random, which is why the exam is hand-picked
text from elsewhere. And run-to-run noise at this scale is large enough to mislead, which
is why every comparison here is paired on shared rounds and the record needs a margin.

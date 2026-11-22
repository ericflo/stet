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
| Tokenizer | Separators (the ink budget) |
| How the exam is made and scored | Everything about how the curator decides |

Training is budgeted in tokens, never seconds, so a faster GPU buys nothing there. The trainer
does not shuffle: the stream the curator writes is the stream the student sees, which makes
curriculum a real move.

## Decisions, and what they are for

**The exam measures what pretrained models are for.** Text has no quality of its own: a page
is worth keeping for what it teaches a model. Three quarters of the score is how well the
student predicts writing people have just published, in four kinds that count the same:
reference, news, technical and conversation. One quarter is skills, in five kinds that
count the same: agents, reading, transformation, logic and grammar. How much each counts is
a choice of value, so it is declared rather than left to how many items of each there
happen to be. [The reasoning](research/exam-first-principles.md#9-season-2-november-2026).

**Only what tells curators apart counts.** A student this size cannot do arithmetic, unit
conversion or program tracing cold, whatever it was trained on: on the pilot such questions
ordered curators no better than a coin, at every size of student. Weight on such a part is
weight on noise, and every extra training is then spent averaging it away. So those parts are
*watched*: sat and published in every round, and no part of the score. A watched part
earns a place when, over at least twenty settled rounds, it tells curators apart. A kind
of writing new to the exam (science abstracts) starts there too. On the pilot, the score
made this way moved a quarter as much between two trainings of one script as a score that
counted those parts, and agreed better with the large student
([measured](EXPERIMENTS.md#how-much-is-noise): the pilot's fresh writing was Wikipedia only).

**The exam is not a slice of the corpus.** On unseen text from the same pool, careful
selection barely beats random, because perplexity rewards matching the test distribution.
So the exam comes from elsewhere: writing published in the day before the draw, and
questions made for the round.

**Nobody can know the exam in advance.** A public exam can be learned, and a secret one needs
someone trusted to hold it. So each round's exam is made at its draw, from things nobody
could aim at before it: human writing from three hours of the day before, which the round's
seed picks, and questions generated from the same seed. Nobody knows it before the draw,
the organisers included, and there is nothing to keep secret for longer than a round. It is
published when the round settles, so every score can be checked. Fresh text that the
round's corpus already holds is dropped from the exam, since it is the exam that came
second.

**The exam's writing is from the day before the draw, so no round waits for its exam.**
Writing published after the draw is writing no curator's author has seen, but a round would
wait for it: hours, with the machines that have trained standing idle. So a round's writing
comes from the 26 hours before its draw, and an entry may be newer than some of it. That is
let be. A curator runs with no network; the only text it can bring is its ink, 16,000 tokens
at the standard size; and the grader refuses a script whose ink holds a passage of the exam.
What is left to an author who reads the day's writing is to guess which three hours the draw
will pick and which few hundred passages of them, and then to find pages like them in a
crawl gathered before any of it was written. The examiner harvests every hour of the clock an hour after it ends and
tells the server the harvest's hash at once. So nobody knows which hours a round will be
given until its draw; nobody writing in an hour can tell what the harvest will keep of it,
because that is chosen with a public random number emitted as the hour ends; and the
organisers cannot make a harvest again to suit a round, because its hash was public first.

**Many sources, each capped.** Every source has its own bias: who writes there, about what,
in which languages. The remedy is not a perfect source but many, chosen by kind of writing
first (reference, news, technical, conversation), each kind counting the same, with the
sources inside a kind taking turns. None needs a key or an account, so anyone can read
them, and none can define the target.

**Every language, by its share of the crawl.** Inside every kind of writing, each language
counts by its share of the round's corpus, in bytes. That is what a curator saves on a byte
of the web. More weight to English would reward throwing away most of the web; equal weights
would reward the smallest languages, where a student this size learns little and the score
is mostly noise. The skills are generated in English only, so English is about 59% of the
score, though less than half of the crawl. The tokenizer is trained on the crawl alone, so
other languages do not cost the student more than they must.

**Scored by likelihood.** At this size a student's choice between answers is chance, but the
likelihood of the right answer, in bits per byte, still moves, and it is the best-validated
small-scale predictor of what larger models can do. So every item is a passage, an answer
given its question, or the right one of two sentences, scored by how likely the student finds
it. An item's bits are never put over fewer than 16 bytes: the bits per byte of a one-word
answer swing by whole bits between two trainings, and would otherwise weigh as much as a
paragraph's. A text longer than the student's window is scored whole, a window at a time.

**The data is drawn after the entries are locked.** The seed comes from drand, a public
randomness beacon, at a time fixed on a public grid, and the corpus comes from the newest
Common Crawl. The crawl itself is public, and billions of pages long. Which few thousand
of its files a round gets is unknown until after entries close, so a curator cannot be
tuned to its sample, and the organisers cannot choose a draw that suits anyone.

**Almost no ink.** If new text were allowed, the winning move would be to ignore the
corpus and ship a dataset, or to have a large model write one. Ink is capped at 0.01% of
the training tokens: enough for separators and nothing else. Everything else the student
reads is copied byte for byte from the corpus. This is also what rules out distilling a
frontier model into the training data. Cleaning a page needs no ink: whole lines of one
page, kept in the page's order, are copies however short they are. And since ink is the
only way a curator's own words reach the student, a script whose own text holds a passage
of its round's exam is refused.

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
above a measured floor run curators. What is left over is speed above the floor, which a
curator that works until the clock runs out can use. That is disclosed, bounded by taking
the score over runs on different machines, and cheaper than the alternative: counting a
curator's work instead of its seconds would mean running entrants' models for them, and
then the curator is no longer any program.

**Volunteers run the curators, not the organisers.** Every entry is run twice, by two
different people, and each script is graded by someone who did not write it. The
organisers' accounts are verifiers like the rest, with one difference: they start out
established, because someone has to be first.

**Trust is earned slowly and never spent alone.** Accounts are free, so an account's
word counts only once it has agreed with other people's a number of times and is tied to
a GitHub identity with some age. Even then no score rests on one volunteer: a runner
never grades a script that only it wrote, runs are compared with each other, grades with
each other, and the server deals the work at random. Trust that one volunteer's grade could
set a score would be cheap to earn honestly and then spend once, on the entry that matters.
What holds up is needing two people who were not able to choose each other.

**A round does not wait for people who are not there.** With few volunteers the honest
options are to stall or to settle with less and say so. Stet settles: an entry that only
one person was around to run is scored on that run and marked *run once*; one that two
people ran and agreed on is marked *confirmed*.

**Scores are gains, on shared rounds.** Rounds differ in difficulty by far more than
curators differ from each other. Every score is a difference from the unedited corpus on
the same round with the same initial weights, ratings are adjusted for which rounds a
curator happened to get, and the top curators run in every round so that any two of them can
be compared directly. The unedited corpus trains the least steadily of all, so the round's
baseline is the mean of four established graders' measurements, item by item. Four and not
every grader's: each costs a whole training, and the mean of all of them is no better a
baseline than the mean of a few. The organisers' reference curators are yardsticks, and a
yardstick need not run every round: `fineweb2` and `dclm` do, and tie the rounds together,
and each of the others runs in one round in eight.

**A rating says how sure it is.** A few rounds can be luck. So a rating is drawn towards
the field's mean while a curator's rounds are few, the ladder is filled in order of rating
less 1.28 standard errors, and neighbours too close to tell apart share a place on the
board. Three lucky rounds do not take a place from many steady ones. And failing is never
better than finishing: a failed entry counts as the lowest gain any valid entry got in its
round, or zero if that is lower.

**The record is hard to take by luck.** A challenger is tested against the holder at 5, 10,
20 and 40 shared rounds, and must win by more than a margin set from measured noise.
Copying the leader and hoping for a good draw does not work, and neither does entering
many near-copies.

**No multiplier yet.** "Worth twice the data" is easier to read than a gain in bits per
byte, but it has to be read off a measured curve of the unedited corpus's score against
training tokens, and that curve has not been measured on this exam. Until it is, the board
shows gains.

**Rounds at the standard student; finals at a larger one.** Rounds train an 8 × 512 student
on 160M tokens: about 18 minutes on an RTX 3090. Smaller students rank curators
differently: they reward narrow English data, which loses as the student grows
([measured](research/exam-first-principles.md#75-scale-rounds-at-the-standard-student-finals-at-the-large-one-decide-the-season)).
The standard student ranks them as the large one does. Finals train the season's top eight
at 10 × 640 on 640M tokens and decide the table. The top three are also tried one size up,
which says whether the next season should move up too.

## What could go wrong

| Attack | What stops it |
|---|---|
| Tune a curator to a known corpus | The crawl is public, but which files are drawn is unknown until after commit |
| Tune a curator to the exam | Its writing is three hours of the day before, picked by the draw, and its questions come from the draw's seed; a curator can carry in next to no text, and what it carries is checked against the exam |
| Write a passage into a curator, then publish it where a harvest may take it | What a harvest keeps of an hour is chosen with a random number emitted as the hour ends, and a script whose own text holds a passage of its round's exam is refused |
| Ship a dataset instead of editing | The ink budget. A model's weights can be large; the text it may add cannot |
| Have a large model write the data | The same: 0.01% ink, and no network while the curator runs |
| Curator reads the exam or phones home | The sandbox has no network, and until the round settles the exam goes only to machines grading it |
| A source, or the organisers, bias the exam | Many sources, each capped, none with a key. The examiner is published code, every harvest's hash is public before the draw that picks from it, and the exam is published when the round settles |
| Fail on purpose rather than score badly | A failed run counts as the entry's worst finished run when that is below zero, and a failed round as the lowest valid gain in it, or zero if that is lower |
| A hostile curator attacks the machine running it | Sandbox: no files, no network, no view of the host. Running curators is opt-in, and the guide says what a sandbox cannot promise |
| A curator swaps its weights after it is checked | Assets are pinned to a commit and named by sha256; every machine checks the hash |
| A curator takes more than its share of the machine | Cores, memory and scratch are enforced by the kernel. The GPU memory limit is not; lifting it is against the rules and curators are public |
| A hostile script attacks the grader | Scripts are parsed as data with hard limits; nothing in one is executed |
| A runner hands back a script the curator did not write | A second person runs the same curator. Runs that differ by more than a tolerance get a third, and the middle one counts |
| A runner makes up its script's score | A runner never grades a script that only it wrote |
| A runner copies the other runner's script | Scripts are sealed until the round settles, and nobody who has been shown one is asked to run that entry |
| A verifier invents a grade | Compared with the other grades of the same script. Outside the tolerance it is a strike, and it is published |
| A grader passes the exam on | Until the round settles its items go only to machines grading it, and a curator has no network to be passed anything |
| A new account invents grades | Its grades set no score until it is established: five that agreed, and a GitHub identity |
| Two verifiers agree on a lie | Not caught, if both are established and both are assigned to the entry. Nobody chooses their work, and everything is published and can be re-run |
| A verifier takes jobs and sits on them | A round waits only for accounts with an agreed grade, and an abandoned job costs a day without work |
| A verifier runs or grades their own entry | Never offered it, by account or by GitHub identity. The organisers' machines are offered the organisers' reference curators only when nobody else who is around could take the job |
| Someone uploads under another person's key | An upload's signature is checked from its headers before any of it is read or counted |
| A verifier copies a grade | Nothing is visible until the round settles |
| Resubmit the leader for a lucky draw | Four fixed looks, a margin, paired on shared rounds |
| Organiser picks a good seed | drand, a ten-minute grid, the crawl named in advance, checked by every client |
| Flood the queue | GitHub-linked accounts, three entries in play, six submissions a day |

What is left is in [Rules](RULES.md#what-is-trusted-and-what-is-not).

## A round's exam

1. **The harvests.** An organisers' machine (`stet examine --follow`) harvests every hour of
   the clock, an hour after it ends: the hour is for the sources that publish late (Common
   Crawl's news files, edits on Hacker News). What it keeps of an hour is chosen with the
   public random number emitted as the hour ends. It tells the server the harvest's
   sha256 at once, and the server lists them (`/api/v1/harvests`).
2. **The draw.** The round's seed picks three of the hours that began in the 26 before the
   draw and whose harvests the server had been told of, and the round names them
   (`exam_hours`). Curators run as soon as the
   round is drawn. A round drawn while the server knows of no harvest is given hours once
   there are some, and says so (`exam_hours_late`).
3. **The examiner** makes the exam from exactly those hours, at once. It:
   - builds the round's corpus, measures its languages, and drops the fresh items the
     corpus already holds;
   - generates the round's questions from its seed;
   - posts the exam: the items, what each counts toward (pillar, part, language, format and
     kind of task), the items' digest, the languages' shares and the hours it was made
     from. The server refuses an exam made from any other hours.
4. **Grading.** A grader is handed its script as soon as a curator has written it, and
   trains the student on it. Four established graders a round, and every grader not yet
   established, are asked to train the unedited corpus as well. Once the exam is posted a
   grader fetches the items, checks them against the round's digest, has its students sit
   them, and reports one score per item.
5. **The server adds it up.** The round's baseline is the item-by-item mean of the
   established graders' baselines. A grade's score is the exam's aggregate (writing three
   quarters, skills one quarter) of its item scores less that baseline. The server computes
   it with a Go port of `src/stet/exam/aggregate.py`, checked against the Python on shared
   test vectors (`tools/aggregate_vectors.py`). From there, agreement, settlement, the board
   and the record work on that score.
6. **Settlement.** Every item score is published (`/api/v1/rounds/<id>/scores`), and so is
   the exam: the generated items and Wikipedia's text whole, and every other source's items
   as their source, id and hash. Other sources' text is not ours to republish; anyone can
   check it at the source. `stet run --round <id>` replays the round on what was published.

The exam is made at the draw, so a round takes as long as its runs and its grades do.
Rounds never overlap. A practice round's exam is made the same way, from hours harvested on
the machine that runs it (`stet harvest`); with none harvested it is the generated items
alone.

## How it is built

- **The harness** (Python, `src/stet`) is what decides a score: rounds from seeds, the
  corpus, the script format, the student, the trainer, the exam. It is the same code in
  `stet run`, `stet verify`, the examiner and the gym. The exam's generators, harvest and
  aggregate are in `src/stet/exam`.
- **The server** (Go, `server/`) keeps the books: accounts, entries, rounds, jobs, grades,
  ratings, and each round's exam. It trains nothing. One stateless process serves the
  website, the API and the downloads.
- **Records** live in [durable-db](https://github.com/ericflo/durable-db), a versioned
  key-value database over object storage. The server holds every record in memory and
  writes each change through as one conditional commit. A commit that fails or conflicts
  changes nothing in memory: the server reloads and answers "try again", and if it
  cannot reload it stops showing numbers until it can. There is no SQL and no migration
  step. An open round's exam and item scores are in it only while the round is open; at
  settlement they move to the file store.
- **Files** (entries, edit scripts, each round's exam and item scores) live in a bucket,
  addressed by hash or by round. Model weights do not: an entry names them where they
  already are.
- **Identity** is an Ed25519 key made by `stet login`. Every request that matters is
  signed. There are no passwords.

Limits worth knowing: every record is loaded at start (tens of megabytes after a year at
four rounds a day), and readers wait while a change is being committed. Both are fine at
this size and are the first things to revisit if Stet outgrows one process.

## The gym

`stet.gym.Env` is the same harness as an environment: the observation is a corpus, the
action is an edit script, the reward is the gain. Its rounds come from a different dataset
(DCLM-Pool) than official rounds, so a policy trained in the gym and entered in the
competition is tested on data of a kind it has not seen. Its exams are generated items.

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
space is open (cut, reorder, repeat, repair), a score costs well under an hour on a
consumer GPU, and every score is made by more than one volunteer's machine and can be
re-run by anyone.

Three findings from that literature shaped the design. On unseen text from the same pool,
careful selection is almost indistinguishable from random, which is why the exam comes from
elsewhere. The best data depends on the compute it is for, which is why the size of the
student was chosen by measurement and finals train a larger one. And run-to-run noise at
this scale is large enough to mislead, which is why every comparison here is paired on
shared rounds and the record needs a margin. [The research](research/prior-art.md).

# A new exam: proposal

Status: proposal, revised 2026-11-18 with the organisers' decisions. Nothing here is in force
yet. It runs as a shadow score beside season 0 and becomes the exam in season 1.

## Decisions taken

| | Decision |
|---|---|
| Weighing | Balanced across formats and kinds of task; inside each, every corner the same |
| Languages | Every language, in proportion to the crawl |
| Sources | Fresh text and generated items only. No public benchmark or dataset, ever: too easy to cheat with, even by accident |
| Organisers' curators | Compete, marked, and barred from any prize. The organisers dogfood and bootstrap the competition |
| Scale | Micro for rounds, plus a check of the ladder at the standard scale each season |
| When | A shadow score beside season 0 now; the exam in season 1 |

## The problem

The exam is six kinds of openly licensed English writing: encyclopedia, textbooks, science
abstracts, Q&A, books and news. The leaders of season 0 win by matching those six kinds.
`compass`, its variants and `compass-clf` say so in their own descriptions: they pick the
pages that read most like "the exam's six kinds", judged by statistics of public Common Pile
files from the same collections the exam is cut from. This is within the rules. It is also
Goodhart's law in action: the score has stopped measuring better pretraining data and
started measuring resemblance to six collections.

The held-out exam does not catch it. It is drawn from the same six collections.

What Stet is for is a curator whose data makes a model better in general: at code, math,
conversation, agents' work, stories, knowledge, every kind of writing and every language.
The exam has to measure that, and nobody, the organisers included, may be able to see it
before it is used.

## What the exam has to do

1. **Extreme diversity.** Tens of thousands of distinct corners of human text and skill,
   in every language, every item in a round from a different one.
2. **Stability.** Separate curators that are a few hundredths of a nat apart, on a 6-layer
   student trained on 40M tokens, without seeds or reruns.
3. **Speed.** Seconds per grade on a consumer GPU, next to minutes of training.
4. **Unknowable in advance.** To entrants and organisers alike.
5. **Checkable.** Anyone can rebuild and re-grade any settled round.

## The design in one paragraph

A round's exam is made at its draw, from two things nobody can know before it: **text first
published after the draw**, harvested from a published list of openly licensed live sources
in every language, and **generated items**, made by public generators from the round's drand
seed. Every candidate is placed on a season's map of tens of thousands of fine-grained
**corners** (a topic, in a format, for a kind of task, in a language), and the round keeps
about 1,500 items, each from a different corner: balanced across formats and kinds of task,
and shared among languages as the round's own corpus is. Items are scored only by
continuous log-likelihood (bits per byte of a passage, or of an answer given its question),
never by accuracy. Every curator in a round sits the same items against the same unedited
baseline, so only paired differences count. Each corner's gain is put into common units,
clipped, and averaged by a trimmed mean.

## 1. An exam nobody can know: fresh and generated

Nothing in the exam exists, as an exam item, before the draw. So nothing is secret, and
nothing has to be trusted to anyone.

**Fresh text, published after the draw.** In the hours after a round's draw, a harvester
(published code) collects new text from a fixed, published list of openly licensed sources
that publish all the time, in every language:

- new Wikipedia articles and revisions, in every language edition;
- new questions on StackExchange's ~170 sites, with their answers;
- new commits to permissively licensed code repositories: a message and its change, in
  every programming language;
- new arXiv papers in every category;
- court opinions, government publications (the Federal Register, the Congressional Record,
  European Union documents in its 24 languages), patents;
- open-access biomedical articles;
- public-domain news, such as Voice of America's ~40 language services;
- new public-domain books and transcriptions.

Every item is pinned to an immutable id (a revision id, a commit hash, a version), so anyone
can check it against its source. The harvest is snapshotted, hashed and released when its
window closes; grading begins after. Fresh text gives answer items as well as passages,
without any benchmark: a question and its accepted answer, a commit message and its change,
a title and its abstract, an infobox and the sentence it supports.

**Generated items, seeded at the draw.** Generators whose code is public, run from the
round's seed, so their instances cannot be looked up or computed in advance:

- arithmetic, word problems, algebra, units and dates, each with a checkable answer;
- programs and what they print ("predict the output");
- algorithmic tasks: sorting, counting, tracing, parsing;
- logic puzzles with a unique answer;
- facts from Wikidata as cloze sentences, in every language its labels have;
- simulated agent episodes: a scripted shell, file system and web environment produces
  realistic tool calls and their results, and the student predicts the results and the
  next action;
- grammatical minimal pairs from templates, in many languages.

This is your "prefill the math question, read the answer's logits": the question is the
context, and the score is the bits per byte of the correct answer.

**Decontamination turns around.** A round's corpus comes from a crawl made before the draw,
so fresh text cannot be in it. Any exam item whose sentences do appear in the round's corpus
is dropped from the exam, instead of the page being dropped from the corpus.

**Why this beats a hidden pool.** A hidden pool has to be held by someone, who could leak it
and whose own curators then look suspect. Here there is nothing to hold. The organisers know
the source list and the generators' code, as everyone does, and nothing else, which is what
lets them keep entering curators to dogfood and bootstrap the competition, marked and barred
from prizes.

What it does not stop, by design: a curator learning from past rounds' published items what
kinds of text matter, or keeping math pages because math matters. That is choosing good
data. What it stops is fitting the particular sources, passages and phrasing it is judged on.

## 2. Extreme diversity: select for it, at the finest grain

The unit of diversity is a **corner**: one topic, in one format, for one kind of task, in one
language. "Rust async error handling", "paediatric dosage questions", "chess opening
commentary", "Kubernetes YAML", "Swahili football reporting", "sea shanties", "spreadsheet
formula help", "a tool call and its result in an agent's log". A season's map has tens of
thousands of them.

1. **The map.** At the start of a season, a large sample of fresh and generated candidates is
   embedded with a small multilingual text-embedding model and given facets: format (prose,
   code, dialogue, table, list, math notation, logs, markup, transcript, verse, structured
   data), kind of task (passage, answer, pair), language, register, length, source. Within
   each format, kind and language, the embeddings are clustered finely into corners, and
   corners far from everything already mapped are added on purpose (farthest-point
   sampling), so the map has no empty regions. The map is published.
2. **Every item from a different corner.** At each round, every candidate is placed in its
   corner, and the round keeps about 1,500 items from about 1,500 different corners, chosen
   by the seed.
3. **Weighed as decided.** Formats and kinds of task each get a fair share of the round, and
   inside each, every corner counts the same. Languages get the share they have in the
   round's own corpus.
4. **Groups are a view, not a unit.** "Math", "code", "agents" and so on are only ways of
   reading a result: a curator's page shows where it is strong and where it is weak.
5. **Diversity is measured, not claimed.** Each round reports how many corners, formats,
   languages and sources it touched.

With one item per corner, a curator cannot aim at anything smaller than "good, broad data".
That is also the right goal for a curator: diversity in training data is part of what makes
it good.

## 3. Stability: pairing, continuous scores, and a robust average

**Continuous scores only.** At this scale, choosing the right answer is chance. We trained the
official student on `compass`'s round-12 script and put it through 45 candidate measurements
built from public benchmarks (as instruments only; none of them will be in the exam).
Accuracy on seven multiple-choice sets was at or barely above chance. But the log-probability
of the correct answer moved a long way from an untrained model's (bits per byte, higher is
better):

| | untrained | trained on `compass`'s data |
|---|---|---|
| a math word problem's worked solution, given the question | −5.27 | −2.25 |
| its final answer, given the solution | −5.73 | −3.94 |
| a short factual answer, given the question | −4.43 | −2.52 |
| a function's body, given its signature and docstring | −4.99 | −2.89 |
| the last word of a passage | −4.64 | −1.61 |

So an answer item scores the answer's bits per byte. That matches AI2's *Signal and Noise*
(2025) and OLMo 3's small-scale suite, which use bits per byte on code, math and QA for this
reason. Whether a family of items separates one curator's data from another's by more than
its noise is measured family by family in the calibration study.

**Pairing.** Every curator in a round sits the same items, from the same initial weights, and
is compared with the same unedited baseline. Item difficulty, which dominates the variance of
an absolute score, cancels in a difference.

**Measured noise decides what is kept and how much of it.** The calibration study measures the
paired noise of each family (a format, a kind of task, a language). A family whose signal does
not clear its noise is dropped, and the round's items are shared across families by Neyman
allocation: more items where the paired difference is noisier. With one item per corner, a
round's score is the mean of 1,500 paired differences, so its sampling error is the per-item
spread divided by about 39.

**Common units.** Each item's gain is divided by how much the unedited baseline improves on its
family when given twice the data. The unit is *doublings of raw data*: the "worth N× the data"
headline, measured per family. The slopes are measured once a season.

**A robust average.** Per-corner gains are clipped to a few noise widths and combined by a
trimmed mean. A curator that gains hugely on a few corners and loses on the rest gets little;
one that gains a little everywhere wins.

**Noise in the draw.** Different rounds draw different corners and different news, which adds
variance to a single round. That variance is the curator-by-corner interaction, the
specialisation the exam should see, and it averages out over the rounds a rating and the
record test already use. Scoring the student at its last two or three checkpoints and
averaging is a further damper that costs almost nothing.

## 4. Speed

Scoring is one forward pass of a ~10M-parameter model. A 4090 trains the student at about
340,000 tokens a second, forward and backward, so scoring runs at very roughly a million
tokens a second. 1,500 items are under a million tokens: a second or two. The current exam
is about 2.8 million. Generating items takes seconds; a harvest window's snapshot is a few
megabytes. Training remains the cost.

## 5. Languages

Every language, in proportion to the crawl: each round gives each language the share of
corners it has in that round's corpus, so a curator is rewarded for doing well by every
language the web writes in, in the amounts it writes in them. Fresh sources and Wikidata
cover hundreds of languages.

One consequence: season 1 needs a tokenizer trained on the crawl's language mix. The current
8,192-token vocabulary leans English, which makes other languages expensive to model. Bits per
byte keeps the score fair across tokenizations, but the student should be able to learn them.

## 6. Practice

`stet run` needs an exam without waiting for a draw. The generators run locally from any seed,
and every settled round's fresh items are published, so the practice exam is generated items
plus the published fresh items of past rounds, as many as anyone wants.

## 7. Calibration, each season

Before a season starts, and again in the shadow run, train the student on data known to
differ and keep what separates it: the published scripts of past rounds (raw, `english`,
`tidy`, `compass` and its variants); selections shown at large scale to make better models
(the top pages by the FineWeb-Edu classifier and by the DCLM filter); a deliberate aiming
selection toward the published items of past rounds, to measure how much aiming buys; reruns
of one script in shuffled order, for training noise; and the raw corpus at twice the budget,
for the per-family slopes. Public benchmarks may appear here as an outside check, never in
the exam.

## 8. The scale check

Rounds are graded at micro. Once a season, the ladder's curators are rerun at the standard
scale (8 × 512, 160M tokens, about ten times the training cost) on the same exam, and both
rankings are published. If they disagree, that is the first thing to look at.

## 9. Shadow now: what to build

The shadow score is computed for every settled entry of season 0, shown beside the official
score, and ranks nothing.

1. **Scorers.** Passage, answer and pair scoring in the harness, with the corner-balanced
   trimmed mean. `stet run --exam next` uses them.
2. **Generators.** The first families, each with tests that pin what a seed produces.
3. **Harvester.** The first fresh sources, with immutable ids, a window after each draw, and a
   hashed snapshot.
4. **Map.** The embedding model, the facets, the corners, and the per-round draw.
5. **Shadow grading.** At first the organisers' machine retrains each settled script after the
   harvest window closes and scores it (about two minutes a script on a 4090). Later, graders
   score it as they grade.
6. **Site.** The shadow score beside the official one, and a curator's strengths by format,
   kind of task and language.
7. **Calibration.** On the first shadow rounds, which fixes the allocation and the slopes.

## 10. What changes in season 1

- **Rules:** the exam and the held-out exam are replaced by this one; the score is the trimmed
  mean of normalised per-corner gains; the reference curve is per family; the organisers'
  curators are marked and barred from prizes.
- **Harness:** the scorers; the round's exam is built from the snapshot and the seed;
  decontamination drops exam items found in the corpus; a multilingual tokenizer.
- **Server:** stores each round's snapshot, starts grading when the window closes, publishes
  the items when the round settles.
- **Site:** a curator's page shows its gain by format, kind of task and language, and each
  round shows the corners it drew.

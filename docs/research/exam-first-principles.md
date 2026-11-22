# What the exam should measure, from first principles

Stet exists to find the best pretraining data there is. This note asks what an exam must
be to point the competition at that, and what that means for Stet. It was written while the
first season's exam (six kinds of English prose, a fixed public pack) was the score, and the
exam proposed to replace it (here, "the new exam") was being trained and measured beside
it. Section 7 is what was decided; it is season 1's exam. Section 9 is what season 1 taught and
what season 2 does about it.

## The answer, in one paragraph

A pretraining dataset is good when the model trained on it, **at the sizes people actually
train**, becomes capable at **what pretrained models are for**. So the exam should be:

- a public, broad, declared mix of capabilities, scored by how likely the model finds good
  answers and good writing, with the individual items unknowable in advance;
- trusted only where it reproduces what is already known about data at scale. The reference
  curators that rebuild published recipes must come out in the order the literature found;
- decided at the largest scale we can afford. Rounds become a fast proxy, and the proxy is
  checked against larger runs.

Targeting such an exam is not cheating. It is the job.

**What we decided** (section 7):

- The score is seven capabilities, each counting the same: knowledge, reading, reasoning,
  code, agents, writing and conversation. Languages count by their share of the crawl, inside
  every one of them.
- Fresh text comes from many keyless sources, chosen by kind of writing first, so that no
  source defines the target.
- Rounds train the standard student. Micro misranks curators on this target: it rewards the
  narrow English data that loses at scale. Finals at the large size decide the season.

**What the scale pilot found** (section 6):

- Curators keep their order from 14M to 54M parameters, on both exams.
- The two exams order curators in nearly opposite ways at every size. So the choice of
  target matters more than the size of the student.
- The new exam's verdict is decided above all by how much each language and capability
  counts. Under the weights the pilot used, the curator to beat cleans every language rather
  than choosing English.

## 1. What a pretraining dataset is for

Text has no quality of its own. A page is worth keeping only for the difference it makes
to a model, which depends on three things:

- **What the model will be used for.** Today that means knowledge, reasoning, mathematics,
  code, reading and learning from context, writing and explaining, conversation, and acting
  with tools, in the languages people use. Pretraining is where those abilities come from;
  post-training mostly draws them out.
- **How big the model is, and how long it trains.** A small model cannot hold seventy
  languages or many facts, so the best data for it is narrow and easy. A large model can,
  so the best data for it is broad. Aggressive filtering is right at low compute and wrong
  at high compute, because good data repeated loses its value faster than the next-best
  data
  ([Goyal et al., 2024](https://arxiv.org/abs/2404.07177)).
- **How much data there is to choose from.** A dataset can be too small to train on.

So the only direct measure is the one Stet already uses: train a model on the data, then
test the model. Everything below is about making that test point at the right purpose, at
a scale whose answer carries over, precisely, and without a way round it.

## 2. Why compass won, in these terms

Compass and its variants keep about a tenth of the pages, chosen to look like the season's
exam (six kinds of English prose), and train on them twice. That is the right choice for:

- a narrow exam, which rewards resembling it;
- a model of about 14M parameters trained on 3 tokens a parameter, for which narrow, easy,
  English data is the best use of a tiny capacity.

Neither condition holds for a real model. The new exam, which is broad, already reverses
the order. `english` keeps every English page and beats compass on code, programs, other
languages and fresh text, by 0.14 to 0.22 in all four trainings of round 12. The fault was
never that compass targeted the exam. The exam was a narrow, unfaithful picture of the
purpose.

## 3. What the exam must have, in order of importance

1. **The right target.** A faithful, broad picture of what pretrained models are for.
2. **The right scale.** Its verdict must hold for models people train.
3. **Items nobody can know.** No answer can be memorised or copied in.
4. **Precision.** Differences between curators must stand clear of noise.

The first two are about validity, and they matter most. The first proposal for the next
exam handled the third and fourth well, and its trial runs showed what was missing from the
first two.

## 4. The design

### 4.1 Target: a declared mix of capabilities

The exam's weights are a choice of what data is for. Make that choice openly, by
capability, and let sources and generators only supply items to each one. That also
answers the worry about biased sources: no source defines the target.

| Pillar | What it asks | Where items come from |
|---|---|---|
| Knowledge | Facts the web knows, as cloze and short answers | Wikidata, in every language its labels have |
| Reading and learning in context | Questions about a new text, table or rule given in the context; few-shot novel mappings | Generated, so it cannot be memorised |
| Mathematics and logic | Worked problems and their answers | Generated and checked (exists) |
| Code | What programs print; the change a commit makes | Generated (exists); fresh public commits |
| Agents and tools | The next command, the tool's answer, the fix | Generated sessions (exists); permissively licensed real traces |
| Writing | Fresh human writing across genres: news, reference, how-to, discussion | Fresh sources, each capped |
| Conversation and instructions | A reply to a question or request | Fresh discussions; generated instructions |

Languages cut across every pillar, in proportion to the crawl, as decided. Every pillar
counts the same unless that is decided otherwise.

**How it is scored.** At small scale, a model's choice between answers is chance. The
likelihood of the right answer, in bits per byte, still moves, and that is the
best-validated small-scale predictor of large-scale capability. With it, benchmarks
including MMLU, ARC, HellaSwag, MBPP and HumanEval become more than 80% predictable at 1B
([DataDecide, 2025](https://arxiv.org/abs/2504.11393)). So most items are answers given
their question, scored by likelihood. Passages of good writing are the rest.

**Formats vary.** No single "Question: … Answer:" template. The model should be rewarded
for knowing the answer, not for having seen the template. A curator that keeps text which
answers people's questions is doing what the best published filter does
([DCLM, 2024](https://arxiv.org/abs/2406.11794)), and that is wanted.

### 4.2 Trust: the known-answer test

We already know a good deal about which data makes better large models. Here is the order
at 1B to 7B parameters on English benchmarks:

1. DCLM's classifier-filtered web, and FineWeb-Edu's educational filter, the best;
2. then heuristic quality filtering with deduplication (RefinedWeb, C4-style);
3. then raw crawl text.

Each recipe can be rebuilt as a reference curator on Stet's own pool:

- `heuristic`: Gopher and C4-style rules plus deduplication;
- `fineweb-edu`: the published educational classifier;
- `dclm`: the published fastText classifier, keeping the top tenth.

An exam that ranks them in the published order at a given scale has earned some trust at
that scale. An exam that ranks them otherwise is measuring something else, however sensible
it looks. This needs no public benchmark in the score: the reference curators are entries,
and the literature only says what order to expect.

These classifiers are English, so the check reads the English slice of the exam. The weight
of other languages is a separate choice.

### 4.3 Scale: a ladder, and finals

- **Rounds train the standard student** (section 7.5). On the capability target, micro
  misranks curators: a 14M-parameter student does best on narrow English data, which loses as
  the student grows. The standard student (29M) orders curators as the large one (54M) does.
- **Micro stays for practice** and the gym: fast feedback, not a verdict.
- **Finals decide the season.** The top curators are trained at the large size, twice each,
  on a fresh round. The top three also go one rung further up, about 150M parameters, as a
  check on the ladder.
- **Calibration** (section 7.8). For each pillar: does its score at the rounds' size predict
  the finals? A pillar that does not gets more items or a bigger student.

### 4.4 Precision

- **A stable reference.** The unedited corpus is the least stable student we train: its
  season-exam loss varies by 0.046 nats between trainings, against 0.004 to 0.015 for
  curated corpora. Every gain is measured from it. So a round's baseline is the mean of
  every established grader's own, item by item.
- **Pairs.** A pair of curators varies by about 0.03 from training alone, and 0.015 to 0.02
  from the draw of items. Report both. Train the close ones twice.
- **Items nobody can know.** Fresh text from after the draw, items generated from the draw's
  seed, and any fresh item the round's corpus already holds dropped from the exam.

## 5. What has been measured so far

- **The two exams disagree.** On round 12:
  - Season exam, as officially graded: compass v1, compass v2, compass-clf-nudge,
    compass-clf, compass-linear, tidy, english.
  - New exam, mean of four trainings: english first, by 0.13 or more over every other;
    compass v1 last; the rest within noise of each other.
- **The two halves of the new exam agree on the top, not below it.** Generated items and
  fresh text both put english first. Below it they differ: tidy and compass-linear do
  better on fresh text than on generated items.
- **Training noise.** The same script, trained four times on the organisers' rented GPUs
  from the same initial weights, does not give the same student: GPU arithmetic is not
  deterministic, and the differences grow over training. At micro, on round 12:

  | | Season exam loss, sd (nats) | New exam score, sd (bits per byte) |
  |---|---|---|
  | The unedited corpus | 0.046 | 0.081 |
  | Curated scripts (7) | 0.004 to 0.015 | 0.010 to 0.037 |

  The unedited corpus is by far the least stable student, and every gain is measured from
  it: one training of it moves every entry's gain in a round together.

## 6. The scale pilot (round 12, November 2026)

Nine curators were run on round 12's pool at three sizes of student, each trained by the
organisers' rented machines, and each student sat both exams:

- the organisers' curators: `english`, `tidy`, `compass` v1 and v2;
- four reference curators that rebuild published recipes: `dclm`, `fineweb-edu`, `heuristic`
  and the multilingual `fineweb2`;
- the raw crawl.

The new exam is version 5, with 1,897 items: 1,094 generated (knowledge and reading in
context included) and 803 fresh. The raw crawl's student is averaged over four trainings, or
two at the large scale.

| | micro | standard | large |
|---|---|---|---|
| Student | 14M parameters, 40M tokens | 29M, 160M | 54M, 640M |
| The raw crawl's spread between trainings (season exam) | 0.081 | 0.028 | 0.004 |

**The new exam's score**, in bits per byte over the raw crawl:

| | micro | standard | large |
|---|---|---|---|
| fineweb2 (every language, cleaned) | +0.055 | −0.041 | −0.052 |
| english | +0.063 | −0.180 | −0.123 |
| fineweb-edu | −0.015 | −0.185 | −0.201 |
| dclm | −0.022 | −0.193 | −0.254 |
| compass v2 | −0.129 | −0.205 | −0.259 |
| tidy | −0.174 | −0.241 | −0.279 |
| compass v1 | −0.226 | −0.286 | −0.297 |
| heuristic | −0.270 | −0.317 | −0.354 |

**The season exam's gain**, in nats over the raw crawl. Compass leads at every size:

| | micro | standard | large |
|---|---|---|---|
| compass v1 | 1.131 | 0.476 | 0.421 |
| fineweb-edu | 1.104 | 0.436 | 0.415 |
| compass v2 | 1.135 | 0.479 | 0.412 |
| dclm | 1.063 | 0.443 | 0.392 |
| tidy | 0.930 | 0.396 | 0.345 |
| english | 0.916 | 0.376 | 0.337 |
| heuristic | 0.990 | 0.321 | 0.247 |
| fineweb2 | 0.375 | 0.189 | 0.205 |

**The new exam at the large scale, by kind of item** (bits per byte over the raw crawl):

| | english | fineweb-edu | dclm | fineweb2 | compass v2 | heuristic |
|---|---|---|---|---|---|---|
| Fresh writing, English | +0.17 | +0.22 | +0.20 | +0.09 | +0.25 | +0.23 |
| Fresh writing, other languages | −0.59 | −0.74 | −0.75 | +0.06 | −0.67 | −0.90 |
| Maths and logic | −0.09 | +0.03 | +0.12 | +0.07 | +0.03 | +0.10 |
| Code, tools, structure | +0.09 | +0.01 | −0.08 | −0.15 | −0.16 | −0.21 |
| Knowledge, ten languages | −0.27 | −0.32 | −0.29 | −0.03 | −0.24 | −0.46 |
| Reading in context | −0.05 | −0.01 | −0.02 | −0.16 | −0.23 | −0.25 |

What it says:

1. **Curators keep their order as the student grows.** Rank agreement from micro to standard
   to large is 0.98 and 1.00 on the new exam as it was then weighted, and 0.90 and 0.88 on
   the season's. That weighting was mostly fresh text in every language, which is measured
   precisely at every size. On the capability target decided in section 7, micro is no longer
   a sound proxy, and the standard student is.
2. **What curation is worth does not hold.** Every English-only curator beats or matches the
   raw crawl at micro, and falls below it by 54M parameters. The season exam's gains shrink
   from about 1.1 nats to about 0.4. As the literature says, filtering pays most when the
   model is small.
3. **The two exams disagree at every size.** Their rank agreement is −0.45, −0.31 and −0.29.
   Which exam is used decides the competition more than the size of the student does.
4. **Quality filters trade capabilities, as published work found.** They help English prose
   and mathematics. They cost code and tools (the C4 rules delete lines with braces), and
   everything not in English: knowledge in other languages, and fresh text in them. The
   season exam, all English prose, sees only the gains. The new exam sees both sides.
5. **On the new exam, languages decide.** With languages weighed as the crawl has them, the
   only curator near the raw crawl at 54M is the one that keeps every language. Under this
   exam, the winning curator has to clean every language, not choose one. That follows from
   the choice "every language, in proportion to the crawl". Season 0's 8,192-token,
   English-leaning tokenizer makes other languages costlier for the student, and so the
   effect larger.
6. **The known-answer test has to be read per capability.**
   - The published order (DCLM and FineWeb-Edu above heuristic filtering, above raw) is
     about English benchmarks.
   - The season exam reproduces it at every size.
   - On English fresh writing, the new exam has every recipe above raw, as published.
   - On code and on other languages, those recipes really are worse. That is why model
     builders add code, mathematics and other languages to filtered web text separately.

## 7. Decisions (November 2026)

What we want to elicit is a way of choosing data that makes the most capable model at the
sizes people train. Each decision below follows from that, and from what the pilot measured.

### 7.1 The target: seven capabilities, each counting the same

The score is the mean of seven pillars:

| Pillar | What it asks | Items |
|---|---|---|
| Knowledge | Facts the web knows, in ten languages | `knowledge` |
| Reading | Reading and learning from the context | `context` |
| Reasoning | Mathematics and logic, in thirteen languages | `arith`, `word`, `units`, `algebra`, `logic`, `algo`, `multi` |
| Code | Programs and structured data | `prog`, `transform`, `struct` |
| Agents | An agent at work with tools | `agent` |
| Writing | Fresh human writing of every kind, in every language | fresh text, and `grammar` |
| Conversation | Replies in fresh discussions | fresh text |

Inside a pillar each part (a generator family, or a kind of writing) counts the same. Inside a
part, each language counts by its share of the round's corpus. Inside a language, formats and
kinds of task count the same.

Why:

- **How much each capability counts is a value choice, so it must be declared, not left to
  accident.** Before, the generated and fresh items counted by how many of each there happened
  to be, and formats counted the same whatever they were. So 42% of the score was fresh
  Wikipedia text, and a stratum of 14 markup answers counted as much as 490 maths answers.
- **Equal pillars are the choice that favours no use of a model over another.** Calibration
  (7.8) can move the weights later, and only for a measured reason: a pillar whose small-scale
  score does not predict the finals.
- **It bounds every source.** No source, generator or language can weigh more than its part
  of its pillar. That is the answer to the fear that a biased source defines the target.

### 7.2 Languages: by the crawl's share of bytes, in every part

Kept, as decided, and now applied inside every part of every pillar.

- **It is coherent with the score.** Items are scored in bits per byte. A language's share of
  the corpus's bytes, times the bits a curator saves on each of its bytes, is the bits saved
  on a byte of the web. A share by documents would undercount the languages whose scripts take
  two or three bytes a character, since their bits per byte are correspondingly small.
- **It is measured, not chosen.** Each round counts its own corpus.
- **The alternatives fail on first principles.** More weight to English rewards dropping the
  other two-thirds of the web: the season exam's mistake. Equal or tempered weights reward
  curators for the smallest languages, where a student of this size learns little and the
  score is mostly noise.

### 7.3 Fresh text: kinds of writing, each an equal part, from many keyless sources

Fresh text is chosen by kind of writing first:

- reference (Wikipedia's new articles and additions);
- news (Common Crawl's news crawl);
- technical (GitHub issues, pull requests and releases);
- social (public posts on Bluesky and Mastodon);
- discussion (replies on Hacker News, Lemmy, GitHub and Wikipedia's talk pages, and to
  social posts). Discussion is the conversation pillar.

Each kind gets an equal part of the round's fresh items. Inside a kind, languages get their
share of the corpus, and the sources take turns, so one source with many documents cannot
fill a language alone. No source needs a key or an account. A kind without a source yet (how-to
writing, government, fiction) is a known gap, not a reason to let one source fill in.

Every source has its own bias: who posts there, about what, and in which languages. The
remedy is not a perfect source. It is many sources, each capped, none deciding the target.

A live harvest of two hours (09:00 to 11:00 UTC, 19 November) took 26 minutes and gave 1,683
documents. Chosen for an exam, they made 901 items:

| Kind | Items | Sources (items) | Languages |
|---|---|---|---|
| Reference | 255 | Wikipedia additions 151, new articles 104 | 36, 29% English |
| Discussion | 255 | Wikipedia talk 70, Bluesky replies 65, Lemmy 47, Mastodon replies 39, Hacker News 34 | 20, 65% English |
| News | 140 | Common Crawl's news crawl 140, from 110 sites | 39, 45% English |
| Social | 129 | Bluesky 74, Mastodon 55 | 13, 53% English |
| Technical | 122 | GitHub 122 | 12, 72% English |

News, social and technical had fewer candidates than their part, and gave the rest to the
other two. GitHub's comments are left out: by our reading about half of those written in 2026
come from coding agents under people's own accounts.

### 7.4 The tokenizer: trained on the crawl alone

Season 0's 8,192-token vocabulary was trained on equal parts raw crawl and the season exam's
English passages. Trained on round 12's crawl alone, at the same size, a vocabulary moves
other languages toward English:

| | Season 0 (characters a token) | On the crawl alone |
|---|---|---|
| English | 3.35 | 3.03 |
| German | 2.40 | 2.58 |
| Russian | 2.03 | 2.14 |
| Japanese | 1.04 | 1.12 |
| Hebrew | 1.10 | 1.33 |

German goes from costing 1.40 times English's tokens for the same characters to 1.17. Season 1
trains its tokenizer on the crawl alone and keeps 8,192 tokens. A larger vocabulary would close
more of the gap, but would cost a student this size more of its compute than it saves.

### 7.5 Scale: rounds at the standard student; finals at the large one decide the season

With the target in 7.1, micro is no longer a sound proxy. Every curator was trained again, three
times at micro and twice at standard. Each training was scored on the new target:

| | micro | standard |
|---|---|---|
| Curators' score, sd between trainings of the same script | 0.025 | 0.013 |
| Spread between curators | 0.095 | 0.077 |
| Signal to noise | 3.8 | 5.8 |
| Rank agreement with the large student (Spearman) | 0.69 | 0.95 |

Micro's disagreement is not noise. Averaged over three trainings, micro puts `dclm`
(+0.22) far ahead of `fineweb2` (+0.07); at the standard and large sizes `fineweb2` is first.
A 14M-parameter student learns the English skills best from English-only data, and only a
bigger one gains from breadth. That is the compute-dependence of curation the literature
found, and a contest decided at micro would reward exactly the narrow data that loses at scale.

So:

- **Rounds train the standard student** (29M parameters, 160M tokens). It trains in about 18
  minutes on a rented 3090 or 4070 Ti, under $0.10 a student, against 2 to 3 minutes for micro.
  Its ranking agrees with the large student's.
- **The unedited corpus is averaged over several trainings**, since it is still the least
  steady student: a round's baseline is the mean of every established grader's.
- **Micro stays for practice and the gym:** quick feedback, not a verdict.
- **Finals decide the season.** The top eight curators of the season's rounds are trained at
  the large size (54M parameters, 640M tokens, about an hour on a 4090) on a fresh round, twice
  each. That is the largest size where we have measured the order, and where noise between
  trainings is smallest.
- **Each season also tries the next rung up** (about 150M parameters) for the top three, as a
  check on the ladder. If the order changes there, the next season's rounds move up a size.

### 7.6 Scoring: the likelihood of the right answer, as before

We tested another way to score answers. Each answer was scored against two wrong answers taken
from other items of the same kind, given the same context. It was tried on 20 fresh micro
trainings.

- **It was steadier.** Noise between trainings halved for knowledge and reasoning.
- **It also erased most of what separates curators.** In agents, code and reasoning, most of a
  curator's difference is whether its student knows the form of the answer: a tool's log, a
  program's output, a number. That is a real effect of data, since a corpus without logs and
  code makes a student that has never seen them.
- **It did not carry over better to the large student.**

So answers keep being scored by the likelihood of the right one, as DataDecide validated. The
noise is dealt with by the size of the student (7.5).

### 7.7 What a settled round publishes

Wikipedia's text is openly licensed and is published with the round. The other sources'
text (news, GitHub, Hacker News, Lemmy, Bluesky, Mastodon) is held as private exam items only.
For them, a round publishes each item's source, its id there and the item's hash, and anyone
can check them at the source.

The harvest respects what people asked for:

- **Bluesky:** authors who keep their posts from logged-out readers, or say no to AI use.
- **Mastodon:** only authors who opted in to search, only public posts, and no server whose
  terms forbid this.

### 7.8 Calibration

Each season, every pillar's score at the rounds' size is compared with its score at the finals'
size. A pillar whose order does not carry over gets more items or a bigger student, not less
weight. Its weight is a statement of value, and noise is a reason to measure better, not to care
less.

## 8. Open questions

- Likelihood of answers at 150M is validated against benchmarks, not against what a model
  becomes after post-training.
- A fine-tuning probe deserves a place in calibration: how quickly a trained student
  learns a small new task. Pretraining is for a model that learns well afterwards.
- Coding agents now write much of what is new on GitHub, and some of what is new elsewhere.
  Fresh text is not all human writing any more, and no keyless signal tells the two apart.
- Some kinds of writing have no keyless source that publishes all the time: how-to writing,
  government and legal text, fiction.

## 9. Season 2 (November 2026)

Season 1 fixed the first exam's two faults: nobody can know the exam, and it is not six kinds of
English prose. It paid for that in steadiness and in compute. Four separate assessments of the
competition were written from first principles at the end of the season, and every measurement
below was made on the scale pilot of section 6 (nine curators, three student sizes, repeated
trainings, every item's score kept), so none of it cost a training.

The goal they were held to: the order this competition gives curators should be the order
they would have at real scale, where a selection of 500B tokens is asked to beat 40T raw ones,
and it should be found with as few trainings as possible.

### 9.1 What was wrong with the score

Section 7.1 gave seven capabilities the same weight and said of a pillar whose order does not
carry over that it "gets more items or a bigger student, not less weight. Its weight is a
statement of value." The pilot shows where that goes wrong. Item by item, at the official size:

| part | items | spread over noise | order carries to the large student |
|---|---|---|---|
| discussion (fresh) | 74 | 23.5 | 0.95 |
| reference (fresh) | 729 | 16.3 | 0.88 |
| agent | 32 | 4.5 | 0.90 |
| transform | 28 | 3.4 | 0.90 |
| grammar | 26 | 2.2 | 0.81 |
| context | 72 | 1.6 | 0.71 |
| logic | 44 | 1.6 | 0.57 |
| knowledge | 200 | 2.1 | 0.57 |
| algorithms, algebra, word problems | | 1.6 to 2.6 | 0.17 to 0.43 |
| arithmetic, multi-step, units, structure, programs | | 0.8 to 1.8 | at chance at every size |

Nine of the fourteen generated families order curators no better than a coin at any size we
can train. A student of this size cannot do them cold, whatever it was trained on. Their weight
was not a statement of value: it was three sevenths of the score given to noise, and the noise
is what every extra training in season 1 was spent averaging away.

Two faults in the arithmetic made it worse:

- **A one-byte answer weighed as much as a paragraph.** Bits per byte of " 4" is all of an
  item's score, and it swings by whole bits between two trainings.
- **A long answer was cut from the left.** When a question and its answer overflowed the
  window, the answer's own beginning went, and what was left was divided by all of its bytes.

### 9.2 The rule, fixed before the numbers

*Adopt the aggregate with the best spread over noise at the official size, provided its rank
agreement with the large student is no lower than season 1's; within 5%, the simpler one.*

| aggregate | noise | spread over noise | agrees with large |
|---|---|---|---|
| season 1 | 0.0196 | 3.4 | 0.90 |
| pooled bits in a stratum | 0.0157 | 4.0 | 0.95 |
| square-root weights | 0.0172 | 3.8 | 0.98 |
| 16-byte floor | 0.0097 | 6.0 | 0.90 |
| fresh text only | 0.0049 | 28.0 | 0.88 |
| generated only | 0.0225 | 2.8 | 0.93 |
| **season 2: floor, 3/4 writing, 1/4 the skills that pass, small languages pooled** | **0.0052** | **21.7** | **0.95** |

Fresh text alone is the steadiest, and it carries a little less well than the mix: the skills
that pass add something the writing does not. So season 2 keeps both.

### 9.3 What season 2's score is

- **Two groups with fixed shares: writing three quarters, skills one quarter.** Inside a group
  every pillar counts the same, and a pillar missing from a round gives its share to its own
  group, never to the other.
- **Writing:** reference, news, technical, conversation (discussion and social). All of it
  written by people after every entry in the round was frozen. Science abstracts are new to
  the exam and start on the watch list.
- **Skills:** agents, reading, transformation, logic, grammar: the five generated families that
  pass three tests on the pilot (spread over noise above 1.5, order carried to the large
  student at 0.55 or better, and not at chance at the official size) and whose answers follow
  from what the item itself shows.
- **Knowledge passes the same tests narrowly and is watched all the same.** Its questions come
  from a fixed, public list of facts, so a curator can go looking for the pages that state
  them: that measures a curator's aim at the list, not its data. What a student knows of the
  world is measured by the reference and news writing, which nobody can aim at.
- **Watched:** every other family and every kind of writing the season does not name. They are
  sat in every round, scored and published part by part, and count for nothing. This is where
  7.1's statement of value now lives. A watched part earns a place in the next rules version
  when, over at least twenty settled rounds, its spread over noise is above 1.5 and its order
  agrees with the counted score's at 0.5 or better; the few-shot forms of the nine families
  (several worked examples, then the question) start the season there.
- **An item's bits are never put over fewer than 16 bytes**, and an answer or passage longer
  than the window is scored whole, a window at a time.
- **No share of the score rests on one or two items.** A language or a format with fewer than
  eight items in a part counts in proportion, and a part's small languages are measured
  together as one.

The price is stated plainly: under this score a curator that keeps only English loses. Skills
were English only and writing is in every language, and the two ordered the pilot's curators
in opposite directions; English held 76% of season 1's exam weight and holds about 59% of
this one. On the pilot, the multilingual reference filter is the only curator above the
unedited corpus, and every English-only filter is 0.18 to 0.38 bits per byte below it. That is
what 7.2 always said the target was. Season 1's skills were quietly voting the other way.

Two cautions on the evidence. The pilot's exam held Wikipedia's writing only: new articles,
additions to old ones, and talk pages. News, technical writing and social posts were not in
it, and are taken to behave as the same kind of thing: long passages of human text, which is
where the steadiness comes from. Every item's score is public once a round settles, so each
part's spread over noise can be worked out as the season runs, and that is where the
assumption is checked. And with eight curators, most of them English-only filters that the
writing score nearly ties, rank agreement cannot tell a writing share of 67% from one of 90%.
The share is a declared value, like 7.1's; what the pilot settles is that the floor and the
watch list cost no agreement and remove three quarters of the noise.

### 9.4 Contamination, closed further

- **The overlap check counted the letters a to z.** A passage in Russian, Japanese, Chinese,
  Arabic or Hindi gave no keys, so nothing in those languages was ever dropped for being in the
  corpus already. Keys now count letters in every script and sentences end at 。！？ too.
- **Ink falls tenfold**, to one ten-thousandth of the budget (16,000 tokens at the official
  size). Ink is the only place a curator's own words can enter the training set, and nothing a
  curator legitimately does needs more.
- **Deleting lines is never ink.** Whole lines of one page, kept in the page's order, are
  copies however short. Cleaning a page was the most useful thing ink was being spent on.
- **The agent family's answers follow from the session shown**, in a project drawn from the
  seed. Season 1's listings could be learned by heart.
- **The exam's writing is harvested before the draw, and chosen by it.** Season 1 waited two
  hours after a draw for writing published after it. That bought the guarantee that no curator's
  author had seen the exam, and cost every round two hours. Season 2 takes the writing from
  before the draw. The examiner harvests every clock hour, and commits each harvest's hash to
  the server at once, where anyone can read it. A draw's seed picks three of the 26 hours
  before it; the exam is made from exactly those, and the server refuses one that is not. The
  examiner cannot choose which hours a round gets, and cannot change an hour after committing
  it. What still rests on trust is that each harvest is what the sources said: section 7.3's
  answer stands, many sources, each capped, each harvest seeded by the public beacon.
- **No freeze on entries.** An entry may be submitted after some of its exam's writing was
  published. A first draft of the season made every entry wait 26 hours so that none could;
  it was taken out, because the wait cost every entrant a day and bought almost nothing. A
  curator has no network when it runs. The pool it chooses from was crawled before any of the
  writing existed, and what the pool shares with the exam is dropped from the exam. The one
  way text gets from an author's hands into the training set is ink, and there are 16,000
  tokens of it.
- **A curator cannot write the exam into its script.** The grader holds the ink and the exam:
  a script whose ink, with 256 bytes either side, has the opening of a sentence of a fresh
  item is refused. With ink at 16,000 tokens this is a check on a few pages of text, and it
  is what the freeze was for.
- **Three more keyless sources.** Forums that run Discourse (31 of them, about 100 replies an
  hour), the kernel's mailing lists (first versions of patches: code written by people this
  hour), and Crossref (abstracts of works registered this hour, about 40). Stack Exchange was
  tried and left out: its whole network is asked about four questions an hour now, and without
  a key it allows 300 requests a day.

### 9.5 The board

A score is one thing and a ranking another. Season 1's exam was noisy, and its board made the
noise worse in four ways. Each is fixed by a rule that was stated before any round ran.

- **Three lucky rounds outranked thirty steady ones.** A rating is now the adjusted mean drawn
  towards the field's: `m + n/(n+k) (adjusted - m)`, where k is the noise of one round over the
  spread between curators, kept between 1 and 8. The ladder takes curators in order of rating
  less 1.28 standard errors, which is what each is nine times in ten at least as good as.
  Ratings within two standard errors of the one above share its place.
- **The first rounds set their own error bars.** With one curator at two rounds, the spread of
  a round's gain was estimated from two numbers. What the season expects a round to move by
  now counts beside its own rounds, as eight rounds' worth. Season 1's board gives 0.0195 a
  round at the standard size (eleven curators, pooled). Between two trainings the pilot's
  noise fell from 0.0196 to 0.0052, but a round also moves with the pages it draws, which the
  pilot did not measure. The prior is 0.015: erring high makes early ties more common, which
  is the safe direction, and after twenty rounds the season's own numbers carry it.
- **Failing could be the better move.** An entry that failed counted zero, above every curator
  that finished below the baseline. It counts as the round's lowest valid gain, or zero if
  that is lower. The same inside an entry: a failed run counts as the entry's worst finished
  run when that is below zero, so failing one run in two does not halve a loss. Neither is
  used to work out how hard a round was: a penalty is not a measurement.
- **Every grader measured the baseline.** That doubled the cost of a first grade and made each
  score rest on a baseline from one machine. Four established graders measure a round's
  baseline; every score is taken against the mean of theirs.
- **Reference curators held six of the ladder's eight places**, and ran every round. They are
  there to tie rounds together and to mark the scale, and two do that: `fineweb2` and `dclm`
  run every round, each of the others one round in eight. That is two and a half runs a round
  where there were six.

Every item's score from every grade is public once a round settles. Anyone can recompute a
round's gains, try another weighting, or measure a part's spread over noise without asking.

### 9.6 What is decided against, and what waits on a number

Season 2 takes everything section 9 found wanting and does it now. Three things are decided
against, each on principle:

- **Raw HTML as the curator's input.** A crawl's pages with their markup are fifteen times the
  size of their text (CC-MAIN-2026-43: about 960 MB a file against 65 MB). At a fixed
  download, a curator choosing among fifteen times the pages has more to work with than one
  reading the markup of a fifteenth. The bandwidth goes to a bigger pool, and deleting lines
  for nothing gives a curator most of what the markup would: it can cut a page's furniture.
- **A track where a served model rewrites the corpus.** The score would then measure the
  model that was served. Ink stays small, and a curator's own GPU may write within it.
- **A different student.** The student is the measuring instrument; changing it with the exam
  would leave nothing to compare the two seasons by. Its ranking is checked at a larger size
  instead (the finals, and the audit of scale in between).

Three more were put to a measurement, each with its rule written first:

- **The pool's size.** As many pages as a curator can read and a verifier can fetch inside the
  time box on the machines the fleet rents, measured, with a third to spare; and only if the
  larger pool changes what is measured (9.8).
- **Averaging the student's last weights: not adopted.** The rule was that the noise between
  trainings fall by a quarter and spread over noise rise by a fifth at micro, and by 15% each
  at standard. Six ways of averaging (an exponential average over 0.5%, 2% and 5% of the
  steps; the plain mean of the last 5%, 10% and 20%) were carried through every training
  of 9.7 and sat the same exam. At micro, seven scripts trained four times each, the
  season's score had a noise of 0.0575 bits per byte between trainings with the last
  weights and between 0.0584 and 0.0613 with an average: every one is noisier, by 1.5% to
  6.5%, and every one lowers spread over noise (2.74 with the last weights, 2.55 to 2.66
  with an average). None changes the curators' order. The student's noise is not the
  jitter of its last steps, which is what an average removes; the trainer stays as it is.
- **Which parts count.** A draft kept parts that were sat and did not count, to be promoted
  after twenty rounds. 9.7 replaces that with a measurement made before the season.

### 9.7 Every part is in the score or off the exam

A first draft of the season kept a *watch list*: parts that every exam asked, whose scores
were published, and that counted for nothing until twenty rounds had shown what they were
worth. It is gone. A part that counts for nothing still costs every grade its time and
every reader a line to understand, and "we will see" is a decision put off. Each part is
measured before the season starts and is then in the score or off the exam, its generator
deleted.

**The measurement.** One practice round (`s2-p00000`), its corpus at the micro and the
standard size. Seven curators that differ in kind: the unedited corpus, `tidy`, `english`,
the two reference filters that run every round (`fineweb2`, `dclm`), and two red-team
curators, `scribe` (tidy's selection, and the whole ink allowance spent on worked problems
in the forms the exam's skills take) and `mimic` (no ink; the pages of the corpus that look
most like the exam). Each is trained four times at micro and three times at standard, on a
rented RTX 4090; two trainings of one script differ as two machines do (the initial weights
moved by one part in a million). Every student sits one candidate exam: every generated
family of generator version 9, cold and with worked examples first, in eleven languages,
and about 1,500 passages of writing of every kind the harvest takes (science abstracts
among them) from three harvested hours, by the corpus's language shares.

**The rule, written before any of these trainings had finished.** For a part, *gain* is
its mean over items of the unedited students' bits per byte less the student's; *noise* is
the standard deviation of that between trainings of one script, pooled over the scripts;
*spread* is the standard deviation of the six edited curators' mean gains.

1. *It tells curators apart.* Spread over noise is at least 1.5 at the standard size. A part
   on which curators did not differ at all would show about 0.6 here (one over the square
   root of three trainings), so 1.5 is well clear of nothing.
2. *It carries.* The order it puts the curators in at micro and at standard agree, at a
   rank correlation of 0.55 or more. Where trainings at the large size are made, the same
   between standard and large.

**Amended while the trainings ran, before any curator's large student but tidy's had
finished.** The second test as first written cannot be read. The micro corpus was 8 files
for a budget of 40 million tokens, twelve times the budget; the standard one 16 files for
160 million, six times; the large one 64 files for 640 million, six times. So micro and
standard did not play the same game: at micro a curator kept one page in twelve and at
standard one in six, and a curator that keeps only prose lines (`tidy`, and `scribe` with
it) fed the micro student so narrow a diet that it fell 1.8 bits per byte behind on the
agent items, where at standard it is 0.17 behind. The first standard trainings showed the
effect on every part, the kinds of writing among them: it is a property of the two pools,
not of a part. Standard and large have the same ratio of pool to budget, and the season's
claim is that the order at standard carries to large. So the second test is the agreement
between **standard and large** alone, at the same 0.55, with one large training of each
curator; micro against standard is reported beside it and decides nothing. (9.8 gives every
size the same ratio from here on, so that this cannot happen again.)

**The reason given was then checked, and it is only part of the truth.** Every script was
trained four more times at micro on a pool of 4 files, six times the budget as at standard,
and sat the same exam. The kinds of writing do agree with standard better on the matched
pool (discussion 0.66 to 0.77, news 0.60 to 0.83, reference 0.60 to 0.71, technical 0.54 to
0.71, social 0.89 to 1.00), and the micro score is steadier (0.054 between trainings to
0.035). But the score's order at micro still agrees with standard's at only 0.43, and agent
sessions and logic still run the other way (−0.37 and −0.26): `tidy` is last at micro and
in the middle at standard whichever pool it is given. Most of the disagreement is the micro
student's own, which is what 7.5 found in season 1, and not the pool's. The amended test
stands, for that reason and not the one first given: the season's claim is that the order
at standard holds one size up, and micro is not a size the season scores at.

A part that passes both is in the score. A part that fails either is off the exam. The
thresholds are the ones 9.3 used on the pilot. The few-shot forms, science abstracts, each
kind of writing and each cold family face the same two tests; nothing is in for its name.
A skill pillar is the mean of its parts that are in, and a pillar with none is dropped.

Also fixed before the standard trainings were complete: a family that is asked cold and
with worked examples first is one skill asked two ways, so at most one form of it counts,
the one with the larger spread over noise at standard among those that pass. And the
families that had no pillar go, if they pass, into three: *numbers* (arithmetic, several
steps of it, units, word problems, algebra), *programs* (running a program, an algorithm's
output, structured data) and *knowledge*; beside agents, reading, transformation, logic and
grammar. Science abstracts, if they pass, are a sixth kind of writing with a pillar of
their own.

3. *Ink does not buy it.* `scribe` is `tidy` plus 16,000 tokens of the exam's own forms. If
   scribe's score is above tidy's by more than twice the standard error of the difference
   at the standard size, the exam can be bought with ink, and ink goes to nothing: a script
   is copies of the crawl and no more. If it is not, ink stays at 16,000 tokens, shown to
   buy nothing.

`mimic` has no rule of its own. Choosing the pages that most resemble what the exam asks
is curation, and if it wins, it won by selection.

### 9.8 The pool

A curator at real scale keeps about one page in eighty. A standard round's pool was 16 files
of the crawl, six times what the student is trained on, so the most selective script a
curator could write kept one page in six. A larger pool costs every machine that touches a
round its download, so it is taken only as far as it fits, and only if it changes the
measurement.

**What fits**, timed on a rented machine at 1.28 times the season's processor floor, with
the reference curators reading the pool on the six cores a curator is given (`task.map`):
at 64 files (24 times the budget, 1.35 million pages, 10.2 GB of text, a 4.1 GB download)
the corpus is built in 116 seconds and the slowest reference curator finishes in 157 of its
1,200 seconds. Two rounds on disk are 21 GB of the 60 GB a machine must have free. At 128
files two rounds are 41 GB, more than two thirds of the disk, so 64 files is the most that
fits with a third to spare.

**The rule, written before any student was trained on the larger pool.** One standard
student of each of the seven scripts, made on 64 files of the same practice round, against
the three of each trained on 16 files, on the same exam. The pool grows to 64 files if the
spread between the six edited curators' scores is at least a quarter larger on 64 files, or
if their order differs between the two (a rank correlation below 0.8): either way the small
pool was hiding something that the real regime would show. If neither, the pool stays at 16
files, since four times the download would buy nothing. Every size keeps the same ratio of
pool to budget, so that what is compared across sizes is the same game.

**The result: the pool stays at 16 files.** The season's score, in bits per byte gained on
the unedited corpus (16 files: the mean of three trainings, which differ by 0.005; 64 files:
one training):

| | mimic | fineweb2 | scribe | tidy | english | dclm | spread |
|---|---|---|---|---|---|---|---|
| 16 files | +0.117 | +0.057 | +0.021 | +0.011 | −0.239 | −0.299 | 0.171 |
| 64 files | +0.121 | +0.060 | −0.040 | −0.050 | −0.234 | −0.177 | 0.135 |

The spread is a fifth smaller on the larger pool, where the rule asked for a quarter larger.
The order is the same but for the last two changing places, a rank correlation of 0.94,
where the rule asked for one below 0.8. Neither holds.

Single scores did move, by far more than the noise. `tidy` keeps the pages with the most
prose in them; with four times the pages to choose from it kept only the most prose-like,
and its student fell 0.06, two thirds of that on conversation, news and reference writing.
`dclm` rose 0.12. A larger pool is a different game for some curators. It is not one that
tells them apart better or puts them in another order, and it would cost every machine
four times the download. So it is not taken.

Micro's pool was 8 files for 40 million tokens, twelve times its budget. It becomes 4, six
times like the others, with 300 seconds to edit it in: the same time for each page as at
standard and large. Measured on it (seven scripts, four trainings each), the micro score is
steadier, 0.035 between trainings where it was 0.054, and its spread over noise is 4.1
where it was 2.9; every curator finished inside the 300 seconds. (Nano is a smoke test of
two files, the least a round can be, and is compared with nothing.)

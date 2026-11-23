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
competition were written from first principles at the end of the season. The measurements of
9.1 to 9.5 were made on the scale pilot of section 6 (nine curators, three student sizes,
repeated trainings, every item's score kept), so none of them cost a training. 9.7 and 9.8
are a measurement made for the purpose before the season began: 91 trainings on rented GPUs.

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
- **Writing:** reference, news, technical, conversation (discussion and social posts). All
  of it written by people in the day before the draw.
- **Skills:** agents, reading, logic, grammar, numbers, programs and knowledge. Numbers is
  arithmetic, units, word problems and algebra; programs is an algorithm's output. Both are
  asked after a few worked examples, the only form in which a student this small can be
  told apart on them. (Transformation and structured data passed 9.7's tests and were
  taken off by 9.10, which set them against a harder noise.)
- **Every part earned its place, and nothing else is on the exam.** The pilot chose the two
  tests: curators spread on a part by at least 1.5 times its noise at the official size, and
  the order it puts them in holds at the large student at 0.55 or better. 9.7 put every
  part to them on a measurement made for the purpose, with every kind of writing in it and
  every generated family asked both cold and after examples. What failed is not sat and
  its generator is deleted. A draft of the season kept such parts on a *watch list*, sat
  in every round and counting for nothing; 9.7 says why that went.
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
it, and were taken to behave as the same kind of thing: long passages of human text, which is
where the steadiness comes from. 9.7 then measured them, and they do. And with eight
curators, most of them English-only filters that the writing score nearly ties, rank
agreement cannot tell a writing share of 67% from one of 90%. The share is a declared value,
like 7.1's; what the measurements settle is that the floor, and leaving out the parts that
fail, cost no agreement and remove three quarters of the noise.

### 9.4 Contamination, closed further

- **The overlap check counted the letters a to z.** A passage in Russian, Japanese, Chinese,
  Arabic or Hindi gave no keys, so nothing in those languages was ever dropped for being in the
  corpus already. Keys now count letters in every script and sentences end at 。！？ too.
- **A script is copies of the crawl and no more.** Season 1 let a script hold a little text
  of its own, called ink: 160,000 tokens at the official size. A draft of season 2 cut that
  tenfold; 9.7 then measured what 16,000 tokens buy when they are aimed at the exam, and
  ink went to nothing. A copy is at least 64 bytes long, so that text cannot be spelled out
  of scraps.
- **Deleting lines is always allowed.** Whole lines of one page, kept in the page's order,
  are copies however short, as long as 64 bytes of the page are kept. Cleaning a page was
  the most useful thing ink was being spent on.
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
  writing existed, and what the pool shares with the exam is dropped from the exam. And a
  script holds no text of its own, so there is no way for anything an author has read to
  get from the author's hands into the training set.
- **Two more keyless sources.** Forums that run Discourse (31 of them, about 100 replies an
  hour) and the kernel's mailing lists (first versions of patches: code written by people this
  hour). Crossref (abstracts of works registered this hour, about 40) was added and taken
  out again when science abstracts failed 9.7's test. Stack Exchange was tried and left out:
  its whole network is asked about four questions an hour now, and without a key it allows
  300 requests a day.

### 9.5 The board

A score is one thing and a ranking another. Season 1's exam was noisy, and its board made the
noise worse in four ways. Each is fixed by a rule that was stated before any round ran.

- **Three lucky rounds outranked thirty steady ones.** A rating is now the adjusted mean drawn
  towards the field's: `m + n/(n+k) (adjusted - m)`, where k is the noise of one round over the
  spread between curators, kept between 1 and 8. The ladder takes curators in order of rating
  less 1.28 standard errors, which is what each is nine times in ten at least as good as.
  Ratings within two standard errors of the one above share its place.

  *Changed after round 2, on 23 November.* The pull towards the field's mean was taken out: a
  rating is the adjusted mean alone, and its standard error `sqrt(s2/n)`. The field's mean is
  set by whoever is on the board, and half of the organisers' references are far below the
  unedited pages. After two rounds it stood at -0.113 and k at its floor of 1, so three new
  curators that had each gained about 0.07 in their one round were rated -0.02, under
  references they had beaten in it. The ladder's order and the shared places already do what
  the pull was for: at this season's round noise (a spread of 0.0136), three rounds have to
  beat thirty steady ones by 0.007 to take a ladder place from them.
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

Season 2 takes everything section 9 found wanting and does it now. Two things are decided
against, each on principle:

- **Raw HTML as the curator's input.** A crawl's pages with their markup are fifteen times the
  size of their text (CC-MAIN-2026-43: about 960 MB a file against 65 MB). At a fixed
  download, a curator choosing among fifteen times the pages has more to work with than one
  reading the markup of a fifteenth. The bandwidth goes to a bigger pool, and deleting lines
  for nothing gives a curator most of what the markup would: it can cut a page's furniture.
- **A different student.** The student is the measuring instrument; changing it with the exam
  would leave nothing to compare the two seasons by. Its ranking is checked at a larger size
  instead (the finals, and the audit of scale in between).

Five more were put to a measurement, each with its rule written first:

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
- **Ink.** Whether 16,000 tokens of a curator's own text, aimed at the exam, buy it (9.7).
  They do, and a script now holds none.
- **Text a model writes.** Two of the four assessments ask for a second track in which one
  fixed open model, served by the harness, rewrites pages or writes new ones, on a budget of
  10% to 15% of the training tokens. A third asks for a pilot of it at 0.1%, 1% and 5%, beside
  a curator that only selects. An earlier draft of this note called such a track decided
  against on principle, because the score would measure the model that was served. That was
  wrong. If the model is fixed and the same for everyone, what differs between two curators
  is what they have it write, and that is curation. What stands against it in a round is
  something else: a teacher's time on every curator's machine, text whose style can be known
  in advance, and that nobody had measured whether it teaches this student anything. 9.9
  measures that.

### 9.7 Every part is in the score or off the exam

A first draft of the season kept a *watch list*: parts that every exam asked, whose scores
were published, and that counted for nothing until twenty rounds had shown what they were
worth. It is gone. A part that counts for nothing still costs every grade its time and
every reader a line to understand, and "we will see" is a decision put off. Each part is
measured before the season starts and is then in the score or off the exam, its generator
deleted.

**The measurement.** One practice round (`s2-p00000`), its corpus at the micro, the
standard and the large size. Seven curators that differ in kind: the unedited corpus, `tidy`, `english`,
the two reference filters that run every round (`fineweb2`, `dclm`), and two red-team
curators, `scribe` (tidy's selection, and the whole ink allowance spent on worked problems
in the forms the exam's skills take) and `mimic` (no ink; the pages of the corpus that look
most like the exam). Each is trained four times at micro, three times at standard and once
at large, on rented GPUs (RTX 4090s; five of the large students on an RTX 5090); two
trainings of one script differ as two machines do (the initial weights moved by one part
in a million). Every student sits one candidate exam: every generated
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
0.71, social 0.89 to 1.00), and the micro score is steadier (0.045 between trainings to
0.030). But the score's order at micro still agrees with standard's at only 0.43, and agent
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

**The result.** At the standard size, with the order at standard against the order at large
(six curators, rank correlation):

| Part | Items | Spread over noise | Carries to large | |
|---|---|---|---|---|
| News | 283 | 93.4 | 0.83 | in |
| Reference | 294 | 61.2 | 0.77 | in |
| Discussion | 294 | 46.6 | 0.77 | in |
| Social posts | 233 | 38.1 | 1.00 | in |
| Technical writing and code | 273 | 24.5 | 1.00 | in |
| Science abstracts | 113 | 19.4 | 0.49 | **out**: does not carry |
| Logic | 288 | 12.1 | 0.71 | in |
| Agent sessions | 304 | 5.7 | 1.00 | in |
| Knowledge | 200 | 5.7 | 0.89 | in |
| Reading | 384 | 5.4 | 0.94 | in |
| Grammar | 198 | 4.9 | 0.89 | in |
| Transformation | 204 | 2.1 | 0.83 | in |
| Arithmetic, after worked examples | 192 | 3.8 | 0.77 | in |
| Arithmetic, cold | 192 | 1.6 | 0.03 | out |
| Arithmetic, as a pair | 192 | 1.4 | 0.09 | out |
| An algorithm's output, after examples | 56 | 3.8 | 0.94 | in |
| An algorithm's output, cold | 56 | 3.4 | 0.77 | passes; the other form counts |
| Word problems, after examples | 48 | 2.6 | 0.94 | in |
| Word problems, cold | 48 | 1.4 | 0.26 | out |
| Units, after examples | 56 | 2.3 | 0.83 | in |
| Units, cold | 56 | 0.8 | 0.94 | out |
| Units, as a pair | 44 | 1.0 | −0.09 | out |
| Algebra, after examples | 40 | 1.6 | 0.66 | in |
| Algebra, cold | 40 | 0.8 | −0.09 | out |
| Structured data, after examples | 72 | 1.6 | 0.60 | in |
| Structured data, cold | 72 | 1.6 | −0.09 | out |
| Running a program, after examples | 84 | 2.4 | 0.54 | **out**: 0.54 is not 0.55 |
| Running a program, cold | 84 | 0.5 | 0.31 | out |
| Several steps of arithmetic, cold | 144 | 1.2 | 0.66 | out |
| Several steps of arithmetic, after examples | 144 | 0.4 | −0.71 | out |

So the score is: **writing**, four pillars (reference, news, technical, and conversation,
which is discussion and social posts); **skills**, eight (agents, reading, transformation,
logic, grammar; numbers, which is arithmetic, units, word problems and algebra after worked
examples; programs, which is an algorithm's output and structured data after worked
examples; and knowledge). Science abstracts, running a program, several steps of
arithmetic, every pair form and every cold form of a family that also has examples are off
the exam and their generators deleted; Crossref is not harvested. (9.10 takes two more
off, transformation and structured data, and gives the score's numbers without them.)

What the measurement says about the parts, beyond in and out:

- **Asked cold, a student this small is too near chance to tell curators apart.** Eight
  families were asked both ways. With worked examples first, six pass; cold, one does.
- **Writing is where the signal is.** The five kinds of writing spread curators by 24 to 93
  times their noise; the best skill by 12. The pilot's assumption that news, technical
  writing and posts behave as Wikipedia's writing does (9.3) holds: they are the steadiest
  parts on the exam.
- **Science abstracts are steady and say little.** Their noise is the smallest of any kind
  of writing and so is their spread (0.03 bits per byte, where the other kinds have 0.14 to
  0.29): every curator's student reads them about as well as the next one's, and the small
  differences do not hold at the large size.
- **Knowledge counts.** A draft kept it out because its questions come from a public list,
  which a curator could aim at. The list is 16,617 facts, each put in up to ten languages (152,405 sentences), and an
  exam asks 200 of them; a
  script holds no text of its own, so the only way to aim at them is to keep the pages
  that state such facts, in ten languages. That is curation, and the part passes both
  tests with room.

**The score these parts make**, on the same trainings: 0.0041 bits per byte between two
trainings of one script at standard, a spread between the six curators of 0.165, forty
times the noise (the pilot's mix: 0.0052 and 21.7). The six curators stand in the same
order at standard and at large; with the unedited corpus counted as a seventh, the two
orders agree at 0.89.

| | mimic | fineweb2 | scribe | tidy | unedited | english | dclm |
|---|---|---|---|---|---|---|---|
| standard | +0.108 | +0.053 | +0.012 | +0.006 | 0 | −0.233 | −0.295 |
| large | +0.079 | +0.044 | −0.013 | −0.030 | 0 | −0.276 | −0.323 |

**Ink goes to nothing, and part of that is a judgement.** `scribe` spent its 16,000 tokens
on the five skills the draft counted, a fifth on each. On the score as it stood when the
standard trainings finished it beat `tidy` by 0.0102 bits per byte, 2.6 standard errors:
by the rule, ink goes. Where the ink worked it worked strongly: 0.10 bits per byte on
agent sessions (5.6 standard errors) and 0.10 on transformation (3.2); on reading, logic
and grammar it bought nothing. Then the part list changed. The final score has eight
skill pillars where the draft had five, three of them forms `scribe` never wrote, so each
pillar it aimed at weighs less; on that score the same students differ by 0.0061, which
is 1.8 standard errors and short of two. Read against the final score, the letter of the
rule would leave ink at 16,000 tokens. It goes all the same, for three reasons. The test
asked whether text aimed at the exam buys the parts it is aimed at, and it does. Nothing a
curator does for the data needs ink once deleting lines is free: 16,000 tokens are a
ten-thousandth of what the student reads, and can matter only where they are aimed. And
with no text of its own in a script, nothing an author knows can be carried into the
training set, which is what lets an entry run in the next round drawn with no wait.

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
the unedited corpus (16 files: the mean of three trainings, which differ by 0.004; 64 files:
one training):

| | mimic | fineweb2 | scribe | tidy | english | dclm | spread |
|---|---|---|---|---|---|---|---|
| 16 files | +0.108 | +0.053 | +0.012 | +0.006 | −0.233 | −0.295 | 0.165 |
| 64 files | +0.110 | +0.058 | −0.043 | −0.049 | −0.225 | −0.178 | 0.130 |

(On the score 9.7 ended with. The rule was applied when the standard trainings finished,
on the score as it then stood: 0.171 and 0.135, and the same two answers.)

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
steadier, 0.030 between trainings where it was 0.045, and its spread over noise is 4.3
where it was 3.1; every curator finished inside the 300 seconds. (Nano is a smoke test of
two files, the least a round can be, and is compared with nothing.)

### 9.9 Text a model writes: a pilot

Written on 22 November 2026, after the teacher's first pages were read and before any
student of the pilot was trained.

**The question.** A script in a round is copies of the crawl (9.7). Does text that one fixed,
small, open model writes from the round's own pages teach the student more about fresh human
writing than the pages a selecting curator had already chosen?

**The setting.** Practice round `s2-p00000` at standard, the corpus of 9.7 (337,313 pages),
and the candidate exam of 9.7, sat whole. The code is the code 9.7 measured, from before a
script lost its text, with the allowance for a curator's own text raised on the pilot's
machine. The parent of every arm is `tidy`'s script, made again there. It is two lines
shorter than the one 9.7 trained (51,762 lines), so the control is trained again on the same
machine, five times, and 9.7's three `tidy` students are only looked at beside it.

**The teacher.** `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0, downloaded with no key), in
16-bit floats, at temperature 0.7, top-p 0.8, top-k 20, one fixed seed. It is given a page
3,500 bytes at a time and told to write the page's content anew in the page's language, in
its own words, in full sentences; to tell everything the page tells and add nothing; to
leave out menus, links and advertisements; and to answer `EMPTY` where there is nothing but
those. A first instruction ("rewrite it as clear writing, keep every fact") was dropped
after its first 384 answers were read: the model gave the page back nearly letter for letter,
menus and all, and a copy tests nothing. No student had been trained on either.

**The arms.** The dose is the share of the tokens trained on that the teacher wrote.

- `rewrite-1`, `rewrite-5`, `rewrite-15`: pages of `tidy`'s script that are trained on, taken
  in one seeded order, so that each arm holds the one below it. The teacher is given the
  whole page, not only the lines `tidy` kept. Its text stands in the script where `tidy`'s
  lines of that page stood.
- `rescue-5`: pages `tidy` did not choose, that pass its own test for looking written, of
  1,500 to 12,000 bytes, of which its rule keeps less than three tenths, no two with the same
  opening (40,583 pages, 140 MB). The teacher's text of them, in a seeded order, takes the
  place of the lowest-ranked pages `tidy` trained on, token for token, spread through the
  script.

The teacher runs on rented GPUs by the hour. If it has not written 15% by the time the
students must start, the largest arm is the largest dose it has written, and is named for
it. A piece of a page that is cut off at the teacher's limit of 2,048 tokens spoils its
page: an in-place page then keeps `tidy`'s lines, and a rescued one is not used.

Three students of each arm, with the starts of 9.7 (the first as it is, the others moved by
one part in a million). Text with no page behind it, a larger teacher, and a teacher tuned
for the task are not tested.

**The score.** The four kinds of fresh writing, a quarter each: reference, news, technical,
conversation. The skills are left out of the rule, because text can be written to a
generated format and 9.7 showed that it pays; the whole score is given beside it.

**The rule.** The noise is the spread between students of one script, pooled over the
pilot's five scripts. An arm passes if its mean is above `tidy`'s by more than twice the
standard error of that difference. If an arm passes, a second track for written text is
worth designing, and its first step is the control two of the assessments ask for: the
teacher's time spent choosing pages and not writing them. Four arms are four chances, about
one in eight that one passes with nothing there, so an arm that passes by less than three
standard errors is trained three more times before anything is built on it. If none passes,
Season 2 has no written text, and this section says how far each arm fell short.

**Added the same morning, before any arm's student had finished.**

*A fifth arm, `rescue-15`.* The teacher had the time. Five arms are five chances, about one
in six that one passes with nothing there; the rule for a narrow pass stands.

*What the order of a script is worth.* The control trained on the pilot's machine scores
0.017 lower on writing than 9.7's `tidy`: five students at −0.000 (from −0.004 to +0.003)
against three at +0.016 (+0.010 to +0.020), where students of one script differ by 0.003.
It is not the machine or the code. The unedited script's tokens are the same to the byte,
and an unedited student trained on the pilot's machine scores +0.004 against 9.7's three,
inside their spread. It is the script. `tidy` chooses a quarter more than the budget holds
and shuffles its choice with a fixed seed. 9.7's standard scripts were made a few minutes
before `tidy`'s test of a page's characters moved into the regular-expression engine, a
change meant to leave its script as it was. At this size it does not quite: the two choose
the same pages but for four of 51,764 (made again from both versions on the pilot's
machine: three pages only in the old script, one only in the new). The same seed then
shuffles a list two shorter into another order: 160 pages stand in the same place, and of
the 40,600 pages each trains on, 2,300 are not trained on by the other. So the
noise between students of one script, which is what a round's error bars are made of, says
nothing of how far a curator's score moves when its script is put in another order. That
is measured here too: `tidy`'s own lines in two more orders (seeds 1 and 2), two students
each.

Every arm keeps `tidy`'s order and changes only the pages named, so most of this cancels
between an arm and the control. Not all of it does: the teacher's text of a page is a fifth
longer than `tidy`'s lines of it, so an arm's budget is full a little earlier in the script.
The rule therefore gains a second condition, written before any arm's result: an arm passes
only if its gain over `tidy` is also larger than the widest gap between two of `tidy`'s
three orders on this machine. A gain smaller than that is not told apart from an accident
of order.

**Added at 10:35 UTC, after the first `rescue-5` student and before any other arm's.** That
student is 0.028 above `tidy` on writing. A rescued page differs from the page it replaces
in two ways: the teacher wrote it, and it is a kind of page `tidy` never chooses. To tell
the two apart each rescue arm gets a twin, `rescueraw-5` and `rescueraw-15`: the same pages
in the same places, copied whole and not rewritten, with `tidy`'s lowest-ranked pages
removed by as many tokens. The in-place arms have the same twin (`rewriteraw-N`, the whole
page copied where the teacher's text of it would stand), trained if an in-place arm passes.
A pass is credited to the teacher's writing only if the arm also beats its twin by more
than twice the standard error of their difference. If the twin does as well, the gain
belongs to the pages, which a script of copies can have today, and it is a finding about
`tidy` and not about writing.

**Added at 11:20 UTC.** By now 22 students are trained: `tidy` five times, three more
orders of it, three each of `rescue-5` and `rescueraw-5`, two or three of `rewrite-1` and
`rewrite-5`, two of `rewriteraw-1`, one of `tidy-whole`. Every arm is 0.010 to 0.035
above `tidy`. Two things found in them change how the rest is run. Both are written here
before the students they govern have finished.

*The exam asks for a mark the crawl does not have.* The exam's fresh passages hold 3,032
empty lines in 1,490 items, 1.6 in every 1,000 bytes: a title and its body are joined by
one, and paragraphs keep theirs. The crawl's pages hold 0.002 in 1,000 bytes (11 pages in
3,000 have one at all) and `tidy`'s lines hold none. The teacher's text holds 1.8 in 1,000
bytes. A student that has never seen an empty line pays some twenty bits for each one it
meets. Regressing each item's gain on what the item holds, `rewrite-1`'s gain over `tidy`
is 9.2 bits for every empty line, 28,000 of its 39,000 bits; `rescue-5`'s is 9.5 bits a
line, 29,000 of 45,000. So most of what the arms gained is a mark of layout that the
teacher's text happens to share with the exam, and a script of copies cannot have it
because the crawl does not.

That is a fault of the exam and not a use for a teacher. The students' weights were kept,
so every student sits the exam a second time with its empty lines closed up (each run of
line ends becomes one), with no training. The rule is applied on both exams, and the
verdict is the one on the exam with the lines closed, because that is what the exam will
be. One arm is added to show the mark alone: `layout-1` has no teacher's text; the 396
pages `rewrite-1` rewrites stand as `tidy` keeps them, with an empty line after every
line. If the empty line is what the teacher gave, `layout-1` gains on the exam as it was
about what `rewrite-1` gains there, and nothing on the exam with the lines closed.

*A script's order is worth more than was measured.* Yesterday's `tidy` script, trained on
this machine, is 0.025 above today's (one student; yesterday's three students of it were
0.016 above). The two scripts hold the same lines but for four pages, in different orders,
and build the same text from a line. Seeds 1 and 2 of today's script are within 0.003 of
it. So three orders agree and a fourth does not, and the second condition of the rule
(the widest gap between `tidy`'s three orders) was measured too narrow. More orders are
trained: the pages `tidy` trains on in another order (`tidy-o1`), two more full orders
(`tidy-s3`, `tidy-s4`), yesterday's script with its trained lines in another order
(`tidyv1-o1`). The second condition becomes: larger than the widest gap between any two
orders of `tidy`'s lines trained on this machine, yesterday's script among them. And the
1% and 5% arms are trained once more in a second order, with `tidy` in that same order as
their pair (every shared page in the same relative place), so that an arm is compared with
`tidy` twice, in two orders.

The 15% arms and their twins are trained as registered, once or twice each as the
machine's hours allow.

**The result (22 November, 12:30 UTC).** 69 students, trained on one machine of seven RTX
4090s rented for 2 hours 47 minutes, $7.80. No arm passes, and Season 2 has no written
text. The pilot found two other things on the way, and 9.10 acts on them: the exam's
writing was laid out in a way no page of the crawl is, and a script's order is worth more
than the noise between two trainings of it.

*What was not run as written.* `tidy-s4` was never trained. `tidy-s1` and `tidy-s2` have one
student each, not two, and `rewriteraw-1` two, not three. The 15% arms have two students in
the first order and one in a second. A third order was trained for `rewrite-1`, `rescue-5`,
`rescue-15` and their twins, which nothing above asked for. The exam the verdict is read on
went a step past the one registered: not only its empty lines closed, but all of its white
space as the crawl has it (9.10); both are given. On the two corrected exams the unedited
student is the one trained on the pilot's machine, since 9.7's weights were not kept. And
the pilot's own weights were left on the machine but for one student, the link being too
slow to fetch them, so none of this can be sat again.

*The marks.* For every 1,000 bytes, the exam's fresh writing and the crawl's pages hold:

| | The exam's writing | The crawl's pages | |
|---|---|---|---|
| Empty lines | 1.97 | 0.007 | 300 times |
| Tabs | 0.37 | 0.001 | 600 times |
| No-break spaces | 0.072 | 0.001 | 58 times |
| Two spaces together | 0.106 | 0.007 | 15 times |
| Lines that begin with a space | 0.44 | 0.10 | 4 times |

(An empty line here is a line with nothing or only spaces in it. The 0.002 and 1.6 above
counted two line ends with nothing between them.) The exam's sources give paragraphs,
indented code and tables as their authors typed them; Common Crawl's text has every run of
white space folded to one space or one line end.

*The teacher's text against a copy of the same pages.* Writing, in bits per byte: an arm
less `tidy` in the same order, for the first, second and third order it was trained in; its
twin the same; and the teacher's text less the twin, in standard errors of that difference.
On the exam with its white space as the crawl's:

| Arm | The teacher's text | The same pages copied whole | The teacher's less the copy |
|---|---|---|---|
| `rewrite-1` | +0.013, −0.010, +0.018 | +0.005, −0.013, +0.008 | +0.008 (2.9), +0.003 (0.6), +0.011 (2.6) |
| `rewrite-5` | +0.007, −0.001 | +0.017, +0.002 | −0.010 (−4.2), −0.003 (−0.8) |
| `rewrite-15` | +0.014, −0.000 | +0.020, +0.017 | −0.006 (−2.0), −0.018 (−4.3) |
| `rescue-5` | +0.012, +0.001, +0.008 | +0.017, +0.000, +0.013 | −0.005 (−2.1), +0.001 (0.3), −0.005 (−1.2) |
| `rescue-15` | +0.021, +0.003, +0.025 | +0.022, +0.010, +0.022 | −0.001 (−0.3), −0.008 (−1.8), +0.003 (0.7) |

Over the thirteen pairs the teacher's text is 0.002 below its copy, and above it in five.
On the exam as registered, with only the empty lines closed, it is 0.005 below and above in
four. On the exam as it was, it is 0.008 above and above in twelve of thirteen.

*The rule.* Three conditions: above `tidy` by twice the standard error; above it by more
than the widest gap between two arrangements of `tidy`'s own lines; and above its twin by
twice the standard error. The widest gap is 0.029 on the exam as it was, 0.024 with the
empty lines closed, 0.023 with the crawl's white space.

- *On the exam as it was*, two arms pass all three: `rewrite-15` in the first order (+0.031
  over `tidy`, +0.008 over its twin, 2.5 standard errors) and `rewrite-1` in the third
  (+0.032, and +0.021 over its twin, 4.7). Had the exam been left as it was, this section
  would be proposing a track for written text.
- *With the empty lines closed*, the exam the verdict was registered on, only `rescue-15`
  clears the widest gap (+0.027 and +0.028 in two orders of three), and its twin is 0.008
  to 0.016 *above* it. By the rule the gain is the pages', not the writing's.
- *With the crawl's white space*, only `rescue-15` clears it, once (+0.025), and its twin is
  level (+0.003, 0.7 standard errors).

`layout-1`, which has no teacher's text and an empty line after every line of 396 pages,
gains 0.013 over `tidy` on the exam as it was, half of what `rewrite-1` gains there in the
same order (0.026), and 0.004 and 0.001 on the two corrected exams. The prediction was "about
what `rewrite-1` gains"; it is half, and the rest of `rewrite-1`'s lead also goes when the
marks do.

*How far each arm fell short.* One dose is ahead of its copy on the corrected exam: at 1%
the teacher's text is above its twin in all three orders, by 0.003 to 0.011. At 5% and at
15% the rewritten pages are below their copies in all four pairs, by 0.003 to 0.018. A gain
that is there at the smallest dose and turns to a loss as the dose grows is what a mark
does (a little of it teaches all there is to learn) and not what better writing does. What
mark is left after the white space is not known; it is worth about 0.007, and no arm built
on it clears the gap between two orders of `tidy`. The rescued pages, rewritten, do about
as well as the same pages copied (six pairs: −0.005, +0.001, −0.005, −0.001, −0.008,
+0.003). So the teacher's time bought nothing that copying the page does not, and
at 15% in place it cost 0.006 to 0.018.

*What the twins found instead.* The twins hold no written text: they are `tidy` with some
pages copied whole, in place of its lines of them or of its lowest-ranked pages. They beat
`tidy` in the same order in twelve of thirteen comparisons, by up to 0.022. That is the rule's other branch, "a finding about
`tidy` and not about writing", and 9.10 follows it.

*The order of a script.* Ten arrangements of `tidy`'s lines, one student each but the first
(five), writing on the exam with the crawl's white space:

| `tidy` | seed 1 | seed 2 | seed 3 | trained lines reordered | keyed order 1 | keyed order 2 | less one page | yesterday's trained lines reordered | yesterday's script |
|---|---|---|---|---|---|---|---|---|---|
| +0.019 | +0.021 | +0.021 | +0.026 | +0.023 | +0.027 | +0.018 | +0.022 | +0.019 | +0.041 |

Nine of them differ from each other as two trainings of one script do (0.0032 against
0.0029). Yesterday's order stands 0.019 above their mean, six times that, and falls back
among the others when its trained lines are put in another order. So what was written above on the
morning of the pilot, that the control fell 0.017 "because of the script", was right that it
is the order and wrong in what it suggested: most orders agree, and one does not. Why that
one is better is not known. It is not its pages (the same lines reordered are ordinary),
not the code that builds a script (old and new build the same text and the same tokens),
not a damaged page it holds (taking the one garbled page out moves nothing), not a spike in
the training loss, and not what the student read last (it survives averaging the last
weights).

The data, every script and the teacher's text are kept in
`/home/ericflo/stet-measurements/s2-decision/pilot` (1.4 GB): each student's score on every
item of each of the three exams, the 53 scripts, the teacher's 232 blocks of answers, the
code as it ran, and `tables.txt`.

### 9.10 What the pilot changed

Written on 22 November 2026, after 9.9's result. Each of these is a change to the season
made on the pilot's evidence, which was gathered for another question; each says what it
rests on.

**The exam's writing is laid out as the crawl lays out a page.** Before an item is made
from a harvested document, every run of spaces, tabs and no-break spaces becomes one space
and every run of line ends, with or without spaces between them, one line end
(`crawl_space`, in `exam/fresh/select.py`). Nothing a student is asked to predict in a
passage is now a mark its training text cannot hold. The cost is that indented code on the
exam loses its indentation, as it does in every page of the crawl.

What the fault did while it was there: a student paid about twenty bits for an empty line
it had never seen, and a curator's place depended on whether its script happened to show
the student one. A script that cut pages to their lines of prose never did, and `tidy`'s
writing score rises by 0.023 against the unedited corpus's when the marks go (−0.004 to
+0.019). A script of copies could also have made empty lines on purpose, by joining two
pieces at a line end, and bought about what `layout-1` did. The generated items are left
as they are: closing the empty lines in their contexts moved no skill pillar by more than
0.002.

9.7's students sat the exam as it was, and their weights were not kept. Its verdicts on the
kinds of writing stand, since those passed by 24 to 93 times the noise and the fault is
worth 0.02. Its table of the six curators would move for the two that cut lines (`tidy`
and `scribe`, up by about 0.02 against the others); by that estimate no two curators
change places.

**Two parts that passed 9.7 are taken off: transformation, and lookups in structured
data.** 9.7's noise was the spread between trainings of one script. A round gives a curator
one script, and the order that script happens to be in is luck of the same kind (9.9), so
the noise a part has to clear is the spread between trainings of one curator's pages in
different arrangements. The pilot has eighteen of those in three groups (`tidy`'s lines ten
ways, the same pages whole four ways, the raw pages four ways), which gives the spread
within a group fifteen degrees of freedom. 9.7's rule, unchanged but for that noise, on
9.7's spreads between the six curators:

| Part | Spread over the noise between trainings of one script | Spread over the noise between arrangements | |
|---|---|---|---|
| News | 93 | 47 | in |
| Discussion; social posts | 47; 38 | 29; 20 | in |
| Reference | 61 | 25 | in |
| Technical writing and code | 25 | 3.5 | in |
| Logic | 12 | 6.2 | in |
| Reading | 5.4 | 5.5 | in |
| Agent sessions | 5.7 | 4.0 | in |
| Knowledge | 5.7 | 3.3 | in |
| Grammar | 4.9 | 3.2 | in |
| An algorithm's output, after examples | 3.8 | 2.3 | in |
| Arithmetic; word problems; algebra; units, after examples | 3.8; 2.6; 1.6; 2.3 | 1.9; 1.8; 1.7; 1.6 | in |
| Transformation | 2.1 | 1.3 | **out** |
| Structured data, after examples | 1.6 | 0.8 | **out** |

On transformation one script in another order moves by 0.062 bits per byte, and the six
curators stand 0.081 apart: taking the luck out of that spread leaves curators about as far
apart as one of them is from itself. On structured data they are closer than that. Both
generators are deleted. Technical writing's 3.5 is one raw-page arrangement that scored
0.15 above the other three; without the raw pages it is 8.9.

This is a rule's verdict changed after its results were in, and it is said here as that.
The change takes parts off and puts none on; the threshold is the one written before; what
changed is the noise, to the one a round really has. With that noise a part on which
curators did not differ would show about 0.9 where 9.7 reckoned 0.6, so 1.5 is less clear
of nothing than it was meant to be, and the four parts of *numbers* pass narrowly.

The score is now four kinds of writing and seven skills (agents, reading, logic, grammar,
numbers, programs, knowledge), programs being an algorithm's output alone. On 9.7's
students: 0.0036 between two trainings of one script; the six curators 0.165 apart; in the
same order at standard and at large, and with the unedited corpus as a seventh the two
orders agree at 0.89.

| | mimic | fineweb2 | scribe | tidy | unedited | english | dclm |
|---|---|---|---|---|---|---|---|
| standard | +0.100 | +0.050 | +0.006 | +0.002 | 0 | −0.239 | −0.301 |
| large | +0.077 | +0.045 | −0.017 | −0.025 | 0 | −0.280 | −0.326 |

Between two arrangements of one curator's pages the score moves by 0.008 (0.0084 over
`tidy`'s ten, 0.0056 over the whole pages' four), twice what training again does, so the
six curators are about twenty of those apart and not forty-five. The season's file carries
both numbers (`noise.standard.gain_sd` and `order_sd`), and a practice run prints both.

*Ink, read again.* Transformation was one of the two skills `scribe`'s text bought. On
this score `scribe` is above `tidy` by 0.0040, 1.3 standard errors, where 9.7 gave 1.8 on
the score before this and 2.6 on the one it was aimed at. Those standard errors are of the
noise between trainings of one script; the two scripts share their order, so the order's
luck is mostly common to them, but not wholly. So that a script holds no text of its own
rests on 9.7's three reasons and on 9.9, which found that text written for the purpose
teaches no more than the pages it was written from; it does not rest on the letter of
9.7's third test.

**`tidy` copies its pages whole.** `tidy` kept only the lines of a page that read as
sentences. The same pages in the same order, each copied whole, were trained in four
orders against `tidy` in the same four:

| | first order | keyed order 1 | keyed order 2 | seed 1 |
|---|---|---|---|---|
| Writing, whole pages less `tidy`'s lines | +0.025 | +0.018 | +0.029 | +0.034 |
| The score, the same | +0.025 | +0.021 | +0.032 | +0.035 |

Nearly all of it is technical writing and code (−0.13 for the lines, −0.03 for the pages,
against the unedited corpus): a line of code is not a sentence, and `tidy` cut it. On
reference and news the lines are 0.003 ahead, and on conversation 0.009 behind. Two things
tried beside it did worse than `tidy` as it now is: only as many of its top-ranked pages as
fill the budget, whole (0.011 and 0.014 lower on writing in two orders), and the same of
its lines (0.015 lower in one). `tidy` chooses a quarter more than the budget holds, and
the student reads a shuffled part of that.

**The starter is not said to beat the raw pages.** Its test for a page that reads as
writing keeps 314,649 of the round's 337,313 pages, so its script is the raw pages in a
shuffle. Three shuffles of it scored −0.008, +0.002 and +0.034 against the crawl's own
order. The site said it "already beats the raw pages"; it now says that it runs, and that
the gain is the entrant's to find.

**Those three shuffles are the largest thing the pilot found and did not look for.** A
student reads a sixth of the raw pages. Which sixth, and in what order, is worth up to 0.04
bits per byte, more than `tidy`'s whole gain over them, and a round measures every curator
against one such draw: its four
baseline trainings are four trainings of the crawl's pages in the crawl's order. The four
are not changed into four orders. A lucky or unlucky baseline moves every curator of the
round by the same amount, and the board fits a difficulty to each round and takes exactly
that out (`fitRatings` in `server/rating.go`), so no curator's place depends on it. What it does move is the gain a
round's page shows, and "worth N times the raw data" read from one round. That is a reason
to read both over many rounds, and the site's glossary says so under *gain*.

**What is still not known.** Why one order of `tidy`'s lines is 0.02 better than nine
others, and whether a curator could find such an order on purpose. If one could, ordering
would be worth more than most filters measured here, and it is allowed: a script's order
is the curator's. And what mark, if any, the teacher's text still shares with the exam.

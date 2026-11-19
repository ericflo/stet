# What the exam should measure, from first principles

Stet exists to find the best pretraining data there is. This note asks what an exam must
be to point the competition at that, and what that means for Stet. It builds on
[the exam proposal](exam-proposal.md) and on what the first shadow runs measured.

## The answer, in one paragraph

A pretraining dataset is good when the model trained on it, **at the sizes people actually
train**, becomes capable at **what pretrained models are for**. So the exam should be:

- a public, broad, declared mix of capabilities, scored by how likely the model finds good
  answers and good writing, with the individual items unknowable in advance;
- trusted only where it reproduces what is already known about data at scale. The reference
  curators that rebuild published recipes must come out in the order the literature found;
- decided at the largest scale we can afford. Micro rounds become a fast proxy, and the
  proxy is checked against larger runs.

Targeting such an exam is not cheating. It is the job.

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

Neither condition holds for a real model. The next exam, which is broad, already reverses
the order. `english` keeps every English page and beats compass on code, programs, other
languages and fresh text, by 0.14 to 0.22 in all four trainings of round 12. The fault was
never that compass targeted the exam. The exam was a narrow, unfaithful picture of the
purpose.

## 3. What the exam must have, in order of importance

1. **The right target.** A faithful, broad picture of what pretrained models are for.
2. **The right scale.** Its verdict must hold for models people train.
3. **Items nobody can know.** No answer can be memorised or copied in.
4. **Precision.** Differences between curators must stand clear of noise.

The first two are about validity, and they matter most. The proposal handled the third and
fourth well, and the shadow runs showed what was missing from the first two.

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

- **Rounds stay at micro.** They are fast, cheap, and give the steady feedback a competition
  needs.
- **A pilot, now.** The organisers' curators and the reference curators are run at micro, at
  standard, and at a scale about ten times larger, each sitting both exams. The question is
  how far micro's order holds as the model grows. That says how much the round scores can be
  trusted.
- **Finals, each season.** The top curators are run on a much larger pool and train a model
  of about 150M parameters on a private exam of the same design. On the organisers' rented
  machines that is a few dollars a curator. Finals decide the season.
- **Calibration.** For each pillar: does its micro score predict the finals? Pillars that do
  get more weight in next season's rounds.

### 4.4 Precision

- **A stable reference.** The unedited corpus is the least stable student we train: its
  season-exam loss varies by 0.046 nats between trainings, against 0.004 to 0.015 for
  curated corpora. Every gain is measured from it. Average it over four trainings, or
  measure from a reference curator that trains steadily.
- **Pairs.** A pair of curators varies by about 0.03 from training alone, and 0.015 to 0.02
  from the draw of items. Report both. Train the close ones twice.
- **Items nobody can know.** Fresh text from after the draw, items generated from the draw's
  seed, private families for finals, decontamination against the pool, and a check of every
  item against each script's own literal text.

## 5. What has been measured so far

- **The two exams disagree.** On round 12:
  - Season exam, as officially graded: compass v1, compass v2, compass-clf-nudge,
    compass-clf, compass-linear, tidy, english.
  - Next exam, mean of four trainings: english first, by 0.13 or more over every other;
    compass v1 last; the rest within noise of each other.
- **The two halves of the next exam agree on the top, not below it.** Generated items and
  fresh text both put english first. Below it they differ: tidy and compass-linear do
  better on fresh text than on generated items.
- **Training noise** is in [the proposal, section 3](exam-proposal.md#3-stability-pairing-continuous-scores-and-a-robust-average).

## 6. Open questions

- Likelihood of answers at 150M is validated against benchmarks, not against what a model
  becomes after post-training.
- How much each pillar should count is a value choice, not a measurement.
- A fine-tuning probe deserves a place in calibration: how quickly a trained student
  learns a small new task. Pretraining is for a model that learns well afterwards.

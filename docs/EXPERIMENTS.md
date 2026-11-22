# Experiments

The measurements the rules rest on, and what is still to be measured. Each says what was
run, what came out, and what it decided. Scores are in bits per byte; higher is better. Each
says which exam it was measured on: Season 1's (seven capabilities, each counting the same)
or Season 2's ([the rules](RULES.md#the-exam)). The reasoning behind the exam, and the pilot
that chose the student's size, are in [the research](research/exam-first-principles.md).

## How much is noise?

GPU arithmetic is not exactly repeatable, and the differences grow over training: the same
script, trained twice from the same initial weights, does not give the same student. Nine
curators on one official round's corpus were trained again on rented NVIDIA GPUs, three times
at micro and twice at standard, and each training sat the exam. On Season 1's exam:

| | micro | standard |
|---|---|---|
| A curator's score, sd between trainings of the same script | 0.025 | 0.013 |
| Spread between curators | 0.095 | 0.077 |
| Signal to noise | 3.8 | 5.8 |
| Rank agreement with the large student (Spearman) | 0.69 | 0.95 |

The unedited corpus trains least steadily of all.

**What it decided.** Rounds train the standard student: it is steadier, and it orders
curators as the large one does, where micro does not
([why](research/exam-first-principles.md#75-scale-rounds-at-the-standard-student-finals-at-the-large-one-decide-the-season)).

**The pilot, scored other ways.** The scale pilot
([research](research/exam-first-principles.md#6-the-scale-pilot-round-12-november-2026):
nine curators, three sizes of student, repeated trainings) kept every item's score, so it
could be scored again under any way of adding the items up, with nothing trained again.
The rule was fixed before the numbers: take the one with the best spread over noise at the
official size, provided it agrees with the large student no worse than Season 1's.

| How the items are added up | Noise: sd between two trainings of one script | Spread between curators, over noise | Rank agreement with the large student |
|---|---|---|---|
| Season 1's | 0.0196 | 3.4 | 0.90 |
| Bits pooled within a stratum | 0.0157 | 4.0 | 0.95 |
| Square-root weights | 0.0172 | 3.8 | 0.98 |
| No item's bits put over fewer than 16 bytes | 0.0097 | 6.0 | 0.90 |
| Fresh writing only | 0.0049 | 28.0 | 0.88 |
| Generated items only | 0.0225 | 2.8 | 0.93 |
| **Season 2's: the 16-byte floor, writing three quarters, the skills that pass one quarter, small languages pooled** | **0.0052** | **21.7** | **0.95** |

Part by part, on the same pilot:

| Part | Items | Spread over noise | Order carries to the large student |
|---|---|---|---|
| Discussion (fresh) | 74 | 23.5 | 0.95 |
| Reference (fresh) | 729 | 16.3 | 0.88 |
| Agents | 32 | 4.5 | 0.90 |
| Transformation | 28 | 3.4 | 0.90 |
| Grammar | 26 | 2.2 | 0.81 |
| Reading | 72 | 1.6 | 0.71 |
| Logic | 44 | 1.6 | 0.57 |
| Facts | 200 | 2.1 | 0.57 |
| Algorithms, algebra, word problems | | 1.6 to 2.6 | 0.17 to 0.43 |
| Arithmetic, multi-step, units, structure, programs | | 0.8 to 1.8 | at chance at every size |

**The pilot's fresh writing was Wikipedia only**: new articles, additions to old ones, and
talk pages. News, technical writing and posts were not in it. They are taken to behave as
the same kind of thing, long passages of human text, and that is checked as the season
runs. And with eight curators, most of them English-only filters that the writing nearly
ties, rank agreement cannot tell a writing share of 67% from one of 90%: three quarters is
a declared value, not a measured one.

**What it decided.** Season 2's score: the 16-byte floor, writing three quarters and
skills one quarter, and a watch list. A family of questions counts if curators spread on
it by more than 1.5 times its noise, its order carries to the large student at 0.55 or
better, and it is not at chance at the official size: agents, reading, transformation,
logic and grammar. Facts pass narrowly and are watched all the same: they come from a
fixed public list, so a curator could go looking for the pages that state them. Fresh
writing alone is steadier still, and carries a little less well than the mix, so the
skills stay. The season takes two trainings of one script to differ by 0.005 at standard
and 0.013 at micro (`noise` in its file). At the official scale two grades of a script
must agree within 0.025, a measurement of the unedited baseline within 0.04 of the
others', and two runs within 0.04. All three tolerances are *provisional* until grades
from many machines, of both makes, can be compared: every round's page publishes each
grade's difference from the others.

## Where a round's training went

In Season 1 every grading machine trained the unedited corpus for itself each round, and
all seven reference curators ran in every round. A third of the season's training was
baselines, and half was the references.

**What it decided.** Four established graders measure a round's baseline, and its score on
each item is the mean of theirs: the mean of every grader's is no better a baseline than
the mean of a few, and it is still several, so that one unsteady training of the unedited
corpus does not move every gain in the round. Two references, `fineweb2` and `dclm`, run
in every round and tie the rounds together; each of the others runs in one round in eight.

## What the record rule can tell apart

The record changes hands on a one-sided paired test against a margin of 0.005, at looks after
5, 10, 20 and 40 shared rounds ([the rule](RULES.md#the-record)). How often does a
challenger with a given true lead over the holder end up with the record? It depends on how
much two curators' gains differ from round to round, which is not measured yet, so here it is
at four guesses. The season takes one curator's gain to move by 0.015 from round to round
(`noise.standard.round_sd`: Season 1's board, scaled by the pilot), which puts the
difference of two near 0.02 if they move independently.

| Spread between rounds \ true lead | 0 (a copy) | 0.005 | 0.01 | 0.015 | 0.02 | 0.03 | 0.04 |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.1% | 4% | 82% | 100% | 100% | 100% | 100% |
| 0.02 | 0.6% | 4% | 30% | 82% | 99% | 100% | 100% |
| 0.03 | 1.0% | 4% | 17% | 48% | 82% | 100% | 100% |
| 0.05 | 1.6% | 4% | 10% | 22% | 40% | 82% | 98% |

```
uv run --no-project --with numpy --with scipy python tools/record_power.py --sd 0.02 --margin 0.005 --leads 0 0.005 0.01 0.015 0.02 0.03 0.04
```

**What it decided.** A lead equal to the margin takes the record about one time in
twenty-five, inside the 5% the rule allows. A copy of the holder takes it once in a hundred
tries or less while rounds differ by 0.03 or less, and 1.6 times in a hundred if they
differ by 0.05. A lead of 0.02, four times the margin, is confirmed nearly always if rounds
differ by 0.02 or less, and four times in ten if they differ by 0.05. The margin is
*provisional* until the first rounds measure the spread between them.

## How long a round takes

| Step | When |
|---|---|
| The draw. Its seed picks three of the hours already harvested | 0 |
| The examiner builds the round's corpus, makes the exam from those hours' harvests and posts it. It reads no source | From the draw |
| Curators run, two runs each of up to 1,200 seconds | From the draw, on the machines around |
| Grades: one standard training each, about 18 minutes on a rented RTX 3090 or 4070 Ti, under $0.10 | From when a curator has written a script |
| Each grader sits the exam on its trained students | Once the exam is posted |
| The round settles | When its runs and its grades are in |

A large student, for the finals, trains in about an hour on an RTX 4090.

Season 1 harvested the hour after each draw and then waited an hour more for the sources
that publish late. Its exam was posted about 2 h 15 m after the draw and its rounds settled
at about 2 h 20 m, while the machines that had trained sat idle.

**What it decided.** The harvest moved before the draw: every hour of the clock is
harvested an hour after it ends, a draw picks three of the 26 before it, and no round waits
for its exam.

## What a harvest gives

On Season 1's sources, a live harvest of a two-hour window took 26 minutes and gave 1,683
documents. Chosen for an exam, they made 901 items:

| Kind of writing | Items | Sources (items) | Languages |
|---|---|---|---|
| Reference | 255 | Wikipedia additions 151, new articles 104 | 36, 29% English |
| Discussion | 255 | Wikipedia talk 70, Bluesky replies 65, Lemmy 47, Mastodon replies 39, Hacker News 34 | 20, 65% English |
| News | 140 | Common Crawl's news crawl 140, from 110 sites | 39, 45% English |
| Social | 129 | Bluesky 74, Mastodon 55 | 13, 53% English |
| Technical | 122 | GitHub 122 | 12, 72% English |

News, social and technical had fewer candidates than their part, and gave the rest to the
other two. GitHub's comments are left out: about half of those written now come from coding
agents under people's own accounts.

**One hour is enough.** Round s1-o00003's window (20 November, from 11:40 UTC), harvested
again with every source read at once: the two hours took 645 seconds and gave 1,592
documents; its first hour took 629 seconds and gave 1,156. Chosen for an exam, both made 900
items, in nearly the same parts (discussion 257 and 267, reference 257 and 267, news 143 and
141, social 126 and 111, technical 118 and 114), the hour in 70 languages to the two hours' 51.
Each source still gave most of its share in the hour (Hacker News 97 of 150, Lemmy 107,
Mastodon 83), and the parts are kept equal by the choosing, not by the harvest.

**What it decided.** A harvest is of one hour of the clock. A Season 2 exam is made from
three of them.

**Three more sources, and one left out.** Forums that run Discourse, 31 of them, give about
100 replies an hour. The kernel's mailing lists give first versions of patches: fresh code
written by people. Crossref gives about 40 new works' abstracts an hour. Stack Exchange was
tried and left out: four questions an hour. Science abstracts are a kind of writing the
pilot never measured, so they are watched: 60 items an exam, counting for nothing until
they pass the season's test.

## What a small model can read in twenty minutes

A curator that uses a language model is limited by the clock before it is limited by
memory. Measured on the reference machine (Radeon 8060S, ROCm, PyTorch 2.15) with nothing
else running, half precision, batches of 128 passages of 128 tokens, one forward pass:

| Model | Parameters | Tokens read per second | In 19 minutes |
|---|---|---|---|
| MiniLM-L6 (encoder) | 22M | 669,000 | 763M |
| ModernBERT-base (encoder) | 149M | 76,000 | 87M |
| SmolLM2-135M | 135M | 65,000 | 74M |
| SmolLM2-360M | 362M | 29,000 | 33M |
| Qwen3-0.6B | 596M | 15,800 | 18M |
| Qwen2.5-1.5B | 1,544M | 7,900 | 9.0M |
| SmolLM2-1.7B | 1,711M | 6,300 | 7.2M |

A standard round's corpus is about 2.5 GB in 335,000 documents. So a small encoder can
read most of it, a 0.6-billion-parameter model the opening of the tens of thousands of
documents cheaper code liked best, and nothing larger much at all. Writing is far slower
than reading, 350 to 1,250 tokens a second in batches of 16, and the ink allowance leaves
almost nothing for written text to do.

That machine is about 1.2 times the floor for running curators
(`runner.gpu_tokens_per_second` in the season file is this measurement's cousin: a fixed
110M-parameter model, 80,000 tokens a second).

**What it decided.** A curator is given 8 GB of GPU memory, and model files may total
4 GiB: enough for any model in this table.

## Still to measure

- Noise and spread, part by part, on writing that is not Wikipedia's: news, technical
  writing, forums and posts. The pilot had none of it. Every settled round publishes every
  item's score, so the data arrives by itself.
- How far two curators' gains differ from round to round: what the record's margin should
  be set from.
- Grades of the same scripts on NVIDIA and AMD machines: whether the tolerances hold across
  makes. The data arrives by itself, on every round's page.
- The unedited corpus's score against training tokens, on this exam: the curve a multiplier
  would be read from.
- For each watched part, over at least twenty settled rounds: whether curators spread on it
  by more than 1.5 times its noise, and whether its order agrees with the counted score's
  at 0.5 or better. A part that does earns a place in the next rules.
- What share the writing should have. The pilot cannot tell two thirds from nine tenths.
- The finals' check one size up: whether the order holds at about 150M parameters.
- How long a round takes from draw to settlement, now that it does not wait for its exam.
- What shipped text inside the 0.01% ink allowance is worth on this exam.

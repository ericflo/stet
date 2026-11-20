# Experiments

The measurements the rules rest on, and what is still to be measured. Each says what was
run, what came out, and what it decided. Scores are in bits per byte; higher is better. The
reasoning behind the exam, and the pilot that chose the student's size, are in
[the research](research/exam-first-principles.md).

## How much is noise?

GPU arithmetic is not exactly repeatable, and the differences grow over training: the same
script, trained twice from the same initial weights, does not give the same student. Nine
curators on one official round's corpus were trained again on rented NVIDIA GPUs, three times
at micro and twice at standard, and each training sat the exam:

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
The tolerances are first values set from the standard figure: two grades of a script agree
within 0.06, a grader's baseline within 0.1 of the others', and two runs within 0.09. The
round's baseline is the mean of every established grader's, item by item, so that one
unsteady training of the unedited corpus does not move every gain in the round. All three
tolerances are *provisional* until grades from many machines, of both makes, can be
compared: every round's page publishes each grade's difference from the others.

## What the record rule can tell apart

The record changes hands on a one-sided paired test against a margin of 0.01, at looks after
5, 10, 20 and 40 shared rounds ([the rule](RULES.md#the-record)). How often does a
challenger with a given true lead over the holder end up with the record? It depends on how
much two curators' gains differ from round to round, which is not measured yet, so here it is
at three guesses:

| Spread between rounds \ true lead | 0 (a copy) | 0.01 | 0.02 | 0.03 | 0.04 | 0.06 |
|---|---|---|---|---|---|---|
| 0.02 | 0.1% | 4% | 82% | 100% | 100% | 100% |
| 0.03 | 0.3% | 4% | 48% | 97% | 100% | 100% |
| 0.05 | 0.8% | 4% | 22% | 63% | 93% | 100% |

```
uv run --no-project --with numpy --with scipy python tools/record_power.py --sd 0.03 --margin 0.01 --leads 0 0.01 0.02 0.03 0.04 0.06
```

**What it decided.** A copy of the holder takes the record less than once in a hundred
tries, and a lead equal to the margin about one time in twenty-five, inside the 5% the rule
allows. A lead three times the margin is confirmed nearly always if rounds differ by 0.03
or less, and about two times in three if they differ by 0.05. The margin is *provisional*
until the first rounds measure the spread between them.

## How long a round takes

| Step | When |
|---|---|
| The draw | 0 |
| Curators run, two runs each of up to 1,200 seconds | Within the first hour or so, on the machines around |
| The harvest window closes | 1 h |
| The sources that publish late have caught up | 2 h |
| The examiner harvests every source at once (about ten minutes), checks the corpus and posts the exam | About 2 h 15 m |
| Grades: one standard training each, about 18 minutes on a rented RTX 3090 or 4070 Ti, under $0.10 | Half an hour or so, a machine for each |
| The round settles | About 3 h |

A large student, for the finals, trains in about an hour on an RTX 4090.

## What a harvest gives

A live harvest of a two-hour window took 26 minutes and gave 1,683 documents. Chosen for an
exam, they made 901 items:

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
Mastodon 83), and the parts are kept equal by the choosing, not by the harvest. So the window
is an hour from s1-o00004 on, and a round's exam is made two hours after its draw, not three.

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

- How far two curators' gains differ from round to round: what the record's margin should
  be set from.
- Grades of the same scripts on NVIDIA and AMD machines: whether the tolerances hold across
  makes. The data arrives by itself, on every round's page.
- The unedited corpus's score against training tokens, on this exam: the curve a multiplier
  would be read from.
- For each pillar, whether its order at standard carries over to the finals' large student.
  A pillar that does not gets more items or a bigger student, not less weight.
- The finals' check one size up: whether the order holds at about 150M parameters.
- What shipped text inside the 0.1% ink allowance is worth on this exam.

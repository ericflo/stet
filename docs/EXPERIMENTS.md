# Experiments

The measurements the rules rest on. Each one says what was run, what came out, and what
it decided. Numbers marked *running* are being measured now and are not yet used.

Everything here was run on the reference machine (Radeon 8060S, ROCm, PyTorch 2.15) with
`tools/exp.py`, on practice rounds, which anyone can rebuild. The command is under each
table. Losses are exam losses in nats per token; lower is better.

## The reference editors

Three editors ship with Stet. `identity` changes nothing. `english` keeps documents that
look like English prose. `tidy` also cuts navigation and boilerplate lines out of each
page, drops duplicates and ranks what is left.

Nano scale (4 layers × 192, 8M tokens), five practice rounds:

| Round | identity | english | tidy | english gain | tidy gain |
|---|---|---|---|---|---|
| s0-p00000 | 6.382 | 5.669 | 5.552 | +0.713 | +0.830 |
| s0-p00001 | 6.330 | 5.728 | 5.515 | +0.603 | +0.815 |
| s0-p00002 | 6.492 | 5.704 | 5.603 | +0.788 | +0.889 |
| s0-p00003 | 6.415 | 5.639 | 5.535 | +0.776 | +0.880 |
| s0-p00004 | 6.335 | 5.739 | 5.598 | +0.597 | +0.737 |
| **mean** | | | | **+0.695** | **+0.830** |

Official scale (micro: 6 layers × 384, 40M tokens), three practice rounds and the first
official one:

| Round | identity | english | tidy | english gain | tidy gain | tidy − english |
|---|---|---|---|---|---|---|
| s0-p00000 | 5.355 | 4.418 | 4.389 | +0.937 | +0.966 | +0.029 |
| s0-p00001 | 5.399 | 4.456 | 4.403 | +0.942 | +0.996 | +0.054 |
| s0-p00002 | 5.354 | 4.437 | 4.473 | +0.917 | +0.882 | −0.036 |
| s0-o00001 | 5.342 | 4.430 | 4.410 | +0.912 | +0.933 | +0.020 |
| **mean** | | | | **+0.927** | **+0.944** | **+0.017** |

`tidy`'s figure on s0-p00000 is the mean of three gradings of its output in shuffled
order (4.388, 4.400, 4.378).

```
python tools/exp.py --scale nano --rounds 0 1 2 3 4 --editors identity english tidy
python tools/exp.py --scale micro --rounds 1 2 --editors identity english tidy
```

**What it decided.** A few lines of filtering are worth 0.7 to 1.0 nats, so there is room
to compete. The gap between `english` and `tidy` nearly closes at micro: 0.017 on average
against 0.135 at nano, and on one round in four `english` is ahead. A bigger student is
less hurt by boilerplate, and the easy wins shrink. At the official scale the two
reference editors cannot be told apart in four rounds, which is the rule working: neither
would take the record from the other. Beating both by more than the margin is the
season's first target.

An earlier version of this page gave `tidy` 5.513 at nano and 4.375 at micro on round
s0-p00000. Those were measured with an earlier version of the editor, before a change to
how many bytes it keeps. Every `tidy` number on this page is now from the editor in
`examples/tidy` as published.

## Is the student learning anything?

A check that the exam measures more than word counts: a trigram count table built from
the same training stream sits the same exam.

| Scale, editor | Student | Trigram table |
|---|---|---|
| nano, identity | 6.382 | 6.435 |
| nano, tidy *(earlier version)* | 5.513 | 5.804 |
| micro, identity | 5.355 | 6.003 |
| micro, tidy *(earlier version)* | 4.375 | 5.501 |

```
python tools/exp.py --scale micro --rounds 0 --table
```

**What it decided.** At nano, a student trained on the unedited crawl is barely better
than a lookup table. Nano is for testing that an editor runs; it is not for ranking. At
micro the student is 0.65 to 1.1 nats ahead of the table, which is why micro is the
official scale.

## How much data is an edit worth?

The same unedited corpus, trained for more or fewer tokens. This is the curve the
leaderboard's **multiplier** is read from.

Nano, round s0-p00000:

| Tokens | 4M (0.5×) | 8M (1×) | 16M (2×) | 32M (4×) | 64M (8×) |
|---|---|---|---|---|---|
| Unedited loss | 6.670 | 6.382 | 6.030 | 5.519 | 5.198 |

On that round `english` (5.669) matches 3.3× the data and `tidy` (5.552) matches 3.8×.

Micro, round s0-p00000:

| Tokens | 20M (0.5×) | 40M (1×) | 80M (2×) | 160M (4×) | 320M (8×) |
|---|---|---|---|---|---|
| Unedited loss | 6.057 | 5.355 | 4.828 | 4.522 | *running* |

On that round `english` (4.418) and `tidy` (4.389) are both past the last measured point:
each is worth more than four times the data. Until the 8× point is in, the leaderboard
shows gains only and leaves the multiplier blank.

```
python tools/exp.py --scale nano --rounds 0 --editors identity --train-tokens 32000000
```

**What it decided.** The multiplier is read between measured points and never
extrapolated. A gain past the last point is shown as "more than 8×".

## Does the answer depend on the learning rate?

The trainer is fixed, and so is its learning rate. If a different one gave a different
answer, the leaderboard would be measuring the trainer. Micro, round s0-p00000, at half
and twice the season's rate:

| Peak learning rate | 0.00075 (½×) | 0.0015 (the season's) | 0.003 (2×) |
|---|---|---|---|
| Unedited | 5.170 | 5.355 | 5.546 |
| `tidy` | 4.400 | 4.389 | 4.373 |
| `tidy` gain | +0.770 | +0.966 | +1.173 |

```
python tools/exp.py --scale micro --rounds 0 --editors identity tidy --lr 0.00075
```

**What it decided.** Two things, and the second is a limit on what this page can claim.

An edited corpus's loss hardly moves: 0.027 nats across a fourfold change in learning
rate, about two standard deviations of order noise. Editors are ranked against each other
by their losses on the same rounds, so the ranking of editors that clean the corpus does
not hang on this choice.

The unedited corpus's loss moves a great deal, 0.38 nats, and it is better at half the
season's rate. So a *gain* is a gain over a baseline that this trainer does not train as
well as it could, and part of every gain here is the edited corpus being easier to train
on at this rate. The size of a gain, and the "worth N× the data" figure read from it,
belong to this trainer. A baseline trained at its own best rate would make both smaller:
`tidy`'s gain would be nearer 0.77 than 0.97. The season keeps its rate, because changing
it would change what every settled round measured. The next season should choose the
rate at which the unedited corpus does best.

Only `tidy` was run at the other rates. Whether two editors that are close together keep
their order has not been measured.

## How much is noise?

**The same script on the same machine: none.** Grading is deterministic on one machine.
Two grades of one script on the reference machine agree to the last printed digit.

**The order of the documents: 0.011.** `tidy`'s output, shuffled with four other seeds
and graded on the same round at nano: 5.537, 5.524, 5.549, 5.541, against 5.552 in its
own order. Standard deviation 0.011. At micro, three shuffles gave 4.388, 4.400 and
4.378: also 0.011. An editor's order matters about this much by chance.

**The round: 0.05 to 0.09.** Across the five nano rounds above, the unedited loss has a
standard deviation of 0.066, `english`'s gain 0.092 and `tidy`'s gain 0.061. Compared on
the *same* rounds, `tidy` minus `english` has a standard deviation of 0.046, against 0.11
if the two were measured on different rounds. That is why the record is decided on
shared rounds.

**Two different machines:** not yet measured. It needs a second kind of GPU, which the
reference machine cannot stand in for. As a stand-in, runs that start from initial
weights changed by one part in a million are *running*. Until verifiers on other hardware
have graded the same scripts, the tolerance (0.08 nats at micro) is a guess. Every
grade is published with its difference from the others of the same script, and its
unedited loss with it, so the first rounds with verifiers on other GPUs are the
measurement. Something does ride on the guess now: grades further apart than the
tolerance call for a third. Too tight costs extra grades; too loose lets a wrong one
through. Both show on the round pages.

```
python tools/exp.py --scale nano --rounds 0 --editors tidy --reseed 1
```

## What the record rule can tell apart

The record changes hands on a one-sided paired test against a margin, at fixed looks
([the rule](RULES.md#the-record)). Simulated as the server applies it, with the paired
standard deviation measured above (0.046) and the nano margin (0.03): how often does a
challenger with a given true lead over the holder end up with the record?

| True lead, nats | 0 (a copy) | 0.03 | 0.04 | 0.05 | 0.06 | 0.08 | 0.135 |
|---|---|---|---|---|---|---|---|
| Looks at 5, 10, 20 rounds | 0.0% | 4% | 16% | 44% | 76% | 99.4% | 100% |
| Looks at 5, 10, 20, 40 rounds | 0.0% | 4% | 24% | 70% | 97% | 100% | 100% |

0.135 is `tidy`'s measured lead over `english` at nano: it would be confirmed at the first
look about 7 times in 8, and by the second look always.

```
uv run --no-project --with numpy --with scipy python tools/record_power.py --sd 0.046 --margin 0.03
```

**What it decided.** A copy of the holder does not take the record, and a lead equal to
the margin does so 4% of the time, inside the 5% the rule allows. Three looks were too
few: a challenger that really was 0.05 better lost more often than it won. The rule now
has a fourth look at 40 shared rounds (ten days).

At the official scale the paired standard deviation is, so far, the same: `tidy` minus
`english` over the four micro rounds above has a standard deviation of 0.04. With 0.046
and micro's margin of 0.02:

| True lead, nats | 0 (a copy) | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.08 |
|---|---|---|---|---|---|---|---|
| Looks at 5, 10, 20, 40 rounds | 0.2% | 4% | 24% | 70% | 97% | 99.9% | 100% |

```
uv run --no-project --with numpy --with scipy python tools/record_power.py --sd 0.046 --margin 0.02 --leads 0 0.02 0.03 0.04 0.05 0.06 0.08
```

So at the official scale a challenger 0.05 nats better than the holder takes the record
nearly always within ten days, and one 0.03 better takes it one time in four. Four rounds
is a thin estimate of the noise; it is measured again as official rounds accumulate.

## How long a round takes

On the reference machine (an integrated Radeon 8060S), one training run at the official
scale takes about ten minutes. The first official round, s0-o00001, was four training
runs (the unedited corpus and three entries) and three edits of 1, 45 and 56 seconds:
45 minutes from the first job to the last grade. One machine did all of it.

A full round is 17 entries. Done by one machine, once each, that is 18 training runs and
17 edits: about three and a quarter hours when the editors are quick, and six if every
one of them uses its whole ten minutes, which is all the time there is between draws.
Done as the rules now ask, twice by different people with each script graded by someone
else, it is up to twice the work, shared out: with three machines of that speed a full
round of slow editors takes about four hours. Editors that write the same script on both
runs are graded once.

## What a small model can read in ten minutes

An editor that uses a language model is limited by the clock before it is limited by
memory. Measured on the reference machine with nothing else running, PyTorch 2.15 on
ROCm, half precision, batches of 128 passages of 128 tokens, one forward pass:

| Model | Parameters | Tokens read per second | In 9 minutes |
|---|---|---|---|
| MiniLM-L6 (encoder) | 22M | 669,000 | 361M |
| ModernBERT-base (encoder) | 149M | 76,000 | 41M |
| SmolLM2-135M | 135M | 65,000 | 35M |
| SmolLM2-360M | 362M | 29,000 | 16M |
| Qwen3-0.6B | 596M | 15,800 | 8.5M |
| Qwen2.5-1.5B | 1,544M | 7,900 | 4.3M |
| SmolLM2-1.7B | 1,711M | 6,300 | 3.4M |

A micro corpus is about 320 million tokens in 170,000 documents, and `tidy`'s cheap first
pass leaves about 47,000 candidates. So a 0.6-billion-parameter model can read the first
180 tokens of every candidate, a 135M one the first 750, and nothing above 22M can read
the whole corpus. Writing is far slower than reading, 350 to 1,250 tokens a second in
batches of 16, and the ink allowance leaves almost nothing for written text to do.

That machine is about 1.2 times the floor for running editors
(`runner.gpu_tokens_per_second` in the season file is this measurement's cousin: a fixed
110M-parameter model, 80,000 tokens a second).

**What it decided.** The edit limit at the official scale went from 300 to 600 seconds,
an editor is given 8 GB of GPU memory, and model files may total 4 GiB: enough for any
model in this table.

## Does a small language model beat rules?

One trial so far, and the answer was no. `tidy`'s candidates were ranked by a model
instead of by `tidy`'s hand-written score: Qwen3-0.6B, as downloaded, shown the first 480
bytes of each candidate and asked whether it is well-written, informative prose worth
learning from; the rank is how strongly it says yes. Everything else is `tidy`: the same
candidates, the same budget rule, the same shuffle. Micro, round s0-p00000:

| Ranking | Exam loss |
|---|---|
| `tidy`'s own score | 4.355 |
| the model, every candidate judged | 4.360 |
| the model's answer and `tidy`'s score, equal weight | 4.352 |
| the model, as many as it judged in 300 s (12,288), the rest by `tidy`'s score | 4.372 |
| the model, as many as it judged in 600 s (28,672), the rest by `tidy`'s score | 4.377 |

The same documents in another order move the loss by about 0.01, so none of these is a
gain, and the two that ran out of time are losses: they put whatever the model had got
to ahead of everything it had not. Judging all 47,497 candidates took 930 seconds on a
shared GPU, more than the limit.

**What it decided.** Nothing about the rules: this is one model, one question and one
round. It is here so that nobody mistakes "a language model is allowed" for "a language
model wins". In published work the small models that do curate data well were trained
for it, on labels from a much larger model. That is open to an entry (train it
beforehand, name the weights) and has not been tried here.

## What the exam filter catches

A page is removed from the corpus when it contains the first 64 bytes of an exam sentence
that is at least that long and mostly letters. In the public pack, 71.8% of sentences are
long enough to carry such a key, and they hold 90.3% of the exam's text. A short sentence
quoted on its own is not caught; a passage of several sentences nearly always is.

## The held-out exam

The held-out pack is 48,000 passages, 8,000 from each of the six collections. It shares
no document and no sentence opening with the public pack; the builder refuses any
candidate that does.

Pages that quote it stay in the corpus, because removing them would say which passages
it holds. How many are there? In the corpus of round s0-o00001, 47 of 168,080 pages
contain the opening of any held-out sentence, and 2 contain one from the passages that
round drew. Measured on the organisers' machine, without publishing which pages.

## Why so little ink?

The threat: ignore the corpus and ship good text inside the editor. To measure it, `tidy`'s
output was mixed with passages from the same six sources as the exam (ones this round is
not examined on), taking a share of the training budget. Nano, round s0-p00000:

| Share of the budget that is shipped text | 0 | 0.1% | 1% | 10% |
|---|---|---|---|---|
| Exam loss | 5.524 to 5.552 | 5.541 | 5.458 | 5.218 |

```
python tools/exp.py --scale nano --rounds 0 --editors tidy --canned 0.01
```

Micro, the same round, against `tidy` at 4.355:

| Share of the budget that is shipped text | 0.01% | 0.03% | 0.1% |
|---|---|---|---|
| Exam loss | 4.357 | 4.346 | 4.334 |

**What it decided.** At nano, 0.1% of shipped text cannot be told from order noise, and
1% is worth about 0.08 nats, more than half of `tidy`'s lead over `english`. The limit is
0.1%. At micro the same 0.1% is worth about 0.02 nats: twice the order noise, and the
size of the record's margin. It is open to every entry alike, so it moves no ranking,
but it is not nothing, and a lower limit (0.02%) is under consideration for that reason.

## Still to measure

- Micro: the 8× point of the reference curve.
- Two runs of one editor on two machines: how far apart honest runs land, which is what
  the run tolerance (0.12 nats, a guess) should be set from.
- A language model trained for the job, against `tidy`.
- Micro: whether two close editors keep their order at another learning rate, and at
  what rate the unedited corpus trains best.
- Two machines: the real tolerance, from verifiers on NVIDIA hardware. The data arrives
  by itself, on every round's page.
- Whether the held-out exam and the public one rank editors the same way. Known only to
  the organisers until the season closes.
- The first official rounds, which use a different crawl sample every six hours.

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

Official scale (micro: 6 layers × 384, 40M tokens), one practice round so far:

| Round | identity | english | tidy | english gain | tidy gain |
|---|---|---|---|---|---|
| s0-p00000 | 5.355 | 4.418 | 4.389 | +0.937 | +0.966 |

`tidy`'s micro figure is the mean of three gradings of its output in shuffled order
(4.388, 4.400, 4.378). More micro rounds: *running*.

```
python tools/exp.py --scale nano --rounds 0 1 2 3 4 --editors identity english tidy
```

**What it decided.** A few lines of filtering are worth 0.7 to 1.0 nats, so there is room
to compete. The gap between `english` and `tidy` is much smaller at micro than at nano
(0.03 against 0.12 on the same round): a bigger student is less hurt by boilerplate, and
the easy wins shrink. Beating `tidy` at the official scale is the season's first target.

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

Micro: *running*. Until it is measured, the leaderboard shows gains only and leaves the
multiplier blank.

```
python tools/exp.py --scale nano --rounds 0 --editors identity --train-tokens 32000000
```

**What it decided.** The multiplier is read between measured points and never
extrapolated. A gain past the last point is shown as "more than 8×".

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
volunteer's grade is published with its difference from the anchor's, and its unedited
loss with it, so the first rounds with volunteers on other GPUs are the measurement.
Nothing rides on the guess meanwhile: only an anchor's grade is ever a score.

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
has a fourth look at 40 shared rounds (ten days). The margin and the looks for the
official scale will be set the same way once micro's paired noise is measured: *running*.

## How long a round takes

On the anchor, one training run at the official scale takes about ten minutes. The first
official round, s0-o00001, was four runs (the unedited corpus and three entries) and
three edits of 1, 45 and 56 seconds: 45 minutes from the first job to the last grade.

A full round is 17 entries: 18 runs and 17 edits. That is about three and a quarter
hours when the editors are quick, and four and a half if every one of them uses its
whole five minutes. Draws are six hours apart.

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
round drew. Measured on the anchor, without publishing which pages.

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

**What it decided.** At 0.1%, shipped text cannot be told from order noise. At 1% it is
worth about 0.08 nats, more than half of `tidy`'s lead over `english`. The limit is 0.1%.
Micro: *running*.

## Still to measure

- Micro: the reference curve, order noise, more rounds, shipped text at 0.1%.
- Micro: whether the ranking of editors survives a learning rate half or twice as large.
  If it does not, the student is tuned to one kind of data and the scale needs changing.
- Two machines: the real tolerance, from verifiers on NVIDIA hardware. The data arrives
  by itself, on every round's page.
- Whether the held-out exam and the public one rank editors the same way. Known only to
  the anchor until the season closes.
- The first official rounds, which use a different crawl sample every six hours.

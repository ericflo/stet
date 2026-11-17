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
| s0-p00000 | 6.382 | 5.669 | 5.513 | +0.713 | +0.869 |
| s0-p00001 | 6.330 | 5.728 | 5.515 | +0.603 | +0.815 |
| s0-p00002 | 6.492 | 5.704 | 5.603 | +0.788 | +0.889 |
| s0-p00003 | 6.415 | 5.639 | 5.535 | +0.776 | +0.880 |
| s0-p00004 | 6.335 | 5.739 | 5.598 | +0.597 | +0.737 |
| **mean** | | | | **+0.695** | **+0.838** |

Official scale (micro: 6 layers × 384, 40M tokens), one practice round so far:

| Round | identity | english | tidy | english gain | tidy gain |
|---|---|---|---|---|---|
| s0-p00000 | 5.355 | 4.418 | 4.375 | +0.937 | +0.980 |

More micro rounds: *running*.

```
python tools/exp.py --scale nano --rounds 0 1 2 3 4 --editors identity english tidy
```

**What it decided.** A few lines of filtering are worth 0.7 to 0.9 nats, so there is room
to compete. The gap between `english` and `tidy` is much smaller at micro than at nano
(0.04 against 0.16 on the same round): a bigger student is less hurt by boilerplate, and
the easy wins shrink. Beating `tidy` at the official scale is the season's first target.

## Is the student learning anything?

A check that the exam measures more than word counts: a trigram count table built from
the same training stream sits the same exam.

| Scale, editor | Student | Trigram table |
|---|---|---|
| nano, identity | 6.382 | 6.435 |
| nano, tidy | 5.513 | 5.804 |
| micro, identity | 5.355 | 6.003 |
| micro, tidy | 4.375 | 5.501 |

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

On that round `english` (5.669) matches 3.3× the data and `tidy` (5.513) matches 4.0×.

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

**The order of the documents: 0.013.** `tidy`'s output, shuffled with four other seeds
and graded on the same round at nano: 5.537, 5.524, 5.549, 5.541, against 5.513 in its
own order. Standard deviation 0.013. An editor's order matters about this much by chance.

**The round: 0.05 to 0.09.** Across the five nano rounds above, the unedited loss has a
standard deviation of 0.066, `english`'s gain 0.092 and `tidy`'s gain 0.063. Compared on
the *same* rounds, `tidy` minus `english` has a standard deviation of 0.045, against 0.11
if the two were measured on different rounds. That is why the record is decided on
shared rounds.

**Two different machines:** not yet measured. It needs a second kind of GPU, which the
reference machine cannot stand in for. As a stand-in, runs that start from initial
weights changed by one part in a million are *running*. Until verifiers on other hardware
have graded the same scripts, the tolerance (0.08 nats at micro) is a guess on the safe
side, and anchors check any record before it stands.

```
python tools/exp.py --scale nano --rounds 0 --editors tidy --reseed 1
```

## What the record rule can tell apart

The record changes hands on a one-sided paired test against a margin, at fixed looks
([the rule](RULES.md#the-record)). Simulated as the server applies it, with the paired
standard deviation measured above (0.045) and the nano margin (0.03): how often does a
challenger with a given true lead over the holder end up with the record?

| True lead, nats | 0 (a copy) | 0.03 | 0.04 | 0.05 | 0.06 | 0.08 | 0.14 |
|---|---|---|---|---|---|---|---|
| Looks at 5, 10, 20 rounds | 0.0% | 4% | 16% | 45% | 78% | 99.5% | 100% |
| Looks at 5, 10, 20, 40 rounds | 0.0% | 4% | 25% | 72% | 97% | 100% | 100% |

0.14 is `tidy`'s measured lead over `english`: it would be confirmed at the first look,
19 times in 20.

```
uv run --no-project --with numpy --with scipy python tools/record_power.py --sd 0.045 --margin 0.03
```

**What it decided.** A copy of the holder does not take the record, and a lead equal to
the margin does so 4% of the time, inside the 5% the rule allows. Three looks were too
few: a challenger that really was 0.05 better lost more often than it won. The rule now
has a fourth look at 40 shared rounds (ten days). The margin and the looks for the
official scale will be set the same way once micro's paired noise is measured: *running*.

## Why so little ink?

The threat: ignore the corpus and ship good text inside the editor. To measure it, `tidy`'s
output was mixed with passages from the same six sources as the exam (ones this round is
not examined on), taking a share of the training budget. Nano, round s0-p00000:

| Share of the budget that is shipped text | 0 | 0.1% | 1% | 10% |
|---|---|---|---|---|
| Exam loss | 5.513 to 5.549 | 5.541 | 5.458 | 5.218 |

```
python tools/exp.py --scale nano --rounds 0 --editors tidy --canned 0.01
```

**What it decided.** At 0.1%, shipped text cannot be told from order noise. At 1% it is
worth 0.07 nats, about half of `tidy`'s lead over `english`. The limit is 0.1%.
Micro: *running*.

## Still to measure

- Micro: the reference curve, order noise, more rounds, shipped text at 0.1%.
- Micro: whether the ranking of editors survives a learning rate half or twice as large.
  If it does not, the student is tuned to one kind of data and the scale needs changing.
- Two machines: the real tolerance, from verifiers on NVIDIA hardware.
- The first official rounds, which use a different crawl sample every six hours.

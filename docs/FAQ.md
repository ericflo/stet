# Questions

## What is this, in one sentence?

A competition to write the best program for editing pretraining data, where the model, the
trainer and the exam are fixed and the data is the only thing an entry can change.

## Why would editing data matter?

Most of a raw web crawl teaches a language model very little. If an automatic editor can
make the same pages worth twice as much, that is half the compute for the same model.
Stet measures how good such an editor can get, on data it has never seen, with nothing
else varying.

## What is the multiplier?

How many times more unedited data the student would have needed to do as well. "2×" means
a student trained on the edited corpus matched one trained on twice as many tokens of the
raw one. It is read off a measured curve and never extrapolated beyond it. The curve for
the official scale is still being measured, so for now the board shows the gain in nats
and leaves the multiplier blank. [Experiments](EXPERIMENTS.md) has the curve so far.

It is a statement about this trainer. The unedited corpus trains better at a lower
learning rate than the season's, so against a baseline tuned for it the same editor
would show a smaller gain and a smaller multiplier
([measured](EXPERIMENTS.md#does-the-answer-depend-on-the-learning-rate)). Editors are
ranked against each other by their losses on the same rounds, and an edited corpus's
loss barely moved with the learning rate.

## Do I need a big GPU?

No. A practice run at nano scale takes about a minute on a laptop-class GPU. The official
scale is about eight minutes on an integrated Radeon. `stet doctor` tells you what your
machine can do.

## Can my coding agent do this?

That is the intended way in. Give it one line:

> Read https://www.stetrun.com/AGENTS.md and enter Stet.

The brief is one file, the loop is one command with JSON output, and practice rounds are
unlimited. Two steps need you: linking a GitHub account, and agreeing that the editor is
published.

## Can my editor use a language model?

Yes, if it fits: 50 MB for the whole bundle, no network, and 300 seconds on the
organisers' machine for the whole corpus. A small classifier fits. A large model does not.

## Can my editor write new training text?

Almost none: 0.1% of the training tokens. Everything else the student reads must be copied
byte for byte from the round's corpus. Without that limit the best entry would be a
dataset written in advance, or by a large model, and the competition would stop being
about editing.

## Is the exam secret?

No. It is a public file, so that anyone can reproduce any score. Your editor cannot read
it while it runs, pages containing the opening of a long exam sentence are removed from
the corpus, and your script's output is checked for them too.

## Then what stops people overfitting the exam?

The exam is six broad kinds of writing, not a set of questions: what transfers is "English
prose that explains things", which is the point. An editor still has to find that in a
crawl it has never seen, in five minutes, with almost no ink. And editors are public, so
anything cleverer than that is in plain view.

There is also a second exam that only the organisers' anchors hold, drawn in secret from
the same collections out of other documents. Its hash is already published. Every
official grade is quietly repeated on it, and when the season closes it is published and
the final table is worked out from it. An editor that learned the public exam's passages,
rather than its kind of text, loses its lead there. See
[the rules](RULES.md#the-held-out-exam).

## Who runs my code?

One of the organisers' machines (an *anchor*), in a sandbox with no network. Nobody else
ever runs it. What your editor writes is an edit script, which is plain data, and that is
what gets graded.

## Who decides my score?

The anchor grades the script, and that grade is the score. Volunteers grade the same
script on their own GPUs, and every one of their results is published beside it on the
round's page, whatever it said. You can re-grade any script yourself with `stet score`.

## Why is my editor published?

So that everyone can learn from the leader, and so that every result can be checked. It is
published from the moment it is first drawn into an official round, under the MIT
licence, along with every script it writes. `stet fork` copies any of them, and the fork credits its parent.

## Why do I need a GitHub account to enter?

Editing and grading an entry takes ten to fifteen minutes of an organiser's machine,
three times over. That is the scarce resource, so one person gets one entry in play.
Practice runs, forking and verifying need no account at GitHub.

## How do I know the organisers are not cheating?

The corpus is chosen by a public randomness beacon at a time fixed in advance. The exam,
the trainer, every editor and every script are public. Any score can be recomputed with
`stet score`. What you do have to trust the organisers for is listed, and it is short:
[what the organisers are trusted with](RULES.md#what-the-organisers-are-trusted-with).

## Why are grades limited to NVIDIA and AMD GPUs?

Two machines must get the same number for the same script, to within a small tolerance.
CUDA and ROCm builds of PyTorch are expected to, and every replication is published with
its difference from the anchor's, so the first weeks will show whether they do. Other
backends can run and submit editors but not grade for the board.

## What happens between seasons?

A season freezes the student, the budgets, the exam and the rules, so that scores stay
comparable. Season 0 is the first and smallest. Later seasons can raise the scale. The
plan is to rerun each season's top editors a scale up, to see whether what won still wins.

## Is there a prize?

No. There is a record, a board, and your name on a technique.

## Where is the RL environment?

`stet.gym.Env`: the observation is a corpus, the action is an edit script, the reward is
the gain. It draws from a different dataset than the official rounds. See
[the design](DESIGN.md#the-gym).

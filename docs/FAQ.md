# Questions

## What is this, in one sentence?

A competition to write the best program for editing pretraining data, where the model, the
trainer and the exam are fixed and the data is the only thing an entry can change.

## Why would editing data matter?

Most of a raw web crawl teaches a language model very little. If an automatic curator can
make the same pages worth twice as much, that is half the compute for the same model.
Stet measures how good such a curator can get, on data it has never seen, with nothing
else varying.

## What is the multiplier?

How many times more unedited data the student would have needed to do as well. "2×" means
a student trained on the edited corpus matched one trained on twice as many tokens of the
raw one. It is read off a measured curve and never extrapolated beyond it: the curve
goes to eight times the data, and past that the board says "more than 8×".
[Experiments](EXPERIMENTS.md#how-much-data-is-an-edit-worth) has the curve.

It is a statement about this trainer. The unedited corpus trains better at a lower
learning rate than the season's, so against a baseline tuned for it the same curator
would show a smaller gain and a smaller multiplier
([measured](EXPERIMENTS.md#does-the-answer-depend-on-the-learning-rate)). Curators are
ranked against each other by their losses on the same rounds, and an edited corpus's
loss barely moved with the learning rate.

## Do I need a big GPU?

No. A practice run at nano scale takes a couple of minutes on a laptop-class GPU. The
official scale is about eight minutes of training on an integrated Radeon. `stet doctor`
tells you what your machine can do. Running other people's curators for the leaderboard
asks for more, a mid-range desktop card; `stet bench` measures yours.

## How many things can I try?

On your own machine, as many as you like. `stet run` is the same harness, exam and
arithmetic the leaderboard uses, nothing is uploaded, and nothing is counted. Practice
rounds never change, so two ideas can be compared on the same data; `--rounds 5` gives a
mean and its error; and every official round can be replayed once it settles
(`stet rounds`, then `stet run --round <id>`). To run many trials from a program there
is `stet.gym`, the same thing as a Python environment.

Official runs cost volunteers GPU time, so those are rationed: three entries in play at
once and six submissions a day. A round is drawn as soon as an entry is waiting and the
last round has settled, so an entry usually has its first score within a couple of
hours, not at the next fixed time.

## Can my coding agent do this?

That is the intended way in. Give it one line:

> Read https://www.stetrun.com/AGENTS.md and enter Stet.

The brief is one file, the loop is one command with JSON output, and practice rounds are
unlimited. Two steps need you: linking a GitHub account, and agreeing that the curator is
published.

## Can my curator use a language model?

Yes. Name its weights in `stet.toml` and they are in place when your curator starts: up to
4 GiB of public files on Hugging Face, which is room for a model of a billion or two
parameters at half precision. `stet asset add <owner>/<repository>` does it. Your curator
gets one GPU with 8 GB of memory, PyTorch and transformers, and ten minutes for the whole
corpus.

Ten minutes is the real limit. A round's corpus is about ten times what the student can
read, so a model cannot read all of every page: spend it where cheaper tests cannot
decide. `examples/reader` is a working curator that does this with a 135M-parameter model.

## Can I fine-tune a model for this and run it over the data?

Yes, and it is what we most hope to see. Train it at home, on any hardware, on anything
except the two exams. Publish the weights in a public Hugging Face repository,
`stet asset add` it, and your curator loads it from `/assets` and runs it over the corpus
however you choose: a classifier over every chunk, a tagger that marks the lines to cut, a
scorer that ranks candidates.

Budget by reading speed. On the slowest machine that runs curators, a 20M-parameter
encoder reads about 670,000 tokens a second and a 0.6B language model about 16,000; a
micro round's corpus is about 490 million tokens and the limit is 600 seconds. So a small
model can see some of every document, and a larger one only what cheaper code sends it.

## Can my curator be an agent?

Yes. The harness is yours: `entry` in `stet.toml` is any command, it may start other
processes, and although there is no network, `127.0.0.1` works. So you can serve a model
inside the sandbox (`transformers serve` speaks the OpenAI API, tool calls included) and
point an agent harness at it. Node.js is installed beside Python for harnesses written in
JavaScript. `examples/agent` is a complete curator that runs the pi harness against a 0.6B
model this way.

The model has to be one you bring: no curator can call an API. And an agent's turn takes
seconds, so it can decide how a round is handled but cannot visit every page.

## What can my curator import?

Python 3.13 and fifty libraries at fixed versions: PyTorch, transformers,
sentence-transformers, tokenizers, safetensors, scikit-learn, numpy, scipy, pandas,
polars, faiss, fastText, tiktoken, LightGBM, XGBoost and more, plus Node.js 24.
`AGENTS.md` has the list. Anything else goes in your bundle: pure Python as it is, a
JavaScript harness as one archive that your curator unpacks when it starts.

## Does it have to work on both NVIDIA and AMD?

Yes. Curators are run by volunteers, and each entry is run on both makes when both are
around. Use the GPU through PyTorch and say `device="cuda"`: that is the right name on
an AMD GPU too. Nothing in the library list is one maker's alone. A run that fails
scores a gain of 0 for that run.

## Can my curator write new training text?

Almost none: 0.1% of the training tokens. Everything else the student reads must be copied
byte for byte from the round's corpus. Without that limit the best entry would be a
dataset written in advance, or by a large model, and the competition would stop being
about editing.

## Is the exam secret?

No. It is a public file, so that anyone can reproduce any score. Your curator cannot read
it while it runs, pages containing the opening of a long exam sentence are removed from
the corpus, and your script's output is checked for them too.

## Then what stops people overfitting the exam?

The exam is six broad kinds of writing, not a set of questions: what transfers is "English
prose that explains things", which is the point. A curator still has to find that in a
crawl it has never seen, in ten minutes, with almost no ink. And curators are public, so
anything cleverer than that is in plain view.

There is also a second exam that only the organisers hold, drawn in secret from the same
collections out of other documents. Its hash is already published. When the season
closes it is published, every script that counted is graded on it, and the final table
is worked out from that. A curator that learned the public exam's passages,
rather than its kind of text, loses its lead there. See
[the rules](RULES.md#the-held-out-exam).

## Who runs my code?

Two volunteers, each on their own machine, in a sandbox with no network: people whose
grades have a record of agreeing with other people's. Never you, and never the same
person twice. What your curator writes is an edit script, which is plain data, and that
is what gets graded.

## Who decides my score?

Nobody alone. Each run's script is graded by someone other than the person who ran it,
and your gain for the round is the mean of your two runs. If the runs disagree, a third
person runs it and the middle one counts. Every run and every grade is published on the
round's page, whoever made it and whatever it said, and you can re-grade any script
yourself with `stet score`.

A round does not wait for people who are not there. When one machine is all there is, it
runs and grades the entry alone, and the score says so: *run once*, graded by its own
runner. It becomes *confirmed* in the next round that two people are around for.

## What if the organisers' machine goes away?

Rounds carry on. The organisers' accounts are established verifiers like any others:
they were the first, because someone has to be, and early in a season their machine may
be doing most of the work. No score needs it. Once a handful of volunteers are
established, they run and grade everything among themselves.

## Why is my curator published?

So that everyone can learn from the leader, and so that every result can be checked. It is
published from the moment it is first drawn into an official round, under the MIT
licence, along with every script it writes. `stet fork` copies any of them, and the fork credits its parent.

## Why do I need a GitHub account to enter?

Running and grading an entry takes volunteers about half an hour of GPU time a round,
for at least three rounds. That is the scarce resource, so one person gets three entries
in play and six submissions a day, and a person has to be someone.
Practice runs, forking and verifying need no account at GitHub.

## How do I know the organisers are not cheating?

The corpus is chosen by a public randomness beacon at a time fixed in advance. The exam,
the trainer, every curator and every script are public. Any score can be recomputed with
`stet score`. Your curator is run and graded by volunteers, not by the organisers. What
you do have to trust, and whom, is listed, and it is short:
[what is trusted, and what is not](RULES.md#what-is-trusted-and-what-is-not).

## Why are grades limited to NVIDIA and AMD GPUs?

Two machines must get the same number for the same script, to within a small tolerance.
CUDA and ROCm builds of PyTorch are expected to, and every grade is published with its
difference from the others of the same script, so the first weeks will show whether they
do. Other backends can build, run and submit curators but not grade for the board.

## What happens between seasons?

A season freezes the student, the budgets, the exam and the rules, so that scores stay
comparable. Season 0 is the first and smallest. Later seasons can raise the scale. The
plan is to rerun each season's top curators a scale up, to see whether what won still wins.

## Is there a prize?

No. There is a record, a board, and your name on a technique.

## Where is the RL environment?

`stet.gym.Env`: the observation is a corpus, the action is an edit script, the reward is
the gain. It draws from a different dataset than the official rounds. See
[the design](DESIGN.md#the-gym).

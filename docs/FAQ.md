# Questions

## What is this, in one sentence?

A competition to write the best program for editing pretraining data, where the model, the
trainer and the way the exam is made are fixed, and the data is the only thing an entry can
change.

## Why would editing data matter?

Most of a raw web crawl teaches a language model very little. If an automatic curator can
make the same pages worth twice as much, that is half the compute for the same model.
Stet measures how good such a curator can get, on data it has never seen, with nothing
else varying.

## Why is there no "worth N× the data" figure?

That figure says how many times more unedited data the student would have needed to do as
well. It has to be read off a measured curve of the unedited corpus's score against training
tokens, and that curve has not been measured on this exam yet. Until it is, the board shows
gains in bits per byte.

## Do I need a big GPU?

No. A practice run at nano scale takes a couple of minutes on a laptop-class GPU. The
official scale, standard, trains for about 18 minutes on an RTX 3090 or 4070 Ti, and it
wants a GPU with 12 GB or more. `stet doctor` tells you what your machine can do. Running
other people's curators for the leaderboard asks for more, a mid-range desktop card;
`stet bench` measures yours.

## How many things can I try?

On your own machine, as many as you like. `stet run` is the same harness, exam and
arithmetic the leaderboard uses, nothing is uploaded, and nothing is counted. Practice
rounds never change, so two ideas can be compared on the same data; `--rounds 5` gives a
mean and its error; and every official round can be replayed once it settles
(`stet rounds`, then `stet run --round <id>`). To run many trials from a program there
is `stet.gym`, the same thing as a Python environment.

Run `stet harvest` once before the first `stet run`. It fetches three hours of recent human
writing from the sources official exams are made from, with no account and no key, and
every practice round's exam is then made as an official round's is: generated questions,
and writing from the hours you hold. Without it a practice exam is the generated skills
alone, a quarter of an official score, and `stet run` says so. `stet harvest --fresh`
replaces the hours with newly written ones: the check on having tuned to the old. A replay
of an official round sits what that round published, its generated questions and its
Wikipedia text.

Official runs cost volunteers GPU time, so those are rationed: three entries in play at
once and six submissions a day. An entry runs in the next round drawn. A round is
drawn every 3 hours. Its exam is made at the draw, and it settles as soon as its runs and
its grades are in.

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
gets one GPU with 8 GB of memory, PyTorch and transformers, and twenty minutes for the whole
corpus.

Twenty minutes is the real limit. A round's corpus is about six times what the student can
read, so a model cannot read all of every page: spend it where cheaper tests cannot
decide. `examples/reader` is a working curator that does this with a 135M-parameter model.

## Can I fine-tune a model for this and run it over the data?

Yes, and it is what we most hope to see. Train it at home, on any hardware, on anything
except published exams. Publish the weights in a public Hugging Face repository,
`stet asset add` it, and your curator loads it from `/assets` and runs it over the corpus
however you choose: a classifier over every chunk, a tagger that marks the lines to cut, a
scorer that ranks candidates.

Budget by reading speed. On the slowest machine that runs curators, a 20M-parameter
encoder reads about 670,000 tokens a second and a 0.6B language model about 16,000; a
standard round's corpus is about 2.5 GB and the limit is 1,200 seconds. So a small model
can see some of every document, and a larger one only what cheaper code sends it.

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
an AMD GPU too. Nothing in the library list is one maker's alone. A run that fails is
never better than one that finishes: it counts 0, or as your worst finished run when that
is below 0.

## Can my curator write new training text?

No. A script is copies of the round's corpus and nothing else, so everything the student
reads is copied byte for byte. A little text of a curator's own was allowed at first. It
was measured, and 16,000 tokens of exam-style text bought 0.004 bits per byte, 1.3
standard errors: too small and uncertain to earn a rule, so it was left out. Cleaning a
page is always allowed: whole lines of one page, kept in order, are copies however short
they are. Any other copy is at least 64 bytes long.

## Is the exam secret?

Nobody knows it in advance, the organisers included. Each round's exam is made at its draw.
Three quarters of the score is human writing from three hours of the day before, which the
draw picks, gathered from many sources that need no key. One quarter is questions generated
from the round's seed. Your curator runs with no network and can add no text of its own:
a script is copies of the round's corpus and nothing else.

Until the round settles, the exam goes only to the machines grading it. Then it is
published: the generated questions and Wikipedia's text whole, and every other source's
items as where they came from and a hash, so anyone can check them at the source. Every
student's score on every item is published with it.

## Then what stops people overfitting the exam?

There is nothing to fit in advance: a round's writing is three hours of the day before,
which its draw picks, and its questions come from its seed. What a curator can learn from
past rounds is what kinds of text matter (news, programs, replies, every language), and
keeping more of those is choosing good data, which is the point. The exam weighs four kinds
of writing and seven skills, and every language by its share of the crawl, so no one source
or kind of text can carry a score.

## Does everything on the exam count?

Yes. The season names every part of the exam. Each part was measured before the season; a
part that did not tell curators apart, or whose order did not hold for a larger student, is
not on the exam. A part is in the score or it is not on the exam at all: nothing on the exam
counts for nothing. Arithmetic, units, word problems, algebra and small algorithms are asked
only after a few worked examples of the same task: asked cold, a student this size is too
near chance for curators to differ.

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

Runs and grades carry on. The organisers' accounts are established verifiers like any
others: they were the first, because someone has to be, and early in a season their machine
may be doing most of the work. Once a handful of volunteers are established, they run and
grade everything among themselves.

Each round's exam is made by the organisers' examiner, from the harvests it makes every
hour. If it stops, curators still run and graders still train, and the trained students
sit the exam when it is posted.

## Why is my curator published?

So that everyone can learn from the leader, and so that every result can be checked. It is
published from the moment it is first drawn into an official round, under the MIT
licence, along with every script it writes. `stet fork` copies any of them, and the fork credits its parent.

## Why do I need a GitHub account to enter?

Running and grading an entry takes volunteers over an hour of GPU time a round, for at
least three rounds. That is the scarce resource, so one person gets three entries in play
and six submissions a day, and a person has to be someone. Practice runs, forking and
verifying need no account at GitHub.

## How do I know the organisers are not cheating?

The corpus is chosen by a public randomness beacon at a time fixed in advance. Each round's
exam is made by published code, from text anyone can read: the hash of every hour's harvest
is public before the draw that picks from it, and the exam and every score on every item
are published when the round settles. The trainer, every curator and every script are
public. Any score can be recomputed with `stet score`. Your curator is run and graded by volunteers, not by the
organisers. What you do have to trust, and whom, is listed, and it is short:
[what is trusted, and what is not](RULES.md#what-is-trusted-and-what-is-not).

## Why are grades limited to NVIDIA and AMD GPUs?

Two machines must get the same number for the same script, to within a small tolerance.
CUDA and ROCm builds of PyTorch are expected to, and every grade is published with its
difference from the others of the same script, so the first weeks will show whether they
do. Other backends can build, run and submit curators but not grade for the board.

## What decides the season?

Finals. When the season closes, announced two weeks ahead, the eight best-rated curators and
one more drawn by lot from the next eight are run on two fresh rounds, and a larger student
(10 layers by 640, 640M tokens) is trained twice on what each writes. Their gain there is the
season's table.

A season freezes the student, the budgets, how the exam is made and the rules, so that scores
stay comparable.

## Is there a prize?

No. There is a record, a board, and your name on a technique.

## Where is the RL environment?

`stet.gym.Env`: the observation is a corpus, the action is an edit script, the reward is
the gain. It draws from a different dataset than the official rounds. See
[the design](DESIGN.md#the-gym).

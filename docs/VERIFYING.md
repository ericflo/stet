# Verifying

A score on the leaderboard is measured by an **anchor**, a machine the organisers run.
That is one machine's word. `stet verify` is how it becomes more than that: your GPU
grades the same edit scripts, and your result is published beside the anchor's. This
page is what that involves, and why a score checked that way can be believed.

## Lend a GPU

```
stet doctor      # checks the GPU, PyTorch and disk
stet login       # makes a key and an account; no email, no password
stet verify      # grades until you press Ctrl-C
```

You need an NVIDIA (CUDA) or AMD (ROCm) GPU and about 10 GB of disk. The installer brings
PyTorch with it, in the NVIDIA build. On an AMD GPU, run the installer with the ROCm wheel
index for your ROCm version instead (`stet doctor` says so if it is needed):

```
curl -fsSL https://www.stetrun.com/install.sh | STET_TORCH_INDEX=https://download.pytorch.org/whl/rocm7.14 sh
```

`stet verify --once` does one job and exits; `--max-jobs 5` does five. One verifier runs
per account. For a second GPU, give the second verifier an account of its own: set
`STET_CONFIG` to another directory and run `stet login` there.

**No one else's code runs on your machine.** A grade job is an edit script, which is plain
data: lists of document numbers and byte ranges. Your machine downloads the round's raw
text from Common Crawl, builds the corpus, trains the fixed student on what the script
says, and reports the exam loss. Entrants' programs run only on the organisers' machines.

What a job costs: at the official scale, a grade is one training run, about ten minutes
on the reference machine, which has an integrated Radeon 8060S. The first job of a
round also downloads the round's raw text (about half a gigabyte; 1.2 GB once built) and
trains on the unedited corpus once, so it takes about twice as long.

What your machine sends with each grade: the losses it measured, hashes of the corpus and
exam it built, how long training took and a thinned loss curve, and a description of the
machine: GPU name and memory, operating system and processor type, and the versions of
Python, PyTorch and the harness.

What is published: your handle and linked GitHub login, the GPU's name, how many of your
grades agreed, your streak, GPU hours, when you last graded, and every grade you report:
the loss you measured, beside the anchor's, on the round's page. Your IP address is used
for rate limits and appears in the server's logs; it is not published.

## What your machine checks for itself

A verifier does not take the server's word for what a round is.

- **The seed.** It recomputes the round's seed from the drand beacon and the crawl, and
  fetches the beacon from drand's own relays, not from Stet.
- **The moment of the draw.** A round drawn from a drand round that is not on the season's
  ten-minute grid is refused.
- **The announcement.** Each time it asks for work, a verifier is told what the next draw
  is committed to, and writes it down. A round that does not match what was announced is
  refused.
- **The data.** Corpus files are fetched from Common Crawl by path, the exam pack is
  pinned by hash, and the script must match the hash the job named.
- **The code.** A verifier running a different version of the harness from the season's
  is asked to upgrade before it is given work.

## Why a score can be believed

**Grading is reproducible.** Same script, same corpus, same exam, same initial weights,
same number of tokens. Two honest machines agree to within a small tolerance
*(provisional: 0.08 nats at the official scale; see [Experiments](EXPERIMENTS.md))*.

**An anchor grades everything.** Every entry in every round is graded by an anchor, and
that grade is the score. No volunteer's grade moves the leaderboard, so there is nothing
to gain by reporting a made-up one.

**Volunteers replicate.** Up to three volunteers grade each entry. You cannot choose
what you grade, you are offered an entry once, you never grade your own, and ties
between jobs are broken at random.

**Grades are sealed.** Nobody can see any grade, or the round's unedited loss, until the
whole round settles. A grade cannot be copied from someone else's.

**Everyone proves the round first.** Every grade comes with your baseline for the round:
the unedited corpus's loss. If your corpus, your exam or your baseline differs from the
anchor's, your result is set aside with a message saying which. That is a broken cache or
a different PyTorch, and it does not count against you. It is listed on the round's page
all the same: if honest machines of one kind keep landing there, the tolerance is wrong,
and that is how anyone would see it.

**Every grade is published.** When a round settles, each volunteer grade is compared with
the anchor's, and the round's page shows all of them with their differences. One within
the tolerance is an agreement: your name goes on the entry, and the grade and its GPU
time on the [verifier board](/verifiers). One outside it differs: either that verifier
or the anchor is wrong, and `stet score` lets a third person find out which.

**Differing alone is a strike; differing together is a question for the anchor.** If
yours is the only grade that differs, it counts against your account. Three of those, and
more than one for every ten grades that agreed, and the account gets no grading work for
two weeks after the latest. Then it may try again: the machine may be honest and merely
different. If another volunteer who has agreed with an anchor before finds what you
found, neither of you is struck, and the entry is marked: its score may be the thing
that is wrong.

**What replication cannot show.** Two scores closer together than the tolerance are not
told apart by it, and an entry that leaves the corpus unedited has the round's unedited
loss, which every grader has already measured, so agreeing about it earns no standing.

**Anyone can check anything.** `stet rounds` lists every official round, and
`stet run --round <id>` or `stet score <script> --round <id>` recomputes any score on it,
with or without an account.

## When a round settles

A round settles as a whole, when an anchor has graded every entry in it. If a volunteer
is part-way through a grade at that moment, the round waits for them, up to 45 minutes
after the anchor's last grade.

It waits only for accounts with a grade that has agreed before. A key costs nothing, and
one that could hold a round back just by taking a job would. So until your first grade
has agreed, start early in a round: a grade that arrives after the round has settled
earns nothing, and for a newcomer costs nothing either.

Taking a job and keeping it is the one thing that does cost. If the round has to give up
waiting for you, or a job's three hours run out with no result and without being handed
back, the account is offered no work for a day.

## What you get

- A place on the [verifier board](/verifiers): grades that agreed, in total and in the
  last two weeks, your streak, and GPU hours given.
- Your name on every entry you replicated.
- **Your own entries go first.** When more editors are waiting than a round can take,
  entries from people who graded for others in the last two weeks run before the rest.

## If something goes wrong

| Message | What it means |
|---|---|
| "your corpus or exam differs from the round's" | A download was cut short or a cache is stale. The verifier drops its copy of the round and rebuilds it once by itself. If it happens twice in a round, it takes no more work from that round. |
| "your unedited baseline for ... is further from the anchor's than the tolerance ..." | Your machine trains differently: usually an old PyTorch or a CPU fallback. `stet doctor` shows what is in use. The difference is listed on the round's page. |
| "sealed until the round settles" | Normal. Your grade is in, and is compared with the anchor's when the round settles. |
| "this entry settled before your result arrived" | The round settled while you were training. No credit. |
| "a job this verifier took was given up on ..." | A job ran out of time with no result and without being handed back. Work is offered again a day later. |
| "this account is given no grading work until ..." | Three grades that differed from the anchor's with nobody else finding the same. It lifts on the date given. If you think your machine is right, tell the organisers. |
| "No work right now" | No round is open, or every open entry has its volunteers. Leave it running; it asks again. |
| HTTP 426 | The harness changed. Reinstall with the command on the [home page](/). On an AMD GPU, set `STET_TORCH_INDEX` as you did the first time. |

Pressing Ctrl-C hands the job back so that someone else can take it. A failed download is
waited out with the job kept, a little longer each time, and handed back after about half
an hour. A job that fails on your machine is tried again twice before it is handed back.
The verifier stops by itself when jobs keep failing on your machine (nine failed tries
in a row), when the server says it is out of date or is giving this account no work, or
when the server asks it to do something it should not.

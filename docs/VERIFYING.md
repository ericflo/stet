# Verifying

Stet has no compute budget. Every score on the leaderboard was computed by someone who
ran `stet verify` on their own GPU. This page is what that involves and why a score
computed that way can be believed.

## Lend a GPU

```
stet doctor      # checks the GPU, PyTorch and disk
stet login       # makes a key and an account; no email, no password
stet verify      # grades until you press Ctrl-C
```

You need an NVIDIA (CUDA) or AMD (ROCm) GPU with PyTorch installed, and about 10 GB of
disk. `stet verify --once` does one job and exits; `--max-jobs 5` does five.

**No one else's code runs on your machine.** A grade job is an edit script, which is plain
data: lists of document numbers and byte ranges. Your machine downloads the round's raw
text from Common Crawl, builds the corpus, trains the fixed student on what the script
says, and reports the exam loss. Entrants' programs run only on the organisers' machines.

What a job costs: at the official scale, one grade is about ten minutes on the reference
machine, which has an integrated Radeon 8060S. The first job of a round also downloads the
round's corpus (about 1.5 GB) and trains the baseline once.

What is recorded about you: your handle, your public key, the GPU's name, and your grades.

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
- **The code.** Every result carries a hash of the files that decide a score. A verifier
  running a different version of the harness is asked to upgrade.

## Why a grade can be believed

**Grading is reproducible.** Same script, same corpus, same exam, same initial weights,
same number of tokens. Two honest machines agree to within a small tolerance
*(provisional: 0.08 nats at the official scale; see [Experiments](EXPERIMENTS.md))*.

**Work is assigned.** You cannot choose what you grade, you never grade your own entry,
and ties between jobs are broken at random.

**Grades are sealed.** Nobody can see any grade for an entry until the entry settles, so a
grade cannot be copied, and making one up is a gamble against someone who did the work.

**Trust is earned.** A new verifier's grades count only when a trusted verifier has graded
the same script and agreed. Five agreements earn trust. After that your grade settles an
entry by itself, and one entry in four is given to a second verifier regardless.

**Everyone proves the round first.** Every grade comes with your baseline for the round:
the unedited corpus's loss. If your corpus, your exam or your baseline differs from the
round's, the result is refused with a message saying which, before it can do any harm.

**Disagreement costs.** A grade that differs from the settled one, on the same script and
the same data, is a strike. One strike ends trust; three end grading. A result computed on
a different corpus or exam is thrown away without a strike: that is a broken cache, not a
lie.

**Anchors have the last word.** The organisers' machines can grade anything, their grade
overrules, and no editor can hold the record until an anchor has graded at least two of
its rounds.

**Anyone can check anything.** `stet rounds` lists every official round, and
`stet run --round <id>` or `stet score <script> --round <id>` recomputes any score on it.

## Who goes first

When volunteers are around, the anchor waits: an entry that a volunteer could grade is
left alone for 45 minutes, and a volunteer who has started is given that long to finish.
The anchor steps in at once only to vouch for a new verifier's grade, or when nobody else
is there.

## What you get

- A place on the [verifier board](/verifiers): grades that agreed, in total and in the
  last two weeks, your streak, and GPU hours given.
- **Your own entries go first.** When more editors are waiting than a round can take,
  entries from people who graded for others in the last two weeks run before the rest.

## If something goes wrong

| Message | What it means |
|---|---|
| "your corpus or exam differs from the round's" | A download was cut short or a cache is stale. Delete the round's directory under `~/.cache/stet/rounds` and run again. |
| "your baseline is ...; the round's is ..." | Your machine trains differently: usually an old PyTorch or a CPU fallback. `stet doctor` shows what is in use. |
| "held until a trusted verifier grades the same script" | Normal for a new verifier. |
| "No work right now" | Every open entry has the grades it needs. Leave it running; it asks again. |
| HTTP 426 | The harness changed. Reinstall with the command on the [home page](/). |

Pressing Ctrl-C hands the job back so that someone else can take it.

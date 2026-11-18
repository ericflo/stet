# Verifying

Nothing on the leaderboard is one machine's word. Curators are run and edit scripts are
graded on volunteers' machines, each piece of work by more than one person, and all of it
is published. `stet verify` is how you lend a machine. This page is what that involves,
and why a score made that way can be believed.

## Lend a GPU

```
stet doctor            # checks the GPU, the libraries and the disk
stet login             # makes a key and an account; no email, no password
stet verify            # grades until you press Ctrl-C
stet verify --edits    # also runs entrants' curators, in a sandbox
```

There are two kinds of work, and you choose whether to take the second.

| | Grading | Running curators (`--edits`) |
|---|---|---|
| What runs on your machine | the fixed trainer, on plain data | entrants' programs, in a sandbox |
| Needs | an NVIDIA (CUDA) or AMD (ROCm) GPU, the season's PyTorch, 10 GB of disk | also Linux, bubblewrap, the season's libraries, a machine at or above the floor, 60 GB of disk |
| Who is given it | anyone | accounts that are established (below) |

The installer brings Python 3.13, PyTorch in the build for your GPU, and the season's
libraries. `stet bench` measures your machine against the floor for running curators and
says what, if anything, falls short. Run it when the GPU is doing nothing else.

`stet verify --once` does one job and exits; `--max-jobs 5` does five;
`--exit-when-idle 15` stops once it has had nothing to do for fifteen minutes, never with a
job in hand, which suits a GPU rented by the hour. One verifier runs
per account, and holds one job at a time. For a second GPU, give the second verifier an
account of its own: set `STET_CONFIG` to another directory and run `stet login` there.

### Leaving it running

```
stet service install            # or: stet service install --edits
stet service status
stet service logs
stet service uninstall
```

On Linux with systemd this makes `stet verify` a service of your own user: it starts
with the machine, starts again if it stops, and writes to
`~/.local/state/stet/stet-verify.log`. It needs no root. If the machine logs you out, it
may ask you to run `sudo loginctl enable-linger $USER` once so that it keeps going.
Elsewhere, leave `stet verify` in a terminal or under your system's own scheduler.

**It keeps itself up to date.** When a new version of stet is out, a verifier that holds
no job installs it and starts again with the same arguments. It does this by running the
site's installer, the same one you ran, fetched over TLS from the same address. Everything
is downloaded into a scratch environment first, so a failed download leaves the version
you have running. Nothing is upgraded in the middle of a job.

To decide for yourself when to upgrade, start it with `stet verify --no-upgrade` (or
`stet service install --no-upgrade`, or set `STET_AUTO_UPGRADE=0`). It then says when a
new version is out, stops being given work once the server needs the new one, and
`stet upgrade` installs it when you are ready.

### What it keeps on your disk

Everything stet downloads goes in one directory, `~/.cache/stet` (`STET_CACHE` moves it),
and all of it can be fetched again. It cleans up after itself:

| What | How long it is kept |
|---|---|
| A round's raw crawl files | Deleted as soon as the round's corpus is built from them |
| Rounds you graded | The newest three |
| Curators' model files | Until the cache passes 40 GB (`STET_ASSET_CACHE_GB`), least recently used first |
| A job's working files | Deleted when the job ends |
| Anything else: practice rounds, model files, scripts | Deleted after 14 days without being used (`STET_CACHE_DAYS`; `0` keeps everything) |
| The service's log | Cut back to its newest megabyte when it passes 8 MB |

```
stet clean --dry-run   # what is there, and how big
stet clean             # delete all of it now
```

`stet clean` is safe while a verifier is running: it leaves what that verifier used in the
last few hours. `stet doctor` shows how much is held.

To remove stet altogether: `stet service uninstall`, `stet clean`, `uv tool uninstall
stet-cli`, and delete `~/.config/stet` (your key: without it the account cannot be used
again) and `~/.local/state/stet` (logs). The packages it was installed from are in uv's
own cache, shared with anything else uv installs: `uv cache clean` frees that.

### Grading

**No one else's code runs.** A grade job is an edit script, which is plain data: lists of
document numbers and byte ranges. Your machine downloads the round's raw text from Common
Crawl, builds the corpus, trains the fixed student on what the script says, and reports
the exam loss.

At the official scale a grade is one training run, about ten minutes on a mid-range GPU.
The first job of a round also downloads the round's raw text (about half a gigabyte;
1.2 GB once built) and trains on the unedited corpus once, so it takes about twice as
long.

### Running curators

**This runs strangers' programs on your machine**, which is why it is a flag you have to
pass. Each one runs in a sandbox:

- no network, at all;
- none of your files: it sees its own bundle, the round's corpus and the model files it
  named, all read-only, and scratch space held in memory;
- not your machine's name, other processes or hardware serial numbers. It can see the
  path of the directories it was given, and so the name of the account `stet` runs as;
- six cores, 12 GB of memory, one GPU with 8 GB of its memory, and the time limit, after
  which it is killed.

`stet verify --edits` tests the sandbox before it takes any work and will not start
without it. What the sandbox cannot do is stand between the curator and your GPU's
driver: using the GPU means talking to it. A flaw in the driver or the kernel is outside
what any of this contains. Curators are public code from accounts tied to a GitHub
identity, which is a deterrent and not a guarantee. Run curators on a machine you could
reinstall without loss.

A curator may name model files, up to 4 GiB an entry. They are fetched from Hugging Face
before it runs, checked against their hashes, and kept in a cache that is trimmed at
40 GB and of whatever has gone unused for two weeks
([what it keeps on your disk](#what-it-keeps-on-your-disk)). A run takes up to ten minutes
at the official scale, plus the download.

### Becoming established

Scores are made from the work of **established** accounts: 5 grades that agreed with
other people's, and a linked GitHub account at least 30 days old (`stet login --github`).
Until then your grades are replications: published and compared, and each one that
agrees counts towards the 5. `stet whoami` shows where you stand.

### What is sent and what is published

With each result your machine sends: the losses it measured, or the script the curator
wrote; hashes of the corpus and exam it built; how long the work took and a thinned loss
curve; and a description of the machine: GPU name and memory, operating system and
processor type, the speeds `stet bench` measured, and the versions of Python, PyTorch and
the harness.

Published: your handle and linked GitHub login, the GPU's name, how many of your grades
agreed, how many curators you have run, your streak, GPU hours, when you were last seen,
and every run and grade you report, on the round's page. Your IP address is used for rate
limits and appears in the server's logs; it is not published.

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

**Every curator is run twice, by different people,** when two are around. A machine is
handed first an entry whose other run was on the other make of GPU, so the two are on
NVIDIA and AMD when machines of both are asking. Nobody runs their own entry or runs one
entry twice. When two runs differ
by more than a tolerance *(provisional: 0.12 nats)*, or one fails, a third person runs it
and the middle one counts. Whoever has seen one of an entry's scripts cannot run it, so a
runner who could give that third run is handed none of them while someone else can grade.

**Every script is graded by someone other than its runner.** A runner who could also say
what its script scored would be taking its own word. Its gain is the median of the
established graders' results, and when two of them disagree a third is taken. Most
curators write the same script on both runs, byte for byte; such a script is nobody's
own, and either runner may grade it.

**Work is assigned.** You cannot choose what you run or grade, you are offered an entry
once, you never work on your own, and ties between jobs are broken at random.

**Everything is sealed.** Nobody can see a script, a grade or the round's unedited loss
until the whole round settles, beyond the job in hand: a script is served only to the
verifier that holds it as a grade job. A grade cannot be copied, and neither can a run:
a person who has been shown a script of an entry is never asked to run that entry, and
an account that is not established is shown a script only once its entry is closed to
further runs.

**Everyone proves the round first.** Every grade comes with your baseline for the round:
the unedited corpus's loss. If your corpus, your exam or your baseline differs from the
round's, your result is set aside with a message saying which. That is a broken cache or
a different PyTorch, and it does not count against you. It is listed on the round's page
all the same: if honest machines of one kind keep landing there, the tolerance is wrong,
and that is how anyone would see it.

**Everything is published.** When a round settles, its page shows every run of every
entry and every grade of every script: who, on what GPU, what they measured, and how far
it was from the script's gain. An entry is marked **confirmed** when two runs by
different people agree, each was graded by someone else, and the grades behind them are
not all one person's; **one grader** when the runs agree and one person graded them all;
**one runner** when they agree and one person made them all; **run once** or **runs
differed** otherwise. The organisers' machines count as one person, however many there are. A script that both runs wrote is graded
twice, by two people, when two are around.

**Differing is a strike.** A grade outside the tolerance of the script's gain counts
against your account. Three of those, and more than one for every ten grades that
agreed, and the account gets no work for two weeks after the latest. Then it may try
again: the machine may be honest and merely different. When exactly two established
grades disagree and nobody is there to break the tie, neither is struck and the entry
says so. And a strike needs something solid to be measured against: a gain that two
established grades agree on, or an organisers' machine's. A grade that differs from one
volunteer's lone grade is published as *unresolved* and counts against nobody.

**What this cannot show.** Two scores closer together than the tolerance are not told
apart. A runner who hands back a script the curator did not write is caught only as far
as the second run differs from it. Two verifiers assigned to the same entry who agree on
a lie are not caught at all; what stands in the way is that nobody chooses their work.
[The rules](RULES.md#what-is-trusted-and-what-is-not) list all of it.

**Anyone can check anything.** `stet rounds` lists every official round, and
`stet run --round <id>` or `stet score <script> --round <id>` recomputes any run or any
score on it, with or without an account.

## When a round settles

A round settles as a whole, when every entry has its runs and every script its grades.

It does not wait for people who are not there. If nobody who is at work on a job, or has
asked for one in the last ten minutes, could give an entry its second run, the entry
settles with one. If a
verifier is part-way through a job, the round waits for them: for an established
verifier until the job is reported, handed back or out of time (three hours at most), and
for anyone else up to 45 minutes, and only if a grade of theirs has agreed before. A key
costs nothing, and one that could hold a round back just by taking a job would. So until your first grade has
agreed, start early in a round. A grade that arrives after the round has settled earns
nothing, and for a newcomer costs nothing either.

Taking a job and keeping it is the one thing that does cost. If the round has to give up
waiting for you, or a job's time runs out with no result and without being handed back,
the account is offered no work for a day.

## What you get

- A place on the [verifier board](/verifiers): grades that agreed, in total and in the
  last two weeks, curators run, your streak, and GPU hours given.
- Your name on every entry you ran or graded.
- **Your own entries go first.** When more curators are waiting than a round can take,
  entries from people who worked for others in the last two weeks run before the rest.

## If something goes wrong

| Message | What it means |
|---|---|
| "your corpus or exam for ... differs from the round's ..." | A download was cut short or a cache is stale. The verifier drops its copy of the round and rebuilds it once by itself. If it happens twice in a round, it takes no more work from that round. |
| "your unedited baseline for ... is further from what the round's established graders measured than the tolerance ..." | Your machine trains differently: usually another PyTorch or a CPU fallback. `stet doctor` shows what is in use. The difference is listed on the round's page. |
| "sealed until the round settles" | Normal. Your run or grade is in, and is compared with everyone else's when the round settles. |
| "Not given curators to run: ..." | You asked for curators with `--edits` and the server gave its reason: the account is not established yet, has no GitHub account linked, or the machine is under the floor. It grades meanwhile, and is told when that changes. |
| "this entry settled before your result arrived ..." or "this job ended before your result arrived: the round settled first ..." | The round settled while you were training. No credit, and nothing against you. |
| "this job ended before your result arrived: it was given up on ..." or "a job this verifier took was given up on ..." | A job ran out of time with no result and without being handed back. Work is offered again a day later. |
| "this account is given no work until ..." | Three grades that differed from what established verifiers found for the same scripts. It lifts on the date given. If you think your machine is right, tell the organisers. |
| "curators are run by established verifiers only" | The account lost its standing between taking a curator and reporting on it. The run is not recorded. |
| "the result could not be delivered ..." | The server was away or failing when your grade was ready. The job is handed back; nothing counts against you. |
| "No work right now" or "Nothing in the open round is this machine's to do ..." | No round is open, or nothing in the open one is yours: you have had each job once, it is your own entry or a script only your run wrote, or it has the runs and grades it wants. Leave it running; it asks again. |
| "this verifier is out of date" (HTTP 426) | The harness changed. A verifier upgrades itself and carries on. With upgrades turned off, or an install the installer did not make, run `stet upgrade` or the installer again. |
| "This account holds a curator it took to run ..." | The verifier was restarted without `--edits` while it held a curator. It is handed nothing until that job is reported, handed back or out of time: start it with `--edits` again. |
| "... handed it back, or let the job run out, so it is offered no more of that round's scripts" | An account that is not established yet gave a script back. It is offered work again in the next round. |

Pressing Ctrl-C hands the job back so that someone else can take it, and so does stopping
any other way the verifier gets a say in. If the server cannot be told, it says that the
job is still held and until when: run `stet verify` again before then and it is picked up
where it was. A failed download is waited out with the job kept, a little longer each
time, and handed back after about half an hour. A job that fails on your machine is tried again twice before it is handed back.
The verifier stops by itself when jobs keep failing on your machine (nine failed tries
in a row), when the server says it is out of date or is giving this account no work, or
when the server asks it to do something it should not.

# Operating Stet

For whoever runs the competition. Entrants and verifiers do not need this page.

## What runs where

| Piece | Where | How it gets there |
|---|---|---|
| Website and API | Epsilon, `apps/stet` | Push to `main` → Woodpecker builds the image → Flux rolls it out (about 4 minutes) |
| Records | durable-db, `apps/stet-db`, on its own B2 bucket | Written by the server, one conditional commit per change |
| Files (bundles, scripts, each round's exam and item scores, the CLI wheel) | B2 bucket `maxint-stet-files` | Written by the server |
| The organisers' verifier | A workstation with a GPU | systemd user unit `stet-anchor` |
| The examiner | An organisers' machine | systemd user unit `stet-examine` ([below](#the-examiner)) |

The server is stateless: it can be restarted or rolled at any time. `/api/v1/season` says
which build is live (`server`) and which harness version it accepts (`harness`).

## The organisers' verifier

The organisers run one verifier of their own. It is not needed for a score: curators are
run and scripts graded by any established accounts. It is there because someone has to
be first. Until volunteers are established it does all the work, alone, and entries it
ran alone are marked *run once*. If it is down and nobody else is established, rounds
are drawn but nothing settles.

```sh
systemctl --user status stet-anchor
tail -f ~/.local/state/stet/anchor.log
systemctl --user restart stet-anchor
systemctl --user disable --now stet-anchor      # stop it for good
```

It runs the installed release (`~/.local/bin/stet verify --edits`), the same wheel and
the same libraries volunteers install, with the account whose key is in
`~/.config/stet/key.json`. The server knows an organisers' account by its public key:
`STET_ANCHOR_KEYS` in Epsilon's `apps/stet/deployment.yaml`. Those accounts start out
established, may draw and post exams, and, when nobody else who is around could do it, may
run and grade the organisers' own reference curators and grade a script they wrote. That
is all the name buys.

It has to meet the floor like anyone's (`stet bench`, on a quiet GPU; the measurement is
kept for a week). Edit time is wall-clock time, so do not run other GPU or heavy CPU
work on it while it is running a curator.

One verifier runs per key. A second `stet verify` on the same key refuses to start.

A verifier that fails a job keeps it and tries again after one minute and after five
before handing it back. A failed download is waited out with the job kept for about half
an hour, then handed back. That matters because the server counts every job it hands
out: after 3 failed edit jobs or 5 failed grade jobs an entry is void for the round.
For the organisers' account a restart by systemd costs no attempt: it gets the same job
back. Ctrl-C when run by hand does hand the job back, and costs one.

The organisers' verifier stops by itself, with the job still held, in two cases: it
cannot vouch for the round (the seed, the moment of the draw or the announcement does
not check out), or a finished result could not be delivered after many tries (the server
is away, or its file store is failing). Neither spends an attempt. systemd starts it
again a minute later, it is given the same job, and it checks the round or does the job
afresh. The log says which it was. A volunteer's verifier hands the job back instead.

**To add an organisers' account:** install the CLI there, `stet login --handle <name>`,
add its public key (`stet whoami` there, or `GET /api/v1/admin/accounts/<name>` signed by
an organisers' key) to `STET_ANCHOR_KEYS`, and then start `stet verify` (or
`stet service install --edits`). A verifier that was already running has to be started
again to notice. Whoever holds such a key is established from the start, can settle an
entry alone, can post a round's exam, and is given every open round's exam, so there
should be few.

**It upgrades itself.** Like any verifier, it installs a new release when it holds no
job and starts again, so a deploy that changes the harness needs nothing done here. The
log says `upgraded to <version>. Starting again.` To do it by hand: `stet upgrade`, then
`systemctl --user restart stet-anchor`.

## The examiner

Every official round's exam is made by an organisers' machine running `stet examine
--follow`. It does two things.

- **Every hour it harvests an hour of the clock**: the writing people published in it, once
  the hour has settled (an hour after it ends, when the sources that publish late have
  caught up). It keeps the harvest under `~/.cache/stet/examine/hours/` and tells the
  server its hash at once; `GET /api/v1/harvests` lists them. It reads every source at
  once, so a harvest takes as long as its slowest source (Lemmy, about ten minutes).
- **When a round is drawn it makes the round's exam**, within a few minutes. The server
  gives the round three of the hours harvested in the day before the draw, chosen by the
  round's seed. The examiner takes exactly those, builds the round's corpus, drops the
  fresh items the corpus already holds, generates the round's questions, and posts the
  exam. No round waits for its exam.

So it has to be running for a day before a season's first draw, and a stop of more than a
day leaves the next round without hours to draw from. Started again, it harvests the
newest hours first.

What it needs:

- an organisers' key (`stet login` there, the key in `STET_ANCHOR_KEYS`): only the
  organisers may post an exam;
- the stet client with the season's libraries, PyTorch among them, and the same version of
  the exam's generators as the season (an older client refuses, and says to upgrade);
- about 3 GB of disk a round, under `~/.cache/stet/examine/<round>/` and the round's corpus,
  and the hours it has harvested;
- the network: Wikipedia, Common Crawl's news crawl, GH Archive, Hacker News, Lemmy,
  Bluesky's Jetstream, Mastodon, the forums that run Discourse and the kernel's mailing
  lists, none with a key or an account. A harvest downloads about a gigabyte of news an
  hour.

Leave it running as a user service, like the verifier. It does not upgrade itself: after
a deploy that changes the harness, `stet upgrade` and restart it.

```ini
# ~/.config/systemd/user/stet-examine.service
[Unit]
Description=Stet examiner: harvests every hour, and makes each official round's exam at its draw
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=%h/.local/bin/stet examine --follow
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=60
StandardOutput=append:%h/.local/state/stet/examine.log
StandardError=append:%h/.local/state/stet/examine.log

[Install]
WantedBy=default.target
```

```sh
systemctl --user daemon-reload && systemctl --user enable --now stet-examine
tail -f ~/.local/state/stet/examine.log
stet examine --round s1-o00003      # one round, by hand, once its exam can be made
```

**If it stops, grading waits.** Curators still run and graders still train, but no student
sits the exam until it is posted, and the timers that give up on missing grades count from
then, so nothing is lost meanwhile. Started again, it makes the exam of any open round still
without one. What it harvested is kept, so a second try does not harvest again. Do not
leave it down for long: Bluesky's relay holds about a day and a half of posts, and a source
that can no longer give the window is left out of that round's exam. A round it has failed
five times is left alone until it is restarted; the log says why.

A source that fails during a harvest is left out of that round, and the rest is kept. The
log names it.

It also keeps each round's whole exam beside the harvest (`items.jsonl.zst` and
`languages.json` under `~/.cache/stet/examine/<round>/`), with the text of the sources that
are never published. [The organisers' measurements](#the-curve-and-the-audit) read it from
there.

## The early days: only the organisers' machines

With one established machine, every entry is run once and graded by the machine that ran
it, and marked *run once*. Rounds do not wait for a second person who is not there, so
the board fills at the speed of that one machine: a run of up to twenty minutes and a
grade of about 18 minutes on a 3090-class GPU, an entry a round. The scripts are published
when the round settles, so every one of those scores can be re-run by anyone later.

More of the organisers' machines help in one of two ways, and `STET_RUNS_PER_ENTRY`
chooses which:

| `STET_RUNS_PER_ENTRY` | With two organisers' machines | Use it when |
|---|---|---|
| 2 (the default) | Each entry is run by both and each grades the other's script: two runs on two machines that have to agree. Same speed as one machine | A second machine's agreement matters more than speed |
| 1 | Each entry is run once by whichever machine is free, and graded by the other when it is around: about twice the speed | The queue is long and the machines are the organisers' own |

Each machine needs its own key (one verifier per key) and each key in `STET_ANCHOR_KEYS`.
A machine that only grades (`stet service install`, no `--edits`) needs no sandbox and no
floor, and is enough to turn *run once* into a second opinion on every script.

The organisers' machines count as one person, however many there are. An entry that only
they ran and graded is marked *one grader*, never *confirmed*: that word is kept for two
people who did not choose each other.

When volunteers arrive nothing needs changing: an established volunteer who runs curators
is preferred for the second run, and the organisers' machines stop grading their own
scripts as soon as somebody else can.

## Rented machines

While there are few volunteers, the organisers can rent GPUs by the hour to run curators
and grade: `stet-cloud` ([its README](../cloud/README.md)) rents them from five markets
(Vast.ai, RunPod, Lium, Clore.ai and Salad) when a round has work, and lets them go when it
has none. It is built on its own, from `cloud/`, and is not part of the stet command line or
the server.

RunPod's machines are containers that cannot make bubblewrap's namespaces (Docker's
system-call filter refuses them, and there is no systemd), so a rented machine runs
curators in the **container sandbox** instead (`stet verify --edits --sandbox container`,
`src/stet/cage.py`): a user of the curator's own, a system-call filter that hands it only
TCP sockets bound to the loopback device, Landlock to keep it off the machine's other
ports (Docker's DNS resolver among them), and the season's limits. What differs for a
curator is in [the rules](RULES.md#the-curators-machine). The server takes curators run in
it from the organisers' machines only (`runsEditors`), and each run says which sandbox it
was in. `python tools/cage_check.py`, as root on such a machine, tries to get out of it the
ways a curator might, and says what held.

A host where the container sandbox cannot be made (an old kernel, no Landlock) still
grades: the machine falls back to `stet verify` without `--edits`.

Each rented machine has a key of its own from a pool kept by `stet-cloud`, and one of the
organisers' own keys (in `STET_ANCHOR_KEYS`) **vouches for it**: it signs a grant naming the
machine's key, and the server keeps the grant (`POST /api/v1/admin/workers`). Nothing in
the server's configuration changes. The grant is kept with its signature and checked each
time the server reads it, and it counts only while the key that signed it is still one of
the organisers': a row in the database cannot make an organisers' machine by itself, and
taking an organisers' key out of `STET_ANCHOR_KEYS` takes its grants with it. A machine
vouched for is one of the organisers' machines in everything the judge decides: established from the
start, one person with the rest of the organisers' machines, given the organisers' own
entries only when nobody else can take them. It is nothing more: it administers nothing
(no draws, no exams, no account lookups). Stopped, a rented machine hands back the job it
holds, as a volunteer's does.

Setting it up:

1. `stet-cloud keys --add 6`, on a machine that holds an organisers' own key: six keys,
   each vouched for with that key and given its account.
2. `stet-cloud scale --follow` there (it asks the server what is waiting with the same
   key), or `stet-cloud up 2` to rent two by hand. A key in the pool that nobody has
   vouched for is vouched for when it is first rented.

A rented machine installs stet when its pod starts (about two and a half minutes) into
`/opt/stet`, where the curator's user can read it, keeps its key and cache under `/root`,
where it cannot, works until it has had nothing to do for fifteen minutes
(`stet verify --exit-when-idle 15`), and then ends its own pod. `stet-cloud scale` rents
more when a round has curators to run or scripts to grade, ends those doing nothing as
soon as nothing at all is waiting, and ends any that went quiet. When rounds are drawn on
demand (`STET_ROUND_MIN_GAP`) and an entry is waiting for one with nobody who runs curators
around, it rents one, so that the round is drawn early as it would be for anyone. It never rents past `--max` machines or
`--max-hourly` dollars an hour, never a machine dearer than `--max-price`, and stops
renting once `--daily-budget` is spent.

Each machine rented is the cheapest that will do at that moment, from any of the markets,
and on RunPod from either of its clouds (`--cloud any`): nearly always strangers' machines.
The keys are on them while they work, and a host could read one. Anyone holding a worker
key can grade as the organisers, so a key that may have leaked is taken back (`stet-cloud
keys --revoke stet-rp-03`, or `DELETE /api/v1/admin/workers/<key>`), and the pool gets a
new one. `--cloud secure` rents only from RunPod's own data centres, at two or three times
the price.

### The farm

A machine of the organisers' with nothing official to do trains the farm's students: the
finals', and the organisers' own measurements. The organisers' machine posts them as a
batch ([the API](API.md#the-organisers-farm)); any organisers' machine running
`stet verify`, rented ones included, takes one when no official job is waiting, and sends
back how the student did on every item. Nothing here touches a grade, and a student is
never handed to a machine an official job could use.

`stet-cloud` rents for the farm too, after the official work. With `--long-farm MINUTES`
the machines it rents also take the bigger students, held for hours (they are told
`STET_FARM_MAX_MINUTES`), one machine to each. Those machines are chosen for what a
student costs on them (price over speed, the quicker where that is about the same), under
`--long-max-price` rather than `--max-price`.

What the commands that post to the farm rely on ([the curve and the audit](#the-curve-and-the-audit),
[the finals](#finals)):

- A batch is named by its round, the items its students sit and their size, so asking for
  the same student twice trains it once, and two measurements on one round share students.
- A student of a size the season has no name for (the curve's `standard-x2`, `large-x4`) is
  posted with its whole specification. The machine trains it on the corpus of the season's
  size with as many units, which it may hold already. A machine on an older client builds
  the same corpus again under the new name: upgrade the machines first.
- A machine holds a student for 45 minutes unless its batch says longer. The commands ask
  for longer when the slowest GPU that may run curators would need it: a quarter more than
  that GPU's time, and an hour for the corpus. A student held for more than an hour goes
  only to a `--long-farm` machine allowed that many minutes; each plan says how many.
- A student three machines have failed at is left undone under that name. The commands ask
  for it again under the next one (`unedited#3`), up to nine of one script.
- The server keeps the last 60 batches. The commands keep every result on disk as it comes
  back, and post again only what they do not hold.

## Pausing the rounds

When the machines are wanted for something else (finals, a measurement), and nobody else is
entered, the organisers can pause the rounds: `stet-cloud pause`
(`POST /api/v1/admin/pause {"paused": true}`, organisers only). No round is drawn while they
are paused, by the clock, early or by hand; the round under way finishes, exam and all. The
machines then have nothing official to do and turn to the farm's students, and
`stet-cloud` rents only for those. The site says the rounds are paused, and the season's
`next_round` is empty. `stet-cloud resume` starts them again: the next round is drawn at
the next slot on the grid.

## The curve and the audit

Three measurements are made on settled rounds with [the farm](#the-farm): the unedited
curve, the scale audit, and [the finals](#finals). Their commands are the organisers' (not
in `stet --help`), run on a machine with an organisers' key, one command a stage.
Everything a stage makes is kept under `~/stet-measurements/` (`--out`, or
`STET_MEASUREMENTS`), by round and size. A stage that was stopped is run again and carries
on, and nothing is trained twice. That directory holds whole exams, with text that is
never published: keep it private.

Each needs the round's whole exam. On the examiner's machine it is found where the
examiner kept it. Elsewhere, copy `items.jsonl.zst` and `languages.json` from the
examiner's `~/.cache/stet/examine/<round>/` and pass them as `--items` and `--shares`. It
is checked against the round's `exam_digest` either way.

What a training costs, on the slowest GPU that may run curators. These are worked out from
the season's floor (`runner.gpu_tokens_per_second`, 80,000 in `stet bench`) and the
arithmetic of each student, not measured; every plan prints them for what it is about to
train, and `--speed N` shows them for a card N times the floor. Rented cards are quicker:
a 3090 trains the standard student in about 18 minutes and a 4090 the large one in about
an hour, a half and a quarter of the floor's time.

| student | tokens | GPU hours, one training | held (minutes) | memory for its tokens |
|---|---|---|---|---|
| standard at ½× | 80M | 0.31 | 45 | 0.4 GB |
| standard | 160M | 0.62 | 45 | 0.9 GB |
| standard at 2× | 320M | 1.25 | 155 | 1.8 GB |
| standard at 4× | 640M | 2.50 | 250 | 3.6 GB |
| large at ½× | 320M | 2.27 | 230 | 1.8 GB |
| large | 640M | 4.53 | 400 | 3.6 GB |
| large at 2× | 1,280M | 9.06 | 740 | 7.2 GB |
| large at 4× | 2,560M | 18.13 | 1,420 | 14.3 GB |

Rent for the longest hold of what is posted (`stet-cloud --long-farm 1420` for a curve at
large), and give the large student at 4× a machine with more than the floor's 16 GB of
memory: its tokens alone take 14.

### The unedited curve

A multiplier is read from the unedited corpus's student at multiples of the training
budget. `stet curve` trains it at ½, 1, 2 and 4 times (`--multiples`), twice at 1× and once
elsewhere (`--repeats`, `--repeats-elsewhere`), and gives each point as a gain over the
mean student at 1×, under the round's own score.

```sh
stet curve s2-o00012 --plan                           # what would be trained and what it costs; posts nothing
stet curve s2-o00012                                  # post the students, follow them, print the curve
stet curve s2-o00012 --scale large                    # the same at another size
stet curve s2-o00012 s2-o00015 s2-o00019 --average    # the mean of rounds already measured, with the spread between them
```

One round's curve is 5.3 GPU hours at standard and 38.5 at large, at the floor. It prints
the gain at each multiple, the gain per doubling of the data between each pair, and, on
standard output, the line for the season file:

```
"official/standard": [[0.5, 0.04], [1, 0], [2, -0.035], [4, -0.06]]
```

Each point is a multiple and minus the gain there, 0 at 1×. The multiplier reads only
differences from the point at 1×, so the client and the server read it as they stand. Put
the line of several rounds' average under `reference_curves` in
`src/stet/seasons/<season>.json`, then `python tools/export_season.py > server/season.json`
and deploy. A curve is not part of the season's digest, so no verifier has to upgrade, and
`has_multiplier` in `/api/v1/season` turns true by itself.

The student at 4× may come to the end of the round's corpus and read some of it twice,
which is not four times the data. The plan says how much of the corpus each multiple is,
from a count of its tokens; the training says whether it reached the end. Such a point is
shown, marked, and left out of the line. A curve on which more data did not always do
better is said to be one, and cannot be read as a multiplier: average more rounds.

### The scale audit

The audit trains a settled round again, at its own size and one size up, and asks whether
the order of curators at the rounds' size carries to the finals' size.

```sh
stet audit plan s2-o00012       # the round's entries, the sizes, the trainings, the GPU hours
stet audit scripts s2-o00012    # build the corpora here, and run each curator on the larger one in the sandbox
stet audit farm s2-o00012       # post the students at both sizes, and follow them
stet audit report s2-o00012     # collect them and add up
stet audit report --over s2-o00012 s2-o00015 s2-o00019    # every round audited so far, as one
```

`plan` takes `--entries` (only these curators), `--sizes`, `--trainings` (2 of each script)
and `--baselines` (4 of the unedited corpus). At the round's own size the scripts the round
scored are used. A script is made for one corpus, and the larger size has another, so
`scripts` runs every curator again on that one, with that size's time limit (4,800 s at
large). It is the only stage that runs anybody's code and the only one that needs a corpus
on the machine: run it by hand, on a machine at or above the floor with nothing else
running. A curator that writes no usable script at the larger size, or one that keeps the
corpus as it is, is left out of the comparison and named.

With N curators an audit is 2N + 4 trainings at each size: for eight, 12.5 GPU hours at
standard and 90.6 at large at the floor, and up to 10.7 hours of curators on the machine
that makes the scripts. `--sizes standard` audits a round at its own size alone, for its
noise and spread there: `scripts` then only builds the round's corpus, no curator is run,
and the 12.5 hours are all of it.

For the score and each pillar, the report gives:

- the **noise**: how far two trainings of one script differ;
- the **spread**: how far the curators differ, and the spread over the noise, at each size;
- how far the order at one size **carried** to the next (rank agreement: 1 is the same
  order, 0 none).

One round has a handful of curators, so its rank agreement is rough. `--over` gives the
rounds audited so far as one: noise and spread pooled over the rounds, and the rank
agreement averaged, with how many rounds each rests on.

## Finals

The season's table is decided by finals ([the rules](RULES.md#finals)). The commands are
`stet finals <stage>`, built on the same stages as the audit, and everything is kept under
`~/stet-measurements/finals-<season>/`.

1. **Announce** the finals, and the season's close two weeks ahead: the rules promise it.
   Entries close with the announcement.
2. **Freeze the board** when the last round has settled: `stet finals plan`. The first run
   decides who is in the finals, and later runs only add to the plan. It keeps the board as
   it stood, the eight finalists (the board's order, one per person), the curators the lot
   is drawn among (the next eight, with five rounds or more), the yardsticks (the
   organisers' curators on the board, from the accounts the server marks as the
   organisers'; `--organisers` names them instead), and the newest crawl there was. It
   prints what the finals will cost.
3. **Draw the lot.** Name a drand round still to come and publish it with the frozen board
   (`finals-<season>/plan.json`). Once it has been emitted: `stet finals plan --lot-beacon
   <drand round>`. The value is fetched from drand and checked, the pick is kept in the
   plan, and the lot is not drawn again from another round. The command refuses a round
   emitted before the freeze. That the round was named before it was emitted is shown by
   having published it, which no command can do for you.
4. **Wait for a crawl** that Common Crawl publishes after the freeze, and for the server to
   adopt it (`crawl` in `/api/v1/season`, newer than `crawl` in the plan). The rounds can
   be [paused](#pausing-the-rounds) meanwhile.
5. **Draw the two finals rounds.** Each is an ordinary official round on that crawl. The
   server names a draw's drand round before the draw (`next_round` in `/api/v1/season`):
   publish it as a finals draw before it is emitted, and record it with `stet finals plan
   --draws <drand round> <drand round>` (every one, each time). When both rounds have
   settled: `stet finals plan --rounds <round> <round>`. It refuses a round drawn from a
   drand round that was not recorded, or from a crawl no newer than the newest at the
   freeze.
6. **Run the curators**: `stet finals scripts`, by hand, on a machine at or above the floor
   with nothing else running. It builds each round's large corpus, leaves out the exam
   items that corpus holds, and runs every finalist, the lot's pick and every yardstick on
   it in the sandbox, with the large size's 4,800 s.
7. **Train the students**: `stet finals farm` posts them and follows them. Rent for them
   with `stet-cloud --long-farm 1420`.
8. **The table**: `stet finals table`. It can be run while students are still out: it says
   what is missing and that the table is not final. It writes what is published to
   `finals-<season>/published/`: the plan (the frozen board, the lot's beacon round and its
   pick), the table, each round's exam as a settled round's is published, and every
   student's score on every item. Publish those with the scripts
   (`~/stet-measurements/<round>/large/scripts/`), and put the table on the site as a page.

What it trains, for n curators (eight finalists, the lot's pick and the yardsticks): in
each of the two rounds, 2n + 4 large students and the unedited corpus once at ½, 2 and 4
times the budget. With all eight of the organisers' curators on the board that is 76 large
students, 344 GPU hours at the floor, and 59 for the two curves: 403, or about 89 on 4090s.
The curators themselves take up to 45 hours on the machine that runs them (two rounds, 17
curators, 4,800 s each), and each large corpus is built first.

Rehearse first. `stet finals plan --rehearsal --scale standard --rounds <round> <round>
--lot-beacon <a drand round already emitted>`, and then the other stages with
`--rehearsal`, go through all of it on any settled rounds at a size that costs little. A
rehearsal is kept apart under `finals-<season>-rehearsal/` and says that it is one in
everything it writes. Rehearse once at large on one round too, before the finals depend on
it: the large corpus, the curators' 4,800 s, the long holds and the memory of the student
at 4× are met nowhere else.

## Deploying

1. Deploy between rounds, and roll back only between rounds. A round in flight is graded
   by one harness version; an older server cannot finish a round a newer one opened.
2. Push to `main`. Check `gh api repos/ericflo/stet/commits/<sha>/status`, then
   `/api/v1/season` for the new `server` revision.
3. If the harness version changed (`HARNESS_VERSION` in `src/stet/spec.py`), the server now
   refuses the old client. Upgrade the organisers' verifier and the examiner straight away:

   ```sh
   curl -fsSL https://www.stetrun.com/install.sh | sh    # picks the PyTorch build for the GPU it finds
   stet doctor && stet bench                             # libraries at the season's versions; at or above the floor
   systemctl --user restart stet-anchor stet-examine
   ```

Bump the harness version whenever a change could alter a corpus, a script's meaning, a
score, the exam, or what a curator finds when it runs (the library list, the machine it is
given). Volunteers on the old version are told to upgrade.

**Changing the library list.** The list is `environment` in `src/stet/seasons/s1.json`.
After editing it: `python tools/sync_environment.py` (rewrites the `run` extra in
`pyproject.toml`), `python tools/export_season.py > server/season.json`, update the list
in `src/stet/templates/AGENTS.md` (a test compares them), and bump the harness version.
Every version must have wheels for the season's Python on Linux x86_64 at least.

Epsilon's `apps/stet` manifest and the server image deploy separately. A manifest that
relies on new server behaviour (the readiness probe on `/healthz` is one) goes in after
`/api/v1/season` shows the new `server` revision, and comes out in the same change as any
rollback of the image.

## A new season

A season freezes the student, the budgets, the exam and the rules. The next one is a new
file in `src/stet/seasons/`, and the switch is:

1. Announce it, and finish the current season with its finals. (Season 1 had one entrant
   and was paused after round 14: it has no finals, and Season 2 begins when its change is
   merged.)
2. In one commit: `CURRENT_SEASON` and a new `HARNESS_VERSION` in `src/stet/spec.py`;
   `python tools/export_season.py <season> > server/season.json`; the rules in
   `docs/RULES.md`. Then the tests, and push.
3. The server keeps one season, and rounds and the record are not kept by season, so the
   new one starts on its own database: set `STET_DB_NAME` to a new name (and `STET_SCALE`,
   if the official scale changes) in Epsilon's `apps/stet` manifest, and deploy. The old
   database stays as it was.
   Archive the finished season's site from it ([below](#archiving-a-season)), so that its
   pages stay readable under `/seasons/<id>/`.
4. Enter the reference curators from the organisers' account, before the first draw:
   `for d in identity english tidy heuristic dclm fineweb2 fineweb-edu mimic; do stet submit examples/$d --yes; done`.
   Three new entries are admitted a round, in the order they were submitted.
5. Restart the organisers' verifier and the examiner on the new client. The examiner
   (`stet examine --follow`) harvests an hour of writing every hour and commits to it, and a
   round's exam is drawn from the day before its draw: so it runs for a day before the first
   round is drawn (`GET /api/v1/harvests` lists what it has committed).
6. Watch the first round through: its exam posted (`/api/v1/rounds/<id>` shows
   `exam_digest`), grades arriving, the round settling.

## Archiving a season

The server runs one season. A finished one keeps its pages as they stood when it closed,
under `/seasons/<id>/`: what each page said, shown inside today's masthead and footer under a
bar that names the season. `/seasons` lists them, and the footer links there. Each season is
one file, `server/archive/<id>.jsonl.zst`, built into the server. Making one takes the
season's last server build, a copy of its records, and its curators' code:

1. **Its records.** Export the season's database (`stet` for season 0, `stet-<id>` after)
   as JSON Lines of `{"key", "value"}`: a scan of every key at one revision, with a token
   that can only read that database. Drop any round that is not settled (one announced and
   never drawn): the archiver refuses a season with one.
2. **Its curators' code.** Copy `bundles/<editor id>.tar.zst` for every `editor/*` key from
   the bucket into `<data>/blobs/bundles/`. Curator pages show their code from these.
3. **Its server.** Build the season's last server commit in a worktree and run it on the copy,
   with nothing real at hand: no `STET_DB_URL`, no `STET_S3_*`.

   ```sh
   STET_ARCHIVE_SEED=export.jsonl STET_DATA_DIR=<data> STET_ADDR=127.0.0.1:18080 \
   STET_BASE_URL=https://www.stetrun.com STET_API_URL=https://api.stetrun.com \
   STET_ANCHOR_KEYS=<as in production> STET_DOCS_DIR=<worktree>/docs \
   STET_AGENTS_FILE=<worktree>/src/stet/templates/AGENTS.md ./stet-server
   ```

   `STET_ARCHIVE_SEED` reads the records into memory, schedules nothing, and writes times as
   dates rather than "2d ago". Season 0's build is older than it; its archive was made with
   the same change applied to a copy.
4. **The pages.** With the stet CLI's Python (it needs `zstandard`):

   ```sh
   python tools/archive_season.py --server http://127.0.0.1:18080 --season s1 \
     --title "Season 1" --about "One line for /seasons." --out server/archive/s1.jsonl.zst
   ```

   It keeps the title, description and content of every page it can reach from the
   leaderboard, which becomes the season's front page: rounds, curators with each of their
   files, accounts, verifiers, glossary, documents. Links stay inside the archive, links to
   the guides go to the live site, and the API and downloads are left as plain text.
5. Commit the `.jsonl.zst`, never the export or the bundles: the export holds every account
   record. The server's tests check every archived page and link.

## Things only the organisers hold

- `~/.config/stet/key.json`: the organisers' account's private key. There is no other copy.
- The server's database key and bucket keys, SOPS-encrypted in Epsilon (`apps/stet`,
  `apps/stet-db`).

## Server settings

Set in `apps/stet/deployment.yaml`. The ones that are rules:

| Variable | Default | Meaning |
|---|---|---|
| `STET_LADDER_SIZE` | 8 | Curators that run in every round |
| `STET_MAX_NEW_PER_ROUND` | 3 | New entries admitted per round |
| `STET_ROUND_INTERVAL` | 3h | A round is drawn this often whether or not anybody is waiting: the season's cadence, and what it costs a day |
| `STET_ROUND_MIN_GAP` | 0 | When above 0, a round is also drawn early for a waiting entry, this long after the last draw at the soonest, while a runner is around. `0` keeps to the clock |
| `STET_EARLY_WINDOW` | 10m | A machine the organisers rent is handed grades before an exam only once they, and the further ones it could take, would keep it busy until no sooner than this before the exam is due; it is not paid to wait. `0` hands them out as soon as there are any |
| `STET_ENTRIES_IN_PLAY` | 3 | Entries one person may have waiting or qualifying at once |
| `STET_SUBMISSIONS_PER_DAY` | 6 | Submissions one person may make in a day |
| `STET_QUALIFY_ROUNDS` | 3 | Rounds a new entry runs |
| `STET_RATED_ROUNDS` | 3 | Rounds before a curator is ranked |
| `STET_GITHUB_MIN_AGE_DAYS` | 30 | Age of the GitHub account an entrant must link |
| `STET_GITHUB_TOKEN` | none | Optional. Without it GitHub lookups share 60 requests an hour |
| `STET_ANCHOR_KEYS` | | Public keys of the organisers' accounts, comma-separated. They vouch for the machines the organisers rent through the API ([above](#rented-machines)) |
| `STET_ESTABLISH_AFTER` | 5 | Agreed grades before an account's work counts towards scores |
| `STET_RUNS_PER_ENTRY` | 2 | Different people who run each entry (a third when two differ) |
| `STET_ACTIVE_WINDOW` | 10m | How recently a verifier must have asked for work to be waited for |
| `STET_REPLICA_WAIT` | 45m | How long a round waits for verifiers still at work, and for someone to grade a script nobody else is there for |
| `STET_DB_NAME` | stet | The database the server keeps its records in: one a season |
| `STET_SCALE` | the season's | The scale rounds are run at: `standard` |

A full round is at most 8 + 3 × 3 = 17 entries, and fewer in most rounds: of the
organisers' reference curators two run every round and each of the others one round in
eight. Each entry is run twice, up to twenty minutes a run. Each script is trained once,
about 18 minutes on a 3090-class GPU and 9 on a 4090, and four machines train the round's
unedited baseline. The exam is posted a few minutes after the draw, so a student sits it
as soon as it is trained. Raise the sizes only when rounds settle well before the next
is due.

## When something is wrong

| Sign | Likely cause | Do |
|---|---|---|
| A round stays `open` | Nobody established is verifying, or a verifier is stuck | Check the organisers' unit and its log; look at the verifiers page for who was last seen |
| A round has no `exam_digest` half an hour after its draw | The examiner is down, failed on the round, or has not harvested the hours the round was given | `systemctl --user status stet-examine` and its log. Started again, it makes the exam; `stet examine --round <id>` does it by hand |
| The examiner's log says a source failed | The source was away or refused | It is left out of that round's exam and the rest is kept. If one source keeps failing, find out why before the next round |
| Every entry is marked *run once* | Only one established account is running curators | Expected early in a season. It ends when volunteers are established and pass `--edits` |
| The log says "this verifier is out of date" | The server was deployed with a newer harness | It upgrades itself within ten minutes. If the log says the upgrade did not work, it says why: run `stet upgrade` by hand |
| A runner that crashed is handed nothing | It held a curator and was started again without `--edits` | Start it with `--edits`, or wait for the job's lease to end |
| `stet verify --edits` says the machine is slower than the floor | The GPU was busy when `stet bench` ran, or the machine really is below it | Run `stet bench` again on a quiet GPU. A machine below the floor can still grade |
| `stet verify --edits` says the libraries differ | An old install, or another Python | Run the installer again |
| The site says "back in a moment" | The server cannot reach durable-db | It recovers by itself; alert `StetDBUnavailable` fires meanwhile |
| An entry is "not run" | Verifiers failed to run it three times, or to grade a script five times, or one volunteer ran it and nobody else was there to grade it | Read the round page and the log. The round is void for that entry. Three such rounds in a row, with other entries running, and it stops being drawn |
| Every entry of a round is "not run" | The machines doing the work were in trouble (disk, network, GPU) | Nothing is held against any curator for a round in which nothing ran |
| An entry is marked *runs differed* | Two people's runs of one curator scored further apart than the run tolerance and nobody was there for a third | Look at the two scripts. A curator that depends on the GPU's make or the machine's speed does this honestly; the tolerance is provisional until there are enough such pairs to measure it |
| An entry shows a dispute, or the log has a `disputed` event | Two established grades of one script disagreed and nobody was there for a third | Grade the script again (`stet score <hash> --round <id>`). If the two share nothing but a GPU type, the grade tolerance is too tight for it: the round pages have the differences |
| Many verifiers listed as "turned away" on round pages | Their unedited baseline is out of tolerance on some GPU type | The same: the tolerance is a guess until this is measured |
| An entry's submission is refused over an asset | The file is not at that commit, is not public, or its hash or size is not what `stet.toml` says | The message names the file. `stet asset add` again writes the right tables |
| Runs fail with "has not been fetched" or a download error | Hugging Face is away, or an entry's repository was removed after it was submitted | The job is kept and retried, then handed back. A repository that has gone for good makes the entry fail everywhere: its owner resubmits |
| The log repeats "Stopping with the job still held" | It cannot vouch for a round, or cannot deliver a result | Read the line before it. A result that will not deliver is the server or its bucket. A round it will not vouch for needs a person: compare the round's beacon with drand's before anything else |
| `stet verify` says another verifier is running with this key | One verifier per key | Stop the other one, or give the second its own `STET_CONFIG` and account |

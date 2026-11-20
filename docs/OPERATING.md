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
--follow`. Every minute it looks at the open rounds. Three hours after a round's draw (the
two-hour harvest window, then an hour for the sources that publish late) it harvests the
window's fresh text, builds the round's corpus, drops the fresh items the corpus already
holds, generates the round's questions, and posts the exam. It reads every source at once,
and fetches the corpus meanwhile, so the harvest takes as long as its slowest source
(Lemmy, about ten minutes); then the corpus is checked.

What it needs:

- an organisers' key (`stet login` there, the key in `STET_ANCHOR_KEYS`): only the
  organisers may post an exam;
- the stet client with the season's libraries, PyTorch among them, and the same version of
  the exam's generators as the season (an older client refuses, and says to upgrade);
- about 3 GB of disk a round, under `~/.cache/stet/examine/<round>/` and the round's corpus;
- the network: Wikipedia, Common Crawl's news crawl, GH Archive, Hacker News, Lemmy,
  Bluesky's Jetstream and Mastodon, none with a key or an account. A harvest downloads
  about a gigabyte of news an hour of window.

Leave it running as a user service, like the verifier. It does not upgrade itself: after
a deploy that changes the harness, `stet upgrade` and restart it.

```ini
# ~/.config/systemd/user/stet-examine.service
[Unit]
Description=Stet examiner: makes each official round's exam after its draw
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

**If it stops, grading waits.** Curators still run, but no grade job is handed out until
the exam is posted, and the timers that give up on missing grades count from then, so
nothing is lost meanwhile. Started again, it makes the exam of any open round still
without one. What it harvested is kept, so a second try does not harvest again. Do not
leave it down for long: Bluesky's relay holds about a day and a half of posts, and a source
that can no longer give the window is left out of that round's exam. A round it has failed
five times is left alone until it is restarted; the log says why.

A source that fails during a harvest is left out of that round, and the rest is kept. The
log names it.

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
and grade: `stet-runpod` ([its README](../runpod/README.md)) rents them from RunPod when a
round has work and lets them go when it has none. It is built on its own, from `runpod/`,
and is not part of the stet command line or the server.

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

Each rented machine has a key of its own from a pool kept by `stet-runpod`, and one of the
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

1. `stet-runpod keys --add 6`, on a machine that holds an organisers' own key: six keys,
   each vouched for with that key and given its account.
2. `stet-runpod scale --follow` there (it asks the server what is waiting with the same
   key), or `stet-runpod up 2` to rent two by hand. A key in the pool that nobody has
   vouched for is vouched for when it is first rented.

A rented machine installs stet when its pod starts (about two and a half minutes) into
`/opt/stet`, where the curator's user can read it, keeps its key and cache under `/root`,
where it cannot, works until it has had nothing to do for fifteen minutes
(`stet verify --exit-when-idle 15`), and then ends its own pod. `stet-runpod scale` rents
more when a round has curators to run or scripts to grade, ends those doing nothing as
soon as nothing at all is waiting, and ends any that went quiet. When an entry is waiting
for a round and nobody who runs curators is around, it rents one, so that the round is
drawn early as it would be for anyone. It never rents past `--max` machines or
`--max-hourly` dollars an hour, never a machine dearer than `--max-price`, and stops
renting once `--daily-budget` is spent.

Each machine rented is the cheapest that will do at that moment, from either of RunPod's
clouds (`--cloud any`), and that is nearly always the community one: strangers' machines.
The keys are on them while they work, and a host could read one. Anyone holding a worker
key can grade as the organisers, so a key that may have leaked is taken back (`stet-runpod
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

`stet-runpod` rents for the farm too, after the official work. With `--long-farm MINUTES`
the machines it rents also take the bigger students, held for hours (they are told
`STET_FARM_MAX_MINUTES`), one machine to each. Those machines are chosen for what a
student costs on them (price over speed, the quicker where that is about the same), under
`--long-max-price` rather than `--max-price`.

## Pausing the rounds

When the machines are wanted for something else (finals, a measurement), and nobody else is
entered, the organisers can pause the rounds: `stet-runpod pause`
(`POST /api/v1/admin/pause {"paused": true}`, organisers only). No round is drawn while they
are paused, by the clock, early or by hand; the round under way finishes, exam and all. The
machines then have nothing official to do and turn to the farm's students, and
`stet-runpod` rents only for those. The site says the rounds are paused, and the season's
`next_round` is empty. `stet-runpod resume` starts them again: the next round is drawn at
the next slot on the grid.

## Finals

The season's table is decided by finals ([the rules](RULES.md#finals)). Announce the close
on the site at least two weeks ahead: the rules promise it. After the last round settles,
the eight best-rated curators are run on a fresh round, and the **large** student (10 × 640,
640M tokens, about an hour on a 4090) is trained on what each writes, twice, and on the
unedited corpus four times. Those students are trained on [the farm](#the-farm): rent
machines for them with `--long-farm`. The three best are also trained one size up, about
150M parameters, as a check on the ladder that decides nothing this season. Publish the
table with every script behind it and the round's exam, and put it on the site as a page.

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

1. Announce it, and finish the current season with its finals.
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
   `for d in identity english tidy heuristic dclm fineweb2 fineweb-edu; do stet submit examples/$d --yes; done`.
   Three new entries are admitted a round, in the order they were submitted.
5. Restart the organisers' verifier and the examiner on the new client, and watch the first
   round through: its exam posted (`/api/v1/rounds/<id>` shows `exam_digest`), grades
   arriving, the round settling.

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
| `STET_ROUND_INTERVAL` | 6h | A round is drawn this often whether or not anybody is waiting |
| `STET_ROUND_MIN_GAP` | 1h | When an entry is waiting and a runner is around, the next round is drawn this long after the last draw at the soonest. `0` turns drawing on demand off |
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

A full round is 8 + 3 × 3 = 17 entries. Their runs, two of up to twenty minutes each, fit in
the three hours before the exam. Grading comes after it: one standard training for every
script graded, about 18 minutes on a 3090-class GPU, and one for each grader's baseline.
Raise the sizes only when rounds settle well within four hours of their draw.

## When something is wrong

| Sign | Likely cause | Do |
|---|---|---|
| A round stays `open` | Nobody established is verifying, or a verifier is stuck | Check the organisers' unit and its log; look at the verifiers page for who was last seen |
| A round has no `exam_digest` more than six hours after its draw | The examiner is down, or failed on the round | `systemctl --user status stet-examine` and its log. Started again, it makes the exam; `stet examine --round <id>` does it by hand |
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

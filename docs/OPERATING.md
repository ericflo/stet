# Operating Stet

For whoever runs the competition. Entrants and verifiers do not need this page.

## What runs where

| Piece | Where | How it gets there |
|---|---|---|
| Website and API | Epsilon, `apps/stet` | Push to `main` → Woodpecker builds the image → Flux rolls it out (about 4 minutes) |
| Records | durable-db, `apps/stet-db`, on its own B2 bucket | Written by the server, one conditional commit per change |
| Files (bundles, scripts, exam pack, the CLI wheel) | B2 bucket `maxint-stet-files` | Written by the server |
| The organisers' verifier | A workstation with a GPU | systemd user unit `stet-anchor` |

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
established, may draw and publish, and, when nobody else who is around could do it, may
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
again to notice. It fetches the held-out exam by itself, sealed to its own
key ([below](#sharing-it-between-the-organisers-machines)). Whoever holds such a key is
established from the start, can settle an entry alone, and is given the held-out exam, so
there should be few.

**It upgrades itself.** Like any verifier, it installs a new release when it holds no
job and starts again, so a deploy that changes the harness needs nothing done here. The
log says `upgraded to <version>. Starting again.` To do it by hand: `stet upgrade`, then
`systemctl --user restart stet-anchor`.

## The early days: only the organisers' machines

With one established machine, every entry is run once and graded by the machine that ran
it, and marked *run once*. Rounds do not wait for a second person who is not there, so
the board fills at the speed of that one machine: about a quarter of an hour an entry a
round at the official scale. The scripts are published when the round settles, so every one of
those scores can be re-run by anyone later.

More of the organisers' machines help in one of two ways, and `STET_RUNS_PER_ENTRY`
chooses which:

| `STET_RUNS_PER_ENTRY` | With two organisers' machines | Use it when |
|---|---|---|
| 2 (the default) | Each entry is run by both and each grades the other's script: two runs on two machines that have to agree. Same speed as one machine | A second machine's agreement matters more than speed |
| 1 | Each entry is run once by whichever machine is free, and graded by the other when it is around: about twice the speed | The queue is long and the machines are the organisers' own |

Each machine needs its own key (one verifier per key) and each key in `STET_ANCHOR_KEYS`;
the held-out exam reaches each by itself. A machine that only grades (`stet service install`, no
`--edits`) needs no sandbox and no floor, and is enough to turn *run once* into a second
opinion on every script.

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
entries only when nobody else can take them. It is nothing more. It administers nothing
(no draws, no files, no account lookups), and **it is never given the held-out exam**: the
exam stays on hardware the organisers own. So the grades rented machines make are gaps in
the held-out record, filled at the close like volunteers' grades. Stopped, a rented
machine hands back the job it holds, as a volunteer's does.

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

A machine with nothing official to do trains the next exam's students, and `stet-runpod`
rents for those too. With `--long-shadow MINUTES` the machines it rents also take the
bigger students, held for hours (they are told `STET_SHADOW_MAX_MINUTES`), one machine to
each. Those machines are chosen for what a student costs on them (price over speed, the
quicker where that is about the same), under `--long-max-price` rather than `--max-price`.

## Pausing the rounds

While the organisers work on the next exam, and nobody else is entered, they can pause the
rounds: `stet-runpod pause` (`POST /api/v1/admin/pause {"paused": true}`, organisers
only). No round is drawn while they are paused, by the clock, early or by hand; the round
under way finishes. The machines then have nothing official to do and turn to the next
exam's students, and `stet-runpod` rents only for those. The site says the rounds are
paused, and the season's `next_round` is empty. `stet-runpod resume` starts them again:
the next round is drawn at the next slot on the grid.

Each machine rented is the cheapest that will do at that moment, from either of RunPod's
clouds (`--cloud any`), and that is nearly always the community one: strangers' machines.
The keys are on them while they work, and a host could read one. Anyone holding a worker
key can grade as the organisers, so a key that may have leaked is taken back (`stet-runpod
keys --revoke stet-rp-03`, or `DELETE /api/v1/admin/workers/<key>`), and the pool gets a
new one. `--cloud secure` rents only from RunPod's own data centres, at two or three times
the price.

## Deploying

1. Deploy between rounds, and roll back only between rounds. A round in flight is graded
   by one harness version; an older server cannot finish a round a newer one opened.
2. Push to `main`. Check `gh api repos/ericflo/stet/commits/<sha>/status`, then
   `/api/v1/season` for the new `server` revision.
3. If the harness version changed (`HARNESS_VERSION` in `src/stet/spec.py`), the server now
   refuses the old client. Upgrade the organisers' verifier straight away:

   ```sh
   curl -fsSL https://www.stetrun.com/install.sh | sh    # picks the PyTorch build for the GPU it finds
   stet doctor && stet bench                             # libraries at the season's versions; at or above the floor
   systemctl --user restart stet-anchor
   ```

Bump the harness version whenever a change could alter a corpus, a script's meaning, a
loss, or what a curator finds when it runs (the library list, the machine it is given).
Volunteers on the old version are told to upgrade.

**Changing the library list.** The list is `environment` in `src/stet/seasons/s0.json`.
After editing it: `python tools/sync_environment.py` (rewrites the `run` extra in
`pyproject.toml`), `python tools/export_season.py > server/season.json`, update the list
in `src/stet/templates/AGENTS.md` (a test compares them), and bump the harness version.
Every version must have wheels for the season's Python on Linux x86_64 at least.

Epsilon's `apps/stet` manifest and the server image deploy separately. A manifest that
relies on new server behaviour (the readiness probe on `/healthz` is one) goes in after
`/api/v1/season` shows the new `server` revision, and comes out in the same change as any
rollback of the image.

## The held-out exam

Three files on the organisers' machine. None is ever committed, and none leaves an
organisers' machine in a form anyone else can read before the close:

| File | What |
|---|---|
| `~/.config/stet/heldout-s0.jsonl.zst` | The held-out exam. Its hash is `heldout_exam.sha256` in the season file |
| `~/.config/stet/heldout-s0.seed` | The seed it was built from, which also picks its book shards: 64 hex characters from the system's random source. Anyone with the seed can rebuild the exam, so it is as secret as the pack |
| `~/.config/stet/heldout/s0.jsonl` | One line per official grade this machine made: the loss on both exams, for the script and the baseline |

### Sharing it between the organisers' machines

Every organisers' machine needs the pack, and nobody carries it about. A machine that
holds it and is running `stet verify` seals a copy to each other organisers' key, and the
server keeps those copies. A machine that lacks the pack fetches the copy sealed to it
when `stet verify` starts, checks it against the hash the season committed to, and writes
it to `~/.config/stet/` readable by its owner only. A machine that has just been added
waits a few minutes for its copy, and says that it is waiting.

A sealed copy opens only with the private key of the account it was sealed to
(`src/stet/sealed.py`: X25519 with the account's own key, ChaCha20-Poly1305). The server
and its file store hold nothing they can read, serve the copies to organisers' keys only,
and no public address reaches them. What this trusts is the list in `STET_ANCHOR_KEYS`:
whoever can add a key to it is given the exam. The seed and the record of results are not
shared this way; they stay where they were made.

Back all three up somewhere private, with `key.json`. A machine's own sealed copy is on
the server too, so the pack can be had back with `key.json` alone. If the results file is lost, the
numbers can be measured again, at one training run per line.

The organisers' verifier, in a season with a held-out exam, refuses to start without the
pack, or if the record cannot be appended to, and sits every official student it trains
on it. If one grade's line cannot be written, the log says so in capitals, the public
grade is reported all the same, and the script is a gap to fill in (below).

This machine grades only some scripts now: the rest are graded by volunteers, who do not
hold the pack. So the record is a head start, not the table. At the close the pack is
published and every script that counted is graded on it by volunteers. Before the close,
gaps can be filled here.

`python tools/heldout_report.py ~/.config/stet/heldout/s0.jsonl` prints the table so far:
the public leaderboard's own arithmetic, on held-out losses. Do not publish it, quote it,
or tune a reference curator on it. Remember that you can see it when you choose the closing
date. It exits 0 when the table is whole, 1 when its arithmetic on the public losses does
not give the public leaderboard (stop and find out why), and 2 when that checks out but
official entries are missing from the record.

The report lists any official entry the record has no line for, with the command that
measures it again (`stet score <sha> --round <id> --heldout --json`, on this machine). Save
what that prints beside the record and pass the files with `--also`. Round s0-o00001 was
graded before the held-out exam existed, so its three entries are filled in this way. The
table written with `--out` lists every entry it used and says which came from the record
and which were measured again.

**Closing a season:**

1. Announce the closing date on the site at least two weeks ahead (the rules promise it).
2. After the last round settles, set `STET_SEASON_CLOSED=1` in `apps/stet/deployment.yaml`
   and deploy. Until then the server refuses the held-out pack, so that one command sent
   early cannot publish it.
3. `stet publish ~/.config/stet/heldout-s0.jsonl.zst --close-season s0`. The server accepts
   exactly this file, by hash, and serves it at `/files/heldout-s0.jsonl.zst`.
4. Publish what lets it be checked, under the names the server accepts once closed:
   `heldout-s0.seed`, `heldout-s0.results.jsonl` (this machine's `heldout/s0.jsonl`) and
   `heldout-s0.table.json` (`tools/heldout_report.py --out`), each with
   `stet publish <file> --name <name>`. The build command is in
   `tools/build_exam_pack.py`'s header. Each of the three is published once: the server
   refuses a second upload of the same name, so that nothing published at the close can
   be swapped later. Check each file before sending it. A mistake after that is put right
   in the bucket by hand, and said on the site.
5. Put the final table on the site as a page.

## Starting season 1

Season 1 makes the next exam the score ([the plan](research/season-1-plan.md), [its rules](RULES-s1.md)).
The code for it is in place and does nothing while the season is s0. Starting it:

1. **End season 0 as its rules say.** The close is announced on the site at least two weeks
   ahead, and then closed as above (the held-out pack published, the final table made).
   The organisers may agree with the entrants to end it sooner.
2. **The code.** In one commit:
   - `CURRENT_SEASON = "s1"` and a new `HARNESS_VERSION` (0.4.0) in `src/stet/spec.py`;
   - `python tools/export_season.py s1 > server/season.json`;
   - `git mv docs/RULES.md docs/seasons/RULES-s0.md && git mv docs/RULES-s1.md docs/RULES.md`.

   Then the tests, and push.
3. **The server.** Season 1 starts on its own database, since rounds and the record are not
   kept by season: set `STET_DB_NAME` to a new name and `STET_SCALE=standard` in epsilon's
   `apps/stet` manifest, and deploy the image from step 2. Season 0's database stays as it
   was, for a read-only copy of its site if one is wanted.
4. **The examiner.** On an organisers' machine with the new client: `stet examine --follow`,
   left running like the verifier (a `stet-examine` user service). It makes each round's
   exam five hours after the draw, and nothing is graded until it does: if it stops, rounds
   wait for it. It needs about 3 GB of disk a round and a gigabyte of news downloads an hour
   of window.
5. **The machines.** Restart the organisers' verifier on the new client. stet-runpod's
   machines install the newest client when they start. A standard student needs a GPU with
   12 GB or more, and trains in about 18 minutes on a 3090 or a 4070 Ti.
6. **Then** `stet-runpod resume`, and watch the first round through: its exam posted
   (`/api/v1/rounds/<id>` shows `exam_digest`), grades arriving, the round settling.

## The next exam's shadow

Season 0 is measured again, round by round, on the exam proposed for season 1
([the proposal](research/exam-proposal.md)). It ranks nothing; the site shows it beside the
season's scores (`/next`, and each settled round's page).

An organisers' machine measures it, the faster the better: it trains the student once for the
unedited corpus and once for every entry of the round (about two minutes each on a 4090). It
harvests the round's fresh text three hours after the harvest window closes (five hours after
the draw), once Common Crawl's news files for the window are public and Hacker News's edits
have settled. From a checkout:

```sh
git pull && uv run stet shadow --follow           # every round drawn in the last day, then each new one as it settles
uv run stet shadow --round s0-o00014 --fresh --upload   # one round, by hand
```

With `--farm` (and `--follow --farm`), the students are trained by the organisers' machines
instead, rented ones included: `stet verify` on any of them takes a student when nothing
official is waiting, and stet-runpod rents for them (one machine for every four waiting, under
the same caps). The machine running `stet shadow` only harvests, posts the round's batch, and
adds up what comes back.

```sh
uv run stet shadow --follow --farm                    # every round, its students trained across the fleet
```

It needs the machine's organisers' key (`stet login`), and keeps what it trains under
`~/.cache/stet/shadow/`, so a round can be sat again when the exam grows without training again.
The harvester reads Wikipedia, Common Crawl's news crawl, GH Archive, Hacker News, Lemmy, Bluesky
(Jetstream) and Mastodon, none of them with a key or an account. It asks each politely and
waits when told to, and a source that fails is left out of that round's snapshot. A harvest
downloads about a gigabyte of news per hour of window and takes several minutes.
StackExchange is off: without a key it allows a few hundred requests a day.

## Things only the organisers hold

- `~/.config/stet/key.json`: the organisers' account's private key. There is no other copy.
- The held-out exam files above.
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
| `STET_SEASON_CLOSED` | off | Set when the season is over: the held-out exam may then be published |

A full round is 8 + 3 × 3 = 17 entries. Run once and graded once on one machine, that
is between three and a quarter and four and a half hours of GPU at the official scale
([Experiments](EXPERIMENTS.md#how-long-a-round-takes)); run twice and graded by others
it is about twice the work, spread over whoever is lending a GPU. Raise the sizes only
when rounds settle well inside six hours.

## When something is wrong

| Sign | Likely cause | Do |
|---|---|---|
| A round stays `open` | Nobody established is verifying, or a verifier is stuck | Check the organisers' unit and its log; look at the verifiers page for who was last seen |
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
| Many verifiers listed as "turned away" on round pages | Their unedited loss is out of tolerance on some GPU type | The same: the tolerance is a guess until this is measured |
| An entry's submission is refused over an asset | The file is not at that commit, is not public, or its hash or size is not what `stet.toml` says | The message names the file. `stet asset add` again writes the right tables |
| Runs fail with "has not been fetched" or a download error | Hugging Face is away, or an entry's repository was removed after it was submitted | The job is kept and retried, then handed back. A repository that has gone for good makes the entry fail everywhere: its owner resubmits |
| The verifier exits saying a pack "is not the held-out exam that season s0 committed to" | Wrong or damaged pack | Restore it from backup |
| The organisers' verifier refuses to start without the held-out pack | The pack is not on this machine | Copy it from backup. Only a development server (`STET_DEV=1`) runs without it |
| The log repeats "Stopping with the job still held" | It cannot vouch for a round, or cannot deliver a result | Read the line before it. A result that will not deliver is the server or its bucket. A round it will not vouch for needs a person: compare the round's beacon with drand's before anything else |
| The verifier exits saying the held-out record "cannot be written" | Permissions or a full disk under `~/.config/stet/heldout/` | Put it right; it does not grade until it can keep the record |
| `stet verify` says another verifier is running with this key | One verifier per key | Stop the other one, or give the second its own `STET_CONFIG` and account |

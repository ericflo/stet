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

The organisers run one verifier of their own. It is not needed for a score: editors are
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
established, may draw and publish, may run the organisers' own reference editors, and
may grade a script they wrote when nobody else is around to. That is all the name buys.

It has to meet the floor like anyone's (`stet bench`, on a quiet GPU; the measurement is
kept for a week). Edit time is wall-clock time, so do not run other GPU or heavy CPU
work on it while it is running an editor.

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
add its public key (`stet whoami`) to `STET_ANCHOR_KEYS`, copy the held-out exam to it
(below), and run `stet service install --edits`. Whoever holds such a key is
established from the start and can settle an entry alone, so there should be few.

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
| 2 (the default) | Each entry is run by both and each grades the other's script: entries are *confirmed*. Same speed as one machine | Confirmation matters more than speed |
| 1 | Each entry is run once by whichever machine is free, and graded by the other when it is around: about twice the speed | The queue is long and the machines are the organisers' own |

Each machine needs its own key (one verifier per key), each key in `STET_ANCHOR_KEYS`,
and each the held-out exam. A machine that only grades (`stet service install`, no
`--edits`) needs no sandbox and no floor, and is enough to turn *run once* into a second
opinion on every script.

When volunteers arrive nothing needs changing: an established volunteer who runs editors
is preferred for the second run, and the organisers' machines stop grading their own
scripts as soon as somebody else can.

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
loss, or what an editor finds when it runs (the library list, the machine it is given).
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

Three files on the organisers' machine, none of them ever committed or uploaded before
the close:

| File | What |
|---|---|
| `~/.config/stet/heldout-s0.jsonl.zst` | The held-out exam. Its hash is `heldout_exam.sha256` in the season file |
| `~/.config/stet/heldout-s0.seed` | The seed it was built from, which also picks its book shards: 64 hex characters from the system's random source. Anyone with the seed can rebuild the exam, so it is as secret as the pack |
| `~/.config/stet/heldout/s0.jsonl` | One line per official grade this machine made: the loss on both exams, for the script and the baseline |

Back all three up somewhere private, with `key.json`. If the results file is lost, the
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
or tune a reference editor on it. Remember that you can see it when you choose the closing
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

## Things only the organisers hold

- `~/.config/stet/key.json`: the organisers' account's private key. There is no other copy.
- The held-out exam files above.
- The server's database key and bucket keys, SOPS-encrypted in Epsilon (`apps/stet`,
  `apps/stet-db`).

## Server settings

Set in `apps/stet/deployment.yaml`. The ones that are rules:

| Variable | Default | Meaning |
|---|---|---|
| `STET_LADDER_SIZE` | 8 | Editors that run in every round |
| `STET_MAX_NEW_PER_ROUND` | 3 | New entries admitted per round |
| `STET_ROUND_INTERVAL` | 6h | A round is drawn this often whether or not anybody is waiting |
| `STET_ROUND_MIN_GAP` | 1h | When an entry is waiting and a runner is around, the next round is drawn this long after the last draw at the soonest. `0` turns drawing on demand off |
| `STET_ENTRIES_IN_PLAY` | 3 | Entries one person may have waiting or qualifying at once |
| `STET_SUBMISSIONS_PER_DAY` | 6 | Submissions one person may make in a day |
| `STET_QUALIFY_ROUNDS` | 3 | Rounds a new entry runs |
| `STET_RATED_ROUNDS` | 3 | Rounds before an editor is ranked |
| `STET_GITHUB_MIN_AGE_DAYS` | 30 | Age of the GitHub account an entrant must link |
| `STET_GITHUB_TOKEN` | none | Optional. Without it GitHub lookups share 60 requests an hour |
| `STET_ANCHOR_KEYS` | | Public keys of the organisers' accounts, comma-separated |
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
| Every entry is marked *run once* | Only one established account is running editors | Expected early in a season. It ends when volunteers are established and pass `--edits` |
| The log says "this verifier is out of date" | The server was deployed with a newer harness | It upgrades itself within ten minutes. If the log says the upgrade did not work, it says why: run `stet upgrade` by hand |
| A runner that crashed is handed nothing | It held an editor and was started again without `--edits` | Start it with `--edits`, or wait for the job's lease to end |
| `stet verify --edits` says the machine is slower than the floor | The GPU was busy when `stet bench` ran, or the machine really is below it | Run `stet bench` again on a quiet GPU. A machine below the floor can still grade |
| `stet verify --edits` says the libraries differ | An old install, or another Python | Run the installer again |
| The site says "back in a moment" | The server cannot reach durable-db | It recovers by itself; alert `StetDBUnavailable` fires meanwhile |
| An entry is "not run" | Verifiers failed to run it three times, or to grade a script five times, or one volunteer ran it and nobody else was there to grade it | Read the round page and the log. The round is void for that entry. Three such rounds in a row, with other entries running, and it stops being drawn |
| Every entry of a round is "not run" | The machines doing the work were in trouble (disk, network, GPU) | Nothing is held against any editor for a round in which nothing ran |
| An entry is marked *runs differed* | Two people's runs of one editor scored further apart than the run tolerance and nobody was there for a third | Look at the two scripts. An editor that depends on the GPU's make or the machine's speed does this honestly; the tolerance is provisional until there are enough such pairs to measure it |
| An entry shows a dispute, or the log has a `disputed` event | Two established grades of one script disagreed and nobody was there for a third | Grade the script again (`stet score <hash> --round <id>`). If the two share nothing but a GPU type, the grade tolerance is too tight for it: the round pages have the differences |
| Many verifiers listed as "turned away" on round pages | Their unedited loss is out of tolerance on some GPU type | The same: the tolerance is a guess until this is measured |
| An entry's submission is refused over an asset | The file is not at that commit, is not public, or its hash or size is not what `stet.toml` says | The message names the file. `stet asset add` again writes the right tables |
| Runs fail with "has not been fetched" or a download error | Hugging Face is away, or an entry's repository was removed after it was submitted | The job is kept and retried, then handed back. A repository that has gone for good makes the entry fail everywhere: its owner resubmits |
| The verifier exits saying a pack "is not the held-out exam that season s0 committed to" | Wrong or damaged pack | Restore it from backup |
| The organisers' verifier refuses to start without the held-out pack | The pack is not on this machine | Copy it from backup. Only a development server (`STET_DEV=1`) runs without it |
| The log repeats "Stopping with the job still held" | It cannot vouch for a round, or cannot deliver a result | Read the line before it. A result that will not deliver is the server or its bucket. A round it will not vouch for needs a person: compare the round's beacon with drand's before anything else |
| The verifier exits saying the held-out record "cannot be written" | Permissions or a full disk under `~/.config/stet/heldout/` | Put it right; it does not grade until it can keep the record |
| `stet verify` says another verifier is running with this key | One verifier per key | Stop the other one, or give the second its own `STET_CONFIG` and account |

# Operating Stet

For whoever runs the competition. Entrants and verifiers do not need this page.

## What runs where

| Piece | Where | How it gets there |
|---|---|---|
| Website and API | Epsilon, `apps/stet` | Push to `main` → Woodpecker builds the image → Flux rolls it out (about 4 minutes) |
| Records | durable-db, `apps/stet-db`, on its own B2 bucket | Written by the server, one conditional commit per change |
| Files (bundles, scripts, exam pack, the CLI wheel) | B2 bucket `maxint-stet-files` | Written by the server |
| The anchor | A workstation with the reference GPU | systemd user unit `stet-anchor` |

The server is stateless: it can be restarted or rolled at any time. `/api/v1/season` says
which build is live (`server`) and which harness version it accepts (`harness`).

## The anchor

The anchor runs every entrant's editor in the sandbox and grades every script. If it is
down, rounds are still drawn but nothing is edited and nothing settles.

```sh
systemctl --user status stet-anchor
tail -f ~/.local/state/stet/anchor.log
systemctl --user restart stet-anchor
systemctl --user disable --now stet-anchor      # stop it for good
```

It runs the installed release (`~/.local/bin/stet`), the same wheel volunteers install,
with the account whose key is in `~/.config/stet/key.json`. The server knows an anchor by
its public key: `STET_ANCHOR_KEYS` in Epsilon's `apps/stet/deployment.yaml`.

Edit time is wall-clock time. Do not run other GPU or heavy CPU work on the anchor while a
round is being edited.

The sandbox runs the anchor's own Python, PyTorch and numpy. The rules and the agent
brief name the Python version (3.15 today). If the anchor's changes, change them.

One verifier runs per key. A second `stet verify` on the same key refuses to start.

An anchor that fails a job keeps it and tries again after one minute and after five
before handing it back. A failed download is waited out with the job kept for about half
an hour, then handed back. That matters because the server counts every job it hands an
anchor: after 3 failed edit jobs or 5 failed grade jobs an entry is void for the round.
A restart by systemd costs no attempt: the anchor gets the same job back. Ctrl-C on an
anchor run by hand does hand the job back, and costs one.

An anchor stops by itself, with the job still held, in two cases: it cannot vouch for the
round (the seed, the moment of the draw or the announcement does not check out), or a
finished result could not be delivered after many tries (the server is away, or its file
store is failing). Neither spends an attempt. systemd starts it again a minute later, it
is given the same job, and it checks the round or does the job afresh. The log says which
it was. A round the anchor keeps stopping on stays open until someone looks.

**To add an anchor:** install the CLI there, `stet login --handle <name>`, add its public
key (`stet whoami`) to `STET_ANCHOR_KEYS`, copy the held-out exam to it (below), and run
`stet verify --edits` under systemd. Whoever holds an anchor key sets scores.

## Deploying

1. Deploy between rounds, and roll back only between rounds. A round in flight is graded
   by one harness version; an older server cannot finish a round a newer one opened.
2. Push to `main`. Check `gh api repos/ericflo/stet/commits/<sha>/status`, then
   `/api/v1/season` for the new `server` revision.
3. If the harness version changed (`HARNESS_VERSION` in `src/stet/spec.py`), the server now
   refuses the old client. Upgrade the anchor straight away:

   ```sh
   curl -fsSL https://www.stetrun.com/install.sh | STET_TORCH_INDEX=https://download.pytorch.org/whl/rocm7.14 sh
   systemctl --user restart stet-anchor
   ```

Bump the harness version whenever a change could alter a corpus, a script's meaning or a
loss. Volunteers on the old version are told to upgrade.

Epsilon's `apps/stet` manifest and the server image deploy separately. A manifest that
relies on new server behaviour (the readiness probe on `/healthz` is one) goes in after
`/api/v1/season` shows the new `server` revision, and comes out in the same change as any
rollback of the image.

## The held-out exam

Three files on the anchor, none of them ever committed or uploaded before the close:

| File | What |
|---|---|
| `~/.config/stet/heldout-s0.jsonl.zst` | The held-out exam. Its hash is `heldout_exam.sha256` in the season file |
| `~/.config/stet/heldout-s0.seed` | The seed it was built from, which also picks its book shards: 64 hex characters from the system's random source. Anyone with the seed can rebuild the exam, so it is as secret as the pack |
| `~/.config/stet/heldout/s0.jsonl` | One line per official grade: the loss on both exams, for the entry and the baseline |

Back all three up somewhere private, with `key.json`. If the results file is lost, the
numbers can be measured again on the same anchor, at one training run per line: grading
is deterministic on one machine.

An anchor in a season with a held-out exam refuses to start without the pack, or if the
record cannot be appended to, and sits every official student on it. If one grade's line
cannot be written, the log says so in capitals, the public grade is reported all the same,
and the entry is a gap to fill in (below).

`python tools/heldout_report.py ~/.config/stet/heldout/s0.jsonl` prints the table so far:
the public leaderboard's own arithmetic, on held-out losses. Do not publish it, quote it,
or tune a reference editor on it. Remember that you can see it when you choose the closing
date. It exits 0 when the table is whole, 1 when its arithmetic on the public losses does
not give the public leaderboard (stop and find out why), and 2 when that checks out but
official entries are missing from the record.

The report lists any official entry the record has no line for, with the command that
measures it again (`stet score <sha> --round <id> --heldout --json`, on the anchor). Save
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
   `heldout-s0.seed`, `heldout-s0.results.jsonl` (the anchor's `heldout/s0.jsonl`) and
   `heldout-s0.table.json` (`tools/heldout_report.py --out`), each with
   `stet publish <file> --name <name>`. The build command is in
   `tools/build_exam_pack.py`'s header. Each of the three is published once: the server
   refuses a second upload of the same name, so that nothing published at the close can
   be swapped later. Check each file before sending it. A mistake after that is put right
   in the bucket by hand, and said on the site.
5. Put the final table on the site as a page.

## Things only the organisers hold

- `~/.config/stet/key.json`: the anchor's private key. There is no other copy.
- The held-out exam files above.
- The server's database key and bucket keys, SOPS-encrypted in Epsilon (`apps/stet`,
  `apps/stet-db`).

## Server settings

Set in `apps/stet/deployment.yaml`. The ones that are rules:

| Variable | Default | Meaning |
|---|---|---|
| `STET_LADDER_SIZE` | 8 | Editors that run in every round |
| `STET_MAX_NEW_PER_ROUND` | 3 | New entries admitted per round |
| `STET_QUALIFY_ROUNDS` | 3 | Rounds a new entry runs |
| `STET_RATED_ROUNDS` | 3 | Rounds before an editor is ranked |
| `STET_GITHUB_MIN_AGE_DAYS` | 30 | Age of the GitHub account an entrant must link |
| `STET_GITHUB_TOKEN` | none | Optional. Without it GitHub lookups share 60 requests an hour |
| `STET_ANCHOR_KEYS` | | Public keys of the anchors, comma-separated |
| `STET_REPLICA_WAIT` | 45m | How long a settled-but-for-volunteers round waits for them |
| `STET_SEASON_CLOSED` | off | Set when the season is over: the held-out exam may then be published |

A full round is 8 + 3 × 3 = 17 entries: between three and a quarter and four and a half
hours of one anchor's GPU at the official scale
([Experiments](EXPERIMENTS.md#how-long-a-round-takes)). Raise the sizes only with more
anchors.

## When something is wrong

| Sign | Likely cause | Do |
|---|---|---|
| A round stays `open` | The anchor is down or stuck | Check the unit and its log |
| The anchor logs "this verifier is out of date" once a minute | The server was deployed with a newer harness | Upgrade the anchor (above) |
| The site says "back in a moment" | The server cannot reach durable-db | It recovers by itself; alert `StetDBUnavailable` fires meanwhile |
| An entry is "not run" | The anchor failed to run it three times, or to grade its script five times | Read the anchor's log. The round is void for that entry. Three such rounds in a row, with other entries running, and it stops being drawn |
| Every entry of a round is "not run" | The anchor itself was in trouble (disk, network, GPU) | Fix the anchor. Nothing is held against any editor for a round in which nothing ran |
| An entry shows "differed, together", or the log has a `disputed` event | Two established volunteers measured the same loss as each other and not the anchor's | Grade the script again on the anchor (`stet score <hash> --round <id>`). If the anchor repeats itself and the volunteers share a GPU type, the tolerance is too tight for that GPU: the round pages have the differences. If they do not, look at the two accounts |
| Many volunteers listed as "turned away" on round pages | Their unedited loss is out of tolerance on some GPU type | The same: the tolerance is a guess until this is measured |
| The anchor exits saying a pack "is not the held-out exam that season s0 committed to" | Wrong or damaged pack | Restore it from backup |
| The anchor refuses to start without the held-out pack | The pack is not on this machine | Copy it from backup; official grades without it leave holes in the final table. Only a development server (`STET_DEV=1`) runs without it |
| The anchor's log repeats "Stopping with the job still held" | It cannot vouch for a round, or cannot deliver a result | Read the line before it. A result that will not deliver is the server or its bucket. A round it will not vouch for needs a person: compare the round's beacon with drand's before anything else |
| The anchor exits saying the held-out record "cannot be written" | Permissions or a full disk under `~/.config/stet/heldout/` | Put it right; the anchor does not grade until it can keep the record |
| `stet verify` says another verifier is running with this key | One verifier per key | Stop the other one, or give the second its own `STET_CONFIG` and account |

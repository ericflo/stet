# stet-cloud

Rents GPUs to run curators and grade for Stet while there is more work than the organisers'
own machines and the volunteers can get through, and lets them go when there is none. It
rents single-GPU machines from five markets, each through its own API: **Vast.ai, RunPod,
Lium, Clore.ai and Salad**, each behind one `Market` interface (`market.go`). For the
organisers: entrants and volunteers never need it. It is built on its own, apart from the
stet command line and the server, and uses nothing but Go's standard library.

```sh
cd cloud && go build -o ~/.local/bin/stet-cloud .
```

It began as stet-runpod, which rented from RunPod alone, and keeps every command and setting
stet-runpod had. Its design and plan are in [STET-CLOUD.md](STET-CLOUD.md). It has run the
organisers' fleet as `stet-cloud.service` since 2026-11-21, with all five markets armed.

```sh
stet-cloud markets          # each market: key, what is left to spend, armed or not, what fits now
stet-cloud offers           # what would be rented, in order, every market (unarmed ones marked)
stet-cloud markets arm vast --note "…"   # only with Eric's approval
```

The markets' keys are kept in `~/.config/stet-cloud/credentials.env` (mode 600; `KEY=value`
lines for `RUNPOD_API_KEY`, `VAST_API_KEY`, `LIUM_API_KEY`, `SALAD_API_KEY`, `SALAD_ORG`,
`SALAD_PROJECT` and `CLORE_API_KEY`). It is read again at every look, and the environment wins
over it. Which markets are armed is kept in `markets.json`; with none, RunPod alone is.

## What a rented machine does

It is a RunPod pod running `ubuntu:24.04` and [one script](worker.sh). The script installs
stet from the site into `/opt/stet` (about two and a half minutes), writes the machine's
key under `/root`, and runs `stet verify --edits --sandbox container --exit-when-idle 15`:
it runs curators and grades, and trains the organisers' students when there is nothing
official to do, until it has had nothing to do for fifteen minutes, then ends
its own pod with the key RunPod gives each pod for itself. RunPod restarts a container
whose process exits, and keeps billing for it, so the script never just exits.

RunPod's containers cannot make bubblewrap's namespaces, so curators run in stet's
container sandbox (`src/stet/cage.py`): a user of the curator's own, which cannot enter
`/root`; a system-call filter that hands it only TCP sockets bound to the loopback device;
Landlock, keeping it off the machine's other ports (Docker's DNS resolver among them); and
the season's limits, by a cgroup where the host allows one and otherwise watched. The rules
say what differs for a curator. Where it cannot be made (an old kernel, no Landlock) the
machine only grades. `--no-edits` rents machines that only grade.

Each machine has a key of its own from a pool kept here, and the organisers' own key
(`--key`) vouches for each: it signs a grant the server keeps. Nothing in the server's
configuration changes. A machine vouched for is one of the organisers' machines to the
judge, and nothing more: it administers nothing and posts no round's exam
([Operating: rented machines](../docs/OPERATING.md#rented-machines)). A new machine takes
the key the server has heard from longest ago: the server lets each account run an entry
once, so a machine on the key that just ran a round's curators could take none of their
second runs.

## Setting up

```sh
stet-cloud keys --add 6              # six keys, vouched for with your organisers' key
```

With the markets' keys in `credentials.env` (above), run it where the organisers' key is (`~/.config/stet/key.json`, or `--key`). That is all
the server needs. `stet-cloud keys` says, for each key, who vouched for it and when it was
last at work; `--vouch` vouches for any that are not; `--revoke stet-rp-03` takes one back
and forgets it. A key not vouched for is vouched for when it is first rented.

## Running it

```sh
stet-cloud gpus                      # what would be rented, the fastest first
stet-cloud status                    # what is waiting, what is rented, what it costs
stet-cloud scale --follow            # rent and end to match the work, every minute
stet-cloud up 2                      # or by hand: two now, whatever is waiting
stet-cloud down                      # end those that hold no job
stet-cloud pause                     # draw no more rounds: the machines turn to the organisers' students
stet-cloud resume                    # and start them again
```

`status`, `scale`, `up` and `down` ask the server what is waiting
(`GET /api/v1/admin/work`) with the organisers' own key, `~/.config/stet/key.json` by
default (`--key`). Run them where that key is. `status` judges the work by the settings
`scale --follow` was last started with (kept in `service.json`), so that what it says the
service would rent and end is what the service will; settings given to it are taken over
those.

How many it rents: one for each machine at a grade, which takes no other until it is done
(about 18 minutes on a 3090 at the standard scale), and one more for every `--per-machine`
(6) scripts a new machine could grade now: `--per-machine 1` gives each its own. A grader
may take a grade before the round's exam is posted, train the round's baseline and the
student, and sit the exam once it is. The grades it could take want machines from `--early`
(90 minutes) before the exam is due, about `--exam-after` (2h12m) after the draw, and each
kind of GPU is rented just in time: so that, rented now, it would report between a minute
before the exam and `--settle-within` (15 minutes) after it. A machine that has trained
only waits, so what a grade costs on a kind is its price over the time it is held, and a
kind is rented only if none still to come would cost less: the cheapest is waited for, and
a quicker, dearer kind is rented only once the cheaper ones can no longer report in time.
How long a training pass takes on each kind is learned from the grades of settled rounds
(each says the GPU that trained, and for how long), and how long a machine takes to start
from those this service rents; until a kind has been seen, its pass is a 3090's 18 minutes
over its speed in the table, and a machine is taken to start in 8 minutes. They are kept in
`calibration.json`, and `stet-cloud gpus` shows them. A curator is run in a minute or two,
and the exam its grades wait for is made hours after the draw, so the curators want a
machine for every four entries to run, and never fewer than two, since each entry is run
twice, by two accounts. A machine with nothing official to do trains the organisers' own
students (a few minutes each, on a settled round), so one more for every `--per-farm` (4)
of those waiting, after the official work (`--per-farm 0`: none). `--long-farm 600` rents
machines that also take the bigger students, held for hours, one machine to each. A machine
rented for them is chosen for what a student costs on it, not what it costs an hour: a 4090
at $0.89 trains one in about half the time of a 3090 at $0.50, so the student costs about
the same and is done sooner. Where two cost about the same, the quicker comes first.
`--long-max-price` is the ceiling for those machines (the same as `--max-price` if not
given); `stet-cloud gpus` shows both orders. Never more than `--max` (6), or past
`--max-hourly` dollars an hour (3), and nothing new once `--daily-budget` is spent.
`--max-price` never rents a machine dearer than that, however much is waiting: the work
waits for a cheap one instead. When rounds are drawn on demand and an entry is waiting for
one with nobody who runs curators around, one, so that the round is drawn early. It never ends a machine that
holds a job. It ends at once those doing nothing when nothing at all is waiting, and any
that went quiet or stopped. The rest stop by themselves. A machine ended a moment after it
took a job, or that stopped partway through one, still holds it as the server sees it: what
it holds is handed back on the next look, so nothing waits out its lease.

Every machine rented is the fastest that will do, looked up afresh each time: every kind of
GPU RunPod has free under `--max-price`, in both its clouds, is tried fastest first (the
cheaper first where two are as fast) until one has a machine with room. A round settles
only once its slowest grade is in, and a machine's first grade of a round trains the
round's baseline too: about 16 minutes on a 4090, and 52 on an RTX A4000, which costs less
a grade. Machines to run curators are the cheapest a grade instead: once its runs are done,
such a machine takes a grade before the exam and holds it until then, so its speed buys
nothing. `--cheapest` rents every machine by what a grade costs, its price over its speed:
a 4090 at $0.89 an hour then comes before an A40 at $0.59, and an A4000 at $0.17 before
both. What will do: NVIDIA, 12 GB or more (a machine that runs curators needs 10 and gives
one 8), on a host with the season's floor of 16 GB of memory and 6 vCPUs a GPU, a driver
new enough for CUDA 13 (the season's PyTorch, 2.15.0, is built only for 13.2), bf16
(nothing older than Ampere), and a whole GPU, with a 100 GB disk (at 80, a machine that
stayed up into another round and upgraded found less than the 60 GB a curator run keeps
free, and only graded). `stet-cloud gpus` lists them as they stand.
`--gpu "RTX A5000,RTX 4090"` names them instead.

`--cloud any` (the default) takes the cheaper of RunPod's two clouds, which is nearly
always the community one: strangers' machines, at a third of the price or less. `--cloud
secure` rents only from RunPod's own data centres. The host of a community machine could
read the key it is given, and anyone holding a worker key can grade as the organisers:
take back a key that may have leaked (`stet-cloud keys --revoke HANDLE`).

Leaving it running, as a service of your own user (stet-cloud, which took over from
stet-runpod's service):

```ini
# ~/.config/systemd/user/stet-cloud.service
[Unit]
Description=Stet: rent GPUs to run curators and grade (stet-cloud)
Wants=network-online.target
After=network-online.target

[Service]
ExecStart=%h/.local/bin/stet-cloud scale --follow --cloud any --max 20 --max-price 1.10 --max-hourly 16 --daily-budget 60 --per-machine 1
Restart=always
RestartSec=60

[Install]
WantedBy=default.target
```

The markets' keys are in `~/.config/stet-cloud/credentials.env`, which stet-cloud reads itself
at every look: the unit sets none of them, since a key in the environment would win over the
file's. `systemctl --user enable --now stet-cloud`. If it stops, the machines on RunPod and
Vast still stop by themselves; only renting more stops. Only one of stet-runpod and stet-cloud
may run with a pool: stop and disable the one before starting the other.

## What it keeps

In `~/.config/stet-cloud` (`STET_CLOUD_HOME`):

| | |
|---|---|
| `workers/*.json` | The pool's keys, in stet's own key format. Back them up with the rest, or make new ones |
| `fleet.json` | The pods rented now, and when each was last seen running |
| `spent.jsonl` | Every pod that has ended: GPU, dollars an hour, start, end |
| `curators.json` | Hosts, and CPU models, under the season's floor for running curators this week: none is rented to run them |
| `machines.jsonl` | Each machine's story, an event a line: rented (for what, which round, when it should report, where, on what CPU), its market's phases, first asked for work, what it did, trouble, how it ended and why |
| `history/looks-DAY.jsonl` | One line a look (a minute): all `/admin/work` said, the machines by state and market, $/h and today's spend, what the work wanted and what was rented and ended, the plan, every market's offers and how long they took, every machine tried and what came of it, every call to a market or the server timed, everything the fleet said, and the builds. Days before today are gzipped |
| `history/offers-DAY.jsonl` | Every 10 minutes, what each market offers, by card and kind of host: how many, cheapest, median, and a grade's cost |
| `history/balances.jsonl` | Each market's account, hourly: what it really spent is what that went down by |
| `history/rounds/ID.json` | Each round as the server tells it once it has settled, whole; a line each in `history/rounds.jsonl` |
| `history/backfill/` | What could be put together for before these were kept, from the journal, the markets' own billing and the rounds |

RunPod forgets a pod once it ends and does not say what it cost, so what was spent is
worked out from these, and is an estimate. RunPod's billing page is the account.

`--ssh` lets `~/.ssh/id_ed25519.pub` into the machines it rents; `stet-cloud ssh
stet-rp-01` logs in to one. `stet-cloud logs stet-rp-01` shows what one has said (`-f`
follows it).

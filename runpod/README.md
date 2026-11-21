# stet-runpod

Rents GPUs from RunPod to run curators and grade for Stet while there is more work than the
organisers' own machines and the volunteers can get through, and lets them go when there
is none. For the
organisers: entrants and volunteers never need it. It is built on its own, apart from the
stet command line and the server, and uses nothing but Go's standard library.

```sh
cd runpod && go build -o ~/.local/bin/stet-runpod .
```

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
export RUNPOD_API_KEY=...            # https://www.runpod.io/console/user/settings
stet-runpod keys --add 6             # six keys, vouched for with your organisers' key
```

Run it where the organisers' key is (`~/.config/stet/key.json`, or `--key`). That is all
the server needs. `stet-runpod keys` says, for each key, who vouched for it and when it was
last at work; `--vouch` vouches for any that are not; `--revoke stet-rp-03` takes one back
and forgets it. A key not vouched for is vouched for when it is first rented.

## Running it

```sh
stet-runpod gpus                     # what would be rented, the fastest first
stet-runpod status                   # what is waiting, what is rented, what it costs
stet-runpod scale --follow           # rent and end to match the work, every minute
stet-runpod up 2                     # or by hand: two now, whatever is waiting
stet-runpod down                     # end those that hold no job
stet-runpod pause                    # draw no more rounds: the machines turn to the organisers' students
stet-runpod resume                   # and start them again
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
`calibration.json`, and `stet-runpod gpus` shows them. A curator is run in a minute or two,
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
given); `stet-runpod gpus` shows both orders. Never more than `--max` (6), or past
`--max-hourly` dollars an hour (3), and nothing new once `--daily-budget` is spent.
`--max-price` never rents a machine dearer than that, however much is waiting: the work
waits for a cheap one instead. When an entry is waiting for a round and nobody who runs
curators is around, one, so that the round is drawn early. It never ends a machine that
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
free, and only graded). `stet-runpod gpus` lists them as they stand.
`--gpu "RTX A5000,RTX 4090"` names them instead.

`--cloud any` (the default) takes the cheaper of RunPod's two clouds, which is nearly
always the community one: strangers' machines, at a third of the price or less. `--cloud
secure` rents only from RunPod's own data centres. The host of a community machine could
read the key it is given, and anyone holding a worker key can grade as the organisers:
take back a key that may have leaked (`stet-runpod keys --revoke HANDLE`).

Leaving it running, as a service of your own user:

```ini
# ~/.config/systemd/user/stet-runpod.service
[Unit]
Description=Rent GPUs to grade for Stet

[Service]
EnvironmentFile=%h/.config/stet-runpod/env
ExecStart=%h/.local/bin/stet-runpod scale --follow --max 20 --max-price 1.10 --max-hourly 16 --daily-budget 50 --per-machine 1
Restart=always
RestartSec=60

[Install]
WantedBy=default.target
```

with `RUNPOD_API_KEY=...` in `~/.config/stet-runpod/env` (mode 600), then
`systemctl --user enable --now stet-runpod`. If it stops, the machines still stop by
themselves; only renting more stops.

## What it keeps

In `~/.config/stet-runpod` (`STET_RUNPOD_HOME`):

| | |
|---|---|
| `workers/*.json` | The pool's keys, in stet's own key format. Back them up with the rest, or make new ones |
| `fleet.json` | The pods rented now, and when each was last seen running |
| `spent.jsonl` | Every pod that has ended: GPU, dollars an hour, start, end |

RunPod forgets a pod once it ends and does not say what it cost, so what was spent is
worked out from these, and is an estimate. RunPod's billing page is the account.

`--ssh` lets `~/.ssh/id_ed25519.pub` into the machines it rents; `stet-runpod ssh
stet-rp-01` logs in to one. `stet-runpod logs stet-rp-01` shows what one has said (`-f`
follows it).

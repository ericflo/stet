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
it runs curators and grades, and trains the next exam's students when there is nothing
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
judge, and nothing more: it administers nothing, and is never given the held-out exam
([Operating: rented machines](../docs/OPERATING.md#rented-machines)).

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
stet-runpod gpus                     # what would be rented, cheapest first
stet-runpod status                   # what is waiting, what is rented, what it costs
stet-runpod scale --follow           # rent and end to match the work, every minute
stet-runpod up 2                     # or by hand: two now, whatever is waiting
stet-runpod down                     # end those that hold no job
```

`status`, `scale`, `up` and `down` ask the server what is waiting
(`GET /api/v1/admin/work`) with the organisers' own key, `~/.config/stet/key.json` by
default (`--key`). Run them where that key is.

How many it rents: one for every `--per-machine` (6) jobs a new machine could take now,
counting a curator still to run as two (the run, and a grade of what it writes) and a
script waiting as one. A machine with nothing official to do trains the next exam's
students (a few minutes each, for a settled round's shadow), so one more for every
`--per-shadow` (4) of those waiting, after the official work (`--per-shadow 0`: none).
Never more than `--max` (6), or past `--max-hourly` dollars an hour (3), and nothing new
once `--daily-budget` is spent. `--max-price` never rents a machine dearer than that, however
much is waiting: the work waits for a cheap one instead. When an entry is waiting for a round
and nobody who runs curators is around, one, so that the round is drawn early. It never
ends a machine that holds a job. It ends at once those doing nothing when nothing at all is
waiting, and any that went quiet or stopped. The rest stop by themselves.

Every machine rented is the cheapest that will do, looked up afresh each time: every kind
of GPU RunPod has free, in both its clouds, is tried in order of price until one has a
machine with room. What will do: NVIDIA, 12 GB or more (a machine that runs curators needs
10 and gives one 8), on a host with the season's floor of 16 GB of memory and 6 vCPUs a
GPU, a driver new enough for CUDA 13 (the season's PyTorch, 2.15.0, is built only for
13.2), bf16 (nothing older than Ampere), and a whole GPU, with an 80 GB disk.
`stet-runpod gpus` lists them as they stand. `--gpu "RTX A5000,RTX 4090"` names them
instead. At micro scale a grade trains for a few minutes even on the cheapest, so those are
the cheapest per job.

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
ExecStart=%h/.local/bin/stet-runpod scale --follow --max 5 --max-price 0.22
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

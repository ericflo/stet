# stet-runpod

Rents GPUs from RunPod to grade for Stet while the organisers' own machines have more to
grade than they can get through, and lets them go when there is nothing to grade. For the
organisers: entrants and volunteers never need it. It is built on its own, apart from the
stet command line and the server, and uses nothing but Go's standard library.

```sh
cd runpod && go build -o ~/.local/bin/stet-runpod .
```

## What a rented machine does

It is a RunPod pod running `ubuntu:24.04` and [one script](worker.sh). The script installs
stet from the site (about two and a half minutes), writes the machine's key, and runs
`stet verify --exit-when-idle 10`: it grades until it has had nothing to do for ten
minutes, then ends its own pod with the key RunPod gives each pod for itself. RunPod
restarts a container whose process exits, and keeps billing for it, so the script never
just exits.

It only grades. RunPod's containers cannot make the sandbox a curator runs in (no user
namespaces, no cgroup of their own, no systemd), so curators are run by the organisers' own
machines. Both of RunPod's clouds were checked and are the same.

Each machine has a key of its own from a pool kept here. The server knows them from
`STET_WORKER_KEYS`: one of the organisers' machines to the judge, and nothing more. It
administers nothing, and is never given the held-out exam
([Operating: rented machines](../docs/OPERATING.md#rented-machines)).

## Setting up

```sh
export RUNPOD_API_KEY=...            # https://www.runpod.io/console/user/settings
stet-runpod keys --add 4             # four keys, and the STET_WORKER_KEYS line
```

Put that line in Epsilon's `apps/stet/deployment.yaml` and deploy. `stet-runpod keys` then
says, for each key, whether the server takes it as a rented machine.

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

How many it rents: while curators are still to be run, `--per-runner` (2) for each of
the organisers' machines running them, since a grade takes about twice as long as a
curator; plus one for every `--per-machine` (4) scripts already waiting. Never more than
`--max` (4), or past `--max-hourly` dollars an hour (3), and nothing new once
`--daily-budget` is spent. It never ends a machine that is grading. It ends at once those
doing nothing when nothing at all is waiting, and any that went quiet or stopped. The
rest stop by themselves.

The cheapest GPU RunPod has free that will do is rented first: NVIDIA, 10 GB or more, a
driver new enough for CUDA 13 (the season's PyTorch), bf16 (nothing older than Ampere),
and a whole GPU. `--gpu "RTX A5000,RTX 4090"` names them instead. At micro scale a grade
trains for about two and a half minutes on an RTX 3080, so the cheapest kinds are the
cheapest per grade.

`--cloud secure` (the default) rents only from RunPod's own data centres. `--cloud
community` is strangers' machines, at a third of the price or less, and the host of one
could read the key it is given. Anyone holding a worker key can grade as the organisers:
a key that may have leaked comes out of `STET_WORKER_KEYS`.

Leaving it running, as a service of your own user:

```ini
# ~/.config/systemd/user/stet-runpod.service
[Unit]
Description=Rent GPUs to grade for Stet

[Service]
EnvironmentFile=%h/.config/stet-runpod/env
ExecStart=%h/.local/bin/stet-runpod scale --follow --daily-budget 20
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
stet-rp-01` logs in to one. There is no other way to see a machine's log than RunPod's
console.

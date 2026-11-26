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

It is a single-GPU container on one of the markets armed, running `ubuntu:24.04` and [one
script](worker.sh). The script ensures `gcc`/`g++`/`libc6-dev` are present (for Triton/vLLM/FLA
JIT kernels), installs stet from the site into `/opt/stet` (on Lium, under
`/workspace`, since its container disk is 7.5 GB), writes the machine's key under `/root`, and
runs `stet verify --edits --sandbox container --exit-when-idle 20`: it runs curators and
grades, and trains the organisers' students when there is nothing official to do, until it
has had nothing to do for twenty minutes. Then it ends itself where its market gives it a key
for that alone (RunPod a pod's own, Vast an instance's), and elsewhere waits for stet-cloud to
end it from outside, with the market's own backstop should the fleet be gone: Lium ends a pod
after `termination_hours`, Clore an order after `rental_hours` (both 6), and a Salad container
is made never to restart. A market may restart a container whose process exits, and keep
billing for it, so the script never just exits.

The markets' containers cannot make bubblewrap's namespaces, so curators run in stet's
container sandbox (`src/stet/cage.py`): a user of the curator's own, which cannot enter
`/root`; a system-call filter that hands it only TCP sockets bound to the loopback device;
Landlock, keeping it off the machine's other ports (Docker's DNS resolver among them); and
the season's limits, by a cgroup where the host allows one and otherwise watched. The rules
say what differs for a curator. Where it cannot be made (an old kernel, no Landlock) the
machine only grades. Curators are run on RunPod, and on Vast and Clore where a probe showed
the sandbox to hold (`stet-cloud probe --market M`; `markets.json` keeps the answer). Lium
(Sysbox takes the one seccomp listener allowed) and Salad (WSL2 in people's own PCs) only
grade. stet also refuses curators on a machine under the season's floor (`stet bench`: a CPU
of 100 MB a second on 6 processes, among others): the server says so in `/admin/work`
(`curators_why`), the machine is ended to make way, and its host, and its CPU model where the
CPU fell short, are kept off curators for a week (`curators.json`). `--no-edits` rents
machines that only grade.

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
(about 55 minutes on a 3090, ~30 on a 4090, ~20 on a 5090 at the standard scale), and one more for every `--per-machine`
(6) scripts a new machine could grade now: `--per-machine 1` gives each its own. A grade
trains the student once; the two machines a round asks to measure its unedited baseline
(Season 3's `baselines.per_round = 2`, `baselines_wanted` in `/admin/work`) train that first, on
as many tokens. In Season 3 the exam is built right after the draw from hours of the day before already
harvested (~11–13 minutes using a 1× RTX 5090 `Inkling-2-Mini` vLLM endpoint, within `--exam-after 25m`), so nearly every grade is taken once the
exam is posted, and wants a machine then. A grader may also take a grade before the exam is
posted and hold it until it is: such grades want machines from `--early` (90 minutes) before
the exam is due (when the server says, or else `--exam-after`, 25 minutes, after the draw), and
each kind of GPU is rented just in time, so that, rented now, it would report between a
minute before the exam and `--settle-within` (15 minutes) after it. A machine that has
trained only waits, so what a grade costs on a kind is its price over the time it is held,
and a kind is rented only if none still to come would cost less: the cheapest is waited for,
and a quicker, dearer kind is rented only once the cheaper ones can no longer report in time.
An exam past its time is late, and when it will come is not known: nothing is rented to take
its grades before it, or kept waiting for them, until it is posted. Season 1's exam came
about 2h12m after the draw, and these early grades were most of its grades; the service now
runs with `--early 0`, renting graders only once an exam is posted.
How long a training pass takes on each kind is learned from the grades of settled rounds
(each says the GPU that trained, and for how long), and how long a machine takes to start
from those this service rents; until a kind has been seen, its pass is a 3090's 55 minutes
over its speed in the table, and a machine is taken to start in 8 minutes. They are kept in
`calibration.json`, and `stet-cloud gpus` shows them. The curators want a machine for every
two entries to run, and never fewer than two, since each entry is run twice, by two
accounts. A machine with nothing official to do trains the organisers' own
students (a few minutes each, on a settled round), so one more for every `--per-farm` (4)
of those waiting, after the official work (`--per-farm 0`: none). `--long-farm 600` rents
machines that also take the bigger students, held for hours, one machine to each. A machine
rented for them is chosen for what a student costs on it, not what it costs an hour: a 4090
at $0.89 trains one in about half the time of a 3090 at $0.50, so the student costs about
the same and is done sooner. Where two cost about the same, the quicker comes first.
`--long-max-price` is the ceiling for those machines (the same as `--max-price` if not
given); `stet-cloud gpus` shows both orders. Never more than `--max` (6), or past
`--max-hourly` dollars an hour (4), and nothing new once `--daily-budget` is spent.
`--max-price` never rents a machine dearer than that, however much is waiting: the work
waits for a cheap one instead. In Season 3's on-demand-only mode, rounds draw only when
entries are waiting: when an entry waits with nobody who runs curators around, one 24 GB+
machine is rented to trigger the draw; up to two warm machines are kept across the short
6–16 minute draw notice window (`next_draw_at`), and when nothing is waiting or scheduled
all idle machines are ended immediately down to **0 GPUs**. It never ends a machine that
holds a job. It ends at once those doing nothing when nothing at all is waiting, and any
that went quiet or stopped. The rest stop by themselves. A machine ended a moment after it
took a job, or that stopped partway through one, still holds it as the server sees it: what
it holds is handed back on the next look, so nothing waits out its lease.

Every machine rented for a grade of a posted exam, or to run curators, is the fastest that
will do, looked up afresh each time: every kind of GPU the markets armed have free under
`--max-price` is tried fastest first (the cheaper first where two are as fast) until one has a
machine with room. A kind's price is ranked over how often machines of its kind have started
(`startRates`: some cheap hosts never do), learned per market, card and class of host. A round
settles only once its slowest grade is in: a Season 3 `ERNIE-4.5-0.3B-Base-PT` pass (frozen bottom half, 65.5M tokens, deterministic CUDA kernels) is about 20 minutes
on a 5090, 30 on a 4090 and 55 on a 3090. A machine that runs curators takes grades as soon as its runs are done, and
the first two to take one also measure the round's baseline, two passes. `--cheapest` rents every machine by what a grade costs, its price over its speed.
What will do: NVIDIA, `--min-gpu-memory` 12 GB or more for grading (peak grading VRAM is ≤10 GiB; `--min-curator-gpu-memory` 24 GB
or more for running curators, which receive 22 GB VRAM inside the sandbox; unfilled 24 GB curator
slots fall back to 12 GB+ graders whenever grades or farm tasks are waiting), fast enough to finish
a two-pass baseline inside the server's 5-hour job lease (with a 30-minute margin), on a host with the season's floor of 24 GB of memory (`--min-ram`) and 6 vCPUs a GPU, a driver
new enough for CUDA 13 (the season's PyTorch, 2.14.0, is built for CUDA 13), bf16
(nothing older than Ampere), and a whole GPU, with a 120 GB disk for curator machines (above the 80 GB runner floor; 35 GB for grader-only machines). `stet-cloud gpus` lists them as they stand.
`--gpu "RTX A5000,RTX 4090"` names them instead.

`--cloud` is RunPod's: `any` (the default) takes the cheaper of its two clouds, which is
nearly always the community one, at a third of the price or less, and `secure` only its own
data centres. It does not narrow the other markets. The hosts on Vast, Lium, Salad and Clore,
and RunPod's community ones, are strangers' machines: their owner could read the key a
machine is given, and anyone holding a worker key can grade as the organisers. Take back a
key that may have leaked (`stet-cloud keys --revoke HANDLE`).

Leaving it running, as a service of your own user (stet-cloud, which took over from
stet-runpod's service):

```ini
# ~/.config/systemd/user/stet-cloud.service
[Unit]
Description=Stet: rent GPUs to run curators and grade (stet-cloud)
Wants=network-online.target
After=network-online.target

[Service]
ExecStart=%h/.local/bin/stet-cloud scale --follow --cloud any --max 6 --max-price 1.10 --max-hourly 4 --daily-budget 60 --per-machine 1 --exam-after 25m --early 0
Restart=always
RestartSec=60

[Install]
WantedBy=default.target
```

The markets' keys are in `~/.config/stet-cloud/credentials.env`, which stet-cloud reads itself
at every look: the unit sets none of them, since a key in the environment would win over the
file's. `systemctl --user enable --now stet-cloud`. If it stops, the machines on RunPod and
Vast still stop by themselves, and those on Lium and Clore at their backstop; only renting
more stops. Only one of stet-runpod and stet-cloud
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

What was spent is worked out from the prices rented at, and is an estimate: RunPod forgets a
pod once it ends and does not say what it cost. What each market really charged is what its
balance went down by (`history/balances.jsonl`), less what other projects rented with the same
keys (`history/excluded.jsonl`); each market's billing page is the account.

`--ssh` lets `~/.ssh/id_ed25519.pub` into the machines it rents; `stet-cloud ssh
stet-rp-01` logs in to one. `stet-cloud logs stet-rp-01` shows what one has said (`-f`
follows it).

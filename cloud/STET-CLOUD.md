# stet-cloud: design and plan

*2026-11-21. Steps 1 and 2 are done (§13). Eric approved step 3, the combined ship, and the ~$2 smoke test with Vast first; both are under way (§14).*

stet-cloud grows out of stet-runpod. It is one binary that rents single-GPU machines from five markets through each one's own API: **Vast.ai, Lium, Salad, RunPod and Clore.ai**. It treats every (market, card) pair as its own kind and keeps the fleet's rule: **the cheapest cards that settle a round within `--settle-within` (15 minutes) of its exam**, under the same caps ($1.10 a machine, $16 an hour, $50 a day).

**Hard rule: no aggregators.** gpufinder.dev was used for scouting only. It does not appear in this system in any form: not as a feed, a dependency, a fallback, or a link in output. The same goes for SkyPilot. Every price, stock figure and rental comes from the market's own API.

---

## 1. What stays, what changes

The policy and calibration logic stays as it is; all the RunPod-specific code moves behind one interface. In today's code, RunPod shows up in only a handful of places:

| Today (stet-runpod) | stet-cloud |
|---|---|
| `Fleet.rp *RunPod` | `Fleet.markets map[string]Market`, the enabled markets |
| `rp.gpus()`, which makes a `Choice{GPUType, Cloud, Price}` | `Market.Offers()`, which makes an `Offer{Market, Card, …, Price}`. `Choice` becomes `Offer` |
| `rp.create(NewPod)` / `errNoRoom` | `Market.Rent(Offer, Launch)` / the same `errNoRoom` |
| `rp.pods()`, matched to keys by pod name | `Market.Machines()`, matched to keys by the name or label the market keeps, or else by the ledger |
| `rp.terminate(id)` | `Market.End(id)` |
| `Pod.Status`, `CreatedAt`, `Cost`, `live()` in `Worker.state` | `Machine.State` (starting / running / stopped / gone), `Created`, `Price` |
| `worker.sh` ends its pod through RunPod's API | `worker.sh` ends itself where the market gives the machine a key that can only end that machine (RunPod, Vast). Elsewhere it waits, and the fleet ends it |
| `logs`, `ssh` (RunPod) | Offered per market where its API allows (`Caps`) |
| `~/.config/stet-runpod` | `~/.config/stet-cloud`, moved over once (§7) |

These stay as they are:
- `policy.go`: targets, endings, `decide`.
- `stet.go`: the server API, keys and vouching.
- The worker key pool and the speed table.
- `forExam`, `waitingFor`, `byPace`, `byGrade` and `byStudent`. They take `[]Offer` and look up speed by `Offer.Card`.
- `calibrate.go`, apart from what boot time is keyed by (§4).
- The service shape and nearly every flag.

## 2. The Market interface

```go
// market.go
type Market interface {
	Name() string                                       // "vast", "lium", "salad", "runpod", "clore"
	Offers(ctx context.Context, n Need) ([]Offer, error) // live: what could be rented now, all-in price
	Rent(ctx context.Context, o Offer, l Launch) (*Machine, error) // errNoRoom: gone, try the next offer
	Machines(ctx context.Context) ([]Machine, error)    // ours only, live or stopped
	End(ctx context.Context, id string) error           // idempotent: already gone is nil
	Check(ctx context.Context) (Account, error)         // read-only: key works; balance if the API says
	Caps() Caps
}

type Need struct {
	MinGPUMemGB, MinCPUs, MinRAMGB, DiskGB int
	MinCUDA  string  // "13.0": a host driver of 580 or newer
	MaxPrice float64 // $/h for one machine, all in
	Curators bool    // only offers whose machines can hold cage.py
}

type Offer struct {
	Market, Card string // Card normalized: "RTX 4090", "RTX A6000", "L40S"
	ID           string // the market's own: Vast ask ID, Lium gpu_type, Salad GPU class, RunPod GPU ID + cloud, Clore server ID
	Price        float64 // $/h all in: GPU + 100 GB disk + fees (§3)
	Stock        int     // machines free now; -1 if the market gives only a word
	Level        string  // the market's own word: low/medium/high, or ""
	CUDA         string  // the host's maximum CUDA, if the market says
	Trust        string  // datacenter | verified | community | home
	Region       string
}

type Launch struct {
	Name    string            // the worker key's handle (a random suffix is added where names must be unique)
	Image   string            // ubuntu:24.04; per market where cheaper (Lium: its official Ubuntu 24.04 fast-deploy image)
	Env     map[string]string // STET_WORKER_KEY, STET_API, STET_INSTALL, STET_MARKET, …
	Script  string            // worker.sh
	DiskGB  int
	MaxLife time.Duration     // the market's own backstop, where it has one (Lium termination_hours, Clore rental_hours)
	SSHKey  string
}

type Machine struct {
	Market, ID, Name, Card string
	Price   float64
	State   string // starting | running | stopped | gone
	Created time.Time
}

type Caps struct {
	SelfEnd  bool // the machine can end itself with a key that reaches only itself
	Curators bool // cage.py is known to hold there (set per market only after the smoke test)
	Logs, SSH bool
	Billing  time.Duration // billing granularity: 1s, 1m
}
```

There is one file per market: `runpod.go` (today's, moved behind the interface), `vast.go`, `lium.go`, `salad.go`, `clore.go`. Each uses only Go's standard library `net/http` client. Each gets an `httptest` fake built from that market's recorded responses, as `fleet_test.go` does for RunPod today.

## 3. Kinds, prices and choosing

- **One list, every market.** `choices()` asks every enabled market for offers and merges them. It then keeps the cards that fit, using the same filter as today: NVIDIA, 12 GB or more, Ampere or newer, and not matching `unfitGPU`. A kind is a (market, card) pair, and kinds are ordered with the functions that exist now. `errNoRoom` moves on to the next offer, which is often on another market.
- **All-in price, so markets compare fairly:**

| Market | All-in price |
|---|---|
| Vast | `dph_base` + 100 GB × `storage_cost` ÷ 720 + the host's download price × ~3 GB, spread over the expected life |
| Clore | listed price + the renter's 5% fee + its creation fee spread over the expected life |
| Salad | the GPU class price at the chosen priority (CPU and RAM included) |
| Lium | the GPU price (no volumes) |
| RunPod | the pod cost |

- **Driver.** Vast, Lium, Clore and RunPod report the host's driver or CUDA version, so offers below 13.0 are dropped before renting. Salad doesn't. There, `worker.sh` checks `nvidia-smi`, and if the driver is under 580 it asks Salad's instance metadata service to move it to another node (`POST /v1/reallocate`). The delay counts toward that kind's learned boot time.
- **Curators** are run only on kinds whose market has `Caps.Curators`.
  - At the start that is RunPod alone, where cage.py is proven.
  - Vast and Lium are switched on after `tools/cage_check.py` holds there in the smoke test.
  - Salad (home PCs under WSL2) and Clore (unknown host kernels) start grade-only: `STET_WORKER_EDITS=0`.
- **Each market's own limits:** `--markets vast,lium,salad,runpod,clore` (default: every armed market, §6), `--max-per-market vast=8`, and `--trust` (default `verified`, which is no worse than today's RunPod community cloud; `datacenter` for when it matters).
- **Rate limits** are kept inside each client:
  - Clore takes one order per 5 s, so 20 machines take about 100 s, within one look.
  - Salad allows 240 requests a minute; Lium's public feeds 60 a minute.

## 4. Calibration

- **Training pass:** keyed by card, as now. The same card trains at about the same speed on any market, and the grade detail (`detail.hardware.gpu`, `detail.train.seconds`) names the card but not the market.
  - Per-market correction, from data that exists today (checked by stet-db on round 6). An entry's `detail` comes from one of its graders. `runs[].grades[].by` names the graders, and the ledger knows each handle's market and card at that time. So the grader whose ledger card matches `detail.hardware.gpu` gives the market. This is exact whenever the graders' cards differ. When both had the same card, only the card is known and the sample counts toward the card alone.
  - **Exact, once §11 ships:** `machines[].pass_seconds` in `/admin/work` gives each machine's own latest training time, by worker handle. The ledger knows that handle's market and card, so the pass time is learned per (market, card) directly, every minute, without waiting for a round to settle. The matching through grades above is the fallback until then.
    - A freshly rented machine has an empty disk, so it reports no `pass_seconds` until its first grade. For each rental's first job the fleet uses its per-card figure.
  - The grade-detail contract with stet-db is unchanged.
- **Boot:** keyed by **market**, and by (market, card) once three or more machines of that kind have been seen. Boot differs most between markets: Lium starts in seconds, while Salad pulls over home connections and can take tens of minutes. Until a market has been seen it gets its own default: Lium 3 min, RunPod and Vast 8, Clore 10, Salad 20.
- `lead(kind)` = boot(market, card) + fetch + 2 × pass(card) + sitting. `forExam` runs unchanged over every kind.
- **Starts that fail.** Every rental per (market, card) is recorded as started or never-started (it never asked for work). Round 6 had three $0.18 community 3080 Tis that never started; 8 of 116 RunPod rentals so far have failed this way.
  - A kind's price for ranking is divided by its start rate: a $0.18 card that starts 80% of the time ranks as $0.225.
  - When time to the exam is short, kinds with a poor start rate are passed over first.
- **Grace, per market.** How long a machine may take to first ask for work before it counts as dead is max(10 min, 2 × that market's 80th-percentile boot). It replaces the fixed 25 minutes (`Policy.Grace`). RunPod's boots so far run 1.1–5.8 min, so its Grace drops to about 10 minutes; Salad's will be longer. A dead machine is ended and its worker key goes straight back to the pool.

## 5. Ending machines

- **Vast and RunPod end themselves.** Each gives the machine a key that reaches only that machine:
  - RunPod: `RUNPOD_API_KEY` + `RUNPOD_POD_ID`, as now.
  - Vast: `CONTAINER_API_KEY` + `CONTAINER_ID`, with `DELETE /api/v0/instances/$CONTAINER_ID/`.

  `worker.sh` picks between them by `STET_MARKET`.
- **Lium, Salad and Clore are ended by the fleet.** No account key ever goes onto a rented machine. When `stet verify` exits, `worker.sh` sleeps forever instead of exiting, because Salad restarts a container that exits and keeps billing. The fleet ends the machine:
  - at once if nothing at all is waiting (today's rule);
  - or once the machine has gone quiet. For markets without self-end, "quiet" (`Policy.Silence`) is 5 minutes instead of 20: an idle `stet verify` asks for work far more often than that, and a machine at a job is "grading", never "quiet".
- **The market's own backstop** covers the fleet itself dying:
  - Lium: `termination_hours = 2`.
  - Clore: `rental_hours = 2`.
  - Salad has none, so the fleet's `down` and the 5-minute rule carry it. The prepaid balance is the hard ceiling (§6).
- **Salad:** one container group per machine (`replicas = 1`), named `<handle>-<random>` because names must be unique even against deleted groups. Ending a machine deletes its group.

## 6. Auth and the gate before money

**Credentials can arrive at any time.** stet-cloud reads `~/.config/stet-cloud/credentials.env` (mode 0600, `KEY=value` lines) at every look, and the same names from its environment, which wins. A market whose key is missing is simply off, with one line in `status`, never an error. Adding a key later needs no restart.

```sh
# ~/.config/stet-cloud/credentials.env   (chmod 600; never committed, never copied to a machine)
RUNPOD_API_KEY=...
VAST_API_KEY=...
LIUM_API_KEY=...
SALAD_API_KEY=...
SALAD_ORG=...          # organization name, as in the portal URL
SALAD_PROJECT=...      # project name
CLORE_API_KEY=...
```

Least privilege where the market allows it (exact steps in the sign-up checklist):
- **RunPod:** a separate key for stet-cloud, scoped to `catalog:read`, `pods:read` and `pods:write`, with `account:read` optional. The personal key and today's service key stay as they are.
- **Vast:** a key limited to `misc` (offer search), `instance_read` and `instance_write`.
- **Lium:** a key with scopes `read`, `rent` and `manage` (the defaults; the docs don't say which one deleting a pod needs), never `billing`, and `pod_visibility: own`. Scopes and visibility are fixed when the key is made. Lium also lets a key carry daily and monthly budgets, but **a key that hits its budget has its pods deleted**, which would kill grades mid-training. So the budget is left empty, and the prepaid balance is the cap.
- **Salad:** there is only one key per user, and it reaches every organization that user is in. The service uses the key of a **second Salad account**, invited only to the stet organization.
- **Clore:** account-wide key. It lives only on the fleet's machine.

**Arming:** a market with a key is *read-only until armed*. This makes "spend nothing until approved" part of the code, not a promise.

```sh
stet-cloud markets                 # per market: key found? reaches the API? balance? armed? offers that fit now
stet-cloud markets check lium      # one market, during sign-up: does the key work (read-only)
stet-cloud offers                  # every fitting offer across armed AND unarmed markets, as it would rank them
stet-cloud markets arm vast        # after Eric's approval: recorded in markets.json with time and note
stet-cloud markets disarm salad    # back to read-only; machines it has keep running until they end
```

- **Unarmed markets.** `scale` and `up` never rent on an unarmed market: they log "would rent" lines, like `--dry-run`.
- **RunPod during migration.** It stays armed, because the running service uses it today.
- **Money limits:**
  - **Prepaid balances are the hard ceiling** on Vast, Lium, Salad and Clore (all prepaid). Start each at $20–50 with auto-reload off.
  - **Per-market daily budgets** (`--market-budget vast=10`) sit under the global `--daily-budget`, from the same ledger.

## 7. State and migration

- **`~/.config/stet-cloud`** holds:
  - `workers/` (the key pool, unchanged);
  - `fleet.json` (live machines: `market`, `id`, `handle`, `card`, `price`, `start`, `alive`);
  - `spent.jsonl` (ended machines, now with `market`);
  - `calibration.json` (pass by card; boot by market and kind);
  - `markets.json` (armed state, Lium template ID, Salad GPU class cache);
  - `credentials.env`;
  - `service.json`.
- **First run.** If `~/.config/stet-runpod` exists and `stet-cloud` doesn't, the old folder is copied (never moved). Records without a `market` read as `runpod`. RunPod pods already rented are adopted by name, as today.
- **Commands.** `stet-cloud` keeps every stet-runpod command and setting. `gpus` stays as another name for `offers`, and `--cloud` becomes `--runpod-cloud`; the old spelling is still accepted.
- **The service** becomes `stet-cloud.service`, with `EnvironmentFile=%h/.config/stet-cloud/credentials.env`. The old unit is stopped only when the new one takes over, so there are never two renting at once. Both hold the same lock file once migrated.

## 8. Per-market notes for the build

| | Offers | Rent | Ours | End | Watch |
|---|---|---|---|---|---|
| **Vast** | `POST /api/v0/bundles/` with our floor (`cuda_max_good ≥ 13.0`, `verified`, reliability, `inet_down`; `type: ondemand`) | `PUT /api/v0/asks/{id}/`: image, env, onstart (≤4,048 chars, so `worker.sh` goes in env and onstart unpacks it), disk 100, `label = handle`, `cancel_unavail` | `GET /api/v1/instances/` (trailing slash; without it, a 301), by label | self (`CONTAINER_API_KEY`), or `DELETE /api/v0/instances/{id}/` | Disk bills until destroyed, so destroy rather than stop; `exited`/`offline` means destroy and rent another; the `env` format (Docker flags vs object) is tested first |
| **Lium** | `GET /api/executors?view=summary` (keyless, ETag); per-node driver and `tier` (secure/spot) | `POST /executors/{id}/rent` with `expected_price_per_gpu_hour` (409 if the price rose), or `rent-by-spec`. Both take `pod_name`, `user_public_key`, `template_id`, `termination_hours: 2` and an Idempotency-Key | `GET /pods` (key visibility `own`) | `DELETE /pods/{id}?expected_name=` | Env and start command live only in a template, made once per setup (`POST /templates`). Use the official Ubuntu 24.04 "fast deploy" image: billing starts at RUNNING, so its pull is free. Rent by node, not by spec, so spot-tier nodes (reclaimable at any time) can be skipped |
| **Salad** | `GET …/gpu-classes` (prices per priority); `POST …/availability/sce-gpu-availability` | `POST …/projects/{p}/containers`: `container{image, command, environment_variables, priority: high, resources{cpu, memory, gpu_classes, storage_amount}}`, `replicas: 1`, `restart_policy`, `autostart_policy: true` | `GET …/containers`, groups named `stet-…` | `DELETE …/containers/{name}` | No driver filter (the worker checks); slow boots; the command filter rejects `curl` after `;` or `\|`; group names never reused; replica quota per org |
| **RunPod** | `/v2/catalog/gpus` (as now) | `POST /v2/pods` (as now) | `/v2/pods` by name | self, or `DELETE /v2/pods/{id}` | Unchanged. The help text should point to `console.runpod.io/user/credentials` |
| **Clore** | `GET /v1/marketplace` (keyless; per-day prices, `cuda_version`, `net`, `reliability`, `mrl`) | `POST /v1/create_order`: `currency: USD-Blockchain`, `type: on-demand`, `renting_server`, image, command, env (≤30 vars), `rental_hours: 2`, `required_price` | `GET /v1/my_orders`. Create returns no order ID, so the order is found by server ID and creation time | `POST /v1/cancel_order` | Crypto-funded (USD balance from USDT/USDC); one order per 5 s; errors come back as HTTP 200 with a `code`; fees are the renter's 5% plus $0.10 an order |

The exact request fields are confirmed against each API once keys exist, in step 2 below, before any rental code is trusted.

## 9. Plan, in steps

Steps 1 and 2 cost nothing (funding in step 1 waits for Eric's OK on the amounts). Steps 3–5 each need Eric's approval, and step 4 is the first renting on a new market.

1. **Sign-up.** *Done, 2026-11-21.* Walk through the per-market checklist (separate document). Keys go into `credentials.env`, checked read-only. Funding comes after Eric approves the amounts: $20–50 a market, auto-reload off.
2. **Refactor and read-only clients, with no change in behaviour.** *Done, 2026-11-21, uncommitted (§13).*
   - Bring in `Market`, put RunPod behind it, add arming, credentials, the new state folder and migration.
   - Add read-only Vast, Lium, Salad and Clore clients (`Offers`, `Machines`, `Check`) with recorded-response tests.
   - Check that the policy's decisions on recorded views match today's exactly.
   - `stet-cloud offers` shows the merged list with nothing armed but RunPod.
3. **The combined ship, on RunPod alone (§11).** stet-dc's server and client 0.4.4 land with the fleet's sizing, margin, Grace and start-rate changes, in one round, as one proposal. The service switches from stet-runpod to stet-cloud here, with stet-db told first. stet-db reports the round against rounds 4–7.
4. **A smoke test per new market: about $2.**
   - One machine on each of Vast, Lium, Salad and Clore, rented by hand (`stet-cloud up 1 --market X`), running `worker.sh` on a practice round.
   - Measures boot and install time.
   - Runs `tools/cage_check.py` to see whether curators may run there.
   - Confirms the driver, the ending path, and that billing stops (checked on each market's billing page).
5. **Arm one market at a time.**
   - Order: Vast (ends itself), then Lium, then Salad and Clore (grade-only).
   - After each round, one line: cost, minutes from exam to settle, and grades per market.
   - One proposal at a time, in keeping with the fleet's rule; no drip of tweaks.

## 10. Lessons from round 6 (stet-db's data)

Round 6 (s1-o00006) settled 28 minutes after its exam, against a 15-minute target, at $6.21. Of its 22 grades, 6 had no machine holding them when the exam posted. Per stet-dc's server logs, the server held nothing back: every grade could be taken from 02:42:57. The gap was the fleet's own:

- **The cap arithmetic.** Each machine holds one job, and the 4 machines rented at the draw to run curators count against `--max 20`. So `Launch = min(Target, MaxWorkers) − Live` capped the graders at 16 while 18 were wanted: at most 20 of the 22 grades could ever be held before the exam.
- **Dead starts.** Three $0.18 community 3080 Tis never asked for work and held their keys for 25 minutes. Only 12 rented graders ever held a grade.
- **Slow hosts outside training.** A community 5080 that trains a pass in 10 minutes took 46 minutes from claiming its grade to reporting. About 25 of those minutes went on fetching and tokenizing the corpus.

More markets mean more cheap, less reliable hosts, so these go into the design rather than a later patch:

- **Size the cap from the work, and let money bind.** The machines wanted before an exam are `grades_early` + grades held + curator machines, one job each. stet-cloud keeps `--max` as a safety stop, but the proposal is to set it from that sum plus a margin. The binding limit is then `--max-hourly`: 30 machines at $0.30 is $9 an hour, under the $16 cap. The setting itself stays Eric's.
- **Learn "fetch" per market, not just boot.** The fleet already sees each machine's job state every minute from `/admin/work`. It records when a machine first asks for work, when it claims a grade, and when it reports. Then fetch(market) = report − start − passes × pass(card) − sitting, a median per market (and per kind once there are enough). That replaces the fixed 2 minutes in `fetchAndScripts`; slow home and community hosts get the longer lead they need, with no server change.
  - Not every job has two passes (stet-dc's caveats). A machine's first grade of a round trains the baseline and the student. Its later grades in that round train only the student, because the baseline is cached per round, scale and device.
  - A grade claimed before the exam can't report until the exam posts. So "start" is max(claim, exam − passes × pass). Only jobs that reported well after the exam, where training was the limit, are counted. Otherwise every early grader would look like a slow fetcher.
  - The exact fix, to ask stet-dc for only if these estimates prove too noisy: a "trained at" time per grader in `/admin/work`. The client already logs it, but exposing it would need client 0.4.4 and a server change.
- **Several grades per machine (§11).** Under stet-dc's change, a grader that has trained and is waiting for the exam takes more grades before it. Its baseline is cached, so each extra student is one more pass. "Grades a machine holds before the exam" is one number in the sizing (`official`/`early`) and in `forExam`'s lead model, so the change touches one function.

- **Margin.** Today `forExam` plans a grader to report anywhere up to 15 minutes after the exam, so the whole settle window gets spent and any failure becomes a tail. stet-cloud gets `--aim exam|settle`, which sets the time each grader is rented to report by. `exam` keeps the 15 minutes as margin, at about $0.50–1 a round on cheap cards. Its effect is measured over a few rounds before it becomes the default, as one proposal.
- **Failed starts and Grace:** in §4.
- **Fast machines near the exam.** A fast machine that is idle within about 5 minutes of an exam is not ended while grades are still owed. In round 6 an idle 4090 was ended at 04:11:17 ("nothing is waiting"), just before the exam. This is the fleet's half of §11 B, under which, for about 2 minutes after the exam, a needed grade goes only to the fastest card around.
- **Keys are a cap.** At 03:50 the fleet logged "only 4 keys are free", because dead pods were still holding theirs. Dead machines are ended sooner (Grace per market), and the pool grows itself up to `--max` keys instead of only saying so.
- **stet-dc's commitments.** stet-dc keeps the grade fields the calibration reads stable: `detail.hardware.gpu`, and `detail.train.seconds` (the student's training loop only). It will say before either changes, or `/admin/work` does.
- **GPU names.** New markets bring device names RunPod never showed: "RTX 4090 D", laptop and consumer variants, "RTX PRO 4000 Blackwell" and the like. `shortGPU` and `unfitGPU` are checked against every name each market lists before it is armed. A 4090 D is its own card, slower than a 4090. What `detail.hardware.gpu` and `detail.train.seconds` mean is unchanged; any change goes to stet-db first.
- **Exam time is approximate.** Round 6's exam came at drawn + 2h10m26s, about 2 minutes early. stet-cloud uses the server's `rounds[].exam_due` (§11), the median gap over recent rounds, the same figure the server plans with. `--exam-after` remains only for a server that doesn't send it.

## 11. The server side, with stet-dc: one combined ship

Eric's latest word (2026-11-21): **"Hold for combined ship."** stet-dc's two ideas, client 0.4.4 and the server change, don't go out on their own. They ship together with the fleet's sizing change, in one round, on RunPod. stet-db reports that first combined round against rounds 4–7 as the baseline. The timing is agreed between stet-dc and me. Because nothing ships early, the live 213c753 fleet never runs against a server that lets one machine hold several grades.

Eric confirmed the hold to stet-dc directly. A and B stay unbuilt and undeployed until my ready-to-ship note, and then we pick a round together. The fields and rules below are as stet-dc settled them (2026-11-21). They are free to change until stet-dc builds.

**`/admin/work` additions** (all `omitempty`, so the live 213c753 fleet is unaffected)

| Where | Field | Type | Meaning |
|---|---|---|---|
| top level | `early_grades_per_machine` | int | The most grades one machine may hold before the exam (3) |
| top level | `fast_first_seconds` | int | How long after `exam_at` a needed grade is held for the fastest card (120) |
| `rounds[]` | `exam_due` | int64, unix s | When the exam is expected: `drawn_at` + the median draw-to-exam gap over recent rounds. Omitted once `exam_at` is set |
| `machines[]` | `holding` | int | Grade jobs held (0 up to the cap); `busy` stays "grade" while ≥ 1 |
| `machines[]` | `ready` | bool | Everything it holds is trained and waiting for the exam, so it may take another |
| `machines[]` | `pass_seconds` | float64 | Its latest measured student training time (`train.seconds`), from its hello. 0 or omitted when unknown |

These keep their meaning:
- `grades_early` counts grades, not machines.
- `grades_wanted` also counts grades held back for a faster card during the fast-first window. That is the live fleet's safeguard: it ends idle machines only when nothing is wanted.

**A. Several grades per machine before the exam** (client 0.4.4 and the server)
- After training a grade's student, the client keeps that model and asks for another. Each further grade trains the student only, since the baseline is cached.
- At the exam it sits the baseline once, then each student it holds, and reports each one as soon as it is sat. So holding 3 adds only seconds; sitting is negligible (rounds 5–6: 4–25 s after the exam).
- The server offers an extra early grade only if all four hold:
  - (a) it is a needed grade, never a replica;
  - (b) everything the machine holds is trained;
  - (c) `holding < early_grades_per_machine`;
  - (d) `now + pass_seconds + 2 min ≤ exam_due`. A slow card that finishes early won't take a pass it can't finish, and a machine with no `pass_seconds` yet gets no extra.

**B. The fastest cards first, just after the exam** (server only)
- For `fast_first_seconds` after `exam_at`, a needed grade goes to a machine only if no faster machine seen in the last minute is idle or `ready`.
- Speed is the median `train.seconds` per GPU name from the server's own records. An unknown GPU counts as slowest. After the window, anyone can take it.
- "About to be free" means `ready`, so B relies on A, and the two ship together. In round 6, rp-09 was still training at the exam.

**Dead hosts must be released.** While a round waits for its exam, the server extends early grade claims rather than letting them lapse. So a dead machine's grades, up to 3 under A, wait until the fleet calls `POST /admin/workers/{key}/release`, which hands back every job it holds. In stet-cloud, **every** ending of a machine calls release: dead, quiet, idle, by hand or by its market. Today's code releases only on a later look, and only for keys with no machine. Otherwise one dead host leaves 3 grades stuck for hours.

**The fleet's side** (mine): one function changes, as §10 set up.
- **The exam time is the server's:** `exam_due` replaces `--exam-after` (2h12m) wherever the server gives it.
- **Grades a kind can take by the exam:** `min(early_grades_per_machine, 1 + ⌊(exam_due − 2 min − ready) / pass⌋)`, where `ready = rented + boot(market) + fetch(market) + 2 × pass(card)`. That is a machine's first grade. After it, the server decides by the machine's own `pass_seconds`, so the fleet's estimate is only a plan and the server's count wins.
- **Ranking:** `forExam` ranks kinds by price over held time per grade, so the cost is `price × lead ÷ grades`. A fast card rented early that trains 3 students can beat a slow cheap one.
- **Count:** graders wanted ≈ `grades_early` ÷ the grades each chosen kind can take, so fewer, faster machines, and the machine cap stops binding.
- **Dead hosts:** the cap of 3 limits what one costs. Kinds with a poor start rate (§4) are given fewer grades, or passed over, when time is short.
- **Settings:** the cap and the 2-minute window are read from `/admin/work`, not set in the fleet, so the server owns them and the fleet follows.

**Order:** §9 step 3. The fleet's half is built and tested against fakes of the new `/admin/work` fields once stet-dc settles their names. stet-dc holds its deploy until I send a ready-to-ship note. Then the server, the client and the fleet change in the same round.

## 12. Open questions, answered by the keys or the smoke test

- **Vast:** does `CONTAINER_API_KEY` destroy its own instance (the docs show it, but don't define the key's permissions)? Real boot and install time. Whether `env` is sent as Docker flags or as an object.
- **Lium:** which scope deleting a pod needs (hence read, rent and manage). Whether Landlock and bubblewrap work under Sysbox. Whether a pod whose container has died still bills.
  - Known: a template must keep its container alive (`worker.sh` sleeps forever). Lium reboots a dead container at most once a day.
  - Known: there's no plain `ubuntu:24.04` template. The official Ubuntu 24.04 "fast deploy" image (`daturaai/pytorch:…-ubuntu24.04-…`) is the one whose pull isn't billed.
- **Salad:** what `restart_policy: never` actually does; the replica quota for a new organization; boot times; whether Landlock works under WSL2.
- **Clore:** what happens when the container exits (the troubleshooting docs suggest it restarts, hence `sleep infinity`); host kernels; whether `ubuntu:24.04` or `cloreai/jupyter:ubuntu24.04-v2` starts faster.
  - Known: orders use `currency: USD-Blockchain`; fees are the renter's 5% plus $0.10 an order.
- **Ownership (settled 2026-11-21).** stet-db has handed `runpod/` (now `cloud/`) over. The live 213c753 service and its unit stay exactly as they are until stet-cloud takes over, and stet-db is told before the binary or the unit is replaced, so that its per-round reports don't credit the new fleet's changes to the old one.
- **Shared checkout:**
  - Run `git log @{u}..HEAD` before any push, and push only commits that are all mine (`git push origin HEAD:main`).
  - Stage only `cloud/` (`runpod/` before 2026-11-22; stet-dc keeps uncommitted work in `server/`, `src/`, `tests/` and `tools/`).
  - Run CI's gofmt: `GOTOOLCHAIN=go1.26.11 sh -c '$(go env GOROOT)/bin/gofmt -l .'`.
  - Every push to `main` redeploys the server.

## 13. What step 2 built, and what it found

**Built** (in `runpod/`, now `cloud/`, all Go standard library; `go test -race` passes):
- `market.go`: the `Market` interface (`Offers`, `Rent`, `Rentals`, `End`, `Check`, `Caps`), the credentials file read again at every look (env wins; a file others may read is refused), arming in `markets.json` (RunPod alone by default), and one JSON client for the new markets.
  - The market's machine is called `Rental`, since `Machine` is already the server's account in `/admin/work`.
- `runpod_market.go`: today's RunPod client, unchanged, behind the interface. `vast.go`, `lium.go`, `salad.go`, `clore.go`: `Offers`, `Rentals` and `Check` work. `Rent` and `End` return "not built", and `Caps.Rents` is false, so none of them can be armed.
- The fleet asks every market for its machines. It matches them to keys by the name the market keeps, or by the ledger (Clore keeps no name).
  - A market that doesn't answer is skipped only if it isn't armed and the ledger has nothing on it. Otherwise the look fails, so a key can never be rented twice.
  - Ledger records gain `market`, and a record without one reads as RunPod.
- Commands:
  - `stet-cloud markets` shows each key, its balance, whether the market is armed, and what fits now. `markets check|arm|disarm M` acts on one market; `arm` refuses a market whose renting isn't built.
  - `stet-cloud offers` (`gpus` still works) shows every market, with the unarmed ones marked.
  - `stet-cloud migrate` copies stet-runpod's pool once, and never over an existing pool. Until then, commands that need the pool refuse to start a new one beside stet-runpod's.
- Tests:
  - Each client is tested against that market's own recorded answers (`testdata/`, cut down). The machine lists are made up: nothing was rented.
  - Plus tests for the credentials file, for nothing being rented on an unarmed market, and for a market that doesn't answer.
  - `TestMain` keeps tests away from the real keys file and pool.

**No change in behaviour, checked live (2026-11-21, round 7, RunPod's key from `credentials.env`):**
- The old 213c753 binary and stet-cloud were run at the same moment, with the service's settings.
- They listed RunPod's kinds in the same order at the same prices, three times in each ordering, fastest first and `--cheapest`.
- `status` on twin copies of the pool gave the same decision ("would rent 4 and end 0") and the same state for all 16 machines. The only difference was the tool's name.

**What the keys showed:**
- Funded: Vast $39.66, Lium $100, Clore $100 (USD-Blockchain). Salad: 0 of 50 replicas in use (its API doesn't report a balance).
- Read-only, at the service's settings ($1.10 cap), offers that fit:
  - Vast: 395 offers, 40 kinds of card.
  - Lium: 98 offers, 21 kinds.
  - Salad: 16 GPU classes, hundreds of machines free in most.
  - Clore: 191 offers, 23 kinds.
  - RunPod: about 10 kinds.
  - The new markets' consumer cards often cost a third to a half of RunPod's: a 4090 at $0.35–0.46 against $0.89, and a 5080 at $0.21–0.32 against $0.39.
- **API facts learned:**
  - RunPod: a scoped key's catalog read needs `include=AVAILABILITY` with the product and cloud filters (400 without it). A key without `catalog:read` gets 403.
  - Vast: `/api/v1/instances/` needs its trailing slash (301 without it). The key reads `/api/v0/users/current/` (credit).
  - Lium: `max_cuda_version` is a number. CPU and memory are per node, so they're shared out per GPU.
  - Salad: the public API has no call that reads a project: `GET …/projects/{p}` gives 404 for any name. A project is checked by listing its containers (404 `project_not_found_exception` only when it's missing). GPU prices include CPU and memory. Availability is asked once per class and kept for 10 minutes, as Salad asks.
  - Clore: prices are per day. Errors come back as HTTP 200 with a `code`: 3 means a bad key.

**For step 3's proposal (each would change what is rented, so none is in step 2):**
- The speed table lacks consumer cards common on the new markets: RTX 3060, 4060 Ti, 4070 (SUPER, Ti SUPER), 5060 Ti, 5070 (Ti), A10, A30. An unknown card counts as half a 3090 and ranks last on pace. A 3060 is nearer 0.4, a 5070 Ti about 1.15.
- `unfitGPU` now also leaves out laptop GPUs (Salad lists some). RunPod has none, so its decisions are unchanged.

## 14. Step 3 and the smoke test (2026-11-21, approved by Eric)

**The fleet's half of the combined ship.** Built and tested against fakes of stet-dc's fields; the field list and rules (a)–(d) were re-confirmed unchanged by stet-dc.
- `/admin/work`: `early_grades_per_machine`, `fast_first_seconds`, `rounds[].exam_due`, `machines[].holding/ready/pass_seconds` are read. 0, false or absent means the old server: one grade a machine, `--exam-after`.
- `forExam` plans per kind with `examPlan`:
  - A kind's best time is `full = lead − sitting + 2 min + (most − 1) × pass`, so it can take the most grades the server allows.
  - `grades(left) = min(most, 1 + ⌊(left − (lead − sitting) − 2 min) / pass⌋)`.
  - A grade costs `price × max(left + sitting, lead) ÷ grades`.
  - With `most ≤ 1`, every line reduces to the old code: the old tests pass unchanged.
- The policy counts early grades in machines:
  - New machines needed = ⌈uncovered ÷ the grades the next kind takes⌉.
  - Machines already rented for several (the ledger records `grades`) cover theirs first: a starting or waiting one covers its whole plan, a grading one its plan minus what it holds.
- **Release on every ending:** whatever a machine holds is handed back as soon as the fleet ends it.
- **Boot and Grace per market:** Grace is max(10 min, 2 × that market's 80th-percentile boot).
  - Today's RunPod samples (23 of them: most 1–6 min, two 18–19) give 11 minutes.
  - Unseen markets use their default boot: Lium 3, RunPod and Vast 8, Clore 10, Salad 20 min.
- **Near an exam:** with grades under way, from 5 minutes before the exam through the server's fast-first window after it, the two fastest idle machines are kept, not ended.
- Not in this ship, to come as a later proposal on data: learning fetch time per market, start rates, and `--aim`.
- Live switch: `switch.sh` (in the session's scratchpad) stops and disables stet-runpod, copies the pool with `migrate`, checks `status`, then enables `stet-cloud.service`. The unit sets no keys, since stet-cloud reads `credentials.env` itself.

**Vast renting** (`vast.go`):
- `PUT /api/v0/asks/{id}/` with `runtype: args`, `args: [bash, -c, worker.sh]`, the env as an object (`STET_MARKET=vast` added), the label set to the handle, `disk`, and `cancel_unavail`.
- A 404 or 410 means the offer is gone: try the next.
- `End` is `DELETE /api/v0/instances/{id}/`, and a 404 counts as gone already.
- `worker.sh` ends itself on Vast with `CONTAINER_API_KEY` and `CONTAINER_ID`. On a market with no self-end it sleeps and waits for the fleet.
- Machines off RunPod get `STET_WORKER_EDITS=0`, and the curator batch keeps to markets with `Caps.Curators`, until the sandbox is shown to hold there.
- `up --market M` rents by hand on one market, armed or not; `scale` refuses `--market`.

**Lium, Salad and Clore renting** (built the same morning, tested against fakes, not yet smoke-tested):
- **Lium:** a one-time private template per machine carries its environment, with the worker script base64'd in `STET_WORKER_SCRIPT`.
  - Its `startup_commands` (`workerCommand`) unpacks the script and runs it.
  - The image is Lium's own `daturaai/pytorch:2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind-lium1`.
  - The rent is `POST /executors/{id}/rent` with `termination_hours: 2`, `expected_price_per_gpu_hour` (409 means the price rose: try the next) and an `Idempotency-Key`. `user_public_key` is this machine's SSH key.
  - Offers use `reliable_only=true`.
- **Salad:** one container group per machine, named `<handle>-<5 random>`, with `display_name` = handle, `replicas: 1`, `restart_policy: never` and priority `high`.
  - The command is `[bash, -c, workerInner]`.
  - On a Salad host, `worker.sh` checks the driver itself. Under 580, it asks Salad's in-machine service (`POST 169.254.169.254/v1/reallocate`) to move it.
- **Clore:** `create_order` (on demand, USD-Blockchain) with `required_price` = the listed price per day, and `rental_hours: 2`.
  - The env takes at most 1,536 characters a value, so the script is cut into `STET_WORKER_SCRIPT_0..n` and joined again in the command.
  - Clore returns no order ID: the order is found in `my_orders` by its server, made since it was asked for. Orders are kept 5 s apart.
  - Code 6 means no room, unless it's about the balance.
- Lium, Salad and Clore can't end themselves. When a machine there goes quiet (`quietWithoutSelfEnd`, 5 min holding no job), the fleet ends it; each market's backstop covers the rest.

**Smoke test, first try:**
- Vast 61768181, an RTX 4080 SUPER at $0.22/h, rented 07:04:09 UTC from a separate folder with its own key, `stet-smoke-01`.
- I ended it 38 s later through the fleet's own `End`; Vast no longer listed it within 14 s. stet-db pointed out that round 7 was still handing out grades, and a vouched smoke key could take one.
- **Rule:** a smoke machine is a real grader. Rent one only when no round is open (check `/admin/work` first), and tell stet-db the instance ID and times.

**Smoke test, done (07:22–07:49 UTC, rounds paused, separate folder and keys, about $0.18 in all):**

| Market | Card, $/h | Rented → first asked for work | Ended | Cost |
|---|---|---|---|---|
| Vast.ai | RTX 4080 SUPER, $0.22 | 1¾ min (container up in 31 s; stet 0.4.3 installed in about 1½ min) | by itself, at its 5 idle minutes, with `CONTAINER_API_KEY` | $0.03 |
| Salad | RTX 3060 12 GB, $0.08 | about 2 min | by the fleet (`DELETE` the group), once quiet | $0.016 |
| Lium | RTX 3090, $0.16 | 13 min on this node, after one fix | by the fleet (`DELETE /pods/{id}`) | $0.065 for both tries |
| Clore.ai | RTX 5080, $0.21 | about 20 min | by the fleet (`cancel_order`) | $0.07 |

**Fixes found by renting:**
- **Lium:** a template needs a volume (`/workspace`).
  - The start command must be `bash -c '…'` in single quotes: Lium refuses shell characters otherwise.
  - `created_at` comes without a zone.
  - **The container's own disk is 7.5 GB**, most of it the image, so stet's install ran out of space. It now goes on the 28 GB volume (`STET_WORKER_HOME=/workspace/root`, `STET_WORKER_OPT=/workspace/opt/stet`). That is under the 100 GB the need asks for: before Lium is armed, check what a grade needs on disk, or find how to get a bigger volume.
- **Grace before boots are seen:** with Lium's 3-minute default, the 10-minute floor would have ended its slow first machine before it ever asked for work. A market whose machines all take longer than Grace would then never be seen to start, and would churn. So until 3 boots are seen on a market, its Grace is the policy's 25 min, or twice the default boot if longer.
- Boot defaults were wrong both ways: Salad was 2 min, not 20; Lium 13 and Clore 20, not 3 and 10. Boot is learned per market as machines are seen.

**Not yet tested on the new markets:** an actual grade (nothing was waiting); curators (grade-only until `tools/cage_check.py` holds there); Salad's driver check on an old driver (this host's was new enough). Each market is armed only with Eric's approval, one at a time (§9 step 5): Vast first.


## 15. Curators and the season's floor (round 9, 2026-11-21)

**What happened:**
- At the 12:20 draw, the fleet rented its two curator machines on the curator markets: a Vast RTX 5080 at $0.22/h (host CPU an AMD EPYC 7K62) and a Clore RTX 5080 at $0.21/h (an Intel Xeon E5-2680 v4).
- Both asked for work and got none while 8 runs waited.
- The Vast machine's log said why: `the CPU is slower than the floor (95 MB a second; the floor is 100)`.
- `stet verify --edits` refuses itself on a machine under the season's floor (`stet bench`: GPU, memory, and a text-processing mix on six processes at once). The worker then grades only.
- The fleet counted it as a curator machine all the same. It rented the same host again when the first ended itself.
- At 12:47 curators were turned off on Vast and Clore. RunPod 4090s ran the curators from 12:50, 27 minutes late.

**Why the probes missed it:** `cage_check` proves the sandbox, not the floor. The cheapest hosts on both markets are old server CPUs, and a market's cheapest host is what the fleet rents first.

**The fix, with stet-dc (one batch):**
- The server says, for each machine in `/admin/work`, whether it could be handed curators as it last asked (`curators`), and if not, why (`curators_why`: the floor's shortfalls, e.g. `CPU 95 MB/s, floor 100`).
- Client 0.4.5 sends its saved bench on plain `stet verify` too, so a machine that refused itself still says why.
- The same batch adds `failed`, `refused` and `trouble` per machine: jobs it handed back or let lapse, results the server refused, and the latest one's reason. The counts reset at the fleet's release.
- **Fleet:** a machine rented with curators allowed that the server says can't run them:
  - grades only (status: `grades only: …`, journal event `no curators`);
  - is ended while only curators want machines (no posted grade, no early grade in its window, no student), so a machine that can run them takes its place.
  - Its host is not rented for curators for a week. Nor is its CPU model, if the shortfall was the CPU's speed (`curators.json`; CPU names compared across markets, so `Xeon® E5-2680 v4` and `Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz` are the same).
  - Offers now carry `Host` and `CPU` where the market says: Vast `machine_id` and `cpu_name`, Clore server ID and `specs.cpu`.
- Vast's 400 `no_such_ask` (an ask gone between search and rent) counts as no room, so the next Vast offer is tried instead of passing over Vast.
- Failed and refused jobs are journal events (`trouble`) and appear on the status line.

**Order:** server and fleet ship in the same gap between rounds. Curators go back on for Vast and Clore once both are live.

**Also in this round: slow installs ended too soon.**
- Two of the nine Vast graders (stet-rp-17 at 13:13 and stet-rp-11 at 13:59) were ended at 11 minutes for never asking for work. Both were still downloading stet's libraries on a slow connection.
- The Vast machines that started asked within 1½–5 minutes, so Vast's grace is the 10-minute floor.
- A grace learned from machines that started never sees the slow ones, because they were ended first.
- **Rule:** from 2 minutes before a machine's grace runs out, its log is read each look (where its market keeps one). If it has grown since the last look, with nothing in it saying something didn't work, the machine has up to twice its grace (journal event `slow`). Once its log stops growing, or says it failed, or twice its grace has passed, it's ended as before.
- **Also:** Vast's search lists an ask the fleet rented a moment earlier, which accounts for some of the `no_such_ask` refusals. As no room, the next offer is tried at once.

## 16. Everything the fleet does, kept (Eric, 2026-11-21)

Eric asked for a log of everything, for seeing later the real costs on every market, the markets' and the server's latency, and where the work stalled; "instrument more deeply so that retroactive analysis can be far more complete", and to backfill what can be.

**Kept from now on**, under `~/.config/stet-cloud/history/` (a file a day where they are large, earlier days gzipped):

- **`looks-DAY.jsonl`**, one line per look (each minute):
  - all `/admin/work` said, whole: every round, and every one of the organisers' machines (busy, holding, ready, pass seconds, curators and why not, failed, refused, trouble, harness), so each grade's taking, training and reporting, and idle holding, can be read back minute by minute;
  - the machines by state and by market, and $/h and today's spend;
  - the decision: machines wanted, to rent (for the bigger students, early grades, curators, the rest), ended (each with why), keys whose jobs were handed back, the exam due, the grades a machine rented now is to take, and what the graders still to come wait for;
  - each market's offers: how many, how long they took, and the error if any;
  - each machine tried: handle, what it was for, market, offer, card, price, how long the market took, and what came of it (rented, no room, or the error);
  - every HTTP request to a market or the server: who, method, path (never the query or headers, where keys are), status, milliseconds, error;
  - everything the fleet said, the look's error if it went no further, the fleet's commit and the stet the server hands out.
- **`offers-DAY.jsonl`**: every 10 minutes, each market's offers by card and kind of host (count, cheapest, median, start rate, speed, cost of a grade at the cheapest).
- **`balances.jsonl`**: each market's account, hourly. RunPod's needs a key that may read the account (the scoped key may not).
- **`rounds/ID.json`** and **`rounds.jsonl`**: each round, whole, once it has settled.
- `machines.jsonl`: the "rented" event says what the machine was for, the round, when it was expected to report its last grade, where, and on what CPU; the ledger keeps the same (`for`, `round`, `report`), and so `spent.jsonl` once it ends.

**Also fixed:** a machine is seen starting from the moment it is rented, so one that asks for work before the next look has its boot counted. Lium's first machine asked within 41 s and was never counted, so Lium's grace stayed at 30 min and a dead Lium machine held its place that long in round 10; every market's boots were biased slow the same way.

**Backfill** (`history/backfill/`): the journal since Nov 18, each line classified; each market's billing, where its API gives it; every round from the server; and a machine table joining them.

## 17. Season 2 (merged 2026-11-22, 4ff52af)

Season 2 changes what a round asks of the fleet (stet-89's notice; `src/stet/seasons/s2.json`):

- **The exam is made at the draw**, from hours of the day before already harvested, and posted within a minute or so of it, not about 2h12m after. Nearly every grade is now one of a posted exam, rented for fastest first (`byPace`), and the early grades of §10–§11 are rare.
- **Baselines on four machines a round** (`baselines.per_round`), not on every machine's first grade. `/admin/work` rounds say `baselines_wanted`: how many more machines the round would ask to measure it.
- **Fewer runs:** a ladder of eight needs 9 curator runs and 13 trainings, against Season 1's 16 runs and up to 32 trainings. The runner floor, the curator sandbox's limits, `detail.train.seconds` and the install from the site are unchanged.

What the fleet does about it:

- **`lead()` counts two passes only while a round open wants a baseline** (or the server does not say, or none is open, since the next round wants its baselines first). Once none does, a machine rented trains once for its first grade.
- **A late exam's time is not known.** `examDue` is zero once the exam's time has passed with no exam, so nothing is rented to take its grades early, or kept waiting for them, until it is posted. Before, an overdue exam left `forExam` with no kind in time, and it rented the quickest at any price under `--max-price`, each to hold a trained grade until the exam came: a PR reviewer's finding, confirmed in the code.
- **`--exam-after` is 5 minutes**, for a server that does not yet say `exam_due` (a new database has no rounds to pace from).
- The service runs with `--exam-after 5m --early 0` (Eric, 2026-11-22): graders are rented only once an exam is posted.
- **Round s2-o00001** (stet-db): drawn 21:00, exam posted 21:03:41, curators done about 21:13, settled 21:46:38 (46.6 min), $1.57 on 16 Vast machines. Its tail was two RTX 5070s rented for curators, the cheapest a grade, that took baselines: two passes of about 1,000 s each. A pass cost about the same on Vast's 5070, 5080 and 5090 ($0.041–0.046), so the cheaper card saved nothing; on 5090s alone it would have settled in about 28 min.
- So **curator machines are rented the fastest first too** (Eric, 2026-11-22), among the hosts where curators may run. `--cheapest` still rents every machine by what a grade costs.

## 18. Season 3 (2026-11-24 / 2026-11-25)

Season 3 moves to `baidu/ERNIE-4.5-0.3B-Base-PT` (embeddings + bottom 9 layers frozen, top 9 layers reset + 1-epoch 32-step commercial-permissive multilingual SFT probe, deterministic CUDA kernels), 24 GB curator sandboxes with `vLLM`/`FLA`/`Liger`, a 100%-fresh 7-pillar exam, and dynamic on-demand-only rounds:

- **Hard 6-GPU cap (`--max 6`, `--max-hourly 4`):** rounds draw only when entries are waiting (`STET_ROUND_INTERVAL=0`), chaining back-to-back until every waiting/qualifying entry finishes, and winding down straight to **0 GPUs** when quiet.
- **Warm keep-alive across imminent draws (`nearDraw`, `keptNearDraw = 2`):** when a round is scheduled within 20 minutes (`next_draw_at`), up to 2 idle curator machines are kept warm instead of being killed during the 6–16 minute draw notice window; once nothing is waiting or scheduled (`next_draw_at == 0`), every idle machine is ended immediately.
- **24 GB curator floor vs 12 GB+ grader floor:** `--min-curator-gpu-memory 24` filters curator-capable offers (for the 22 GB sandbox), while `--min-gpu-memory 12` allows 12 GB+ cards to grade `ERNIE-4.5-0.3B-Base-PT` (`8.90 GiB` allocated / `9.33 GiB` reserved across training, SFT, and exam sitting). `--min-ram` rises to 24 GB.
- **Pass pacing, `baselines.per_round = 2`, and `--exam-after 25m`:** a Season 3 pass is ~20 min on RTX 5090, ~29 min on 4090, and ~52 min on 3090; `baselines.per_round` drops from 4 to 2 (saving two unedited passes per round now that student init & kernels are deterministic); `--exam-after` defaults to `25m` to match the local AMD Radeon 8060S examiner's draw-to-publication time (~20–30 min); Lium `termination_hours` and Clore `rental_hours` stay at 6 hours so warm workers cycle through back-to-back rounds without mid-grade cutoffs.
- **Per-season pass calibration & `runsPerMachine = 2`:** `calibration.json` is keyed by active season so Season 2's pass times never skew Season 3 lead times or card ordering; curator machine sizing drops from `4` to `2` runs per machine so 20-minute neural curators parallelize across the 6-GPU cap; grade-only machines pass `STET_WORKER_EDITS="0"` and never count as curator runners.
- **Compiler toolchain in `worker.sh`:** installs `gcc g++ libc6-dev` when missing (failing loudly if the install fails) so Triton, vLLM, and FLA kernels compile cleanly inside containers.

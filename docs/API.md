# API

Everything the website shows and everything the `stet` command does goes through this
API. Base URL: `https://api.stetrun.com`. Responses are JSON. Errors are
`{"error": "..."}` with a message written to be shown to a person.

The API can change between harness versions. The season endpoint says which version the
server speaks.

A curator is called an `editor` in paths and field names: that was its name when they
were given, and changing them would break every client for a word.

Scores are in bits per byte, and higher is better.

## Reading

No key needed.

| | |
|---|---|
| `GET /api/v1/season` | The season: student, budget, tolerances, the record rule, the next and open rounds, the current CLI version, the `unit` scores are in, and `has_multiplier` (false until the multiplier's curve is measured); and what a curator is promised: its `machine`, the floor for machines that run it (`runner`), the limits on `assets`, and the `environment` (Python and every library, with versions) |
| `GET /api/v1/leaderboard` | `editors` (ranked) and `provisional` (fewer than `rated_rounds` rounds) |
| `GET /api/v1/rounds?limit=30` | Recent rounds, newest first, and `next`: what the next draw is committed to |
| `GET /api/v1/rounds/{id}` | One round with its entries, those with a score first, best first, then those that failed. A round id looks like `s2-o00012`. Once it is drawn it has `exam_hours`, the harvests its exam's writing is made from (`hour`, `sha256`), with `exam_hours_late` when they were chosen after the draw. Once the round's exam is posted it has `exam_digest` and `exam_at`, and once it has settled and its item scores are published, `scores`: their address. Before its draw a round has no entries; it has `expected`, the curators the draw would enter if it were made now, and `later`, the queued ones it would leave for a later round. Neither is a promise. Of the organisers' reference curators on the ladder, `expected` has those the season runs in every round (`references.every_round`) and those whose turn this round is: each of the rest runs in one round in `references.one_in` (8), and is in neither list. Each row has `editor_id`, `name`, `version`, `owner`, `why` (`ladder`, `qualifying` with `played` rounds so far, `new`, or `later`) and `public`, whether its code is public yet |
| `GET /api/v1/rounds/{id}/exam` | The round's exam, zstd-compressed JSON Lines. Once the round has settled, to anyone: the generated items and Wikipedia's text whole, and every other item as its source, id, kind, format and language, with `withheld` set. The items' ids, in order, hash to `exam_digest`. Before that, the whole exam, only to a request signed by an account holding a grade job in the round, or by the organisers: `403` otherwise, `404` before it is posted |
| `GET /api/v1/rounds/{id}/scores` | Every item score of a settled round, as JSON, to anyone: see [below](#a-rounds-item-scores). `404` until the round has settled and its scores are published, a minute or so after |
| `GET /api/v1/harvests?limit=72` | `harvests`: the hours of writing the examiner has harvested, newest first, at most 504 (three weeks of them). Each has `hour` (Unix seconds, on the hour: the harvest is of the hour that begins then), `sha256` (of the harvest's file, which the examiner keeps), `documents`, `sources` (how many documents from each source) and `at` (when the server was told). Beside them, the season's `exam.fresh`: `hours` (3: how many a round's exam is made from), `reach_seconds` (93,600: a round is given hours that began no longer than this before its draw) and `settle_seconds` (3,600: how long after an hour ends it is harvested). A round's `exam_hours` follow from these and its seed: of the harvests within reach that the server had been told of before the draw, the `hours` whose `sha256(seed, then "\|fresh/hour\|<hour>")` is smallest, in the hours' order (`stet.exam.fresh.harvest.choose_hours`) |
| `GET /api/v1/editors/{ref}` | One curator. `ref` is an id, the first 8 or more characters of one (the oldest curator that fits), or `handle/name` with the slash written `%2F` |
| `GET /api/v1/editors/{ref}/bundle` | The curator's files (`.tar.zst`), once it is public |
| `GET /api/v1/scripts/{sha256}` | An edit script, zstd-compressed JSON Lines. Public once a round that holds it has settled. Before that it is sealed: `403`, unless the request is signed by the account that holds the script as a grade job |
| `GET /api/v1/verifiers` | The verifier board: the organisers' accounts, and every volunteer with a grade that has agreed or a curator run. A row has `handle`, `validated` (grades that agreed), `runs` (curators run), `established`, `gpu_hours`, `gpu` and `last_seen`; `trusted` and `anchor` mark an organisers' account |
| `GET /api/v1/records` | Every time the record changed hands |
| `GET /files/{name}` | Downloads: the CLI wheel |

A round is **sealed** until it settles: before that an entry shows its `stage` (`editing`,
`grading`) and nothing about its runs, the scripts they wrote or any grade of them, and
the round does not show its unedited score.

A settled entry:

| Field | |
|---|---|
| `valid`, `gain`, `loss` | The score. `gain` is taken over the entry's runs: the mean of two, the median of three. A failed run counts 0, or as the entry's worst finished run when that is below 0. An entry is `valid` when at least half of its runs finished. `loss` is the unedited student's mean bits per byte less the gain |
| `counts_as` | Only on an entry that failed (`valid` false, `gain` 0, not `void`): what the round counts as for it in its rating and against the record. It is the lowest `gain` of the round's valid entries, or 0 if that is lower |
| `n_runs` | How many runs the score is made of |
| `confirmed` | Two different people's runs agree, to within the season's tolerance for runs, neither rests on a grade by its own runner, and more than one account's grades decide them |
| `one_grader` | Its runs agree, and one person's grades decide them all: nobody else was there to grade. The organisers' machines count as one person |
| `one_runner` | Its runs agree, and one person's machines made them all (the organisers'): nobody else was there to run it |
| `runs_differ` | It was run more than once and no two runs agree |
| `runs` | Each run: see below |
| `dropped_runs` | Runs that count for nothing because nobody else was there to check them, in the same shape with `status` `dropped` and a `reason` |
| `script`, `edit` | Those of the run whose gain is nearest the entry's, for anything that wants one script |
| `by_source` | The gain on each pillar, made as `gain` is: the mean, pillar by pillar, over the runs `gain` is the median of (the middle run of three, both of two; a failed run counts on every pillar as it does in `gain`: 0, or as the run that stands in for it), of each run's script's, which is the mean over the grades that script's gain is the median of. A grade's score is a fixed weighted sum of its pillars' scores (the season's `exam`), so the same sum of these is `gain`, to rounding |
| `void`, `reason` | A round that does not count for the entry, and why; or why it is invalid |
| `detail` | `verifiers` (whose grades the score is made of, then everyone else whose grade agreed), `disputed` (grades that differed), `disputed_together` (two established accounts graded one of its scripts, disagreed, and nobody was there to break the tie), and `script`, `train`, `hardware` from the grade nearest the score |

A run:

| Field of a run | |
|---|---|
| `n`, `by`, `gpu`, `backend` | Which run, whose machine, and its GPU (`cuda` is NVIDIA, `rocm` is AMD) |
| `sandbox` | What the curator ran in: `bwrap`, or `container` on the organisers' rented machines ([the rules](RULES.md#the-curators-machine) say how they differ) |
| `edit_seconds` | How long the curator took there |
| `status` | `ok`, or `failed`: the curator wrote no script, or wrote an invalid one. `reason` says which |
| `script`, `script_size` | The sha256 of the edit script it wrote. Two runs that wrote the same bytes name the same script and share its grades |
| `gain`, `loss` | The script's gain: the median of the grades that count |
| `self` | Its only established grade is its own runner's, which only an organisers' machine with nobody else around is ever asked for |
| `grades` | Every grade of the script, those that count first |

A grade has `by`, `valid`, `loss`, `gain` (the exam's aggregate of that grader's item
scores less the round's baseline), and a `verdict`: `agreed`, `outlier`, `unresolved` (it
differs from a gain that rests on one volunteer's grade, or on two that disagree: no
strike), `disputed` (two established grades disagreed and no third was made), `other data`
or `baseline differs` (a set-up problem: no evidence about the script),
`no established grade`, or `no baseline` (an established grade of a script that was given
up because nobody was there to measure the round's unedited baseline: it has no `gain`).
`counted` marks the grades the script's gain is the median of, `established` those by
established accounts, `self` a grade by the only person whose run wrote the script, and
`first` an account's first grade.

A settled round's `differs` lists verifiers who were turned away before their run or
grade was taken: `what` is `corpus` (their machine built another corpus or exam) or
`baseline` (their unedited student was out of tolerance, by `difference`), and
`established` is set for an account whose work would otherwise have counted.

A round that was announced for one draw and then moved lists the beacon rounds it left in
`moved_from`.

A leaderboard row:

| Field | |
|---|---|
| `id`, `name`, `version`, `owner`, `technique` | The curator |
| `rank` | Place among rated curators |
| `tied` | Set when its `rating` is too close to tell apart from that of the first row of the run of tied rows above it: the gap is within two standard errors of the gap, `2·sqrt(se² + se_first²)`. Rows are held against the run's first row, not the row just above, so ties do not chain. The site gives a run of tied rows the place it starts at |
| `rounds` | Official rounds it has settled |
| `rating` | In bits per byte. The board is sorted by this. It is the curator's mean gain adjusted for round difficulty: the mean, over its rounds, of the round's gain less the round's difficulty in `gain = skill + difficulty(round)`. A round the curator failed in counts as that entry's `counts_as`. Who else is on the board is no part of it |
| `adjusted` | The same number as `rating`, under the name it had while a rating was also drawn towards the mean of the board |
| `mean_gain` | The plain mean of the same numbers |
| `se` | The rating's standard error, `sqrt(s2/n)` after `n` rounds. `s2` is the variance of one round's gain about a curator's own mean, pooled over the curators with two rounds or more, together with the square of the season's `noise.<scale>.round_sd`, which counts as eight rounds' worth, so that a season's first rounds do not set it alone |
| `lower` | `rating − 1.28·se`. The ladder is filled in order of this, not of `rating` |
| `multiplier`, `capped` | Null until the multiplier's curve has been measured on this exam |
| `record`, `ladder`, `provisional` | Flags |

`GET /api/v1/editors/{ref}` gives a curator the same fields.

### A round's item scores

`GET /api/v1/rounds/{id}/scores` is what every score of a settled round is made of. Each
list of numbers has one score for each item of the round's exam, in the order of
`GET /api/v1/rounds/{id}/exam`.

| Field | |
|---|---|
| `round`, `items_digest` | The round, and its exam's digest |
| `table` | What each item counts toward: its `pillar`, its `part` of that pillar (both as the season's `exam` names them), `lang` and `stratum` |
| `languages` | The percent of the round's corpus in each language, which weighs the items |
| `baseline` | The unedited student's score on each item: the median of the baselines of the graders named in `baseline_of` (of two, their mean). Until three are in, every established grader's is kept; from three on, one further than the tolerance from the median of them all is left out, whichever came first |
| `baselines` | Every baseline measured for the round, by its grader's handle. Only some graders are asked for one: a grade job says whether (`baseline`, [below](#jobs)) |
| `grades` | Every grade's score on each item, by an id |
| `grades_of` | For each of those ids, what it is a grade of: `editor` (the entry), `script` (its sha256), `runs` (the `n` of the entry's runs that wrote that script), `by` (the grader's handle), and `counted` when it is one of the grades the script's gain is the median of. A grade that found its script invalid sat no exam and is not listed |

A grade's `gain` is the season's aggregate (`score_from_table` in the harness, over `table`
and `languages`) of its item scores less `baseline`, item by item. A script's gain is the
median of its `counted` grades' gains, and an entry's the median of its runs'.

## Signing

Requests that act for an account are signed with its Ed25519 key. Four headers:

```
X-Stet-Key:  <public key, hex>
X-Stet-Time: <unix seconds>
X-Stet-Sig:  <signature, hex>
X-Stet-Body: <sha256 of the request body, hex>
```

`X-Stet-Body` is required on uploads (a bundle, a script, an exam), where it lets the
server check the signature before reading the body. Elsewhere it is optional.

The signed message is five lines joined by `\n`:

```
stet-v1
<METHOD>
<path, with query string>
<the X-Stet-Time value>
<sha256 of the request body, hex>
```

The timestamp must be within 300 seconds of the server's clock. The message does not name
the server, so a key should not be used with a server its owner does not trust.

## Accounts and entries

| | |
|---|---|
| `POST /api/v1/accounts` `{"handle": "..."}` | Create the account for the signing key |
| `POST /api/v1/accounts/github` `{"gist": "<url>"}` | Link GitHub. The gist must be public and contain the line `stet-key:<public key>` |
| `GET /api/v1/me` | Your account, your curators, and under `verifier` your standing: `validated` (grades that agreed), `establish_after` (how many it takes), `established`, `needs_github` (no GitHub account is linked, and being established needs one), `runs` (curators run), `strikes`, `muted`. The account has `worker` set if it is one of the machines the organisers rent |
| `POST /api/v1/editors` | Submit. The body is the bundle: a zstd-compressed tar of the project directory, at most 50 MB. Every file its `stet.toml` names under `[[assets]]` is checked with its host first: `400` says which one is wrong, `503` that the host could not be asked. The reply has the `editor`, `in_play` and `in_play_limit` (how many entries this person now has waiting or qualifying, and how many they may), `replaced` (the id of an entry this one took the place of, if it did) and either `enters` (`round`, `draw_at`: the next round, when the entry will run in it) or `earliest` (the same of the next round, when it may not: the round has no room for it, or its draw is too close for that to be a promise) |

## Jobs

What `stet verify` does. A third-party verifier must use the reference harness to compute
grades: a grade computed any other way will not agree.

| | |
|---|---|
| `POST /api/v1/jobs/claim` | Ask for work. The body says which harness and season you run, what hardware, and whether you will run curators (`edits`, with the sandbox in use, `bwrap` or, on the organisers' rented machines only, `container`, and the speeds `stet bench` measured). Returns `job` (`kind` is `grade` or `edit`; or null, with `retry_after`), `announced` (the next draw's commitment) and, when the body asked for curators, `runner`: `eligible`, and a `reason` if not. A body with `early: true` (stet 0.4.3 and later) says the verifier trains before the exam: it may be handed a grade job as soon as there is a script, while the round's job description has no `exam_digest` yet; it trains, waits for the exam and sits it once it is posted, and its job does not run out while it waits. Anyone else is handed grade jobs once the exam is posted. From stet 0.4.5 it says the speeds `stet bench` measured whenever this machine has them, curators or not; from 0.4.6 they include `disk_gb`, the free disk under its cache, and a body that does not ask for curators says `curators_declined`, why the machine's last `stet verify --edits` ran none, if it said so in the last week. From stet 0.4.4 the body also says `pass_seconds` (how long this machine last took to train a grade's student; absent until it has graded) and `trained` (the grade jobs it holds whose students are trained and wait only for the exam). It is not handed those again, and while they are all it holds, an established verifier may be handed one more grade a score waits for, up to three, while `pass_seconds` and two minutes more still fit before the exam is due. For the first two minutes after an exam is posted, a grade a score waits for goes to a machine only when there are more of them than faster machines that are free or about to be; `retry_after` then says when to ask again. A grade job names its `script` (`sha256`, `size`) and which `run` wrote it; before a round settles that is the only way a script's hash is told to anyone. A grade job also says `baseline`: `true` if this machine is to measure the round's unedited baseline as well (train the unedited corpus, sit it, and send it with the grade), `false` if it is to train the script's student alone. The season says how many machines measure it (`baselines.per_round`, 2 in Season 3): a job says `true` while fewer established accounts than that have measured the round's or hold a grade that asked them to, always to an account that is not established (its baseline is how its machine is checked), and always to an account that has measured the round's or was asked before. A job handed back, run out or turned away takes its promise with it, and the next job handed out says `true`. The same job, handed again, says what it said the first time. A job with no `baseline` (an older server) means `true` |
| `POST /api/v1/jobs/{id}/result` | Report a grade: the script's hash; the exam score, as one score for every item of the round's exam in order (`item_scores`) and their `mean`; your `baseline` for the round the same way, if the job said `baseline: true`; and the corpus and exam hashes. A result of a job that said `baseline: false` leaves `baseline` out, or may send it if the machine has the round's from an earlier job: it is checked like any other. A result that leaves out a baseline its job asked for is refused (`400`). Every grade is scored against the round's baseline, the median of those its established graders sent, whether or not it came with one. A baseline is refused (`409`) only once two others are in and it is further than the tolerance from the median of the three or more. The answer is `pending` until the round settles |
| `POST /api/v1/jobs/{id}/script` | Report a run of a curator and upload the script it wrote, or say that it wrote none. The answer is `edited`. Established accounts only: `403` otherwise |
| `POST /api/v1/jobs/{id}/release` | Hand a job back. `{"reason"}` says why, in words; the organisers see it for their own machines |

Status codes worth handling: `403` (this account is given no work for now: the
message says why and until when), `409` (your data or baseline differs from the round's,
the job was already reported, or it ended before your result arrived), `426` (upgrade
the harness), `429` (slow down), `503` (the server could not make a write durable, or
cannot reach its records; send the same request again).

## Organisers

Signed by an organisers' key; anyone else gets `403`, and so does a machine the organisers
rent (one an organisers' key has vouched for).

| | |
|---|---|
| `POST /api/v1/admin/rounds/{id}/exam` | Post an open round's exam (`stet examine`): `items_digest`; `items_file` (base64 of the zstd JSON Lines graders sit); `public_file` (the same for what is published at settlement); `table`, what each item counts toward (`pillar`, `part`, `lang`, `stratum`); `languages`, each language's percentage of the round's corpus; `hours`, the harvests its writing was made from (`hour`, `sha256`), in the hours' order; `snapshot_sha256`, the sha256 of those hours' `sha256` strings joined in that order; `fresh`, how many items are fresh text; and optional `init_sha256` (authoritative initial student weights digest locked for the round). `hours` must be the round's `exam_hours`, exactly: `400` otherwise. A pillar with zero items is accepted (its share redistributes within its group); a present pillar below `min_pillar_items` or an empty declared group is refused with `400`. A round that was given none at its draw, because the server had been told of no harvest, takes an exam made from any harvests it has been told of since whose hours began no more than `reach_seconds` before the draw, and from then on has them as its `exam_hours`, with `exam_hours_late`. Grades trained before it are sat then, and the rest start. Posting the same exam again answers `posted: false`; another exam for the same round is refused with `409` |
| `POST /api/v1/admin/rounds/{id}/cancel` | Cancel an open round (`{"reason": "...", "force": false}` optional). Voids every open entry as unattended (nothing counts for or against any curator), clears any partial baseline, expires outstanding claims, emits a public `cancel` event, and settles the round with `baseline_loss: null` so the next round can be drawn. Refused with `409` if the round is not open, or if its exam is already posted and `"force": true` was not set. (The server also auto-cancels an un-examined round after `STET_EXAM_TIMEOUT`, default `6h`) |
| `POST /api/v1/admin/harvests` | Tell the server of an hour's harvest (`stet examine --follow`): `hour` (Unix seconds, on the hour: the harvest is of the hour that begins then), `sha256` of the harvest's file, `documents`, and `sources` (how many documents from each source, at most 64 of them). Answers the harvest, with `at`, when the server was told: `201` the first time, `200` when it is told the same harvest again. `400` if it does not add up, or the hour is not over. `409` for another harvest of an hour the server already has: a round may rest on the first. The server keeps the newest 504, and a round keeps its own hours |
| `GET /api/v1/admin/accounts/{handle}` | An account's public key, for adding it to the organisers' keys, and whether it is one of their rented machines (`worker`) |
| `POST /api/v1/admin/workers` `{"key", "at", "sig"}` | Vouch for a machine the organisers rent. `sig` is the signing organisers' Ed25519 signature, in hex, of `stet-worker-v1\n<key>\n<at>`; `at` is now, in Unix seconds, give or take an hour. From then on the key is one of the organisers' machines to the judge, and administers nothing. The grant is kept with its signature, and counts while the key that signed it is one of the organisers' |
| `GET /api/v1/admin/workers` | The grants: each machine's `key`, its `handle` once it has an account, who vouched for it (`by`) and when (`at`), whether it `counts`, `last_seen`, and `busy` if it holds a job |
| `DELETE /api/v1/admin/workers/{key}` | Take a grant back. The account stays, and is nobody's machine |
| `POST /api/v1/admin/workers/{key}/release` | The machine has been ended: every job it holds, official or one of the farm's students, is handed back as if it had handed it back itself, so that nothing waits out its lease. What `admin/work` counts against the key's machine starts again from here. Answers `handle`, and how many `jobs` and `students` were handed back. Only for a key the organisers vouch for (404 otherwise) |
| `GET /api/v1/admin/work` | What the open rounds are waiting for, as counts: for each round `entries`, `runs_waiting` (entries someone who is around could be handed to run, and nobody is running), `runs_wanted` (entries a runner of the organisers' that had just started could be handed), `edits_under_way`, `grades_wanted` (scripts a score waits on that a grader of the organisers' who had just started could be handed now), `grades_under_way`, `grades_early` (before the exam is posted, the grades a score will wait on that a grader of the organisers' which trains before the exam could be handed now) `exam_at` (when the exam was posted, once it is), `drawn_at`, `exam_due` (before the exam, when it is expected: the draw and the median draw-to-exam time of the latest rounds) and `baselines_wanted` (how many more machines the round would ask to measure its unedited baseline with their next grade: the season's number, less the established accounts that have measured it or hold a grade that asked them to. Such a machine's first grade of the round is two training passes, any other grade is one), `baseline_disputed` (omitted unless established baselines disagree beyond tolerance and need a fresh independent baseline), and `fresh_graders_wanted` (omitted unless grades are waiting that no currently idle organiser machine is eligible to take, e.g. a baseline or script/token-stream tie); `runners` (accounts around that run curators); `waiting` (the next draw would be brought forward for an entry waiting for a round once someone who runs curators is around; false when rounds keep to the clock, while a round is open, and once the draw is as early as it can be); `next_draw_at`; `paused` (when the rounds were paused, if they are); `farm_waiting`, `farm_under_way` and `farm_long_waiting` (the farm's students); and `machines`, the organisers' accounts with `last_seen`, `busy` (`edit`, `grade` or `farm`, a farm student), `long` when the student it trains is one of the bigger ones, `holding` (the grade jobs it holds), `ready` (all of them are trained and wait only for the exam), `pass_seconds` (what it last said a pass takes it), `curators` (whether it could be handed curators as it last asked; always present) and `curators_why` (when it could not, why: the server's reason if it asked, or, if it did not, where the speeds it sent fall short of the floor, e.g. `... floor for running curators: CPU 95 MB/s, floor 100`, else `it did not ask to run curators`; then, after a colon, its own `curators_declined`, or what it sent), `harness` (the stet it last said it runs), and `failed`, `refused` and `trouble` (of the jobs it took within the last day and since its key's last machine was ended: those handed back for a reason other than stopping or let run out, those whose result was refused, and the latest of either in words); and `early_grades_per_machine`, `fast_first_seconds` and `early_window_seconds` (a rented machine is handed grades before an exam only once they, and the further ones it could take, would keep it busy until no sooner than this before the exam is due), the server's limits on all three. What `stet-cloud` rents by |
| `POST /api/v1/admin/draw` | Bring the next draw forward to the next mark on the draw grid (`stet draw`). Refused while the rounds are paused |
| `POST /api/v1/admin/pause` `{"paused": true\|false}` | Pause the rounds, or resume them (`stet-cloud pause`, `resume`). While they are paused no round is drawn, and the round under way finishes; `/season` gives `paused` (when, in Unix seconds) and no `next_round`. Answers `paused` (0 when they are not) and `paused_by` |

## The organisers' farm

Students the organisers train on their own machines, outside any round's grading: the
finals' students, and measurements. Signed by an organisers' key; the ones marked *machines*
also take the machines the organisers rent. Nothing here touches a grade.

| | |
|---|---|
| `POST /api/v1/admin/farm/batches` | Post a batch: a round's exam items (base64 of zstd JSON Lines) and the scripts whose students are wanted, with the scale (or a `scale_spec` for one the season does not name) and how long a task may be held. Posting the same items again adds any new scripts |
| `GET /api/v1/admin/farm/batches/{id}` | A batch: every task, and how each student did on every item |
| `PUT /api/v1/admin/farm/scripts/{sha256}` | Keep a script that is no round's entry, zstd-compressed, for the farm's students |
| `POST /api/v1/farm/claim` | *Machines.* Ask for a student to train, saying how many minutes it will give one (`max_minutes`); a bigger student goes only to a machine set aside for it |
| `GET /api/v1/farm/batches/{id}/items` | *Machines.* A batch's items |
| `GET /api/v1/farm/scripts/{sha256}` | *Machines.* A script kept for the farm |
| `POST /api/v1/farm/batches/{id}/tasks/{i}` | *Machines.* Report how a task's student did on every item |
| `POST /api/v1/farm/batches/{id}/tasks/{i}/release` | *Machines.* Hand a task back |

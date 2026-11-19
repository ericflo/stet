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
| `GET /api/v1/season` | The season: student, budgets, tolerances, the record rule, the next and open rounds, the current CLI version, the `unit` scores are in, and `has_multiplier` (false until the multiplier's curve is measured); and what a curator is promised: its `machine`, the floor for machines that run it (`runner`), the limits on `assets`, and the `environment` (Python and every library, with versions) |
| `GET /api/v1/leaderboard` | `editors` (ranked) and `provisional` (fewer than `rated_rounds` rounds) |
| `GET /api/v1/rounds?limit=30` | Recent rounds, newest first, and `next`: what the next draw is committed to |
| `GET /api/v1/rounds/{id}` | One round with its entries. A round id looks like `s1-o00012`. Once the round's exam is posted it has `exam_digest` and `exam_at` |
| `GET /api/v1/rounds/{id}/exam` | The round's exam, zstd-compressed JSON Lines. Once the round has settled, to anyone: the generated items and Wikipedia's text whole, and every other item as its source, id, kind, format and language, with `withheld` set. The items' ids, in order, hash to `exam_digest`. Before that, the whole exam, only to a request signed by an account holding a grade job in the round, or by the organisers: `403` otherwise, `404` before it is posted |
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
| `valid`, `gain`, `loss` | The score. `gain` is taken over the entry's runs: the mean of two, the median of three, a failed run counting 0. `loss` is the unedited student's mean bits per byte less the gain |
| `n_runs` | How many runs the score is made of |
| `confirmed` | Two different people's runs agree, to within the season's tolerance for runs, neither rests on a grade by its own runner, and more than one account's grades decide them |
| `one_grader` | Its runs agree, and one person's grades decide them all: nobody else was there to grade. The organisers' machines count as one person |
| `one_runner` | Its runs agree, and one person's machines made them all (the organisers'): nobody else was there to run it |
| `runs_differ` | It was run more than once and no two runs agree |
| `runs` | Each run: see below |
| `dropped_runs` | Runs that count for nothing because nobody else was there to check them, in the same shape with `status` `dropped` and a `reason` |
| `script`, `edit`, `by_source` | Those of the run whose gain is nearest the entry's, for anything that wants one script. `by_source` is the gain in each pillar |
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
or `baseline differs` (a set-up problem: no evidence about the script), or
`no established grade`. `counted` marks the grades the script's gain is the median of,
`established` those by established accounts, `self` a grade by the only person whose run
wrote the script, and `first` an account's first grade.

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
| `rounds` | Official rounds it has settled |
| `rating` | Mean gain adjusted for round difficulty, in bits per byte. The board is sorted by this |
| `mean_gain`, `se` | Plain mean gain and its standard error |
| `multiplier`, `capped` | Null until the multiplier's curve has been measured on this exam |
| `record`, `ladder`, `provisional` | Flags |

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
| `POST /api/v1/editors` | Submit. The body is the bundle: a zstd-compressed tar of the project directory, at most 50 MB. Every file its `stet.toml` names under `[[assets]]` is checked with its host first: `400` says which one is wrong, `503` that the host could not be asked. The reply has the `editor`, `in_play` and `in_play_limit` (how many entries this person now has waiting or qualifying, and how many they may), `replaced` (the id of an entry this one took the place of, if it did) and either `enters` (`round`, `draw_at`: the round it is expected to run in) or, when that round has no room for it or is too close, `earliest` (the first round it could run in) |

## Jobs

What `stet verify` does. A third-party verifier must use the reference harness to compute
grades: a grade computed any other way will not agree.

| | |
|---|---|
| `POST /api/v1/jobs/claim` | Ask for work. The body says which harness and season you run, what hardware, and whether you will run curators (`edits`, with the sandbox in use, `bwrap` or, on the organisers' rented machines only, `container`, and the speeds `stet bench` measured). Returns `job` (`kind` is `grade` or `edit`; or null, with `retry_after`), `announced` (the next draw's commitment) and, when the body asked for curators, `runner`: `eligible`, and a `reason` if not. Grade jobs are handed out only once the round's exam is posted. A grade job names its `script` (`sha256`, `size`) and which `run` wrote it; before a round settles that is the only way a script's hash is told to anyone |
| `POST /api/v1/jobs/{id}/result` | Report a grade: the script's hash; the exam score, as one score for every item of the round's exam in order (`item_scores`) and their `mean`; your baseline for the round the same way; and the corpus and exam hashes. The answer is `pending` until the round settles |
| `POST /api/v1/jobs/{id}/script` | Report a run of a curator and upload the script it wrote, or say that it wrote none. The answer is `edited`. Established accounts only: `403` otherwise |
| `POST /api/v1/jobs/{id}/release` | Hand a job back |

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
| `POST /api/v1/admin/rounds/{id}/exam` | Post an open round's exam (`stet examine`): `items_digest`; `items_file` (base64 of the zstd JSON Lines graders sit); `public_file` (the same for what is published at settlement); `table`, what each item counts toward (`pillar`, `part`, `lang`, `stratum`); `languages`, each language's percentage of the round's corpus; `snapshot_sha256`, the harvest's hash; and `fresh`, how many items are fresh text. Grading starts then. Posting the same exam again answers `posted: false`; another exam for the same round is refused with `409` |
| `GET /api/v1/admin/accounts/{handle}` | An account's public key, for adding it to the organisers' keys, and whether it is one of their rented machines (`worker`) |
| `POST /api/v1/admin/workers` `{"key", "at", "sig"}` | Vouch for a machine the organisers rent. `sig` is the signing organisers' Ed25519 signature, in hex, of `stet-worker-v1\n<key>\n<at>`; `at` is now, in Unix seconds, give or take an hour. From then on the key is one of the organisers' machines to the judge, and administers nothing. The grant is kept with its signature, and counts while the key that signed it is one of the organisers' |
| `GET /api/v1/admin/workers` | The grants: each machine's `key`, its `handle` once it has an account, who vouched for it (`by`) and when (`at`), whether it `counts`, `last_seen`, and `busy` if it holds a job |
| `DELETE /api/v1/admin/workers/{key}` | Take a grant back. The account stays, and is nobody's machine |
| `POST /api/v1/admin/workers/{key}/release` | The machine has been ended: every job it holds, official or one of the farm's students, is handed back as if it had handed it back itself, so that nothing waits out its lease. Answers `handle`, and how many `jobs` and `students` were handed back. Only for a key the organisers vouch for (404 otherwise) |
| `GET /api/v1/admin/work` | What the open rounds are waiting for, as counts: for each round `entries`, `runs_waiting` (entries someone who is around could be handed to run, and nobody is running), `runs_wanted` (entries a runner of the organisers' that had just started could be handed), `edits_under_way`, `grades_wanted` (scripts a score waits on that a grader of the organisers' who had just started could be handed now) and `grades_under_way`; `runners` (accounts around that run curators); `waiting` (an entry is waiting for a round); `next_draw_at`; `paused` (when the rounds were paused, if they are); `shadow_waiting`, `shadow_under_way` and `shadow_long_waiting` (the farm's students); and `machines`, the organisers' accounts with `last_seen`, `busy` (`edit`, `grade` or `shadow`, a farm student), and `long` when the student it trains is one of the bigger ones. What `stet-runpod` rents by |
| `POST /api/v1/admin/draw` | Bring the next draw forward to the next mark on the draw grid (`stet draw`). Refused while the rounds are paused |
| `POST /api/v1/admin/pause` `{"paused": true\|false}` | Pause the rounds, or resume them (`stet-runpod pause`, `resume`). While they are paused no round is drawn, and the round under way finishes; `/season` gives `paused` (when, in Unix seconds) and no `next_round`. Answers `paused` (0 when they are not) and `paused_by` |

## The organisers' farm

Students the organisers train on their own machines, outside any round's grading: the
finals' students, and measurements. Signed by an organisers' key; the ones marked *machines*
also take the machines the organisers rent. Nothing here touches a grade.

| | |
|---|---|
| `POST /api/v1/admin/shadow/batches` | Post a batch: a round's exam items (base64 of zstd JSON Lines) and the scripts whose students are wanted, with the scale (or a `scale_spec` for one the season does not name) and how long a task may be held. Posting the same items again adds any new scripts |
| `GET /api/v1/admin/shadow/batches/{id}` | A batch: every task, and how each student did on every item |
| `PUT /api/v1/admin/shadow/scripts/{sha256}` | Keep a script that is no round's entry, zstd-compressed, for the farm's students |
| `POST /api/v1/shadow/claim` | *Machines.* Ask for a student to train, saying how many minutes it will give one (`max_minutes`); a bigger student goes only to a machine set aside for it |
| `GET /api/v1/shadow/batches/{id}/items` | *Machines.* A batch's items |
| `GET /api/v1/shadow/scripts/{sha256}` | *Machines.* A script kept for the farm |
| `POST /api/v1/shadow/batches/{id}/tasks/{i}` | *Machines.* Report how a task's student did on every item |
| `POST /api/v1/shadow/batches/{id}/tasks/{i}/release` | *Machines.* Hand a task back |

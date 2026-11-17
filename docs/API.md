# API

Everything the website shows and everything the `stet` command does goes through this
API. Base URL: `https://api.stetrun.com`. Responses are JSON. Errors are
`{"error": "..."}` with a message written to be shown to a person.

The API can change between harness versions. The season endpoint says which version the
server speaks.

## Reading

No key needed.

| | |
|---|---|
| `GET /api/v1/season` | The season: student, budgets, tolerances, the record rule, the next and open rounds, the current CLI version, the hashes of the exam and of the held-out exam |
| `GET /api/v1/leaderboard` | `editors` (ranked) and `provisional` (fewer than `rated_rounds` rounds) |
| `GET /api/v1/rounds?limit=30` | Recent rounds, newest first, and `next`: what the next draw is committed to |
| `GET /api/v1/rounds/{id}` | One round with its entries. A round id looks like `s0-o00012` |
| `GET /api/v1/editors/{ref}` | One editor. `ref` is an id, the first 8 or more characters of one (the oldest editor that fits), or `handle/name` with the slash written `%2F` |
| `GET /api/v1/editors/{ref}/bundle` | The editor's files (`.tar.zst`), once it is public |
| `GET /api/v1/scripts/{sha256}` | An edit script, zstd-compressed JSON Lines |
| `GET /api/v1/verifiers` | The verifier board: the anchors, and every volunteer with a grade that has agreed. `validated` counts those grades; `trusted` is an old name for `anchor` |
| `GET /api/v1/records` | Every time the record changed hands |
| `GET /files/{name}` | Downloads: the CLI wheel, the exam pack |

A round is **sealed** until it settles: before that an entry shows its stage (`editing`,
`grading`) and nothing about its grades, and the round does not show its unedited loss.
A settled entry's `loss` is an anchor's grade. Its `detail` holds the rest:

| Field of `detail` | |
|---|---|
| `verifiers` | The anchor, then the volunteers whose own grades agreed with it |
| `replications` | Every volunteer's grade of the entry: `handle`, `verdict` (`agreed`, `differs`, `other data`, `baseline differs`, or `no anchor grade` when the anchors could not grade the script), `valid`, `loss`, `difference` (its loss minus the anchor's), `baseline_difference`, and `first` when no grade of that account's had agreed before |
| `disputed` | How many of those differ |
| `disputed_together` | Two volunteers who have agreed with an anchor before differ from this grade and agree with each other |
| `script`, `train`, `hardware` | The anchor's own report |

A settled round's `differs` lists volunteers who were turned away before grading: `what`
is `corpus` (their machine built another corpus or exam) or `baseline` (their unedited
loss was out of tolerance, by `difference`).

A round that was announced for one draw and then moved lists the beacon rounds it left in
`moved_from`.

A leaderboard row:

| Field | |
|---|---|
| `id`, `name`, `version`, `owner`, `technique` | The editor |
| `rank` | Place among rated editors |
| `rounds` | Official rounds it has settled |
| `rating` | Mean gain adjusted for round difficulty, nats per token. The board is sorted by this |
| `mean_gain`, `se` | Plain mean gain and its standard error |
| `multiplier`, `capped` | Data-efficiency multiplier; `capped` when it is past the end of the measured curve |
| `record`, `ladder`, `provisional` | Flags |

## Signing

Requests that act for an account are signed with its Ed25519 key. Four headers:

```
X-Stet-Key:  <public key, hex>
X-Stet-Time: <unix seconds>
X-Stet-Sig:  <signature, hex>
X-Stet-Body: <sha256 of the request body, hex>
```

`X-Stet-Body` is required on uploads (a bundle, a script, a published file), where it lets
the server check the signature before reading the body. Elsewhere it is optional.

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
| `GET /api/v1/me` | Your account, verifier standing and editors |
| `POST /api/v1/editors` | Submit. The body is the bundle: a zstd-compressed tar of the project directory, at most 50 MB. The reply has the `editor` and either `enters` (`round`, `draw_at`: the round it is expected to run in) or, when that round has no room for it or is too close, `earliest` (the first round it could run in) |

## Jobs

What `stet verify` does. A third-party verifier must use the reference harness to compute
grades: a grade computed any other way will not agree.

| | |
|---|---|
| `POST /api/v1/jobs/claim` | Ask for work. The body says which harness and season you run and what hardware. Returns `job` (or null, with `retry_after`) and `announced`: the next draw's commitment |
| `POST /api/v1/jobs/{id}/result` | Report a grade: exam loss, per-source losses, your baseline for the round, corpus and exam hashes. The answer is `pending` until the round settles |
| `POST /api/v1/jobs/{id}/script` | Anchors only: report an edit and upload its script |
| `POST /api/v1/jobs/{id}/release` | Hand a job back |

Status codes worth handling: `403` (this account is given no grading work for now: the
message says why and until when), `409` (your data or baseline differs from the round's,
the job was already reported, or it ended before your result arrived), `426` (upgrade
the harness), `429` (slow down), `503` (the server could not make a write durable, or
cannot reach its records; send the same request again).

## Organisers

Signed by an anchor's key; anyone else gets `403`.

| | |
|---|---|
| `POST /api/v1/admin/draw` | Bring the next draw forward to the next mark on the draw grid (`stet draw`) |
| `PUT /api/v1/admin/files/{name}` | Publish under `/files/` (`stet publish`). The server accepts the exam pack and, once it has been told the season is closed, the held-out pack, each only with the hash the season gives for it. After the close it also accepts `heldout-<season>.seed`, `.results.jsonl` and `.table.json`, once each: a second upload of any of the three is refused with `409` |

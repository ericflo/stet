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
| `GET /api/v1/season` | The season: student, budgets, tolerances, the record rule, the next and open rounds, the current CLI version |
| `GET /api/v1/leaderboard` | `editors` (ranked) and `provisional` (fewer than `rated_rounds` rounds) |
| `GET /api/v1/rounds?limit=30` | Recent rounds, newest first, and `next`: what the next draw is committed to |
| `GET /api/v1/rounds/{id}` | One round with its entries. A round id looks like `s0-o00012` |
| `GET /api/v1/editors/{ref}` | One editor. `ref` is an id, an id prefix of 8 or more characters, or `handle/name` |
| `GET /api/v1/editors/{ref}/bundle` | The editor's files (`.tar.zst`), once it is public |
| `GET /api/v1/scripts/{sha256}` | An edit script, zstd-compressed JSON Lines |
| `GET /api/v1/verifiers` | The verifier board |
| `GET /api/v1/records` | Every time the record changed hands |
| `GET /files/{name}` | Downloads: the CLI wheel, the exam pack |

An entry in a round is **sealed** until it settles: before that it shows its stage
(`editing`, `grading`) and nothing about its grades.

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

Requests that act for an account are signed with its Ed25519 key. Three headers:

```
X-Stet-Key:  <public key, hex>
X-Stet-Time: <unix seconds>
X-Stet-Sig:  <signature, hex>
```

The signed message is five lines joined by `\n`:

```
stet-v1
<METHOD>
<path, with query string>
<the X-Stet-Time value>
<sha256 of the request body, hex>
```

The timestamp must be within 300 seconds of the server's clock.

## Accounts and entries

| | |
|---|---|
| `POST /api/v1/accounts` `{"handle": "..."}` | Create the account for the signing key |
| `POST /api/v1/accounts/github` `{"gist": "<url>"}` | Link GitHub. The gist must be public and contain the line `stet-key:<public key>` |
| `GET /api/v1/me` | Your account, verifier standing and editors |
| `POST /api/v1/editors` | Submit. The body is the bundle: a zstd-compressed tar of the project directory, at most 50 MB |

## Jobs

What `stet verify` does. A third-party verifier must use the reference harness to compute
grades: a grade computed any other way will not agree.

| | |
|---|---|
| `POST /api/v1/jobs/claim` | Ask for work. The body says which harness and season you run and what hardware. Returns `job` (or null, with `retry_after`) and `announced`: the next draw's commitment |
| `POST /api/v1/jobs/{id}/result` | Report a grade: exam loss, per-source losses, your baseline for the round, corpus and exam hashes |
| `POST /api/v1/jobs/{id}/script` | Anchors only: report an edit and upload its script |
| `POST /api/v1/jobs/{id}/release` | Hand a job back |

Status codes worth handling: `409` (your data or baseline differs from the round's, or the
job was already reported), `426` (upgrade the harness), `429` (slow down), `503` (the
server could not make a write durable; send the same request again).

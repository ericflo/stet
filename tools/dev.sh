#!/bin/sh
# Runs Stet on this machine the way production runs it: records in a real durable-db,
# files in a directory standing in for the bucket. Nothing here touches the internet
# except the crawl and the beacon.
#
#   tools/dev.sh                 start durable-db and the server on :8080
#   STET_ANCHOR_KEYS=<hex> ...   make your own key an anchor (see `stet whoami`)
#
# Then, in another terminal:  export STET_API=http://localhost:8080 STET_DEV=1
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEV="$ROOT/.dev"
DDB=${DURABLE_DB_BIN:-$HOME/Development/durable-db/target/release/durable-db}
PORT=${PORT:-8080}
DDB_PORT=${DDB_PORT:-7655}
mkdir -p "$DEV/files" "$DEV/data/blobs/files"

if [ ! -x "$DDB" ]; then
  echo "No durable-db binary at $DDB. Build github.com/ericflo/durable-db, or set DURABLE_DB_BIN." >&2
  exit 1
fi

# The CLI wheel the site's installer hands out, and the exam pack the CLI downloads.
rm -f "$DEV"/files/stet_cli-*.whl
(cd "$ROOT" && uv build --wheel -o "$DEV/files" >/dev/null 2>&1)
EXAM="${STET_CACHE:-$HOME/.cache/stet}/exam/exam-s0.jsonl.zst"
if [ -f "$EXAM" ] && [ ! -f "$DEV/data/blobs/files/exam-s0.jsonl.zst" ]; then
  cp "$EXAM" "$DEV/data/blobs/files/"
fi

"$DDB" serve --listen "127.0.0.1:$DDB_PORT" --data "$DEV/ddb" --key-file "$DEV/ddb.key" &
DDB_PID=$!
trap 'kill $DDB_PID 2>/dev/null || true' EXIT INT TERM
while [ ! -s "$DEV/ddb.key" ]; do sleep 0.1; done

cd "$ROOT/server"
go build -o stet-server .
STET_DEV=1 \
STET_ADDR=":$PORT" \
STET_BASE_URL="${STET_BASE_URL:-http://localhost:$PORT}" \
STET_DB_URL="http://127.0.0.1:$DDB_PORT" \
STET_DB_KEY_FILE="$DEV/ddb.key" \
STET_DATA_DIR="$DEV/data" \
STET_FILES_DIR="$DEV/files" \
STET_DOCS_DIR="$ROOT/docs" \
STET_AGENTS_FILE="$ROOT/src/stet/templates/AGENTS.md" \
STET_SCALE="${STET_SCALE:-nano}" \
STET_ROUND_INTERVAL="${STET_ROUND_INTERVAL:-10m}" \
STET_REPLICA_WAIT="${STET_REPLICA_WAIT:-1m}" \
STET_CRAWL="${STET_CRAWL:-CC-MAIN-2026-43:662f8425bc1e6211d388f8ed0a375835d96ffa277469200053ff3fbbd51db46f:100000}" \
./stet-server

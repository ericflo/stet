#!/bin/sh
# Builds harness.tar.zst: the agent harness, as the one file the bundle carries.
# Run it once before `stet run` or `stet submit`, and again when package.json changes.
#
# A bundle holds at most 2,000 files and never packs node_modules/ as it is, so a
# JavaScript harness goes in as one archive and is unpacked to /scratch when the editor
# starts (about a second). Any Node 22 or newer will do to build it; the editor itself
# runs on the season's Node, which every machine that runs editors has.
set -eu
cd "$(dirname "$0")"
npm ci --ignore-scripts --no-audit --no-fund
tar -cf - node_modules | zstd -19 -T0 -q -f -o harness.tar.zst
ls -l harness.tar.zst

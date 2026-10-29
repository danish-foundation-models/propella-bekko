#!/usr/bin/env bash
# Wait for a data file to appear, then train a config (for queued experiments).
set -euo pipefail
data=$1; cfg=$2
while [ ! -f "$data" ]; do sleep 60; done
sleep 60  # let the writer finish flushing
uv run propella-bekko train --config "configs/$cfg.json" > "logs/train_$cfg.log" 2>&1

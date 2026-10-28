#!/usr/bin/env bash
# Sequential tuning runs (one config per line in the argument list).
set -euo pipefail
for c in "$@"; do
  echo "=== $c $(date -Is)"
  uv run propella-bekko train --config configs/$c.json > logs/train_$c.log 2>&1 || echo "FAILED $c"
done

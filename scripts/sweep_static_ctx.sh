#!/usr/bin/env bash
# Context-length sweep for the static model on train-pool (8192 = configs/static-v1-pool.json).
set -euo pipefail
for L in 256 512 1024 2048 4096; do
  uv run propella-bekko train --config configs/static-ctx$L.json > logs/train_static-ctx$L.log 2>&1
done

#!/usr/bin/env bash
# Bigger training pool for scaling experiments (issue: tuning/scaling).
set -euo pipefail
uv run propella-bekko build-data --dynaword all --n-per-subset 5000 \
  --propella fineweb-2:all finepdfs:all --n-per-lang 20000 --per-group 100 \
  --exclude data/heldout-small.parquet \
  --head-chars 50000 --tail-chars 10000 --output data/train-pool-xl.parquet

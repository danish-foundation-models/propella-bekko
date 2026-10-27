#!/usr/bin/env bash
# Small held-out + training sets for fast iteration (issues #3, #5).
set -euo pipefail
uv run propella-bekko build-data --dynaword all --n-per-subset 50 \
  --propella fineweb-2:all finepdfs:all --n-per-lang 150 --per-group 10 \
  --output data/heldout-small.parquet
uv run propella-bekko build-data --dynaword all --n-per-subset 500 \
  --propella fineweb-2:all finepdfs:all --n-per-lang 2000 --per-group 50 \
  --exclude data/heldout-small.parquet --output data/train-small.parquet

#!/usr/bin/env bash
# Re-extract held-out / training-pool texts with up to 60k chars (50k head + 10k tail, ~propella's
# 50k-char view) reusing the existing labels (same seeds/params -> same docs). Issues #16, #17.
set -euo pipefail
which=${1:?heldout|pool}
if [ "$which" = heldout ]; then
  uv run propella-bekko build-data --dynaword all --n-per-subset 50 \
    --propella fineweb-2:all finepdfs:all --n-per-lang 150 --per-group 10 \
    --labels-from data/heldout-small.parquet --head-chars 50000 --tail-chars 10000 \
    --output data/heldout-small-long.parquet
else
  uv run propella-bekko build-data --dynaword all --n-per-subset 2000 \
    --propella fineweb-2:all finepdfs:all --n-per-lang 10000 --per-group 100 \
    --exclude data/heldout-small.parquet --labels-from data/train-pool.parquet \
    --head-chars 50000 --tail-chars 10000 --output data/train-pool-long.parquet
fi

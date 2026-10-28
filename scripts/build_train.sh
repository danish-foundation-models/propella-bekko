#!/usr/bin/env bash
# Training pool excluding the held-out set. The annotation id-scan cost per
# language is independent of N, so we build one generous pool once and
# subsample it per experiment (TrainConfig.max_train_docs).
set -euo pipefail
uv run propella-bekko build-data --dynaword all --n-per-subset 2000 \
  --propella fineweb-2:all finepdfs:all --n-per-lang 10000 --per-group 100 \
  --exclude data/heldout-small.parquet --output data/train-pool.parquet

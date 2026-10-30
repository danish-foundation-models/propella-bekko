#!/usr/bin/env bash
# Write parquet predictions for the three publishable students, then score vs GLM.
set -uo pipefail
models=(static-v1-pool a8m-v1-pool a25m-xl-3ep)
i=0
for m in "${models[@]}"; do
  CUDA_VISIBLE_DEVICES=$i uv run python scripts/student_predict.py "runs/$m/model" \
    data/heldout-sample1000.parquet "runs/pareto/$m.parquet" > "logs/pred_$m.log" 2>&1 &
  i=$((i + 1))
done
wait
uv run python scripts/mf1_vs_glm.py runs/reference/glm-5.3-flash-pareto300.parquet \
  runs/pareto/static-v1-pool.parquet runs/pareto/a8m-v1-pool.parquet runs/pareto/a25m-xl-3ep.parquet

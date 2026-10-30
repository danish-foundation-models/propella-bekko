#!/usr/bin/env bash
# Score the three publishable students against both references.
set -uo pipefail
models=(static-v1-pool a8m-v1-pool a25m-xl-3ep)
i=0
for m in "${models[@]}"; do
  CUDA_VISIBLE_DEVICES=$i uv run propella-bekko evaluate --model "runs/$m/model" \
    --data data/heldout-1k-deepseek.parquet --output "runs/pareto/$m.deepseek.json" \
    > "logs/eval_ds_$m.log" 2>&1 &
  i=$((i + 1))
done
wait
python3 - <<'PY'
import json

for m in ["static-v1-pool", "a8m-v1-pool", "a25m-xl-3ep"]:
    a = json.load(open(f"runs/pareto/{m}.json"))["paper"]["overall"]
    b = json.load(open(f"runs/pareto/{m}.deepseek.json"))["paper"]["overall"]
    print(f"{m:16s} propella {a:.4f}  deepseek {b:.4f}")
PY

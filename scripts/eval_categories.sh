#!/usr/bin/env bash
# Re-score the small and static models with the new expected-value ordinal decoding.
set -uo pipefail
models=(static-v1-pool a8m-v1-pool)
i=0
for m in "${models[@]}"; do
  CUDA_VISIBLE_DEVICES=$i uv run propella-bekko evaluate --model "runs/$m/model" \
    --data data/heldout-sample1000.parquet --output "runs/pareto/$m.json" > "logs/eval_$m.log" 2>&1 &
  i=$((i + 1))
done
wait
python3 - <<'PY'
import json

for m in ["static-v1-pool", "a8m-v1-pool", "a25m-xl-3ep"]:
    p = json.load(open(f"runs/pareto/{m}.json"))["paper"]
    print(f"{m:16s} overall {p['overall']:.4f} qwk {p['mean_qwk']:.4f} iou {p['mean_iou']:.4f}")
PY

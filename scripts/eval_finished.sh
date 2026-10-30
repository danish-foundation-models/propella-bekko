#!/usr/bin/env bash
# Evaluate the finished runs on the 1k test set, one GPU each.
set -uo pipefail
models=(a25m-v2 a25m-v3 mmbert-small-xl a25m-xl-3ep)
i=0
for m in "${models[@]}"; do
  CUDA_VISIBLE_DEVICES=$i uv run propella-bekko evaluate --model "runs/$m/model" \
    --data data/heldout-sample1000.parquet --output "runs/pareto/$m.json" > "logs/eval_$m.log" 2>&1 &
  i=$((i + 1))
done
wait
python3 - <<'PY'
import json

for m in ["a25m-v2", "a25m-v3", "mmbert-small-xl", "a25m-xl-3ep", "a25m-xl-soft"]:
    try:
        p = json.load(open(f"runs/pareto/{m}.json"))["paper"]
        print(f"{m:16s} overall {p['overall']:.4f} qwk {p['mean_qwk']:.4f} iou {p['mean_iou']:.4f}")
    except Exception:
        print(m, "missing")
PY

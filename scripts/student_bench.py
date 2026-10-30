"""Benchmark a trained student on a parquet and write `runs/pareto/<name>.bench.json`.

Same shape as the file the Pareto scripts read: docs/s on whatever GPU this runs on,
plus the parameter count, dtype and GPU name.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

from propella_bekko.inference import Annotator
from propella_bekko.models import PropellaBekko, count_parameters

run, inp = sys.argv[1:3]
dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
    sys.argv[3] if len(sys.argv) > 3 else "bf16"
]

model, tok, cfg = PropellaBekko.load(f"runs/{run}/model")
ann = Annotator(
    model,
    tok,
    cfg["max_length"],
    cfg["tail_tokens"],
    device="cuda",
    dtype=dtype,
    max_tokens=262144,
)
rows = pq.read_table(inp, columns=["id", "text", "n_chars", "n_words"]).to_pylist()
result = ann.benchmark(rows) | {
    "params_M": count_parameters(model) / 1e6,
    "max_length": cfg["max_length"],
    "dtype": "bf16" if dtype is torch.bfloat16 else "fp16",
    "gpu": torch.cuda.get_device_name(0),
}
out = Path("runs/pareto") / f"{run}.bench.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(run, result["docs_per_s"], "docs/s on", result["gpu"])

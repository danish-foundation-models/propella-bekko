"""How much do the agreement numbers depend on the dtype of the prediction run?

The published throughput is bf16 on an RTX A6000; the agreement numbers were scored from
fp16 predictions on an RTX 8000 (Turing has no bf16). This compares fp16 against fp32 on
the same card, over the same 300 documents, against the same DeepSeek-V4.1-Flash reference:
if the two agree, the dtype of that run is not what the published numbers turn on.

    uv run python scripts/dtype_sensitivity.py [run]
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

from propella_bekko.inference import Annotator
from propella_bekko.metrics import field_metrics
from propella_bekko.models import PropellaBekko
from propella_bekko.schema import fields_from_config

RUN = sys.argv[1] if len(sys.argv) > 1 else "a25m-xl-3ep"
N = 300


def load(path: str) -> dict[tuple, dict]:
    return {(r["source"], r["id"]): r for r in pq.read_table(path).to_pylist()}


def main() -> None:
    rows = pq.read_table("data/heldout-sample1000.parquet").to_pylist()[:N]
    gold_all = load("runs/reference/deepseek-v4.1-flash-pareto300.parquet")
    gold = [gold_all[(r["source"], r["id"])] for r in rows]
    cfg = json.loads(Path(f"runs/{RUN}/model/config.json").read_text())
    fields = fields_from_config(cfg["fields"])

    scores = {}
    for dtype in (torch.float16, torch.float32):
        model, tok, c = PropellaBekko.load(f"runs/{RUN}/model", device="cuda")
        ann = Annotator(
            model,
            tok,
            c["max_length"],
            c["tail_tokens"],
            device="cuda",
            dtype=dtype,
            max_tokens=262144,
        )
        preds = ann.annotate(rows)
        m = field_metrics(preds, gold, fields)
        scores[str(dtype).replace("torch.", "")] = (
            sum(v["macro_f1"] for v in m.values()) / len(m),
            preds,
        )
        del model, ann
        torch.cuda.empty_cache()

    (f16, p16), (f32, p32) = scores["float16"], scores["float32"]
    same = sum(p == q for p, q in zip(p16, p32, strict=True))
    print(f"{RUN}: macro-F1 fp16 {f16:.4f} | fp32 {f32:.4f} | delta {f16 - f32:+.4f}")
    print(f"documents with identical annotations: {same}/{len(rows)}")


if __name__ == "__main__":
    main()

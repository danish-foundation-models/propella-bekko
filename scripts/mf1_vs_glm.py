"""Macro-F1 of student predictions against the 300 documents that have GLM labels."""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.metrics import field_metrics
from propella_bekko.schema import fields_from_config

fields = fields_from_config(
    json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
)
ref = {(r["source"], r["id"]): r for r in pq.read_table(sys.argv[1]).to_pylist()}

for pred_path in sys.argv[2:]:
    preds = {(r["source"], r["id"]): r for r in pq.read_table(pred_path).to_pylist()}
    keys = [k for k in ref if k in preds]
    preds = [{**ref[k], **{f.name: preds[k].get(f.name) for f in fields}} for k in keys]
    gold = [ref[k] for k in keys]
    fm = field_metrics(preds, gold, fields)
    mf1 = sum(v["macro_f1"] for v in fm.values() if v.get("macro_f1") is not None)
    n = sum(1 for v in fm.values() if v.get("macro_f1") is not None)
    print(f"{Path(pred_path).name:28s} n={len(keys)} macro-F1 {mf1 / n:.4f}")

"""Per-field / per-slice breakdown of a prediction parquet against held-out labels."""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.metrics import field_metrics, summarize
from propella_bekko.schema import fields_from_config

pred_path, gold_path = sys.argv[1:3]
fields = fields_from_config(
    json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
)
gold = {(r["source"], r["id"]): r for r in pq.read_table(gold_path).to_pylist()}
preds = {(r["source"], r["id"]): r for r in pq.read_table(pred_path).to_pylist()}
keys = [k for k in gold if k in preds]
P = [preds[k] for k in keys]
G = [gold[k] for k in keys]
for p in P:
    for f in fields:
        if p.get(f.name) is None:
            p[f.name] = "<invalid>" if f.kind == "single" else []
res = field_metrics(P, G, fields)
print(json.dumps(summarize(res), indent=1))
for k, m in res.items():
    if "accuracy" in m:
        extra = f" within1 {m['within_1']:.3f} qwk {m['qwk']:.3f}" if "qwk" in m else ""
        print(
            f"{k:22s} acc {m['accuracy']:.3f} maj {m['majority_baseline']:.3f} mF1 {m['macro_f1']:.3f}{extra}"
        )
    else:
        print(
            f"{k:22s} jacc {m['jaccard']:.3f} exact {m['exact_match']:.3f} mF1 {m['macro_f1']:.3f}"
        )
for col in ("domain", "lang"):
    groups = {}
    for i in range(len(keys)):
        groups.setdefault(G[i].get(col) or "?", []).append(i)
    rows = sorted(
        (
            (
                name,
                summarize(field_metrics([P[i] for i in idx], [G[i] for i in idx], fields))
                | {"n": len(idx)},
            )
            for name, idx in groups.items()
        ),
        key=lambda kv: kv[1]["mean_macro_f1"],
    )
    print(f"\n-- {col} (worst first)")
    for name, m in rows:
        print(
            f"  {name:14s} n={m['n']:4d} acc {m['mean_single_accuracy']:.3f} jacc {m['mean_multi_jaccard']:.3f} mF1 {m['mean_macro_f1']:.3f}"
        )

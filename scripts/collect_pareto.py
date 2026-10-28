"""Collect Pareto numbers (macro-F1 vs propella-1-4b labels, docs/s) for one sample.

Usage: python scripts/collect_pareto.py <students.json> <baselines-dir>
where <students.json> maps run name -> {"pred": parquet, "bench": json, "params_M": float}.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.metrics import field_metrics
from propella_bekko.schema import fields_from_config


def score(pred_path, gold, fields):
    preds = {(r["source"], r["id"]): r for r in pq.read_table(pred_path).to_pylist()}
    keys = [k for k in gold if k in preds]
    P = [preds[k] for k in keys]
    for p in P:
        for f in fields:
            if p.get(f.name) is None:
                p[f.name] = "<invalid>" if f.kind == "single" else []
    return field_metrics(P, [gold[k] for k in keys], fields)


def main():
    students_path, baselines_dir = sys.argv[1:3]
    fields = fields_from_config(
        json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
    )
    gold = {
        (r["source"], r["id"]): r
        for r in pq.read_table("data/heldout-small-long.parquet").to_pylist()
    }
    models = {}
    for name, s in json.loads(Path(students_path).read_text()).items():
        bench = json.loads(Path(s["bench"]).read_text())
        res = score(s["pred"], gold, fields)
        models[name] = {
            "family": "student",
            "params_M": s["params_M"],
            "docs_per_s": bench["docs_per_s"],
            "macro_f1": float(sum(m["macro_f1"] for m in res.values()) / len(res)),
            "n": len(res),
        }
    for f in sorted(Path(baselines_dir).glob("long-*.json")):
        stats = json.loads(f.read_text())
        pred = f.with_suffix(".parquet")
        if not pred.exists():
            continue
        res = score(pred, gold, fields)
        name = f"propella-1-{f.stem.replace('long-', '')}"
        models[name] = {
            "family": "propella",
            "params_M": {"propella-1-0.6b": 600, "propella-1-1.7b": 1700, "propella-1-4b": 4000}[
                name
            ],
            "docs_per_s": stats["docs_per_s"],
            "macro_f1": float(sum(m["macro_f1"] for m in res.values()) / len(res)),
            "n": len(res),
        }
    Path("runs/pareto.json").write_text(json.dumps({"models": models}, indent=2))
    for name, m in sorted(models.items(), key=lambda kv: kv[1]["docs_per_s"]):
        print(
            f"{name:26s} {m['params_M']:7.0f}M {m['docs_per_s']:9.1f} docs/s  mF1 {m['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()

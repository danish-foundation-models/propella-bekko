"""Collect Pareto numbers (macro-F1 vs propella-1-4b labels, docs/s) on the shared comparison set.

Everything is scored on the first N (=300) documents of `data/heldout-sample1000.parquet`,
the same documents the propella-1 baselines were run on.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.metrics import field_metrics
from propella_bekko.schema import fields_from_config

N = 300


def score(pred_path, gold, fields):
    preds = {(r["source"], r["id"]): r for r in pq.read_table(pred_path).to_pylist()}
    keys = [k for k in gold if k in preds]
    P = [preds[k] for k in keys]
    for p in P:
        for f in fields:
            if p.get(f.name) is None:
                p[f.name] = "<invalid>" if f.kind == "single" else []
    res = field_metrics(P, [gold[k] for k in keys], fields)
    single = [m["accuracy"] for m in res.values() if "accuracy" in m]
    return {
        "macro_f1": float(sum(m["macro_f1"] for m in res.values()) / len(res)),
        "accuracy": float(sum(single) / max(1, len(single))),
        "n": len(keys),
    }


def main():
    students_path, baselines_dir = sys.argv[1:3]
    fields = fields_from_config(
        json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
    )
    rows = pq.read_table("data/heldout-sample1000.parquet").to_pylist()[:N]
    gold = {(r["source"], r["id"]): r for r in rows}
    models = {}
    for name, s in json.loads(Path(students_path).read_text()).items():
        bench = json.loads(Path(s["bench"]).read_text())
        r = score(s["pred"], gold, fields)
        models[name] = {
            "family": "student",
            "params_M": s["params_M"],
            "docs_per_s": bench["docs_per_s"],
            **r,
        }
    params = {"propella-1-0.6b": 600, "propella-1-1.7b": 1700, "propella-1-4b": 4000}
    for f in sorted(Path(baselines_dir).glob(f"long{N}-*.json")):
        stats = json.loads(f.read_text())
        pred = f.with_suffix(".parquet")
        if not pred.exists():
            continue
        name = f"propella-1-{f.stem.split('-', 1)[1]}"
        models[name] = {
            "family": "propella",
            "params_M": params.get(name, 0),
            "docs_per_s": stats["docs_per_s"],
            **score(pred, gold, fields),
        }
    Path("runs/pareto.json").write_text(json.dumps({"models": models}, indent=2))
    for name, m in sorted(models.items(), key=lambda kv: kv[1]["docs_per_s"]):
        print(
            f"{name:26s} {m['params_M']:6.0f}M {m['docs_per_s']:9.1f} docs/s "
            f"acc {m['accuracy']:.3f}  mF1 {m['macro_f1']:.3f}  n={m['n']}"
        )


if __name__ == "__main__":
    main()

"""Agreement matrix on the 300-doc comparison set: every annotation set vs every reference."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.metrics import field_metrics, summarize
from propella_bekko.schema import fields_from_config

N = 300
fields = fields_from_config(
    json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
)
rows = pq.read_table("data/heldout-sample1000.parquet").to_pylist()[:N]
gold = {(r["source"], r["id"]): r for r in rows}


def load(path):
    if path is None:
        src = gold
    else:
        src = {(r["source"], r["id"]): r for r in pq.read_table(path).to_pylist()}
    out = {}
    for k in gold:
        r = dict(src.get(k, {}))
        for f in fields:
            if r.get(f.name) is None:
                r[f.name] = "<invalid>" if f.kind == "single" else []
        out[k] = r
    return out


sets = {
    "propella-4b (stored)": load(None),
}
for name, path in [
    ("propella-4b fp16", "runs/baselines/long300-4b.parquet"),
    ("propella-1.7b", "runs/baselines/long300-1.7b.parquet"),
    ("propella-0.6b", "runs/baselines/long300-0.6b.parquet"),
]:
    if Path(path).exists():
        sets[name] = load(path)
for ref, path in [("deepseek-v4.1-flash", "runs/reference/deepseek-v4.1-flash-pareto300.parquet"),
                  ("glm-5.3-flash", "runs/reference/glm-5.3-flash-pareto300.parquet")]:  # fmt: skip
    if Path(path).exists():
        sets[ref] = load(path)
for p in sorted(Path("runs/pareto").glob("*.parquet")):
    if "long300" not in p.name:
        sets[f"student {p.stem}"] = load(str(p))

keys = list(gold)
for ref in ["propella-4b (stored)", "deepseek-v4.1-flash", "glm-5.3-flash"]:
    if ref not in sets:
        continue
    print(f"\nreference = {ref}  (n={len(keys)})")
    for name, s in sets.items():
        if name == ref:
            continue
        sm = summarize(field_metrics([s[k] for k in keys], [sets[ref][k] for k in keys], fields))
        print(
            f"  {name:26s} acc {sm['mean_single_accuracy']:.3f}  "
            f"jacc {sm['mean_multi_jaccard']:.3f}  mF1 {sm['mean_macro_f1']:.3f}"
        )

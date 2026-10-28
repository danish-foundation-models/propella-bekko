"""Agreement matrix on the short200 subset: every annotation set vs every reference.

References: stored propella-1-4b labels (the training teacher) and DeepSeek reference labels.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.metrics import field_metrics, summarize
from propella_bekko.schema import fields_from_config

fields = fields_from_config(
    json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
)
short = {
    (r["source"], r["id"]): r for r in pq.read_table("runs/baselines/short200.parquet").to_pylist()
}


def load(path):
    rows = {(r["source"], r["id"]): r for r in pq.read_table(path).to_pylist()} if path else short
    out = {}
    for k in short:
        r = dict(rows.get(k, {}))
        for f in fields:
            if r.get(f.name) is None:
                r[f.name] = "<invalid>" if f.kind == "single" else []
        out[k] = r
    return out


sets = {
    "propella-4b (stored)": load(None),
    "propella-4b fp16": load("runs/baselines/propella-1-4b.parquet"),
    "propella-4b fp16 rerun": load("runs/baselines/short200-4b-fp16-rerun.parquet"),
    "propella-4b fp32": load("runs/baselines/short200-4b-fp32.parquet"),
    "propella-1.7b fp16": load("runs/baselines/propella-1-1.7b.parquet"),
    "propella-0.6b fp16": load("runs/baselines/propella-1-0.6b.parquet"),
    "deepseek-v4.1-flash": load("runs/reference/deepseek-v4.1-flash-short200.parquet"),
}
for m in ["static-v0", "static-v1-pool", "a8m-v0", "a8m-v1-pool", "a25m-v0"]:
    if Path(f"runs/students/short200-{m}.parquet").exists():
        sets[f"student {m}"] = load(f"runs/students/short200-{m}.parquet")
keys = list(short)
for ref in ["propella-4b (stored)", "deepseek-v4.1-flash"]:
    print(f"\nreference = {ref}")
    for name, s in sets.items():
        if name == ref:
            continue
        sm = summarize(field_metrics([s[k] for k in keys], [sets[ref][k] for k in keys], fields))
        print(
            f"  {name:24s} acc {sm['mean_single_accuracy']:.3f}  jacc {sm['mean_multi_jaccard']:.3f}  mF1 {sm['mean_macro_f1']:.3f}"
        )

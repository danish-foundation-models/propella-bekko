"""Per-property comparison of students against propella-1 baselines on the 1k test set."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.paper_metrics import paper_score
from propella_bekko.schema import fields_from_config

fields = fields_from_config(
    json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
)
rows = pq.read_table("data/heldout-sample1000.parquet").to_pylist()[:1000]
gold = {(r["source"], r["id"]): r for r in rows}


def load(path):
    return {(r["source"], r["id"]): r for r in pq.read_table(path).to_pylist()}


sets = {
    "propella-1.7b": load("runs/baselines/b20k-1.7b.parquet"),
    "propella-0.6b": load("runs/baselines/b20k-0.6b.parquet"),
    "a25m-v1-pool": load("runs/pareto/a25m-v1-pool.parquet"),
    "a8m-v1-pool": load("runs/pareto/a8m-v1-pool.parquet"),
    "static-long": load("runs/pareto/static-long-ctx8192.parquet"),
}
keys = list(gold)


def clean(rows_, ks):
    out = []
    for k in ks:
        r = dict(rows_.get(k, {}))
        for f in fields:
            if r.get(f.name) is None:
                r[f.name] = "<invalid>" if f.kind == "single" else []
        out.append(r)
    return out


g = clean(gold, keys)
scores = {name: paper_score(clean(s, keys), g, fields) for name, s in sets.items()}
props = list(scores["propella-1.7b"]["per_property"])
print(f"{'property':22s}" + "".join(f"{n:>16s}" for n in scores))
for p in props:
    print(f"{p:22s}" + "".join(f"{scores[n]['per_property'][p]:16.3f}" for n in scores))
print(f"{'OVERALL':22s}" + "".join(f"{scores[n]['overall']:16.3f}" for n in scores))

"""Ensemble several student prediction files by per-field voting.

* single-select: majority label (ties -> first model in the list)
* multi-select: labels chosen by more than half of the models (union if none pass)

Throughput of an ensemble is the harmonic-style cost: 1 / sum(1/throughput_i) documents/s,
i.e. running all members sequentially.
"""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from propella_bekko.paper_metrics import paper_score
from propella_bekko.schema import PROPELLA_FIELDS, fields_from_config


def vote(tables, keys):
    out = []
    for k in keys:
        rec = {"source": k[0], "id": k[1]}
        for f in PROPELLA_FIELDS:
            vals = [
                t[k].get(f.name) for t in tables if k in t and t[k].get(f.name) not in (None, [])
            ]
            if not vals:
                continue
            if f.kind == "single":
                counts = {v: vals.count(v) for v in set(vals)}
                best = max(counts.values())
                rec[f.name] = next(v for v in vals if counts[v] == best)
            else:
                counts = {}
                for v in vals:
                    for x in set(v):
                        counts[x] = counts.get(x, 0) + 1
                picks = [x for x, c in counts.items() if c * 2 > len(vals)]
                rec[f.name] = picks or list(max(vals, key=len))
        out.append(rec)
    return out


def main():
    paths = sys.argv[1:]
    fields = fields_from_config(
        json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
    )
    tables = [{(r["source"], r["id"]): r for r in pq.read_table(p).to_pylist()} for p in paths]
    keys = [k for k in tables[0] if all(k in t for t in tables)]
    rows = vote(tables, keys)
    out = "runs/pareto/ensemble.parquet"
    pq.write_table(pa.Table.from_pylist(rows), out)

    gold = {
        (r["source"], r["id"]): r
        for r in pq.read_table("data/heldout-sample1000.parquet").to_pylist()
    }
    ref = [gold[k] for k in keys]
    for r in ref:
        for f in fields:
            if r.get(f.name) is None:
                r[f.name] = "<invalid>" if f.kind == "single" else []
    res = paper_score(rows, ref, fields)
    dps = 0.0
    for p in paths:
        bench = Path(p).with_suffix(".bench.json")
        if bench.exists():
            dps += 1 / json.loads(bench.read_text())["docs_per_s"]
    dps = 1 / dps if dps else float("nan")
    Path("runs/pareto/ensemble.bench.json").write_text(json.dumps({"docs_per_s": dps}))
    print(
        f"ensemble of {len(paths)} models -> {out}: n={len(rows)} overall {res['overall']:.3f} "
        f"(qwk {res['mean_qwk']:.3f} iou {res['mean_iou']:.3f}), {dps:.1f} docs/s"
    )


if __name__ == "__main__":
    main()

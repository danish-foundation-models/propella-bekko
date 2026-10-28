"""Combine several LLM annotation sets into one consensus reference.

Per property:
* ordinal / binary single-select: majority vote over annotators; ties go to the
  first annotator (highest priority), and rank distance breaks nothing else.
* multi-select: union of values predicted by a majority of annotators.

Output has the same columns as a single annotation parquet, so it can be used
directly as the reference in the agreement/Pareto scripts.
"""

import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from propella_bekko.schema import PROPELLA_FIELDS


def consensus(tables: list[dict]):
    keys = set.intersection(*(set(t) for t in tables))
    rows = []
    for k in keys:
        rec = {"source": k[0], "id": k[1]}
        for f in PROPELLA_FIELDS:
            vals = [t[k].get(f.name) for t in tables]
            vals = [v for v in vals if v is not None and v != []]
            if not vals:
                continue
            if f.kind == "single":
                c = Counter(vals)
                best = max(c.values())
                rec[f.name] = next(v for v in vals if c[v] == best)  # first annotator wins ties
            else:
                c = Counter(x for v in vals for x in set(v))
                rec[f.name] = [x for x, n in c.items() if n * 2 > len(vals)]
        rows.append(rec)
    return rows


def main(paths: list[str], out: str):
    tables = [{(r["source"], r["id"]): r for r in pq.read_table(p).to_pylist()} for p in paths]
    rows = consensus(tables)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"consensus of {len(tables)} annotators -> {out} ({len(rows)} docs)")


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])

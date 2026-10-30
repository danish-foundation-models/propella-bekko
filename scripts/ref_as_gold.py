"""Write a copy of a held-out file whose label columns come from a reference annotator.

Lets `evaluate --data <out>` score agreement with DeepSeek / GLM instead of the stored
propella labels:

    uv run python scripts/ref_as_gold.py data/heldout-sample1000.parquet \
        runs/reference/deepseek-v4.1-flash-1k.parquet data/heldout-1k-deepseek.parquet
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from propella_bekko.schema import PROPELLA_FIELDS

data, reference, out = sys.argv[1:4]
rows = {(r["source"], r["id"]): r for r in pq.read_table(data).to_pylist()}
ref = {(r["source"], r["id"]): r for r in pq.read_table(reference).to_pylist()}
names = [f.name for f in PROPELLA_FIELDS]

merged = []
for k, r in rows.items():
    if k not in ref:
        continue
    for n in names:
        r[n] = ref[k].get(n)
    merged.append(r)
pq.write_table(pa.Table.from_pylist(merged), out)
print(f"wrote {out} {len(merged)} rows with {Path(reference).name} labels")

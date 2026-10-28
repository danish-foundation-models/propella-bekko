"""Make a 200-doc subset of the 1,000-doc baseline sample: only docs short enough
(<=15k chars) that the stored text is identical to what propella-1-4b originally saw."""

import random

import pyarrow as pa
import pyarrow.parquet as pq

rows = pq.read_table("data/heldout-small.parquet").to_pylist()
sample = random.Random(0).sample(rows, 1000)  # same sample as scripts/propella_baseline.py
short = [r for r in sample if r["n_chars"] <= 15000]
sub = random.Random(1).sample(short, 200)
pq.write_table(pa.Table.from_pylist(sub), "runs/baselines/short200.parquet")
print(len(short), "short docs in sample ->", len(sub))

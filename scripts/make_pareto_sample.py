"""The 1,000-doc sample used for every Pareto comparison (seeded, matches the baselines)."""

import random

import pyarrow as pa
import pyarrow.parquet as pq

rows = pq.read_table("data/heldout-small-long.parquet").to_pylist()
sample = random.Random(0).sample(rows, 1000)
pq.write_table(pa.Table.from_pylist(sample), "data/heldout-sample1000.parquet")
print(len(sample), "-> data/heldout-sample1000.parquet")

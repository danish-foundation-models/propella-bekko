"""Annotate a parquet (with source/id/text) with a trained student, keeping source+id."""

import sys

import pyarrow as pa
import pyarrow.parquet as pq

from propella_bekko.inference import Annotator
from propella_bekko.models import PropellaBekko

model_dir, inp, out = sys.argv[1:4]
model, tok, cfg = PropellaBekko.load(model_dir)
ann = Annotator(model, tok, cfg["max_length"], cfg["tail_tokens"], device="cuda", max_tokens=262144)
rows = pq.read_table(inp).to_pylist()
preds = ann.annotate(rows)
pq.write_table(
    pa.Table.from_pylist([{"source": r["source"], "id": r["id"], **p} for r, p in zip(rows, preds, strict=True)]),
    out,
)
print("wrote", out, len(rows))

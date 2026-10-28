import pyarrow.parquet as pq

from propella_bekko.schema import PROPELLA_FIELDS

t = pq.read_table("runs/baselines/short200.parquet")
for f in PROPELLA_FIELDS:
    col = t.column(f.name).to_pylist()
    bad = sum(v is None or (f.kind == "single" and v not in f.labels) for v in col)
    if bad:
        print(f.name, bad, sorted({str(v) for v in col if v is None or (f.kind == "single" and v not in f.labels)})[:5])
srcs = t.column("source").to_pylist()
print({s: srcs.count(s) for s in set(srcs)})

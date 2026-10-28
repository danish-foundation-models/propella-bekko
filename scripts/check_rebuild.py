"""Check that a --labels-from rebuild samples exactly the same docs as the original."""

import sys

import pyarrow.parquet as pq

orig, new, source, subset = sys.argv[1:5]
o = pq.read_table(orig).to_pylist()
o = {r["id"]: r for r in o if r["source"] == source and r["subset"] == subset}
n = {r["id"]: r for r in pq.read_table(new).to_pylist()}
print("orig", len(o), "new", len(n), "same ids", set(o) == set(n))
longer = sum(len(n[i]["text"]) > len(o[i]["text"]) for i in n if i in o)
same_hash = all(n[i]["text_sha1"] == o[i]["text_sha1"] for i in n if i in o)
print("texts longer:", longer, "| same full-text hash:", same_hash)

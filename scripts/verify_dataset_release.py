"""Check the assembled release and print the tables the dataset card quotes."""

from __future__ import annotations

import statistics
from collections import Counter

import pyarrow.dataset as ds
import pyarrow.parquet as pq

REPO = "/home/kenneth/githubs/propella-bekko-dataset"
PROPS = [
    "content_integrity", "content_ratio", "content_length", "content_type", "business_sector",
    "technical_content", "content_quality", "information_density", "educational_value",
    "reasoning_indicators", "audience_level", "commercial_bias", "time_sensitivity",
    "content_safety", "pii_presence", "regional_relevance", "country_relevance",
]  # fmt: skip

d = ds.dataset(f"{REPO}/data", format="parquet")
common = ["source", "subset", "lang", "domain", "id", "text", "text_sha1", "n_chars", *PROPS]

splits = {
    "train": d.to_table(columns=common, filter=ds.field("reference_annotator").is_null()),
    "test-deepseek": pq.read_table(f"{REPO}/data/test-deepseek.parquet", columns=common),
    "test-glm-5.3-flash": pq.read_table(f"{REPO}/data/test-glm-5.3-flash.parquet", columns=common),
}

for name, t in splits.items():
    keys = list(zip(t.column("source").to_pylist(), t.column("id").to_pylist(), strict=False))
    print(f"\n== {name}: {t.num_rows} rows, {len(set(keys))} unique (source,id)")
    print("  sources:", dict(Counter(t.column("source").to_pylist())))
    print("  languages:", len(set(t.column("lang").to_pylist())))
    print("  domains:", dict(Counter(t.column("domain").to_pylist()).most_common()))
    texts = t.column("text").to_pylist()
    print(
        f"  text: {sum(1 for x in texts if not x)} empty, max length {max(len(x) for x in texts)}"
    )
    vals = t.column("n_chars").to_pylist()
    print(f"  n_chars median {statistics.median(vals):.0f} mean {sum(vals) / len(vals):.0f}")

tr_keys = set(
    zip(
        splits["train"].column("source").to_pylist(),
        splits["train"].column("id").to_pylist(),
        strict=False,
    )
)
for name in ["test-deepseek", "test-glm-5.3-flash"]:
    t = splits[name]
    k = set(zip(t.column("source").to_pylist(), t.column("id").to_pylist(), strict=False))
    print(f"leak {name} ∩ train: {len(k & tr_keys)}")

print("\nlabel coverage in train:")
print("  fully labelled:", splits["train"].column("content_quality").null_count, "unnlabelled rows")
print(
    "  labelled rows:",
    splits["train"].num_rows - splits["train"].column("content_quality").null_count,
)

# reference columns: check the joined labels match the run that produced them
ref_cols = [f"ref_{p}" for p in PROPS]
for name, path in [
    (
        "test-deepseek",
        "/home/kenneth/githubs/propella-bekko/runs/reference/deepseek-v4.1-flash-1k.parquet",
    ),
    (
        "test-glm-5.3-flash",
        "/home/kenneth/githubs/propella-bekko/runs/reference/glm-5.3-flash-pareto300.parquet",
    ),
]:
    t = pq.read_table(
        f"{REPO}/data/{name}.parquet", columns=["source", "id", "reference_annotator", *ref_cols]
    )
    src = {
        (r["source"], r["id"]): r
        for r in pq.read_table(path, columns=["source", "id", "_ok", *PROPS]).to_pylist()
        if r["_ok"]
    }
    bad = 0
    for r in t.to_pylist():
        s = src[(r["source"], r["id"])]
        if any(r[f"ref_{p}"] != s[p] for p in PROPS):
            bad += 1
    print(
        f"{name}: {t.num_rows} rows, annotator={set(t.column('reference_annotator').to_pylist())}, label mismatches vs run: {bad}"
    )
    print("  rows with no reference label:", max(t.column(f"ref_{p}").null_count for p in PROPS))

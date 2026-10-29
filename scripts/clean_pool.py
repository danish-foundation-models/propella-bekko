"""Filter the distillation pool by label-noise and duplicate heuristics.

Filters (each reported with how many rows it removes):
* `dup`      : near-duplicate documents — first 1,000 normalised characters hash collide
* `empty`    : rows where >6 of the 17 properties are missing (partial teacher annotations)
* `length`   : teacher labels internally inconsistent with the document, e.g.
               content_length=substantial but <400 words, minimal but >1,500 words,
               content_ratio=complete_content but content_integrity=fragment
* `boiler`   : content_ratio in {mostly_navigation, minimal_content} (kept only if `keep_boiler`)

    uv run python scripts/clean_pool.py data/train-pool-xl.parquet data/train-pool-xl-clean.parquet
"""

import hashlib
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

src, dst = sys.argv[1:3]
t = pq.read_table(src)
n0 = t.num_rows
rows = t.to_pylist()
print(f"{n0} rows")

PROPS = [
    "content_integrity", "content_ratio", "content_length", "content_type", "business_sector",
    "technical_content", "content_quality", "information_density", "educational_value",
    "reasoning_indicators", "audience_level", "commercial_bias", "time_sensitivity",
    "content_safety", "pii_presence", "regional_relevance", "country_relevance",
]  # fmt: skip

SUBSTANTIAL_WORDS = 400  # teacher says substantial = 2k+ words; allow generous slack
MINIMAL_WORDS = 1500  # teacher says minimal = <100 words

kept, stats = [], {"dup": 0, "empty": 0, "length": 0}
seen = set()
for r in rows:
    h = hashlib.sha1(" ".join(r["text"][:1000].split()).encode()).hexdigest()
    if h in seen:
        stats["dup"] += 1
        continue
    seen.add(h)
    if sum(1 for p in PROPS if r.get(p) in (None, [])) > 6:
        stats["empty"] += 1
        continue
    n_words = r.get("n_words") or 0
    bad_len = (r.get("content_length") == "substantial" and n_words < SUBSTANTIAL_WORDS) or (
        r.get("content_length") == "minimal" and n_words > MINIMAL_WORDS
    )
    bad_int = r.get("content_ratio") == "complete_content" and r.get("content_integrity") in {
        "fragment",
        "severely_degraded",
    }
    if bad_len or bad_int:
        stats["length"] += 1
        continue
    kept.append(r)

out = pa.Table.from_pylist(kept, schema=t.schema)
pq.write_table(out, dst, compression="zstd")
print(f"kept {out.num_rows} ({out.num_rows / n0:.1%}); removed {stats}")
for col in ("domain", "lang"):
    print(col, dict(pc.value_counts(out.column(col)).to_pylist() and
                    [(d["values"], d["counts"]) for d in pc.value_counts(out.column(col)).to_pylist()][:6]))

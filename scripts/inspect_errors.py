"""Print concrete disagreements for a property so they can be inspected by hand.

    uv run python scripts/inspect_errors.py runs/pareto/a25m-xl.parquet \
        data/heldout-sample1000.parquet runs/reference/deepseek-v4.1-flash-1k.parquet \
        --property content_integrity --n 5
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.schema import PROPELLA_FIELDS, fields_from_config

ap = argparse.ArgumentParser()
ap.add_argument("student")
ap.add_argument("gold")
ap.add_argument("reference")
ap.add_argument("--property", required=True)
ap.add_argument("--n", type=int, default=5)
ap.add_argument(
    "--mode",
    default="student-wrong",
    choices=["student-wrong", "teacher-wrong", "own-error"],
    help="own-error = teacher agrees with the reference but the student does not",
)
ap.add_argument("--chars", type=int, default=400)
a = ap.parse_args()

fields = fields_from_config(
    json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
)
f = next(x for x in PROPELLA_FIELDS if x.name == a.property)
S = {(r["source"], r["id"]): r for r in pq.read_table(a.student).to_pylist()}
G = {(r["source"], r["id"]): r for r in pq.read_table(a.gold).to_pylist()}
R = {(r["source"], r["id"]): r for r in pq.read_table(a.reference).to_pylist()}

shown = 0
for k, ref in R.items():
    if k not in S or k not in G:
        continue
    g, r = G[k], ref
    if g.get("n_chars") is None:
        continue

    def same(x, y):
        if f.kind == "multi":
            return set(x or []) == set(y or [])
        return x == y

    s, t = S[k].get(f.name), g.get(f.name)
    if same(s, r.get(f.name)):
        continue
    if a.mode == "teacher-wrong" and same(t, r.get(f.name)):
        continue
    if a.mode == "own-error" and not same(t, r.get(f.name)):
        continue
    text = g.get("text") or ""
    print("=" * 100)
    print(
        f"{k[0]} | {k[1]} | lang={g.get('lang')} domain={g.get('domain')} "
        f"chars={g.get('n_chars')} words={g.get('n_words')} truncated_in_storage={'[...]' in text}"
    )
    print(f"teacher(propella)={t}   student={s}   reference={r.get(f.name)}")
    print("-" * 40, "head")
    print(text[: a.chars].replace("\n", " "))
    print("-" * 40, "tail")
    print(text[-a.chars :].replace("\n", " "))
    shown += 1
    if shown >= a.n:
        break
print(f"\n{shown} examples")

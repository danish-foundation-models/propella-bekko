"""Error analysis for one student: confusions, label bias, slices.

    uv run python scripts/error_analysis.py runs/pareto/a25m-xl.parquet \
        data/heldout-sample1000.parquet experiments/007-error-analysis.md
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.paper_metrics import ORDINAL, paper_score
from propella_bekko.schema import OTHER, PROPELLA_FIELDS, fields_from_config

pred_path, gold_path, out_path = sys.argv[1:4]
fields = fields_from_config(
    json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
)
byname = {f.name: f for f in fields}
rows = pq.read_table(gold_path).to_pylist()
pred = {(r["source"], r["id"]): r for r in pq.read_table(pred_path).to_pylist()}
gold = {(r["source"], r["id"]): r for r in rows}
keys = [k for k in gold if k in pred]
P = [pred[k] for k in keys]
G = [gold[k] for k in keys]
for r in P:
    for f in fields:
        if r.get(f.name) is None:
            r[f.name] = "<invalid>" if f.kind == "single" else []

L: list[str] = ["# Error analysis — a25m-xl on the 1,000-doc test set", ""]
L.append(
    "Student: `runs/a25m-xl` (25M active, 845k training docs). Reference: stored propella-1-4b labels."
)
p = paper_score(P, G, fields)
L += [
    "",
    f"Overall **{p['overall']:.3f}** (QWK {p['mean_qwk']:.3f}, IoU {p['mean_iou']:.3f}, PII-F1 {p['pii_f1']:.3f})",
    "",
]

# 1. ordinal confusion: which levels get confused, in which direction
L += ["## Ordinal properties: dominant confusions", ""]
for name in ORDINAL:
    f = byname[name]
    cm = Counter(
        (g[name], pr[name]) for g, pr in zip(G, P, strict=False) if g.get(name) in f.labels
    )
    off = [(k, v) for k, v in cm.items() if k[0] != k[1]]
    off.sort(key=lambda kv: -kv[1])
    tot = sum(cm.values())
    if not off or tot == 0:
        continue
    top = ", ".join(f"{a}→{b} ({v})" for (a, b), v in off[:3])
    over = sum(
        v
        for (a, b), v in off
        if f.labels.index(b) > f.labels.index(a)
        if b in f.labels and a in f.labels
    )
    under = sum(
        v
        for (a, b), v in off
        if f.labels.index(b) < f.labels.index(a)
        if b in f.labels and a in f.labels
    )
    direction = (
        "over-rates (predicts a better level than the teacher)"
        if over > under * 1.3
        else ("under-rates" if under > over * 1.3 else "balanced")
    )
    L.append(f"- **{name}** ({sum(v for _, v in off)} of {tot} off): {top}. Model {direction}.")

# 2. multi-select label bias
L += ["", "## Multi-select: per-label bias (predicted vs teacher frequency)", ""]
for name in [f.name for f in PROPELLA_FIELDS if f.kind == "multi"]:
    f = byname[name]
    gp, pp = Counter(), Counter()
    for g, pr in zip(G, P, strict=False):
        for x in g.get(name) or []:
            gp[x if x in set(f.labels) else OTHER] += 1
        for x in pr.get(name) or []:
            pp[x if x in set(f.labels) else OTHER] += 1
    n = len(G)
    diffs = sorted(
        ((lab, pp[lab] / n - gp[lab] / n) for lab in set(gp) | set(pp)),
        key=lambda kv: kv[1],
    )
    worst_under = ", ".join(f"{lab} ({d * 100:+.1f}pp)" for lab, d in diffs[:3])
    worst_over = ", ".join(f"{lab} ({d * 100:+.1f}pp)" for lab, d in diffs[-3:][::-1])
    L.append(f"- **{name}**: under-predicted {worst_under}; over-predicted {worst_over}")

# 3. slices
L += ["", "## Slices (mean QWK over ordinal properties / mean IoU over multi-select)", ""]


def slice_scores(idx):
    sub_p = [P[i] for i in idx]
    sub_g = [G[i] for i in idx]
    return paper_score(sub_p, sub_g, fields)


def bucket(n):
    return "≤1k chars" if n < 1000 else "1k–5k" if n < 5000 else "5k–20k" if n < 20000 else ">20k"


for label, key in [
    ("domain", lambda i, r: r.get("domain")),
    ("language", lambda i, r: r.get("lang")),
    ("length", lambda i, r: bucket(r.get("n_chars") or 0)),
]:
    groups = defaultdict(list)
    for i in range(len(P)):
        groups[key(i, G[i])].append(i)
    rowsout = [(k, len(v), slice_scores(v)) for k, v in groups.items() if len(v) >= 25]
    rowsout.sort(key=lambda kv: kv[2]["overall"])
    L += ["", f"**{label}** (worst first, n≥25)"]
    for k, n, s in rowsout:
        L.append(
            f"- {k}: n={n}, overall {s['overall']:.3f} (QWK {s['mean_qwk']:.3f}, IoU {s['mean_iou']:.3f})"
        )

Path(out_path).write_text("\n".join(L) + "\n")
print("\n".join(L))

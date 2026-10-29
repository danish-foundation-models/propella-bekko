"""Split student disagreement into 'inherited from the teacher' vs 'own' errors.

For each document and property, compare the student and the teacher (propella-1-4b labels)
against an *independent* reference (DeepSeek):
* student wrong, teacher also wrong (student agrees with teacher) -> inherited
* student wrong, teacher right                                  -> own error
* student right, teacher wrong                                  -> student fixed it
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.schema import OTHER, PROPELLA_FIELDS, fields_from_config

student_path, teacher_path, reference_path = sys.argv[1:4]
fields = fields_from_config(
    json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
)
byname = {f.name: f for f in fields}


def load(p):
    return {(r["source"], r["id"]): r for r in pq.read_table(p).to_pylist()}


S, T, R = load(student_path), load(teacher_path), load(reference_path)
keys = [k for k in R if k in S and k in T]
print(f"{len(keys)} documents with all three annotation sets\n")

rows = []
for f in PROPELLA_FIELDS:
    inherited = own = fixed = 0
    for k in keys:
        s, t, r = S[k].get(f.name), T[k].get(f.name), R[k].get(f.name)
        if r is None or t is None:
            continue
        if f.kind == "single":
            if r not in f.labels:
                continue
            s_ok, t_ok = s == r, t == r
        else:
            labels = set(f.labels)

            def norm(v, labels=labels):
                return {x if x in labels else OTHER for x in (v or [])}

            sr, tr, rr = norm(s), norm(t), norm(r)
            if not rr:
                continue
            s_ok = sr == rr
            t_ok = tr == rr
        if s_ok:
            fixed += int(not t_ok)
        elif t_ok:
            own += 1
        else:
            inherited += 1
    total = inherited + own + fixed
    if not total:
        continue
    rows.append((f.name, inherited, own, fixed))

print(f"{'property':22s} {'inherited':>10s} {'own':>6s} {'fixed':>6s}  inherited share of errors")
tot_i = tot_o = tot_f = 0
for name, i, o, fx in rows:
    share = i / (i + o) if (i + o) else float("nan")
    print(f"{name:22s} {i:10d} {o:6d} {fx:6d}  {share:.0%}")
    tot_i += i
    tot_o += o
    tot_f += fx
print(
    f"\nTOTAL inherited {tot_i}, own {tot_o}, fixed {tot_f} -> {tot_i / (tot_i + tot_o):.0%} of errors are inherited from the teacher"
)

"""Does soft (expected-value) ordinal decoding beat argmax on QWK?

argmax picks the most likely level; the paper's metric is quadratic-weighted kappa, which
rewards being close. Decoding to the expected level (round of the probability-weighted rank)
is a cheap, training-free change (System-One-style 'Score' decisions return a distribution).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import cohen_kappa_score

from propella_bekko.inference import Annotator
from propella_bekko.models import PropellaBekko
from propella_bekko.paper_metrics import ORDINAL

model_dir, data = sys.argv[1:3]
model, tok, cfg = PropellaBekko.load(model_dir, device="cuda")
rows = pq.read_table(data).to_pylist()
ann = Annotator(model, tok, cfg["max_length"], cfg["tail_tokens"], device="cuda", max_tokens=262144)
probs = ann.probabilities(rows)
by = {f.name: f for f in model.fields}

print(f"{'property':22s} {'argmax':>8s} {'expected':>9s} {'+?':>5s}")
tot_a, tot_e = [], []
for name in ORDINAL:
    f = by[name]
    ranks = np.arange(f.num_labels)
    pairs = [
        (p, r.get(name)) for p, r in zip(probs[name], rows, strict=False) if r.get(name) in f.labels
    ]
    g = [f.labels.index(b) for _, b in pairs]
    a = [int(np.asarray(p).argmax()) for p, _ in pairs]
    e = [
        int(
            np.clip(
                round(float(np.dot(np.asarray(p) / np.asarray(p).sum(), ranks))),
                0,
                f.num_labels - 1,
            )
        )
        for p, _ in pairs
    ]
    ka = cohen_kappa_score(g, a, weights="quadratic")
    ke = cohen_kappa_score(g, e, weights="quadratic")
    tot_a.append(ka)
    tot_e.append(ke)
    print(f"{name:22s} {ka:8.4f} {ke:9.4f} {ke - ka:+5.3f}")
print(
    f"{'mean QWK':22s} {np.mean(tot_a):8.4f} {np.mean(tot_e):9.4f} {np.mean(tot_e) - np.mean(tot_a):+5.3f}"
)
Path("runs/soft_ordinal.json").write_text(
    json.dumps({"argmax": float(np.mean(tot_a)), "expected": float(np.mean(tot_e))}, indent=2)
)

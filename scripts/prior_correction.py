"""Prior (logit) adjustment against the training-label prior.

The error analysis shows the students over-predict the dominant classes (european +12pp,
non_technical +10pp, denmark +4pp). Standard fix: subtract `alpha * log p_train(label)`
from the logits at decode time (logit adjustment / prior correction).

`alpha` is tuned on the validation split of the training pool (never on the test set);
the tuned value is then reported on the 1k test set.

    uv run python scripts/prior_correction.py runs/a25m-xl/model data/train-pool-xl.parquet \
        data/heldout-sample1000.parquet
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from propella_bekko.inference import Annotator
from propella_bekko.models import PropellaBekko
from propella_bekko.paper_metrics import paper_score
from propella_bekko.schema import encode_labels

model_dir, train_path, test_path = sys.argv[1:4]
model, tok, cfg = PropellaBekko.load(model_dir, device="cuda")
fields = model.fields


def _is_val(row, frac=0.02):
    h = hashlib.sha1(f"{row['source']}/{row['id']}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 < frac


train_rows = pq.read_table(train_path, columns=[f.name for f in fields]).to_pylist()
priors = {}
for f in fields:
    counts = np.zeros(f.num_labels, dtype=np.float64)
    for r in train_rows:
        y = encode_labels(r, [f])[f.name]
        idx = np.atleast_1d(np.asarray(y))
        if (idx < 0).all():
            continue
        counts[idx[idx >= 0]] += 1
    p = (counts + 1) / (counts.sum() + f.num_labels)
    priors[f.name] = np.log(p)  # log prior, negative values

val_rows = [r for r in pq.read_table(train_path).to_pylist() if _is_val(r)][:2000]
test_rows = pq.read_table(test_path).to_pylist()
print(f"prior-correction on {len(val_rows)} val docs, {len(test_rows)} test docs")

ann = Annotator(model, tok, cfg["max_length"], cfg["tail_tokens"], device="cuda", max_tokens=262144)


PROB_CACHE: dict[str, dict] = {}


def adjusted(rows, alpha, tag):
    if tag not in PROB_CACHE:
        PROB_CACHE[tag] = ann.probabilities(rows)
    probs = PROB_CACHE[tag]
    out = []
    for i in range(len(rows)):
        rec = {}
        for f in fields:
            p = np.asarray(probs[f.name][i], dtype=np.float64)
            if f.kind == "single":
                logits = np.log(np.clip(p, 1e-9, 1)) - alpha * priors[f.name]
                rec[f.name] = f.labels[int(logits.argmax())]
            else:
                prob = np.clip(p * np.exp(-alpha * priors[f.name]), 1e-9, 1)
                prob = prob / np.exp(-alpha * priors[f.name]).max()
                hot = prob >= 0.5
                hot[int(prob.argmax())] = True
                rec[f.name] = [lab for lab, on in zip(f.labels, hot, strict=False) if on]
        out.append(rec)
    return out


results = {}
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    val = adjusted(val_rows, alpha, "val")
    results[alpha] = {"val": paper_score(val, val_rows, fields)["overall"]}
    print(f"alpha {alpha}: val overall {results[alpha]['val']:.4f}")

best = max(results, key=lambda a: results[a]["val"])
print(f"\nbest alpha on val: {best}")
test = adjusted(test_rows, best, "test")
ts = paper_score(test, test_rows, fields)
results[best]["test"] = ts
print(
    f"test overall {ts['overall']:.4f} (QWK {ts['mean_qwk']:.4f}, IoU {ts['mean_iou']:.4f}, PII-F1 {ts['pii_f1']:.4f})"
)
Path("runs/prior_correction.json").write_text(
    json.dumps({str(k): v for k, v in results.items()}, indent=2)
)

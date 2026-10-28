"""Tune per-field decision thresholds for multi-select fields on the validation split.

Multi-label decoding uses a fixed 0.5 sigmoid threshold; rare labels are
under-predicted, which costs macro-F1. This sweeps thresholds per field on
in-domain validation data (never the held-out set) and writes thresholds.json.

    uv run propella-bekko tune-thresholds --model runs/a25m-v1-pool/model \
        --data data/train-pool-long.parquet --limit 20000
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

from propella_bekko.inference import Annotator
from propella_bekko.models import PropellaBekko
from propella_bekko.schema import Field, encode_labels


def tune(model, tok, cfg, rows, fields: list[Field], grid=(0.1, 0.5, 0.05)) -> dict[str, float]:
    ann = Annotator(
        model, tok, cfg["max_length"], cfg["tail_tokens"], device="cuda", max_tokens=262144
    )
    probs = ann.probabilities(rows)
    targets = [encode_labels(r, fields) for r in rows]
    out = {}
    lo, hi, step = grid
    thresholds = np.arange(lo, hi + 1e-9, step)
    for f in fields:
        if f.kind != "multi":
            continue
        P = np.asarray(probs[f.name], dtype=np.float32)
        Y = np.asarray([t[f.name] for t in targets], dtype=np.int8)
        valid = Y[:, 0] != -100
        best, best_t = -1.0, 0.5
        for t in thresholds:
            pred = (t <= P).astype(np.int8)
            score = f1_score(Y[valid], pred[valid], average="macro", zero_division=0)
            if score > best:
                best, best_t = score, float(t)
        out[f.name] = round(best_t, 2)
    return out


def main(a):
    model, tok, cfg = PropellaBekko.load(a.model, device="cuda")
    # tune on the same 2% validation split that training never sees
    rows = [r for r in pq.read_table(a.data).to_pylist() if _is_val(r, a.val_frac)]
    if a.limit:
        rows = rows[: a.limit]
    th = tune(model, tok, cfg, rows, model.fields)
    Path(a.output).write_text(json.dumps(th, indent=2))
    print(json.dumps(th, indent=2))


def _is_val(row: dict, frac: float) -> bool:
    import hashlib

    h = hashlib.sha1(f"{row['source']}/{row['id']}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 < frac

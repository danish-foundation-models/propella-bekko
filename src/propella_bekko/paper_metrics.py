"""The propella-1 evaluation score (arXiv 2602.12414, §5.2).

Per property type:
* 11 ordinal properties  -> quadratic weighted kappa (QWK)
* 1 binary property (PII) -> F1
* 5 multi-select         -> intersection-over-union (Jaccard), averaged per sample

overall = 11/17 * mean(QWK) + 1/17 * F1 + 5/17 * mean(IoU)

`one_sentence_description` is excluded, as in the paper.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from propella_bekko.schema import OTHER, Field

ORDINAL = [
    "content_integrity",
    "content_ratio",
    "content_length",
    "content_quality",
    "information_density",
    "educational_value",
    "reasoning_indicators",
    "audience_level",
    "commercial_bias",
    "time_sensitivity",
    "content_safety",
]
BINARY = ["pii_presence"]
MULTI = [
    "content_type",
    "business_sector",
    "technical_content",
    "regional_relevance",
    "country_relevance",
]


def _qwk(pred: list[str], gold: list[str], field: Field) -> float:
    from sklearn.metrics import cohen_kappa_score

    rank = {lab: i for i, lab in enumerate(field.labels)}
    yg = [rank[g] for g in gold]
    yp = [rank.get(p, len(rank) - 1) for p in pred]
    if len(set(yg)) < 2:
        return float("nan")
    return float(cohen_kappa_score(yg, yp, weights="quadratic"))


def _iou(pred, gold, labels: set[str]) -> float:
    p = {x if x in labels else OTHER for x in (pred or [])}
    g = {x if x in labels else OTHER for x in (gold or [])}
    return 1.0 if not (p | g) else len(p & g) / len(p | g)


def paper_score(preds: list[dict], golds: list[dict], fields: list[Field]) -> dict:
    by_name = {f.name: f for f in fields}
    out: dict[str, float] = {}
    for name in ORDINAL:
        f = by_name[name]
        pairs = [
            (p[name], g[name])
            for p, g in zip(preds, golds, strict=False)
            if g.get(name) in f.labels
        ]
        out[name] = _qwk([a for a, _ in pairs], [b for _, b in pairs], f) if pairs else float("nan")
    for name in BINARY:
        f = by_name[name]
        pairs = [
            (p[name], g[name])
            for p, g in zip(preds, golds, strict=False)
            if g.get(name) in f.labels
        ]
        labels = sorted({b for _, b in pairs})
        out[name] = (
            float(
                f1_score(
                    [b for _, b in pairs],
                    [a for a, _ in pairs],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
            if pairs
            else float("nan")
        )
    for name in MULTI:
        f = by_name[name]
        labels = set(f.labels)
        pairs = [(p[name], g[name]) for p, g in zip(preds, golds, strict=False) if g.get(name)]
        out[name] = (
            float(np.mean([_iou(a, b, labels) for a, b in pairs])) if pairs else float("nan")
        )
    qwk = float(np.nanmean([out[n] for n in ORDINAL]))
    pii = out["pii_presence"]
    iou = float(np.nanmean([out[n] for n in MULTI]))
    overall = 11 / 17 * qwk + 1 / 17 * pii + 5 / 17 * iou
    return {
        "overall": float(overall),
        "mean_qwk": qwk,
        "pii_f1": pii,
        "mean_iou": iou,
        "per_property": out,
    }

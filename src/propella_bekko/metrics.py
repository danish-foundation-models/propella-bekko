"""Agreement metrics between student predictions and propella-1-4b (teacher)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from propella_bekko.schema import OTHER, Field


def _norm_multi(v, labels: set[str]) -> set[str]:
    return {x if x in labels else OTHER for x in (v or [])}


def field_metrics(preds: list[dict], golds: list[dict], fields: list[Field]) -> dict[str, dict]:
    res: dict[str, dict] = {}
    for f in fields:
        if f.kind == "single":
            pairs: list[tuple[str, str]] = [
                (p[f.name], g[f.name])
                for p, g in zip(preds, golds, strict=False)
                if g.get(f.name) in f.labels
            ]
            if not pairs:
                continue
            yp, yg = zip(*pairs, strict=False)
            m = {
                "n": len(pairs),
                "accuracy": float(np.mean([a == b for a, b in pairs])),
                "macro_f1": float(
                    f1_score(yg, yp, labels=f.labels, average="macro", zero_division=0)
                ),
                "majority_baseline": float(max(yg.count(lab) for lab in set(yg)) / len(yg)),
            }
            if f.ordinal:
                rank = {lab: i for i, lab in enumerate(f.labels)}
                d = np.array([abs(rank[a] - rank[b]) for a, b in pairs])
                m["mae"] = float(d.mean())
                m["within_1"] = float((d <= 1).mean())
                m["qwk"] = (
                    float(
                        cohen_kappa_score(
                            [rank[x] for x in yg],
                            [rank[x] for x in yp],
                            weights="quadratic",
                        )
                    )
                    if len(set(yg)) > 1
                    else float("nan")
                )
            res[f.name] = m
        else:
            labels = set(f.labels)
            mpairs = [
                (_norm_multi(p[f.name], labels), _norm_multi(g.get(f.name), labels))
                for p, g in zip(preds, golds, strict=False)
                if g.get(f.name)
            ]
            if not mpairs:
                continue
            idx = {lab: i for i, lab in enumerate(f.labels)}
            P = np.zeros((len(mpairs), len(idx)), dtype=int)
            G = np.zeros_like(P)
            for i, (p, g) in enumerate(mpairs):
                for x in p:
                    P[i, idx[x]] = 1
                for x in g:
                    G[i, idx[x]] = 1
            jacc = [len(p & g) / len(p | g) if (p | g) else 1.0 for p, g in mpairs]
            res[f.name] = {
                "n": len(mpairs),
                "exact_match": float(np.mean([p == g for p, g in mpairs])),
                "jaccard": float(np.mean(jacc)),
                "micro_f1": float(f1_score(G, P, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(G, P, average="macro", zero_division=0)),
            }
    return res


def summarize(res: dict[str, dict]) -> dict[str, float]:
    single = [m["accuracy"] for m in res.values() if "accuracy" in m]
    multi = [m["jaccard"] for m in res.values() if "jaccard" in m]
    mf1 = [m["macro_f1"] for m in res.values()]
    base = [m["majority_baseline"] for m in res.values() if "majority_baseline" in m]
    return {
        "mean_single_accuracy": float(np.mean(single)) if single else float("nan"),
        "mean_multi_jaccard": float(np.mean(multi)) if multi else float("nan"),
        "mean_macro_f1": float(np.mean(mf1)) if mf1 else float("nan"),
        "mean_single_majority_baseline": float(np.mean(base)) if base else float("nan"),
    }

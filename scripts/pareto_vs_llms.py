"""Pareto numbers with an LLM annotator as the reference, one file per reference.

Every model is scored against the same 300 reference annotations (macro-F1 over the
annotation fields, the same `field_metrics` used elsewhere), so the y axis answers
"how much of that annotator do you get back, and how fast". The reference model is
the axis, not a point; the other LLM annotators are scored as points, so the two
annotators can be compared against each other as well.

Throughput is documents per second on one RTX 8000 for the local models and
documents per second of wall time for the API models.

    uv run python scripts/pareto_vs_llms.py

Writes `runs/pareto_vs_<reference>.json`.
"""

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from propella_bekko.metrics import field_metrics
from propella_bekko.schema import fields_from_config

N = 300
STUDENTS = ["static-long-ctx8192", "a8m-xl", "a25m-xl"]
PROPELLA = {
    "propella-1-0.6b": ("0.6b", 600),
    "propella-1-1.7b": ("1.7b", 1700),
    "propella-1-4b": ("4b", 4000),
}
# The LLM annotators: any one of them can be the y axis of a figure.
LLMS = {
    "deepseek-v4.1-flash": "runs/reference/deepseek-v4.1-flash-pareto300",
    "glm-5.3-flash": "runs/reference/glm-5.3-flash-pareto300",
}


def load_predictions(stem: str) -> dict | None:
    parquet = Path(stem + ".parquet")
    if not parquet.exists():
        return None
    if "audience_level" not in pq.read_schema(parquet).names:
        return None
    return {(r["source"], r["id"]): r for r in pq.read_table(parquet).to_pylist()}


def chance_predictions(gold: dict, fields) -> dict:
    """Always predict the most frequent single-select label, and no multi-select labels.

    That is the floor a model has to clear: the macro-F1 you get with no signal at all
    beyond the label distribution.
    """
    keys = list(gold)
    constant = {}
    for f in fields:
        if f.kind == "single":
            values = [gold[k].get(f.name) for k in keys if gold[k].get(f.name) is not None]
            constant[f.name] = Counter(values).most_common(1)[0][0] if values else "<invalid>"
        else:
            constant[f.name] = []
    return {k: {**constant, "source": k[0], "id": k[1]} for k in keys}


def score(preds: dict, gold: dict, fields) -> dict:
    keys = [k for k in gold if k in preds]
    P = [dict(preds[k]) for k in keys]
    for p in P:
        for f in fields:
            if p.get(f.name) is None:
                p[f.name] = "<invalid>" if f.kind == "single" else []
    res = field_metrics(P, [gold[k] for k in keys], fields)
    single = [m["accuracy"] for m in res.values() if "accuracy" in m]
    return {
        "macro_f1": float(sum(m["macro_f1"] for m in res.values()) / len(res)),
        "accuracy": float(sum(single) / max(1, len(single))),
        "n": len(keys),
    }


def main() -> None:
    fields = fields_from_config(
        json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
    )
    rows = pq.read_table("data/heldout-sample1000.parquet").to_pylist()[:N]
    wanted = {(r["source"], r["id"]) for r in rows}
    predictions = {name: load_predictions(stem) for name, stem in LLMS.items()}

    for reference in LLMS:
        ref = predictions[reference]
        if ref is None:
            print(f"skip reference {reference}: no per-field predictions")
            continue
        gold = {k: v for k, v in ref.items() if k in wanted}
        print(f"\n=== y axis: {reference} ({len(gold)} documents) ===")

        models = {}
        for name in STUDENTS:
            bench = json.loads(Path(f"runs/pareto/{name}.bench.json").read_text())
            models[name] = {
                "family": "student",
                "params_M": round(bench["params_M"]),
                "docs_per_s": bench["docs_per_s"],
                **score(load_predictions(f"runs/pareto/{name}"), gold, fields),
            }

        for name, (short, params) in PROPELLA.items():
            stem_p = f"runs/baselines/long300-{short}"
            preds = load_predictions(stem_p)
            if preds is None:
                print("skip", name, "(no predictions)")
                continue
            stats = json.loads(Path(stem_p + ".json").read_text())
            models[name] = {
                "family": "propella",
                "params_M": params,
                "docs_per_s": stats["docs_per_s"],
                **score(preds, gold, fields),
            }

        for name, preds in predictions.items():
            if name == reference or preds is None:
                continue
            stats = json.loads(Path(LLMS[name] + ".json").read_text())
            models[name] = {
                "family": "llm",
                "params_M": 0,
                "docs_per_s": stats["docs"] / stats["seconds"],
                **score(preds, gold, fields),
            }

        models["chance"] = {
            "family": "chance",
            "params_M": 0,
            "docs_per_s": 0.0,
            **score(chance_predictions(gold, fields), gold, fields),
        }

        out = Path(f"runs/pareto_vs_{reference}.json")
        out.write_text(json.dumps({"reference": reference, "models": models}, indent=2))
        for name, m in sorted(models.items(), key=lambda kv: kv[1]["docs_per_s"]):
            print(
                f"{name:24s} {m['docs_per_s']:9.2f} docs/s  acc {m['accuracy']:.3f} "
                f"mF1 {m['macro_f1']:.3f}  n={m['n']}"
            )
        print("wrote", out)


if __name__ == "__main__":
    main()

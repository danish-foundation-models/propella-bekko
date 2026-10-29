"""Score every model on the 1,000-doc test set against two references and plot both Paretos.

References
----------
* `propella`: the stored propella-1-4b labels that came with the corpora.
* `llm`:      consensus of the large-LLM annotators (DeepSeek-V4.1-Flash + GLM-5.3-Flash),
              labelled with propella's full annotator prompt + JSON schema (as in the paper).

Models: the propella-1 baselines we ran ourselves and the distilled students.
Metric: the propella-1 overall score (11/17 QWK + 1/17 PII-F1 + 5/17 IoU).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from propella_bekko.paper_metrics import paper_score
from propella_bekko.schema import fields_from_config

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--baseline-prefix", default="long1k")
ap.add_argument("--out", default="docs/assets/pareto.svg")
args = ap.parse_args()
N = args.n
BG, FG, GRID = "#191433", "#ffd45c", "#5e3570"
COLORS = {"propella": "#ff6fb1", "student": "#5cf2ff"}

fields = fields_from_config(
    json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
)


def load(path):
    return {(r["source"], r["id"]): r for r in pq.read_table(path).to_pylist()}


def clean(rows, keys):
    out = []
    for k in keys:
        r = dict(rows.get(k, {}))
        for f in fields:
            if r.get(f.name) is None:
                r[f.name] = "<invalid>" if f.kind == "single" else []
        out.append(r)
    return out


rows = pq.read_table("data/heldout-sample1000.parquet").to_pylist()[:N]
stored = {(r["source"], r["id"]): r for r in rows}

refs = {"propella": stored}
llm_paths = sorted(Path("runs/reference").glob("*-1k.parquet"))
if llm_paths:
    from make_consensus import consensus  # scripts/ dir

    refs["llm"] = {(r["source"], r["id"]): r for r in consensus([load(str(p)) for p in llm_paths])}

params_M = {
    "propella-1-0.6b": 600,
    "propella-1-1.7b": 1700,
    "propella-1-4b": 4000,
    "static-v1-pool": 206,
    "static-long-ctx8192": 206,
    "a8m-v0": 106,
    "a8m-v1-pool": 106,
    "a25m-v0": 118,
    "a25m-v1-pool": 118,
    "a25m-domcap": 118,
    "a25m-pw": 118,
    "a25m-refine": 118,
    "a8m-domcap": 106,
    "a8m-xl": 106,
    "a25m-xl": 118,
}

models = {}
for p in sorted(Path("runs/pareto").glob("*.parquet")):
    bench = p.with_suffix(".bench.json")
    if bench.exists():
        models[p.stem] = (load(str(p)), json.loads(bench.read_text())["docs_per_s"], "student")
for p in sorted(Path("runs/baselines").glob(f"{args.baseline_prefix}-*.parquet")):
    stats = p.with_suffix(".json")
    if stats.exists():
        name = f"propella-1-{p.stem.split('-', 1)[1]}"
        models[name] = (load(str(p)), json.loads(stats.read_text())["docs_per_s"], "propella")

out = {}
for ref_name, ref in refs.items():
    keys = [k for k in stored if k in ref]
    gold = clean(ref, keys)
    scores = {}
    for name, (pred, dps, family) in models.items():
        subs = [k for k in keys if k in pred]
        res = paper_score(clean(pred, subs), clean(ref, subs), fields)
        scores[name] = {
            "family": family,
            "params_M": params_M.get(name, 0),
            "docs_per_s": dps,
            "overall": res["overall"],
            "qwk": res["mean_qwk"],
            "iou": res["mean_iou"],
            "pii_f1": res["pii_f1"],
            "n": len(subs),
        }
    out[ref_name] = scores

Path("runs/pareto_scores.json").write_text(json.dumps(out, indent=2))
for ref_name, scores in out.items():
    print(f"\n== reference: {ref_name}")
    for name, m in sorted(scores.items(), key=lambda kv: kv[1]["docs_per_s"]):
        print(
            f"  {name:24s} {m['params_M']:5.0f}M {m['docs_per_s']:8.1f} docs/s "
            f"overall {m['overall']:.3f} (qwk {m['qwk']:.3f} iou {m['iou']:.3f}) n={m['n']}"
        )

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
fig.patch.set_facecolor(BG)
for ax in axes:  # style both panels even if one has no data yet
    ax.set_facecolor(BG)
    ax.set_xscale("log")
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
for ax, ref_name in zip(axes, refs, strict=False):
    scores = out[ref_name]
    for name, m in sorted(scores.items(), key=lambda kv: kv[1]["docs_per_s"]):
        ax.scatter(
            m["docs_per_s"],
            m["overall"],
            s=80,
            color=COLORS[m["family"]],
            edgecolor=FG,
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            name.replace("propella-1-", "").replace("student ", ""),
            (m["docs_per_s"], m["overall"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            color=FG,
        )
    ax.set_xlabel("documents / second (1× RTX 8000)", color=FG, fontsize=9)
    ax.set_title(f"reference: {ref_name}", color=FG, fontsize=10)
axes[0].set_ylabel("propella-1 overall score", color=FG, fontsize=9)
fig.suptitle("propella-bekko: annotation quality vs throughput", color=FG, fontsize=11)
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, facecolor=BG)
print("\nwrote docs/assets/pareto.svg")

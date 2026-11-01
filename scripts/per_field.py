"""Per-property agreement: one bar per property, student against teacher and chance.

The overall macro-F1 in the cards is one number for 17 very different properties. This
figure shows where it comes from: agreement with a reference annotator (default DeepSeek-
V4.1-Flash on the same 300 held-out documents), one bar per property for the student, a
marker for propella-1-4b on the same documents, and a line for chance.

    uv run python scripts/per_field.py                       # a25m vs DeepSeek
    uv run python scripts/per_field.py --run a8m-xl --name propella-bekko-a8m

Writes `docs/assets/per_property.svg` and `.png`, plus the numbers in
`runs/per_property_<run>.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib.lines import Line2D

from propella_bekko.metrics import field_metrics
from propella_bekko.schema import fields_from_config

BG, FG, GRID = "#191433", "#ffd45c", "#5e3570"
STUDENT_C, TEACHER_C = "#5cf2ff", "#ff6fb1"
LINK_C, FLOOR_C = "#8a7bb5", "#c9b8e8"
N = 300
REFS = {
    "deepseek-v4.1-flash": "runs/reference/deepseek-v4.1-flash-pareto300",
    "glm-5.3-flash": "runs/reference/glm-5.3-flash-pareto300",
}
# propella-1 baselines re-run by us, bf16 on an A6000
TEACHER = "runs/baselines/bf16/propella-1-4b"


def load(path: str) -> dict[tuple, dict]:
    return {(r["source"], r["id"]): r for r in pq.read_table(path + ".parquet").to_pylist()}


def scored(preds: dict, gold: dict, fields) -> dict[str, dict]:
    keys = [k for k in gold if k in preds]
    P = []
    for k in keys:
        row = dict(preds[k])
        for f in fields:
            if row.get(f.name) is None:
                row[f.name] = "<invalid>" if f.kind == "single" else []
        P.append(row)
    return field_metrics(P, [gold[k] for k in keys], fields)


def chance(gold: dict, fields) -> dict[tuple, dict]:
    """Most frequent single-select label, no multi-select labels: the no-signal floor."""
    from collections import Counter

    rows = [gold[k] for k in gold]
    const = {}
    for f in fields:
        if f.kind == "single":
            vals = [r.get(f.name) for r in rows if r.get(f.name) is not None]
            const[f.name] = Counter(vals).most_common(1)[0][0] if vals else "<invalid>"
        else:
            const[f.name] = []
    return {k: {**const, "source": k[0], "id": k[1]} for k in gold}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="a25m-xl-3ep", help="run dir under runs/pareto")
    ap.add_argument("--name", default="propella-bekko-a25m", help="name used in the legend")
    ap.add_argument("--reference", default="deepseek-v4.1-flash", choices=list(REFS))
    ap.add_argument("--out", default="docs/assets/per_property.svg")
    a = ap.parse_args()

    fields = fields_from_config(
        json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
    )
    rows = pq.read_table("data/heldout-sample1000.parquet").to_pylist()[:N]
    wanted = {(r["source"], r["id"]) for r in rows}
    gold = {k: v for k, v in load(REFS[a.reference]).items() if k in wanted}

    mine = scored(load(f"runs/pareto/{a.run}"), gold, fields)
    teacher = scored(load(TEACHER), gold, fields)
    floor = scored(chance(gold, fields), gold, fields)

    order = [f.name for f in fields]
    order.sort(key=lambda n: mine[n]["macro_f1"])
    data = {
        n: {
            "student": mine[n]["macro_f1"],
            "teacher": teacher[n]["macro_f1"],
            "chance": floor[n]["macro_f1"],
        }
        for n in order
    }
    Path(f"runs/per_property_{a.run}.json").write_text(json.dumps(data, indent=1))

    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    y = list(range(len(order)))
    # dumbbell: one row per property, the segment is the gap between student and teacher
    for i, n in enumerate(order):
        ax.plot(
            [data[n]["student"], data[n]["teacher"]],
            [i, i],
            color=LINK_C,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            [data[n]["chance"], data[n]["chance"]],
            [i - 0.28, i + 0.28],
            color=FLOOR_C,
            linewidth=1.4,
            linestyle=(0, (1.2, 1.6)),
            zorder=1,
        )
    ax.scatter(
        [data[n]["student"] for n in order],
        y,
        s=58,
        color=STUDENT_C,
        zorder=3,
        label=a.name,
    )
    ax.scatter(
        [data[n]["teacher"] for n in order],
        y,
        marker="D",
        s=28,
        color=TEACHER_C,
        zorder=4,
        label="propella-1-4b",
    )
    # the number belongs to the student, so it sits on the student's side of the row
    for i, n in enumerate(order):
        s = data[n]["student"]
        inside = s < 0.14
        ax.text(
            s + 0.022 if inside else s - 0.022,
            i,
            f"{s:.2f}",
            va="center",
            ha="left" if inside else "right",
            fontsize=7,
            color=STUDENT_C,
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(order, fontsize=8, color=FG)
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", colors=FG, labelsize=8)
    ax.set_xlabel(f"macro-F1 vs {a.reference}, 300 held-out documents", color=FG, fontsize=9)
    ax.set_title(
        f"{a.name} per property, against the teacher that labelled it",
        color=FG,
        fontsize=10,
        pad=10,
    )
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7, color=STUDENT_C, label=a.name),
        Line2D(
            [], [], marker="D", linestyle="", markersize=5, color=TEACHER_C, label="propella-1-4b"
        ),
        Line2D(
            [],
            [],
            linestyle=(0, (1.5, 1.5)),
            linewidth=1.1,
            color=FLOOR_C,
            label="no-signal floor",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=8,
        labelcolor=FG,
        handletextpad=0.4,
    )
    fig.tight_layout()

    out = Path(a.out)
    fig.savefig(out, facecolor=BG)
    fig.savefig(out.with_suffix(".png"), facecolor=BG)
    print(f"wrote {out} and {out.with_suffix('.png')}")
    mean_gap = sum(data[n]["teacher"] - data[n]["student"] for n in order) / len(order)
    worst = order[:3]
    best = order[-3:]
    print(f"mean gap to the teacher: {mean_gap:+.3f}")
    print("furthest behind:", ", ".join(worst))
    print("closest to / above the teacher:", ", ".join(reversed(best)))


if __name__ == "__main__":
    main()

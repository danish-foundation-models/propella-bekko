"""Per-property agreement, one panel per reference annotator.

The overall macro-F1 in the cards is one number for 17 very different properties. This
figure shows where it comes from: for each reference annotator (DeepSeek-V4.1-Flash on the
left, GLM-5.3-Flash on the right, same 300 held-out documents), one row per property with
the student's macro-F1, propella-1-4b's on the same documents, and the no-signal floor.

    uv run python scripts/per_field.py                          # a25m -> models/propella-bekko-a25m/
    uv run python scripts/per_field.py --run a8m-xl --name propella-bekko-a8m

Writes `docs/assets/per_property_<tier>.svg` and `.png`, and the numbers in
`runs/per_property_<run>.json`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from palette import BG, FG, FLOOR, LINK, PROPELLA, STUDENT, style_axes

from propella_bekko.metrics import field_metrics
from propella_bekko.schema import fields_from_config

N = 300
REFS = {
    "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash",
    "glm-5.3-flash": "GLM-5.3-Flash",
}
PATHS = {
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
    rows = [gold[k] for k in gold]
    const = {}
    for f in fields:
        if f.kind == "single":
            vals = [r.get(f.name) for r in rows if r.get(f.name) is not None]
            const[f.name] = Counter(vals).most_common(1)[0][0] if vals else "<invalid>"
        else:
            const[f.name] = []
    return {k: {**const, "source": k[0], "id": k[1]} for k in gold}


def draw(ax, order: list[str], data: dict, name: str, reference: str) -> None:
    y = list(range(len(order)))
    for i, n in enumerate(order):
        # dumbbell: the segment is the gap between the student and the teacher
        ax.plot(
            [data[n]["student"], data[n]["teacher"]],
            [i, i],
            color=LINK,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            [data[n]["chance"], data[n]["chance"]],
            [i - 0.28, i + 0.28],
            color=FLOOR,
            linewidth=1.4,
            linestyle=(0, (1.2, 1.6)),
            zorder=1,
        )
    ax.scatter([data[n]["student"] for n in order], y, s=58, color=STUDENT, zorder=3, label=name)
    ax.scatter(
        [data[n]["teacher"] for n in order],
        y,
        marker="D",
        s=28,
        color=PROPELLA,
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
            color=STUDENT,
        )

    ax.set_yticks(y)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("macro-F1", color=FG, fontsize=9)
    ax.text(
        0.0,
        1.02,
        f"vs {REFS[reference]}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=FG,
    )
    style_axes(ax, grid_axis="x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="a25m-xl-3ep", help="run dir under runs/pareto")
    ap.add_argument("--name", default="propella-bekko-a25m", help="name used in the legend")
    ap.add_argument("--out", default="docs/assets/per_property_a25m.svg")
    a = ap.parse_args()

    fields = fields_from_config(
        json.loads(Path("runs/static-v0/model/config.json").read_text())["fields"]
    )
    rows = pq.read_table("data/heldout-sample1000.parquet").to_pylist()[:N]
    wanted = {(r["source"], r["id"]) for r in rows}

    mine = load(f"runs/pareto/{a.run}")
    teacher = load(TEACHER)

    scores: dict[str, dict] = {}
    for reference in REFS:
        gold = {k: v for k, v in load(PATHS[reference]).items() if k in wanted}
        s, t, f = (
            scored(mine, gold, fields),
            scored(teacher, gold, fields),
            scored(chance(gold, fields), gold, fields),
        )
        scores[reference] = {
            n: {
                "student": s[n]["macro_f1"],
                "teacher": t[n]["macro_f1"],
                "chance": f[n]["macro_f1"],
            }
            for n in s
        }
    Path(f"runs/per_property_{a.run}.json").write_text(json.dumps(scores, indent=1))

    # one order for both panels, taken from the DeepSeek panel: weakest property at the bottom
    order = sorted(
        scores["deepseek-v4.1-flash"], key=lambda n: scores["deepseek-v4.1-flash"][n]["student"]
    )

    fig, axes = plt.subplots(
        1, len(REFS), figsize=(11.4, 6.0), dpi=200, sharey=True, layout="constrained"
    )
    fig.patch.set_facecolor(BG)
    for i, (ax, reference) in enumerate(zip(axes, REFS, strict=False)):
        draw(ax, order, scores[reference], a.name, reference)
        if i == 0:
            ax.set_yticklabels(order, fontsize=8, color=FG)
        ax.set_yticks(list(range(len(order))))
        if i:
            ax.tick_params(labelleft=False)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7, color=STUDENT, label=a.name),
        Line2D(
            [], [], marker="D", linestyle="", markersize=5, color=PROPELLA, label="propella-1-4b"
        ),
        Line2D(
            [], [], linestyle=(0, (1.5, 1.5)), linewidth=1.1, color=FLOOR, label="no-signal floor"
        ),
    ]
    leg = fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    for text in leg.get_texts():
        text.set_color(FG)
    fig.text(
        0.5,
        1.0,
        f"{a.name} per property, 300 held-out documents, agreement with each reference annotator",
        ha="center",
        va="bottom",
        fontsize=10,
        color=FG,
    )

    # SVG only: the Hub rejects binary files pushed over git, and GitHub renders SVG
    out = Path(a.out)
    fig.savefig(out, facecolor=BG)
    print(f"wrote {out}")
    for reference in REFS:
        d = scores[reference]
        gap = sum(d[n]["teacher"] - d[n]["student"] for n in order) / len(order)
        above = [n for n in order if d[n]["student"] >= d[n]["teacher"]]
        print(
            f"{reference}: mean gap to teacher {gap:+.3f}, at or above on {len(above)} properties"
        )


if __name__ == "__main__":
    main()

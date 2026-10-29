"""Pareto figures: annotation quality vs throughput, one per LLM reference annotator.

Reads `runs/pareto_vs_<reference>.json` (built by scripts/pareto_vs_llms.py) and
writes docs/assets/pareto_<reference>.svg and .png.

Conventions: white background, one hue per model family (Okabe-Ito blue / vermillion),
everything else grey; no title, no subtitle, no footnote - the axes and the direct
labels carry the whole chart. The scatter holds the models whose throughput was
measured on one GPU; the other LLM is a dashed quality rule; chance is a dotted floor;
the solid grey line is the Pareto set (no faster model is at least as good).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

BG = "#ffffff"
FG = "#1a1a1a"
GRID = "#e3e3e3"
MUTED = "#8c8c8c"
STUDENT = "#0072b2"   # Okabe-Ito blue
PROPELLA = "#d55e00"  # Okabe-Ito vermillion
RULE = "#333333"
FRONTIER = "#555555"

REFERENCE_NAME = {
    "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash",
    "glm-5.3-flash": "GLM-5.3-Flash",
}
STUDENT_LABEL = {
    "static-long-ctx8192": "static · 206M",
    "a8m-xl": "a8m · 106M",
    "a25m-xl": "a25m · 123M",
}
PROPELLA_LABEL = {
    "propella-1-0.6b": "propella-1-0.6b · 600M",
    "propella-1-1.7b": "propella-1-1.7b · 1.7B",
    "propella-1-4b": "propella-1-4b · 4B",
}
LLM_LABEL = {"glm-5.3-flash": "GLM-5.3-Flash", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash"}
# (dx, dy, alignment) of each direct label, chosen so no two boxes collide.
LABEL_OFFSET = {
    "static-long-ctx8192": (0, -18, "center"),
    "a8m-xl": (14, 14, "left"),
    "a25m-xl": (-14, 8, "right"),
    "propella-1-4b": (14, 10, "left"),
    "propella-1-1.7b": (14, 0, "left"),
    "propella-1-0.6b": (14, -12, "left"),
}
X_TICKS = [0.1, 1, 10, 100, 1000]


def pareto_set(points: list[dict]) -> list[dict]:
    """Non-dominated models: nothing faster is at least as good."""
    frontier, best = [], -1.0
    for m in sorted(points, key=lambda m: -m["docs_per_s"]):
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            frontier.append(m)
    return sorted(frontier, key=lambda m: m["docs_per_s"])


def render(path: Path) -> None:
    payload = json.loads(path.read_text())
    reference = payload["reference"]
    data = payload["models"]
    ref_name = REFERENCE_NAME.get(reference, reference)

    points = {n: m for n, m in data.items() if m["family"] in ("student", "propella")}
    other_llm = {n: m for n, m in data.items() if m["family"] == "llm"}
    chance = data.get("chance")
    frontier = pareto_set(list(points.values()))
    on_frontier = {m["docs_per_s"] for m in frontier}

    fig = plt.figure(figsize=(9.0, 5.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0.085, 0.13, 0.895, 0.84))
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=FG, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax.set_xscale("log")
    ax.set_xlim(0.035, 1200)
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())

    ys = [m["macro_f1"] for m in points.values()]
    ys += [m["macro_f1"] for m in other_llm.values()]
    low = min(ys + ([chance["macro_f1"]] if chance else []))
    ax.set_ylim(low - 0.045, max(ys) + 0.045)

    if chance:
        ax.axhline(chance["macro_f1"], color=MUTED, linewidth=1.0,
                   linestyle=(0, (1, 2.5)), zorder=2)
        ax.annotate(f"chance — most frequent label ({chance['macro_f1']:.2f})",
                    (0.012, chance["macro_f1"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points", fontsize=8.5,
                    color=MUTED, ha="left", va="bottom")

    # The other LLM: quality level only, no GPU throughput to place on the x axis.
    for name, m in other_llm.items():
        ax.axhline(m["macro_f1"], color=RULE, linewidth=1.2, linestyle=(0, (6, 4)), zorder=2)
        ax.annotate(f"{LLM_LABEL.get(name, name)} — {m['macro_f1']:.2f}",
                    (1.0, m["macro_f1"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points", fontsize=9.5,
                    color=RULE, ha="right", va="bottom")

    if len(frontier) > 1:
        ax.plot([m["docs_per_s"] for m in frontier], [m["macro_f1"] for m in frontier],
                color=FRONTIER, linewidth=1.3, alpha=0.55, zorder=3, solid_capstyle="round")

    for name, m in points.items():
        lead = m["docs_per_s"] in on_frontier
        color = STUDENT if m["family"] == "student" else PROPELLA
        ax.scatter(m["docs_per_s"], m["macro_f1"], s=150 if lead else 80,
                   color=color, edgecolor=BG, linewidth=1.4 if lead else 1.0,
                   alpha=1.0 if lead else 0.35, zorder=4)
        label = (PROPELLA_LABEL if m["family"] == "propella" else STUDENT_LABEL).get(name, name)
        dx, dy, ha = LABEL_OFFSET.get(name, (14, 0, "left"))
        ax.annotate(
            label, xy=(m["docs_per_s"], m["macro_f1"]), xytext=(dx, dy),
            textcoords="offset points", fontsize=9.5,
            color=FG if lead else MUTED, ha=ha, va="center", zorder=5,
            # A leader line, because on a four-decade log axis a label a few points away
            # is ambiguous about which dot it belongs to.
            arrowprops=dict(arrowstyle="-", color=FG if lead else MUTED, linewidth=0.7,
                            alpha=0.35, shrinkA=2, shrinkB=5),
        )

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=STUDENT, markeredgecolor=BG,
                   markersize=9, label="propella-bekko· one forward pass"),
        plt.Line2D([], [], marker="o", linestyle="", color=PROPELLA, markeredgecolor=BG,
                   markersize=9, label="propella-1 · generative"),
        plt.Line2D([], [], color=FRONTIER, linewidth=1.3, alpha=0.55, label="Pareto frontier"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
                    handletextpad=0.6, borderaxespad=1.0)
    for text in leg.get_texts():
        text.set_color(FG)

    ax.set_xlabel("documents per second on one Quadro RTX 8000, log scale  →  faster",
                  color=FG, fontsize=10.5, labelpad=10)
    ax.set_ylabel(f"agreement with {ref_name} (macro-F1)", color=FG, fontsize=10.5, labelpad=10)

    for ext in ("svg", "png"):
        out = Path("docs/assets") / f"pareto_{reference}.{ext}"
        fig.savefig(out, facecolor=BG)
        print("wrote", out)
    plt.close(fig)


def main() -> None:
    Path("docs/assets").mkdir(parents=True, exist_ok=True)
    for path in sorted(Path("runs").glob("pareto_vs_*.json")):
        render(path)


if __name__ == "__main__":
    main()

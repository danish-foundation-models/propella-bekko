"""Pareto figures: annotation quality vs throughput, one per LLM reference annotator.

Reads `runs/pareto_vs_<reference>.json` (built by scripts/pareto_vs_llms.py) and
writes docs/assets/pareto_<reference>.svg and .png.

Conventions: the shared palette in `propella_bekko.plotting` (deep navy ground, yellow
text, cyan for the students, pink for the teacher); no title, no subtitle, no footnote -
the axes and the direct labels carry the whole chart. Label offsets are hand-set; the
renderer reports any two labels that still collide. The scatter holds the models whose throughput was
measured on one GPU; the other LLM is a dashed quality rule; chance is a dotted floor;
the solid grey line is the Pareto set (no faster model is at least as good).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from propella_bekko.plotting import BG, FG, GRID, LINK, MUTED, PROPELLA, STUDENT

RULE = FG  # the dashed quality rule of the other annotator
FRONTIER = LINK  # the Pareto frontier

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
    "static-long-ctx8192": (12, -20, "left"),
    "a8m-xl": (12, 0, "left"),
    "a25m-xl": (12, 20, "left"),
    "propella-1-4b": (-14, 16, "right"),
    "propella-1-1.7b": (14, 8, "left"),
    "propella-1-0.6b": (14, -10, "left"),
}
X_TICKS = [0.1, 1, 10, 100, 1000]
HARDWARE = "one RTX A6000"


def pareto_set(points: list[dict]) -> list[dict]:
    """Non-dominated models: nothing faster is at least as good."""
    frontier, best = [], -1.0
    for m in sorted(points, key=lambda m: -m["docs_per_s"]):
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            frontier.append(m)
    return sorted(frontier, key=lambda m: m["docs_per_s"])


def draw(ax, payload: dict) -> None:
    """Everything inside the axes: reference rules, frontier, points, labels, legend."""
    data = payload["models"]

    points = {n: m for n, m in data.items() if m["family"] in ("student", "propella")}
    other_llm = {n: m for n, m in data.items() if m["family"] == "llm"}
    chance = data.get("chance")
    frontier = pareto_set(list(points.values()))
    on_frontier = {m["docs_per_s"] for m in frontier}

    ax.set_facecolor(BG)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
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
        ax.axhline(
            chance["macro_f1"], color=MUTED, linewidth=1.0, linestyle=(0, (1, 2.5)), zorder=2
        )
        ax.annotate(
            f"chance — most frequent label ({chance['macro_f1']:.2f})",
            (0.012, chance["macro_f1"]),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8.5,
            color=MUTED,
            ha="left",
            va="bottom",
        )

    # The other LLM: quality level only, no GPU throughput to place on the x axis.
    for name, m in other_llm.items():
        ax.axhline(m["macro_f1"], color=RULE, linewidth=1.2, linestyle=(0, (6, 4)), zorder=2)
        ax.annotate(
            f"{LLM_LABEL.get(name, name)} — {m['macro_f1']:.2f}",
            (1.0, m["macro_f1"]),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=9.5,
            color=RULE,
            ha="right",
            va="bottom",
        )

    if len(frontier) > 1:
        ax.plot(
            [m["docs_per_s"] for m in frontier],
            [m["macro_f1"] for m in frontier],
            color=FRONTIER,
            linewidth=1.3,
            alpha=0.55,
            zorder=3,
            solid_capstyle="round",
        )

    for name, m in points.items():
        lead = m["docs_per_s"] in on_frontier
        color = STUDENT if m["family"] == "student" else PROPELLA
        ax.scatter(
            m["docs_per_s"],
            m["macro_f1"],
            s=150 if lead else 80,
            color=color,
            edgecolor=BG,
            linewidth=1.4 if lead else 1.0,
            alpha=1.0 if lead else 0.35,
            zorder=4,
        )
        label = (PROPELLA_LABEL if m["family"] == "propella" else STUDENT_LABEL).get(name, name)
        dx, dy, ha = LABEL_OFFSET.get(name, (14, 0, "left"))
        ax.annotate(
            label,
            xy=(m["docs_per_s"], m["macro_f1"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5,
            color=FG if lead else MUTED,
            ha=ha,
            va="center",
            zorder=5,
            # A leader line, because on a four-decade log axis a label a few points away
            # is ambiguous about which dot it belongs to.
            arrowprops=dict(
                arrowstyle="-",
                color=FG if lead else MUTED,
                linewidth=0.7,
                alpha=0.35,
                shrinkA=2,
                shrinkB=5,
            ),
        )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=STUDENT,
            markeredgecolor=BG,
            markersize=9,
            label="propella-bekko· one forward pass",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=PROPELLA,
            markeredgecolor=BG,
            markersize=9,
            label="propella-1 · generative",
        ),
        plt.Line2D([], [], color=FRONTIER, linewidth=1.3, alpha=0.55, label="Pareto frontier"),
    ]
    return handles


def axis_titles(
    ax,
    reference: str,
    x_sub: bool = True,
    main_fs: float = 10.5,
    sub_fs: float = 8.5,
    main_x: float = -0.165,
    sub_x: float = -0.10,
) -> None:
    """FT-style axis titles: the main label, with its subtitle beneath it on the same axis."""
    ax.text(
        main_x,
        0.5,
        "Annotation Quality (macro-F1)",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=main_fs,
        color=FG,
    )
    ax.text(
        sub_x,
        0.5,
        f"Agreement with {REFERENCE_NAME.get(reference, reference)}",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=sub_fs,
        color=MUTED,
    )
    if x_sub:
        ax.text(
            0.5,
            -0.155,
            "documents per second",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            color=FG,
        )
        ax.text(
            0.5,
            -0.245,
            f"on {HARDWARE}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color=MUTED,
        )


def legend(ax, handles, loc: str = "lower right") -> None:
    """Legend inside one axes (the single-panel figures)."""
    leg = ax.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
        borderaxespad=1.0,
    )
    for text in leg.get_texts():
        text.set_color(FG)


def render(path: Path) -> None:
    payload = json.loads(path.read_text())
    reference = payload["reference"]

    fig = plt.figure(figsize=(9.0, 5.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0.085, 0.13, 0.895, 0.84))
    handles = draw(ax, payload)
    legend(ax, handles)
    axis_titles(ax, reference)

    for ext in ("svg", "png"):
        out = Path("docs/assets") / f"pareto_{reference}.{ext}"
        fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
        print("wrote", out)
    plt.close(fig)


def render_combined(paths: list[Path]) -> None:
    """One figure, one panel per reference annotator, side by side."""
    payloads = [json.loads(p.read_text()) for p in paths]
    fig, axes = plt.subplots(1, len(payloads), figsize=(9.6, 5.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.42, left=0.085, right=0.985, top=0.97, bottom=0.225)
    for i, (ax, payload) in enumerate(zip(axes, payloads, strict=False)):
        handles = draw(ax, payload)
        axis_titles(
            ax,
            payload["reference"],
            x_sub=False,
            main_fs=9.5,
            sub_fs=8.0,
            main_x=-0.16,
            sub_x=-0.095,
        )
        if i:
            ax.tick_params(labelleft=False)

    fig.text(0.5, 0.125, "documents per second", ha="center", va="bottom", fontsize=10.5, color=FG)
    fig.text(0.5, 0.085, f"on {HARDWARE}", ha="center", va="bottom", fontsize=8.5, color=MUTED)

    # one shared legend at the bottom: inside a panel it collides with the reference rule
    shared = fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
        columnspacing=1.6,
    )
    for text in shared.get_texts():
        text.set_color(FG)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # direct labels are placed by hand: say so when two of them still collide
    for i, ax in enumerate(axes):
        # Text.get_window_extent, not Annotation's: the latter includes the leader line
        from matplotlib.text import Text

        boxes = [
            (t.get_text(), Text.get_window_extent(t, renderer))
            for t in ax.texts
            if t.get_text().strip()
        ]
        for a_i, (ta, ba) in enumerate(boxes):
            for tb, bb in boxes[a_i + 1 :]:
                if ba.overlaps(bb):
                    print(f"  label collision in panel {i}: {ta!r} / {tb!r}")
    for i, ax in enumerate(axes):
        for other in axes:
            if other is ax:
                continue
            box = other.get_window_extent(renderer)
            for artist in ax.get_children():
                if not hasattr(artist, "get_window_extent") or not hasattr(artist, "get_text"):
                    continue
                if not artist.get_text().strip():
                    continue
                b = artist.get_window_extent(renderer)
                if b.width and b.overlaps(box):
                    print(f"  overlap: {artist.get_text()!r} of panel {i} into another panel")
    for ext in ("svg", "png"):
        out = Path("docs/assets") / f"pareto_combined.{ext}"
        fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
        print("wrote", out)
    plt.close(fig)


def main() -> None:
    global HARDWARE
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default=HARDWARE, help="what the x axis was measured on")
    HARDWARE = ap.parse_args().hardware
    Path("docs/assets").mkdir(parents=True, exist_ok=True)
    paths = sorted(Path("runs").glob("pareto_vs_*.json"))
    for path in paths:
        render(path)
    render_combined(paths)


if __name__ == "__main__":
    main()

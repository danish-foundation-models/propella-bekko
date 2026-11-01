"""One palette for every figure in the release.

The Pareto figures and the per-property figure are read side by side in the card, so they
share one look: deep navy ground, warm yellow for anything that is text, cyan for the
students this project trains, pink for the propella-1 teacher it distils.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

BG = "#191433"  # ground
FG = "#ffd45c"  # every label, tick and annotation
GRID = "#5e3570"  # gridlines and axes
MUTED = "#b9a6d6"  # secondary text, the other annotator's rule
LINK = "#8a7bb5"  # the segment between two points, the Pareto frontier
FLOOR = "#c9b8e8"  # no-signal floor, dashed rules
STUDENT = "#5cf2ff"  # propella-bekko students
PROPELLA = "#ff6fb1"  # propella-1 baselines


def style_axes(ax, grid_axis: str = "both") -> None:
    """Ground, spines, ticks and grid in the shared palette."""
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=FG)


def dark_figure(*args, **kwargs):
    """plt.subplots with the shared ground colour already applied."""
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(BG)
    return fig, axes

"""Pareto plot: agreement with propella-1-4b vs throughput on the same documents.

Reads `runs/pareto.json` (built by scripts/collect_pareto.py) and writes
docs/assets/pareto.svg.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, GRID = "#191433", "#ffd45c", "#5e3570"
COLORS = {"propella": "#ff6fb1", "student": "#5cf2ff"}

data = json.loads(Path("runs/pareto.json").read_text())["models"]
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.tick_params(colors=FG, labelsize=9)
ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
ax.set_xscale("log")

for name, m in sorted(data.items(), key=lambda kv: kv[1]["docs_per_s"]):
    color = COLORS[m["family"]]
    ax.scatter(
        m["docs_per_s"], m["macro_f1"], s=90, color=color, edgecolor=FG, linewidth=0.8, zorder=3
    )
    ax.annotate(
        f"{name}\n{m['params_M']:.0f}M",
        (m["docs_per_s"], m["macro_f1"]),
        textcoords="offset points",
        xytext=(6, 5),
        fontsize=7.5,
        color=FG,
    )

ax.set_xlabel("documents / second (1× RTX 8000)", color=FG, fontsize=9)
ax.set_ylabel("macro-F1 vs propella-1-4b labels", color=FG, fontsize=9)
ax.set_title("propella-bekko: quality vs speed", color=FG, fontsize=11)
fig.tight_layout()
out = Path("docs/assets/pareto.svg")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=BG)
print("wrote", out)

"""Every published number, read from `runs/pareto_vs_*.json`.

The README table and the three model cards show the same rows; before this module each of
them retyped the numbers, and the cards kept a stale throughput and a stale baseline table
after the last refresh. One source, three consumers: `scripts/refresh_release.py`,
`scripts/model_card.py` and the figures in `scripts/pareto_two.py` all read these files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PARETO = ("deepseek-v4.1-flash", "glm-5.3-flash")
# students in publish order, and the card each one ships as
STUDENTS = ("a25m-xl", "a8m-xl", "static-long-ctx8192")
REPO_OF = {
    "a25m-xl": "propella-bekko-a25m",
    "a8m-xl": "propella-bekko-a8m",
    "static-long-ctx8192": "propella-bekko-static",
}
PROPELLA = ("propella-1-0.6b", "propella-1-1.7b", "propella-1-4b")
# display name -> key inside runs/pareto_vs_*.json
LLMS = {"DeepSeek-V4.1-Flash": "deepseek-v4.1-flash", "GLM-5.3-Flash": "glm-5.3-flash"}


def load(root: Path | str = "runs") -> dict[str, dict[str, dict]]:
    root = Path(root)
    out = {}
    for stem in PARETO:
        path = root / f"pareto_vs_{stem}.json"
        if not path.exists():
            sys.exit(f"missing {path}: run scripts/pareto_vs_llms.py first")
        out[stem] = json.loads(path.read_text())["models"]
    return out


def params(m: dict) -> str:
    """600M / 1.7B / 4B, the way the tables have always written them."""
    n = m["params_M"]
    if n < 1000:
        return f"{n:.0f}M"
    return f"{f'{n / 1000:.1f}'.removesuffix('.0')}B"


def student_rows(data: dict, bold: str | None = None) -> list[str]:
    rows = []
    for key in STUDENTS:
        ds, glm = data[PARETO[0]][key], data[PARETO[1]][key]
        name = REPO_OF[key]
        label = f"**{name}**" if name == bold else name
        rows.append(
            f"| {label} | {params(ds)} | {ds['docs_per_s']:.0f} | {ds['macro_f1']:.3f} "
            f"| {glm['macro_f1']:.3f} |"
        )
    return rows


def baseline_rows(data: dict) -> list[str]:
    """The teacher family, then each reference annotator as a row (scored by the other one)."""
    rows = [
        f"| {key} | {params(data[PARETO[0]][key])} | {data[PARETO[0]][key]['docs_per_s']:.2f} "
        f"| {data[PARETO[0]][key]['macro_f1']:.3f} | {data[PARETO[1]][key]['macro_f1']:.3f} |"
        for key in PROPELLA
    ]
    for name, key in LLMS.items():
        other = PARETO[1] if key == PARETO[0] else PARETO[0]
        # each annotator is scored in the file where the other one is the reference;
        # throughput is left out: an API's docs/s is a property of the serving queue, not the model
        row = data[other].get(key)
        if row is None:
            continue
        cols = (
            f"| — | {row['macro_f1']:.3f} |"
            if key == PARETO[0]
            else f"| {row['macro_f1']:.3f} | — |"
        )
        rows.append(f"| {name} | — | — {cols}")
    return rows

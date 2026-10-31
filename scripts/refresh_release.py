"""Refresh every published number from `runs/pareto_vs_*.json`.

One command after new baselines: regenerate the three model cards with their measured
throughput and agreement, and rewrite the results table in README.md, so the cards, the
figures and the README cannot drift apart.

    uv run python scripts/refresh_release.py --hardware "one RTX A6000"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# run -> (repo name, tier, n_train, extra model_card args)
STUDENTS = {
    "a25m-xl-3ep": (
        "propella-bekko-a25m",
        "large",
        {},
    ),
    "a8m-xl": ("propella-bekko-a8m", "small", {"--train-minutes": "92"}),
    "static-long-ctx8192": ("propella-bekko-static", "static", {}),
}
# key in runs/pareto_vs_*.json for each run
KEY = {
    "a25m-xl-3ep": "a25m-xl",
    "a8m-xl": "a8m-xl",
    "static-long-ctx8192": "static-long-ctx8192",
}
PROPELLA = ["propella-1-0.6b", "propella-1-1.7b", "propella-1-4b"]
PARAMS = {
    "a25m-xl": "123M",
    "a8m-xl": "106M",
    "static-long-ctx8192": "206M",
    "propella-1-0.6b": "600M",
    "propella-1-1.7b": "1.7B",
    "propella-1-4b": "4B",
}
REFS = {"deepseek-v4.1-flash": "DeepSeek-V4.1-Flash", "glm-5.3-flash": "GLM-5.3-Flash"}
N_TRAIN = "844,912 documents"


def load() -> dict[str, dict[str, dict]]:
    out = {}
    for stem in REFS:
        path = Path(f"runs/pareto_vs_{stem}.json")
        if not path.exists():
            sys.exit(f"missing {path}: run scripts/pareto_vs_llms.py first")
        out[stem] = json.loads(path.read_text())["models"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default="one RTX A6000")
    a = ap.parse_args()
    data = load()

    # 1. cards
    for run, (repo, tier, extra) in STUDENTS.items():
        key = KEY[run]
        row = data[next(iter(REFS))][key]
        cmd = [
            sys.executable,
            "scripts/model_card.py",
            f"runs/{run}",
            repo,
            "--tier",
            tier,
            "--docs-per-s",
            f"{row['docs_per_s']:.0f}",
            "--mf1-deepseek",
            f"{data['deepseek-v4.1-flash'][key]['macro_f1']:.3f}",
            "--mf1-glm",
            f"{data['glm-5.3-flash'][key]['macro_f1']:.3f}",
            "--n-train",
            N_TRAIN,
            "--hardware",
            a.hardware,
        ]
        for k, v in extra.items():
            cmd += [k, v]
        subprocess.run(cmd, check=True)

    # 2. the README table
    lines = [
        "| model | params | docs/s | vs DeepSeek-V4.1-Flash | vs GLM-5.3-Flash |",
        "|---|---|---|---|---|",
    ]
    for key in [KEY[r] for r in STUDENTS]:
        ds = data["deepseek-v4.1-flash"][key]
        glm = data["glm-5.3-flash"][key]
        name = (
            f"**propella-bekko-{key.split('-')[0]}**"
            if key == "a25m-xl"
            else f"propella-bekko-{key.split('-')[0]}"
        )
        lines.append(
            f"| {name} | {PARAMS[key]} | **{ds['docs_per_s']:.0f}** | **{ds['macro_f1']:.3f}** "
            f"| **{glm['macro_f1']:.3f}** |"
            if key == "a25m-xl"
            else f"| {name} | {PARAMS[key]} | {ds['docs_per_s']:.0f} | {ds['macro_f1']:.3f} | {glm['macro_f1']:.3f} |"
        )
    for key in PROPELLA:
        ds = data["deepseek-v4.1-flash"][key]
        glm = data["glm-5.3-flash"][key]
        lines.append(
            f"| {key} | {PARAMS[key]} | {ds['docs_per_s']:.2f} | {ds['macro_f1']:.3f} | {glm['macro_f1']:.3f} |"
        )
    # each LLM's own quality: it is the reference of one column and a row in the other
    ds_row = data["deepseek-v4.1-flash"].get("glm-5.3-flash")
    glm_row = data["glm-5.3-flash"].get("deepseek-v4.1-flash")
    if ds_row and glm_row:
        lines.append(f"| DeepSeek-V4.1-Flash | — | — | — | {ds_row['macro_f1']:.3f} |")
        lines.append(f"| GLM-5.3-Flash | — | — | {glm_row['macro_f1']:.3f} | — |")

    readme = Path("README.md")
    text = readme.read_text()
    start = text.index("| model | params | docs/s |")
    end = text.index("\n\n", start)
    text = text[:start] + "\n".join(lines) + text[end:]
    # keep the sentence above the table honest about the hardware
    for a_hw in ("one RTX A6000", "one Quadro RTX 8000"):
        text = text.replace(
            f"on {a_hw}, every model in bf16", f"on {a.hardware}, every model in bf16"
        )
    readme.write_text(text)

    print("\n".join(lines))
    print(f"\ncards + README refreshed for {a.hardware}")


if __name__ == "__main__":
    main()

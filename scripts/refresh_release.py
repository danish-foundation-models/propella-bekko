"""Refresh every published number from `runs/pareto_vs_*.json`.

One command after new baselines: regenerate the three model cards with their measured
throughput and agreement, and rewrite the results table in README.md, so the cards, the
figures and the README cannot drift apart.

    uv run python scripts/refresh_release.py --hardware "one RTX A6000"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from propella_bekko.release import REPO_OF, baseline_rows, load, student_rows

# run directory -> (repo name, tier, extra model_card args). The run is what the card reads
# its own config and training time from; the numbers come from runs/pareto_vs_*.json.
RUNS = {
    "a25m-xl-3ep": ("propella-bekko-a25m", "large", {}),
    "a8m-xl": ("propella-bekko-a8m", "small", {"--train-minutes": "92"}),
    "static-long-ctx8192": ("propella-bekko-static", "static", {}),
}
# run directory -> key in runs/pareto_vs_*.json
KEY = {"a25m-xl-3ep": "a25m-xl", "a8m-xl": "a8m-xl", "static-long-ctx8192": "static-long-ctx8192"}
N_TRAIN = "844,912 documents"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default="one RTX A6000")
    a = ap.parse_args()
    data = load()

    # 1. cards
    for run, (repo, tier, extra) in RUNS.items():
        cmd = [
            sys.executable,
            "scripts/model_card.py",
            f"runs/{run}",
            repo,
            "--tier",
            tier,
            "--pareto-key",
            KEY[run],
            "--n-train",
            N_TRAIN,
            "--hardware",
            a.hardware,
            *[x for kv in extra.items() for x in kv],
        ]
        subprocess.run(cmd, check=True)

    # 2. the README table
    lines = [
        "| model | params | docs/s | vs DeepSeek-V4.1-Flash | vs GLM-5.3-Flash |",
        "|---|---|---|---|---|",
        *student_rows(data, bold=REPO_OF["a25m-xl"]),
        *baseline_rows(data),
    ]

    readme = Path("README.md")
    text = readme.read_text()
    start = text.index("| model | params | docs/s |")
    end = text.index("\n\n", start)
    text = text[:start] + "\n".join(lines) + text[end:]
    # keep the sentence above the table honest about the hardware
    start = text.index("Throughput is documents per second")
    end = text.index("chance", start)
    text = (
        text[:start]
        + f"Throughput is documents per second\non {a.hardware}, in bf16 for every model; "
        + text[end:]
    )
    readme.write_text(text)

    print("\n".join(lines))
    print(f"\ncards + README refreshed for {a.hardware}")


if __name__ == "__main__":
    main()

"""Rewrite the marked regions of the cards and the README from the scores.

The cards are hand-written documents. Two things in them are numbers that follow the
experiments, and only those two are generated: the results table and the sentence under the
per-property figure. Each sits between markers,

    <!-- start results-table -->
    ...
    <!-- end results-table -->

so this script can replace them and leave every other word alone:

    uv run python scripts/update_sections.py           # write
    uv run python scripts/update_sections.py --check   # exit 1 if anything is stale (CI)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

from propella_bekko.release import baseline_rows, load, student_rows

# published repo -> run directory it was trained in
MODELS = {
    "propella-bekko-a25m": "a25m-xl-3ep",
    "propella-bekko-a8m": "a8m-xl",
    "propella-bekko-static": "static-long-ctx8192",
}
RESTATE = "deepseek-v4.1-flash"  # the left panel, and the reference the sentence quotes
OTHER = "glm-5.3-flash"


def replace_region(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- start {name} -->\n).*?(\n<!-- end {name} -->)", re.DOTALL)
    if not pattern.search(text):
        sys.exit(f"no `{name}` region in this file")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text, count=1)


def per_property_note(run: str) -> str:
    path = Path(f"runs/per_property_{run}.json")
    if not path.exists():
        sys.exit(f"missing {path}: run scripts/per_field.py --run {run} first")
    scores = json.loads(path.read_text())
    primary, other = scores[RESTATE], scores[OTHER]

    gap = sorted(primary, key=lambda n: primary[n]["student"] - primary[n]["teacher"])
    widest = ", ".join(
        f"`{n}` ({primary[n]['student'] - primary[n]['teacher']:+.2f})".replace("-", "\u2212")
        for n in gap[:3]
    )
    above = sorted(n for n in primary if primary[n]["student"] >= primary[n]["teacher"])
    mean_gap = sum(primary[n]["teacher"] - primary[n]["student"] for n in primary) / len(primary)
    above_other = [n for n in other if other[n]["student"] >= other[n]["teacher"]]
    mean_other = sum(other[n]["teacher"] - other[n]["student"] for n in other) / len(other)

    text = (
        f"Against DeepSeek-V4.1-Flash the distance to the teacher sits mostly in three properties — "
        f"{widest} — with a mean gap of {mean_gap:+.2f}; {len(above)} of {len(primary)} properties "
        f"are at or above propella-1-4b's own score"
        + (f" ({', '.join(f'`{n}`' for n in above)})" if above else "")
        + f". Against GLM-5.3-Flash it is closer: mean gap {mean_other:+.2f}, "
        f"{len(above_other)} of {len(other)} at or above."
    )
    return textwrap.fill(text, width=100, break_on_hyphens=False, break_long_words=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report stale regions instead of fixing")
    a = ap.parse_args()

    data = load()
    stale = []

    for repo, run in MODELS.items():
        card = Path("models") / repo / "README.md"
        if not card.exists():
            print(f"skip {repo}: no {card} (submodule not initialised?)")
            continue
        text = card.read_text()
        rows = "\n".join(
            [
                "| model | params | docs/s | vs DeepSeek | vs GLM-5.3 |",
                "|---|---|---|---|---|",
                *student_rows(data, bold=repo),
                *baseline_rows(data),
            ]
        )
        new = replace_region(text, "results-table", rows)
        new = replace_region(new, "per-property-note", per_property_note(run))
        if new != text:
            stale.append(str(card))
            if not a.check:
                card.write_text(new)

    readme = Path("README.md")
    text = readme.read_text()
    rows = "\n".join(
        [
            "| model | params | docs/s | vs DeepSeek-V4.1-Flash | vs GLM-5.3-Flash |",
            "|---|---|---|---|---|",
            *student_rows(data, bold="propella-bekko-a25m"),
            *baseline_rows(data),
        ]
    )
    new = replace_region(text, "results-table", rows)
    if new != text:
        stale.append(str(readme))
        if not a.check:
            readme.write_text(new)

    if a.check and stale:
        print("stale generated sections in: " + ", ".join(stale))
        sys.exit(1)
    print("updated: " + ", ".join(stale) if stale else "everything already up to date")


if __name__ == "__main__":
    main()

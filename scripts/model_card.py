"""Generate the model sheet for a trained student.

    uv run python scripts/model_card.py runs/a25m-xl propella-bekko-a25m \
        --tier large --docs-per-s 186 --overall 0.811 \
        --mf1-deepseek 0.523 --mf1-glm 0.522 --train-minutes 211

The sheet itself lives in `docs/model_sheet.md` and is identical for every model: this
script only fills in the numbers, so the three published cards cannot drift apart in
structure or in the wording of the limitations.

Writes `runs/<run>/MODEL_CARD.md`, which scripts/publish.py uploads as README.md.
"""

import argparse
import json
from pathlib import Path

SHEET = Path("docs/model_sheet.md")

TIERS = {
    "static": dict(
        desc="206M parameters with no Transformer layers — 686 documents/s on one RTX 8000",
        backbone="[`bekko-static-embedding-v1`](https://huggingface.co/hotchpotch/bekko-static-embedding-v1) "
        "— static token embeddings pooled into one document vector, no attention",
        params="206M",
        max_length="8,192",
        tail_tokens="512",
    ),
    "small": dict(
        desc="106M parameters, 8M of them active — 348 documents/s on one RTX 8000",
        backbone="[`bekko-embedding-v1-a8m`](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m)",
        params="106M",
        max_length="1,024",
        tail_tokens="512",
    ),
    "large": dict(
        desc="123M parameters, 25M of them active — 186 documents/s on one RTX 8000",
        backbone="[`bekko-embedding-v1-a25m`](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m)",
        params="123M",
        max_length="1,024",
        tail_tokens="512",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("repo_name")
    ap.add_argument("--tier", required=True, choices=list(TIERS))
    ap.add_argument("--docs-per-s", required=True)
    ap.add_argument("--overall", required=True)
    ap.add_argument("--mf1-deepseek", required=True)
    ap.add_argument("--mf1-glm", required=True)
    ap.add_argument("--n-train", default="the training pool (403k documents)")
    ap.add_argument("--train-minutes", help="override when the run kept no results.json")
    a = ap.parse_args()

    run = Path(a.run)
    results = run / "results.json"
    if results.exists():
        minutes = f"{json.loads(results.read_text())['train_seconds'] / 60:.0f} minutes"
    else:
        minutes = f"{a.train_minutes} minutes" if a.train_minutes else "not recorded"
    tier = TIERS[a.tier]

    # Prefer what the run actually used over the tier defaults.
    model_cfg_path = run / "model" / "config.json"
    model_cfg = json.loads(model_cfg_path.read_text()) if model_cfg_path.exists() else {}
    max_length = (
        f"{model_cfg.get('max_length', tier['max_length']):,}"
        if isinstance(model_cfg.get("max_length"), int)
        else tier["max_length"]
    )
    tail_tokens = (
        f"{model_cfg.get('tail_tokens', tier['tail_tokens']):,}"
        if isinstance(model_cfg.get("tail_tokens"), int)
        else tier["tail_tokens"]
    )

    train_cfg_path = run / "train_config.json"
    train_cfg = json.loads(train_cfg_path.read_text()) if train_cfg_path.exists() else {}
    epochs = train_cfg.get("epochs")
    tokens = train_cfg.get("max_tokens_per_batch")
    if epochs and tokens:
        train_setup = f"{epochs} epochs, {tokens:,} tokens per batch"
    elif epochs:
        train_setup = f"{epochs} epochs"
    else:
        train_setup = "training setup not recorded"

    fields = {
        "repo_name": a.repo_name,
        "repo_id": f"danish-foundation-models/{a.repo_name}",
        "tier": a.tier,
        "tier_desc": tier["desc"],
        "backbone": tier["backbone"],
        "params": tier["params"],
        "max_length": max_length,
        "tail_tokens": tail_tokens,
        "train_setup": train_setup,
        "n_train": a.n_train,
        "train_minutes": minutes,
        "overall": a.overall,
        "docs_per_s": a.docs_per_s,
        "mf1_deepseek": a.mf1_deepseek,
        "mf1_glm": a.mf1_glm,
    }
    # Plain replacement rather than str.format: the sheet quotes bibtex, whose braces
    # would otherwise have to be escaped.
    card = SHEET.read_text()
    for key, value in fields.items():
        card = card.replace("{" + key + "}", str(value))
    leftover = [k for k in fields if "{" + k + "}" in card]
    assert not leftover, f"unfilled placeholders: {leftover}"

    out = run / "MODEL_CARD.md"
    out.write_text(card)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

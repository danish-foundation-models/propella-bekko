"""Generate the model sheet for a trained student.

    uv run python scripts/model_card.py runs/a25m-xl-3ep propella-bekko-a25m \
        --tier large --pareto-key a25m-xl --train-minutes 316

The sheet itself lives in `docs/model_sheet.md` and is identical for every model: this
script only fills in the numbers, so the three published cards cannot drift apart in
structure or in the wording of the limitations.

Every number in the card — throughput, agreement, parameters, and the rows for the other
two students — is read from `runs/pareto_vs_*.json`, the same files that feed the README
table and the Pareto figures. Nothing is retyped here.

Writes `models/<repo>/README.md` — the card is a file in this repository, in the directory
that mirrors the Hub repo, so it is edited and reviewed like any other document.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

from propella_bekko.release import PARETO, baseline_rows, load, params, student_rows

SHEET = Path("docs/model_sheet.md")

TIERS = {
    "static": dict(
        backbone="[`bekko-static-embedding-v1`](https://huggingface.co/hotchpotch/bekko-static-embedding-v1) "
        "— static token embeddings pooled into one document vector, no attention",
        max_length="8,192",
        tail_tokens="512",
    ),
    "small": dict(
        backbone="[`bekko-embedding-v1-a8m`](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m)",
        max_length="1,024",
        tail_tokens="512",
    ),
    "large": dict(
        backbone="[`bekko-embedding-v1-a25m`](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m)",
        max_length="1,024",
        tail_tokens="512",
    ),
}


def per_property_note(data: dict[str, dict[str, float]]) -> str:
    """Two sentences from runs/per_property_<run>.json: where the gap is, where there is none."""
    gaps = sorted(data.items(), key=lambda kv: kv[1]["student"] - kv[1]["teacher"])
    widest = ", ".join(
        f"`{name}` ({data[name]['student'] - data[name]['teacher']:+.2f})".replace("-", "\u2212")
        for name, _ in gaps[:3]
    )
    above = sorted(name for name, v in data.items() if v["student"] >= v["teacher"])
    at_or_above = (
        f"{len(above)} of {len(data)} are at or above propella-1-4b's own score: "
        + ", ".join(f"`{name}`" for name in above)
        + "."
        if above
        else "No property reaches propella-1-4b's own score."
    )
    text = (
        f"Most of the distance to the teacher sits in three properties — {widest} — "
        f"while {at_or_above}"
    )
    return textwrap.fill(text, width=100, break_on_hyphens=False, break_long_words=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("repo_name")
    ap.add_argument("--tier", required=True, choices=list(TIERS))
    ap.add_argument(
        "--pareto-key", required=True, help="key in runs/pareto_vs_*.json for this model"
    )
    ap.add_argument("--n-train", default="the training pool (403k documents)")
    ap.add_argument("--n-langs", default="20")
    ap.add_argument("--hardware", default="one RTX A6000", help="where docs/s was measured")
    ap.add_argument("--train-minutes", help="override when the run kept no results.json")
    a = ap.parse_args()

    data = load()
    mine = data[PARETO[0]].get(a.pareto_key)
    if mine is None:
        sys.exit(f"{a.pareto_key} not in runs/pareto_vs_{PARETO[0]}.json")

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
        f"{model_cfg['max_length']:,}"
        if isinstance(model_cfg.get("max_length"), int)
        else tier["max_length"]
    )
    tail_tokens = (
        f"{model_cfg['tail_tokens']:,}"
        if isinstance(model_cfg.get("tail_tokens"), int)
        else tier["tail_tokens"]
    )

    kind = (model_cfg.get("backbone_spec") or {}).get("kind", "hf")
    if kind == "hf":
        encoder_block = """# the encoder, with the fine-tuned weights from the checkpoint
backbone = AutoModel.from_pretrained(cfg["backbone"])
backbone.load_state_dict(
    {k.removeprefix("backbone.model."): v for k, v in state.items() if k.startswith("backbone.model.")}
)


def pool_text(input_ids, attention_mask):
    hidden = backbone(input_ids, attention_mask).last_hidden_state
    mask = attention_mask.unsqueeze(-1)
    return (hidden * mask).sum(1) / mask.sum(1)


dim = backbone.config.hidden_size"""
    else:
        encoder_block = """# the encoder: static token embeddings, mean-pooled. No Transformer layers.
spec = cfg["backbone_spec"]
emb = nn.EmbeddingBag(spec["num_embeddings"], spec["dim"], mode="mean", padding_idx=spec["pad_id"])
emb.weight.data.copy_(state["backbone.emb.weight"])


def pool_text(input_ids, attention_mask):
    return emb(input_ids.masked_fill(attention_mask == 0, spec["pad_id"]))


dim = spec["dim"]"""

    example_file = run / "example_output.json"
    if example_file.exists():
        example = json.loads(example_file.read_text())["output"]
        example_output = json.dumps(example, indent=4, ensure_ascii=False)
    else:
        example_output = "{}  # not recorded for this run"

    thresholds_note = (
        " `thresholds.json` in this repository holds thresholds tuned for macro-F1"
        " if you would rather have rare-label recall."
        if (run / "thresholds.json").exists()
        else ""
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

    per_prop_file = Path(f"runs/per_property_{run.name}.json")
    if not per_prop_file.exists():
        sys.exit(f"missing {per_prop_file}: run scripts/per_field.py --run {run.name} first")
    per_prop = json.loads(per_prop_file.read_text())

    fields = {
        "repo_id": f"danish-foundation-models/{a.repo_name}",
        "backbone": tier["backbone"],
        "params": params(mine),
        "max_length": max_length,
        "tail_tokens": tail_tokens,
        "train_setup": train_setup,
        "n_train": a.n_train,
        "n_langs": a.n_langs,
        "hardware": a.hardware,
        "title": a.repo_name[:1].upper() + a.repo_name[1:],
        "student_rows": "\n".join(student_rows(data, bold=a.repo_name)),
        "baseline_rows": "\n".join(baseline_rows(data)),
        "train_minutes": minutes,
        "thresholds_note": thresholds_note,
        "encoder_block": encoder_block,
        "example_output": example_output,
        "docs_per_s": f"{mine['docs_per_s']:.0f}",
        "per_property_note": per_property_note(per_prop),
    }
    # Plain replacement rather than str.format: the sheet quotes bibtex, whose braces
    # would otherwise have to be escaped.
    card = SHEET.read_text()
    for key, value in fields.items():
        card = card.replace("{" + key + "}", str(value))
    leftover = [k for k in fields if "{" + k + "}" in card]
    assert not leftover, f"unfilled placeholders: {leftover}"

    # the card is a file in this repository, in the directory that mirrors the Hub repo,
    # so it is edited and reviewed like any other document
    out = Path("models") / a.repo_name / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

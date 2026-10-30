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


# Students that were graded against the same two reference annotators (300 docs, macro-F1),
# with throughput on one Quadro RTX 8000. The row for the model being published is bolded.
STUDENTS = {
    "propella-bekko-a25m": ("123M", "186", "0.523", "0.522"),
    "propella-bekko-a8m": ("106M", "348", "0.489", "0.479"),
    "propella-bekko-static": ("206M", "686", "0.479", "0.468"),
}


def student_rows(repo_name: str) -> str:
    rows = []
    for name, (params, docs_per_s, ds, glm) in STUDENTS.items():
        label = f"**{name}**" if name == repo_name else name
        rows.append(f"| {label} | {params} | {docs_per_s} | {ds} | {glm} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("repo_name")
    ap.add_argument("--tier", required=True, choices=list(TIERS))
    ap.add_argument("--docs-per-s", required=True)
    ap.add_argument(
        "--overall", default=None, help="kept for older cards; the sheet no longer uses it"
    )
    ap.add_argument("--mf1-deepseek", required=True)
    ap.add_argument("--mf1-glm", required=True)
    ap.add_argument("--n-train", default="the training pool (403k documents)")
    ap.add_argument("--n-langs", default="20")
    ap.add_argument("--hardware", default="one Quadro RTX 8000", help="where docs/s was measured")
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
        "n_langs": a.n_langs,
        "hardware": a.hardware,
        "title": a.repo_name[:1].upper() + a.repo_name[1:],
        "student_rows": student_rows(a.repo_name),
        "train_minutes": minutes,
        "thresholds_note": thresholds_note,
        "encoder_block": encoder_block,
        "example_output": example_output,
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

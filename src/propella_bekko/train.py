"""Training loop (single GPU, fp16/bf16 autocast)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
import torch

from propella_bekko.encoding import Encoder
from propella_bekko.inference import Annotator, token_budget_batches
from propella_bekko.metrics import field_metrics, summarize
from propella_bekko.models import PropellaBekko, count_parameters, load_backbone
from propella_bekko.schema import PROPELLA_FIELDS, build_country_field

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data: str = "data/train.parquet"
    heldout: str | None = "data/heldout.parquet"
    val_frac: float = 0.02
    backbone: str = "jhu-clsp/mmBERT-small"
    output_dir: str = "runs/mmbert-small"
    max_length: int = 1024
    tail_tokens: int = 128
    max_tokens_per_batch: int = 32_768
    grad_accum: int = 1
    epochs: float = 1.0
    lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_train_docs: int | None = None
    max_eval_docs: int = 5_000
    eval_every: int = 1_000
    country_top_k: int = 64
    use_length_features: bool = True
    dtype: str = "fp16"  # fp16 | bf16 | fp32
    seed: int = 0
    gradient_checkpointing: bool = False
    static_dim: int | None = None  # Matryoshka truncation for static backbones
    head_hidden: int | None = None
    freeze_backbone: bool = False
    extra: dict = field(default_factory=dict)


def _is_val(row: dict, frac: float) -> bool:
    h = hashlib.sha1(f"{row['source']}/{row['id']}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 < frac


def _load_split(path: str, frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rows = pq.read_table(path).to_pylist()
    random.Random(seed).shuffle(rows)
    train = [r for r in rows if not _is_val(r, frac)]
    val = [r for r in rows if _is_val(r, frac)]
    return train, val


def build_fields(train_rows: list[dict], top_k: int):
    counts = Counter(c for r in train_rows for c in (r.get("country_relevance") or []))
    return [f for f in PROPELLA_FIELDS if f.name != "country_relevance"] + [
        build_country_field(counts, top_k)
    ]


def evaluate(model, tok, rows, cfg: TrainConfig, device) -> dict:
    model.eval()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.dtype]
    ann = Annotator(
        model,
        tok,
        cfg.max_length,
        cfg.tail_tokens,
        device=device,
        dtype=torch.float32,
        max_tokens=cfg.max_tokens_per_batch * 2,
    )
    with torch.autocast("cuda", dtype=dtype, enabled=cfg.dtype != "fp32"):
        preds = ann.annotate(rows)
    res = field_metrics(preds, rows, model.fields)
    model.train()
    return {"summary": summarize(res), "fields": res}


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = _load_split(cfg.data, cfg.val_frac, cfg.seed)
    train_rows = train_rows[: cfg.max_train_docs] if cfg.max_train_docs else train_rows
    val_rows = val_rows[: cfg.max_eval_docs]
    fields = build_fields(train_rows, cfg.country_top_k)
    log.info("train=%d val=%d", len(train_rows), len(val_rows))

    backbone, tok = load_backbone(cfg.backbone)
    if cfg.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
    model = PropellaBekko(backbone, fields, use_length_features=cfg.use_length_features).to(device)
    log.info("params: %.1fM", count_parameters(model) / 1e6)

    enc = Encoder(tok, cfg.max_length, cfg.tail_tokens)
    t0 = time.time()
    seqs, trunc = [], []
    for i in range(0, len(train_rows), 4096):
        s, tr = enc.ids([r["text"] for r in train_rows[i : i + 4096]])
        seqs += s
        trunc += tr
    log.info(
        "tokenised %d docs in %.0fs (mean len %.0f)",
        len(seqs),
        time.time() - t0,
        sum(map(len, seqs)) / len(seqs),
    )

    def epoch_batches(ep: int):
        rng = random.Random(cfg.seed + ep)
        # shuffle in mega-chunks, then length-bucket inside each chunk
        idx = list(range(len(seqs)))
        rng.shuffle(idx)
        batches = []
        for c in range(0, len(idx), 50_000):
            chunk = idx[c : c + 50_000]
            for b in token_budget_batches([len(seqs[i]) for i in chunk], cfg.max_tokens_per_batch):
                batches.append([chunk[j] for j in b])
        rng.shuffle(batches)
        return batches

    steps_per_epoch = len(epoch_batches(0)) // cfg.grad_accum
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    warmup = int(total_steps * cfg.warmup_ratio)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": cfg.lr,
            },
            {"params": head_params, "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        s = (
            step / max(1, warmup)
            if step < warmup
            else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        )
        for g, b in zip(opt.param_groups, base_lrs, strict=False):
            g["lr"] = b * s

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.dtype]
    scaler = torch.amp.GradScaler(enabled=cfg.dtype == "fp16")
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    step, ep, history = 0, 0, []
    model.train()
    t_start = time.time()
    while step < total_steps:
        batches = epoch_batches(ep)
        for bi, b in enumerate(batches):
            rows = [train_rows[i] for i in b]
            batch = enc.collate(rows, [seqs[i] for i in b], [trunc[i] for i in b], fields)
            targets = {k: v.to(device) for k, v in batch.pop("targets").items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=dtype, enabled=cfg.dtype != "fp32"):
                logits = model(**batch)
            loss, _ = model.loss(logits, targets)
            scaler.scale(loss / cfg.grad_accum).backward()
            if (bi + 1) % cfg.grad_accum:
                continue
            set_lr(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                log.info(
                    "step %d/%d loss %.4f (%.1f steps/s)",
                    step,
                    total_steps,
                    loss.item(),
                    step / (time.time() - t_start),
                )
            if step % cfg.eval_every == 0 or step == total_steps:
                r = evaluate(model, tok, val_rows, cfg, device)
                r["step"] = step
                history.append({"step": step, **r["summary"]})
                log.info("eval @%d %s", step, r["summary"])
            if step >= total_steps:
                break
        ep += 1

    model.save(
        out / "model",
        tok,
        extra_config={
            "max_length": cfg.max_length,
            "tail_tokens": cfg.tail_tokens,
            "backbone": cfg.backbone,
        },
    )
    final = evaluate(model, tok, val_rows, cfg, device)
    result = {
        "config": asdict(cfg),
        "history": history,
        "val": final,
        "train_seconds": time.time() - t_start,
    }
    if cfg.heldout and Path(cfg.heldout).exists():
        heldout_rows = pq.read_table(cfg.heldout).to_pylist()
        result["heldout"] = evaluate(model, tok, heldout_rows, cfg, device)
        log.info("held-out %s", result["heldout"]["summary"])
    (out / "results.json").write_text(json.dumps(result, indent=2))
    return result

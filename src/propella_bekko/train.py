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

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file

from propella_bekko.encoding import Encoder
from propella_bekko.inference import Annotator, token_budget_batches
from propella_bekko.metrics import field_metrics, summarize
from propella_bekko.models import PropellaBekko, count_parameters, load_backbone
from propella_bekko.schema import PROPELLA_FIELDS, build_country_field, encode_labels

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
    static_dim: int | None = None
    static_pool: str = "mean"  # mean | attn (learned token weighting for static backbones)
    pos_weight: bool = False  # inverse-frequency positive weights for multi-select BCE
    label_smoothing: float = 0.0
    ordinal_smoothing: float = 0.0  # spread ordinal targets onto neighbouring levels
    max_domain_frac: float | None = None  # cap the share of any single domain in training
    max_lang_frac: float | None = None  # cap the share of any single language in training
    tune_thresholds: bool = True  # tune multi-select thresholds on val (macro-F1 target)
    save_every: int = 0  # save a trajectory checkpoint every N optimizer steps
    swa: int = 0  # average the last K trajectory checkpoints before the final save
    swa_finetune: int = 0  # low-LR steps after averaging
    init_from: str | None = None  # start from these weights (e.g. an averaged checkpoint)
    head_hidden: int | None = None
    refine: bool = False  # label-conditional second pass
    rich_length: bool = False  # rubric word-count thresholds as features
    label_init: bool = False  # start output rows at the label names' own embeddings
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


def _cap_groups(rows: list[dict], key: str, frac: float, seed: int) -> list[dict]:
    """Subsample the biggest groups (by `key`) so none exceeds `frac` of the training set."""
    from collections import defaultdict

    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r.get(key) or "?"].append(r)
    cap = int(frac * len(rows))
    rng = random.Random(seed)
    out = []
    for name, group in sorted(by.items()):
        if len(group) > cap:
            log.info("%s %s: %d -> %d rows", key, name, len(group), cap)
            group = rng.sample(group, cap)
        out.extend(group)
    rng.shuffle(out)
    return out


def build_fields(train_rows: list[dict], top_k: int):
    counts = Counter(c for r in train_rows for c in (r.get("country_relevance") or []))
    return [f for f in PROPELLA_FIELDS if f.name != "country_relevance"] + [
        build_country_field(counts, top_k)
    ]


def evaluate(model, tok, rows, cfg: TrainConfig, device, thresholds=None) -> dict:
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
        threshold=thresholds or 0.5,
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
    if cfg.max_domain_frac:
        train_rows = _cap_groups(train_rows, "domain", cfg.max_domain_frac, cfg.seed)
    if cfg.max_lang_frac:
        train_rows = _cap_groups(train_rows, "lang", cfg.max_lang_frac, cfg.seed)
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
    # tokenised sequences are kept as one flat int32 array (lists of ints cost ~30x more memory)
    chunks, lens, trunc = [], [], []
    for i in range(0, len(train_rows), 4096):
        s, tr = enc.ids([r["text"] for r in train_rows[i : i + 4096]])
        flat = np.fromiter((t for seq in s for t in seq), dtype=np.int32)
        chunks.append(flat)
        lens.extend(len(seq) for seq in s)
        trunc.extend(tr)
    tokens = np.concatenate(chunks)
    lens = np.asarray(lens, dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    del chunks
    trunc = np.asarray(trunc, dtype=bool)
    log.info(
        "tokenised %d docs in %.0fs (mean len %.0f, %.1f tokens total)",
        len(lens),
        time.time() - t0,
        lens.mean(),
        tokens.size,
    )

    def seq(i):
        return tokens[offsets[i] : offsets[i + 1]]

    def epoch_batches(ep: int):
        rng = random.Random(cfg.seed + ep)
        # shuffle in mega-chunks, then length-bucket inside each chunk
        idx = list(range(len(lens)))
        rng.shuffle(idx)
        batches = []
        for c in range(0, len(idx), 50_000):
            chunk = idx[c : c + 50_000]
            for b in token_budget_batches([int(lens[i]) for i in chunk], cfg.max_tokens_per_batch):
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

    pw = None
    if cfg.pos_weight:
        counts = {f.name: torch.zeros(f.num_labels) for f in fields if f.kind == "multi"}
        for r in train_rows:
            for name, hot in encode_labels(r, fields).items():
                if name in counts:
                    counts[name] += torch.tensor(hot).clamp(min=0)
        n = len(train_rows)
        pw = {
            name: ((n - c).clamp(min=1) / c.clamp(min=1)).clamp(1.0, 20.0).to(device)
            for name, c in counts.items()
        }
        log.info("pos_weight: %s", {k: [round(x, 2) for x in v.tolist()] for k, v in pw.items()})

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
            batch = enc.collate(rows, [seq(i) for i in b], trunc[b].tolist(), fields)
            targets = {k: v.to(device) for k, v in batch.pop("targets").items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=dtype, enabled=cfg.dtype != "fp32"):
                logits = model(**batch)
            loss, _ = model.loss(
                logits, targets, pos_weight=pw, label_smoothing=cfg.label_smoothing
            )
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
            if cfg.save_every and step % cfg.save_every == 0:
                ck = out / "ckpt"
                ck.mkdir(exist_ok=True)
                save_file(
                    {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()},
                    str(ck / f"step{step:07d}.safetensors"),
                )
                for old in sorted(ck.glob("step*.safetensors"))[: -max(cfg.swa, 1) - 1]:
                    old.unlink()
            if step % cfg.eval_every == 0 or step == total_steps:
                r = evaluate(model, tok, val_rows, cfg, device)
                r["step"] = step
                history.append({"step": step, **r["summary"]})
                log.info("eval @%d %s", step, r["summary"])
            if step >= total_steps:
                break
        ep += 1

    if cfg.swa:
        cks = sorted((out / "ckpt").glob("step*.safetensors"))[-cfg.swa :]
        if cks:
            avg = {}
            for c in cks:
                sd = load_file(str(c))
                for k, v in sd.items():
                    avg[k] = avg.get(k, 0) + v / len(cks)
            model.load_state_dict(avg, strict=False)
            log.info("SWA over %d checkpoints: %s", len(cks), [c.name for c in cks])

    model.save(
        out / "model",
        tok,
        extra_config={
            "max_length": cfg.max_length,
            "tail_tokens": cfg.tail_tokens,
            "backbone": cfg.backbone,
        },
    )
    if cfg.tune_thresholds:
        from propella_bekko.tune import tune

        th = tune(model, tok, cfg, val_rows, fields)
        (out / "thresholds.json").write_text(json.dumps(th, indent=2))
        log.info("tuned thresholds: %s", th)
        model.eval()

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
        if cfg.tune_thresholds:
            th = json.loads((out / "thresholds.json").read_text())
            result["heldout_tuned"] = evaluate(model, tok, heldout_rows, cfg, device, thresholds=th)
            log.info("held-out (tuned thresholds) %s", result["heldout_tuned"]["summary"])
        log.info("held-out %s", result["heldout"]["summary"])
    (out / "results.json").write_text(json.dumps(result, indent=2))
    return result

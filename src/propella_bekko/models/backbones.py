"""Backbones: Transformer encoders and static (Transformer-free) embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

BEKKO_PREFIX = "bekko:"  # ModernBERT backbone inside a Bekko System One checkpoint
STATIC_PREFIX = "static:"  # static (Transformer-free) embedding, e.g. "static:hotchpotch/bekko-static-embedding-v1"


class Backbone(nn.Module):
    """Maps (input_ids, attention_mask) -> pooled document vector [B, hidden_size]."""

    kind: str = ""
    hidden_size: int

    def spec(self) -> dict:
        raise NotImplementedError


class HFBackbone(Backbone):
    """Transformer encoder + masked mean pooling."""

    kind = "hf"

    def __init__(self, model: Any):
        super().__init__()
        self.model = model
        self.hidden_size = int(model.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        hs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        return (hs * m).sum(1) / m.sum(1).clamp(min=1.0)

    def spec(self) -> dict:
        return {"kind": self.kind, "config": self.model.config.to_dict()}


class StaticBackbone(Backbone):
    """Static token embeddings + mean pooling (no attention).

    Cost is linear in tokens with a tiny constant, so whole documents can be
    embedded instead of a truncated prefix.
    """

    kind = "static"

    def __init__(self, num_embeddings: int, dim: int, pad_id: int):
        super().__init__()
        self.emb = nn.EmbeddingBag(num_embeddings, dim, mode="mean", padding_idx=pad_id)
        self.hidden_size = dim
        self.pad_id = pad_id

    def forward(self, input_ids, attention_mask):
        ids = input_ids.masked_fill(attention_mask == 0, self.pad_id)
        return self.emb(ids)

    def spec(self) -> dict:
        n, d = self.emb.weight.shape
        return {"kind": self.kind, "num_embeddings": n, "dim": d, "pad_id": self.pad_id}


def backbone_from_spec(spec: dict, attn_implementation: str = "sdpa") -> Backbone:
    if spec["kind"] == "static":
        return StaticBackbone(spec["num_embeddings"], spec["dim"], spec["pad_id"])
    cfg = dict(spec["config"])
    config = AutoConfig.for_model(cfg.pop("model_type"), **cfg)
    return HFBackbone(AutoModel.from_config(config, attn_implementation=attn_implementation))


def _load_static(repo: str, dim: int | None = None):
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_file=hf_hub_download(repo, "tokenizer.json"),
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
    )
    w = load_file(hf_hub_download(repo, "model.safetensors"))["embedding.weight"].float()
    if dim:  # Matryoshka truncation
        w = w[:, :dim].contiguous()
    bb = StaticBackbone(w.shape[0], w.shape[1], tok.pad_token_id)
    with torch.no_grad():
        bb.emb.weight.copy_(w)
    return bb, tok


def load_backbone(name: str, attn_implementation: str = "sdpa", static_dim: int | None = None):
    """Load a backbone + tokenizer.

    * `static:<repo>` - static embedding table (sentence-transformers StaticEmbedding
      layout, e.g. hotchpotch/bekko-static-embedding-v1), optionally truncated
      to `static_dim` dimensions.
    * `bekko:<repo>` - ModernBERT backbone of a Bekko System One checkpoint.
    * anything else - `AutoModel.from_pretrained(name)` (e.g. hotchpotch/bekko-embedding-v1-a8m).
    """
    if name.startswith(STATIC_PREFIX):
        return _load_static(name[len(STATIC_PREFIX) :], static_dim)
    if name.startswith(BEKKO_PREFIX):
        from huggingface_hub import hf_hub_download, snapshot_download

        repo = name[len(BEKKO_PREFIX) :]
        cfg = json.loads(
            Path(hf_hub_download(repo, "0_BekkoInference/backbone_config.json")).read_text()
        )
        config = AutoConfig.for_model(cfg.pop("model_type"), **cfg)
        model = AutoModel.from_config(config, attn_implementation=attn_implementation)
        sd = load_file(hf_hub_download(repo, "0_BekkoInference/model.safetensors"))
        prefix = "encoder.backbone."
        sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "rotary" not in m]
        if missing or unexpected:
            raise RuntimeError(
                f"bekko backbone mismatch: missing={missing} unexpected={unexpected}"
            )
        tok_dir = snapshot_download(repo, allow_patterns=["0_BekkoInference/tokenizer/*"])
        tok = AutoTokenizer.from_pretrained(Path(tok_dir) / "0_BekkoInference" / "tokenizer")
        return HFBackbone(model), tok
    model = AutoModel.from_pretrained(
        name, attn_implementation=attn_implementation, trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return HFBackbone(model), tok

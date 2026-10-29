"""Multi-head model that predicts all propella-1 properties in one forward pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoTokenizer

from propella_bekko.models.backbones import Backbone, backbone_from_spec
from propella_bekko.schema import Field, fields_from_config, fields_to_config


def length_features(
    n_chars: torch.Tensor, n_words: torch.Tensor, truncated: torch.Tensor, rich: bool = False
) -> torch.Tensor:
    """Cheap document-level features the encoder cannot see after truncation.

    `rich` adds the rubric's own word-count thresholds as indicators: the teacher's
    `content_length` is defined by them (100 / 500 / 2000 words), and inspecting the
    errors showed the model under-rates long documents without them.
    """
    w = n_words.float()
    if not rich:
        return torch.stack(
            [torch.log1p(n_chars.float()) / 10.0, torch.log1p(w) / 10.0, truncated.float()],
            dim=-1,
        )
    base = [
        torch.log10(w.clamp(min=1)) / 6.0,
        torch.log10(n_chars.float().clamp(min=1)) / 7.0,
        truncated.float(),
    ]
    return torch.stack([*base, *[(w > t).float() for t in (100, 500, 2000, 5000)]], dim=-1)


N_LENGTH_FEATURES = 3
N_LENGTH_FEATURES_RICH = 7


class PropellaBekko(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        fields: list[Field],
        dropout: float = 0.1,
        use_length_features: bool = True,
        head_hidden: int | None = None,
        refine: bool = False,
        rich_length: bool = False,
        ordinal_decode: str = "expected",
    ):
        super().__init__()
        self.backbone: Backbone = backbone
        self.fields = fields
        self.use_length_features = use_length_features
        self.rich_length = rich_length
        self.ordinal_decode = ordinal_decode
        d = backbone.hidden_size
        h = head_hidden or d
        self.head_hidden = h
        n_len = (
            (N_LENGTH_FEATURES_RICH if rich_length else N_LENGTH_FEATURES)
            if use_length_features
            else 0
        )
        extra = n_len
        self.proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d + extra, h), nn.GELU(), nn.LayerNorm(h)
        )
        self.heads = nn.ModuleDict({f.name: nn.Linear(h, f.num_labels) for f in fields})
        # multi-select label-count head: the model otherwise under-predicts the number of labels
        # second pass: let every field see the first-pass distribution of all fields
        # (labels are strongly correlated, e.g. quality <-> reasoning <-> educational value)
        self.refine = refine
        if refine:
            n_out = sum(f.num_labels for f in fields)
            self.refiner = nn.Sequential(
                nn.Linear(h + n_out, h),
                nn.GELU(),
                nn.Linear(h, n_out),
            )

    def forward(self, input_ids, attention_mask, n_chars=None, n_words=None, truncated=None, **_):
        pooled = self.backbone(input_ids, attention_mask)
        if self.use_length_features:
            assert n_chars is not None and n_words is not None and truncated is not None
            pooled = torch.cat(
                [
                    pooled,
                    length_features(n_chars, n_words, truncated, self.rich_length).to(pooled.dtype),
                ],
                -1,
            )
        z = self.proj(pooled)
        logits = {name: head(z) for name, head in self.heads.items()}
        if not self.refine:
            return logits
        feats = [z]
        for f in self.fields:
            lg = logits[f.name].float()
            feats.append(lg.sigmoid() if f.kind == "multi" else lg.softmax(-1))
        delta = self.refiner(torch.cat(feats, -1)).to(z.dtype)
        out, i = {}, 0
        for f in self.fields:
            out[f.name] = logits[f.name] + delta[:, i : i + f.num_labels]
            i += f.num_labels
        return out

    # ------------------------------------------------------------------ #
    def loss(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
        pos_weight: dict[str, torch.Tensor] | None = None,
        label_smoothing: float = 0.0,
        ordinal_smoothing: float = 0.0,
    ):
        total, parts = 0.0, {}
        for f in self.fields:
            lg, y = logits[f.name].float(), targets[f.name]
            if f.kind == "single":
                if (y != -100).sum() == 0:
                    continue
                if ordinal_smoothing and f.ordinal:
                    # spread a little mass onto neighbouring levels: the teacher's
                    # ordinal judgements are noisy at the boundaries
                    soft = F.one_hot(y.clamp(min=0), f.num_labels).float()
                    shift = (
                        torch.roll(soft, 1, dims=1) + torch.roll(soft, -1, dims=1)
                    ) / 2 * ordinal_smoothing + soft * (1 - ordinal_smoothing)
                    tgt = ((1 - label_smoothing) * shift + label_smoothing / f.num_labels).clamp(
                        min=0
                    )
                    l = -(tgt * F.log_softmax(lg, -1)).sum(-1)
                    valid = y != -100
                    l = l[valid].mean()
                else:
                    l = F.cross_entropy(lg, y, ignore_index=-100, label_smoothing=label_smoothing)
            else:
                valid = y[:, 0] != -100
                if valid.sum() == 0:
                    continue
                l = F.binary_cross_entropy_with_logits(
                    lg[valid],
                    y[valid].float(),
                    pos_weight=None if pos_weight is None else pos_weight[f.name],
                )
            w = (weights or {}).get(f.name, 1.0)
            total = total + w * l
            parts[f.name] = l.detach()
        return total, parts

    @torch.no_grad()
    def init_labels_from_text(self, tokenizer: Any, max_length: int = 24) -> None:
        """Initialise classification rows from the label names' own embeddings.

        GLiClass feeds label tokens through the encoder and scores the document against them;
        the cheap version of that idea is to start each output row at the pooled embedding of
        the label's name, so rare classes (a country name, a niche sector) begin from their
        meaning instead of from noise. Trainable afterwards like any other head.
        """
        self.eval()
        heads: Any = self.heads
        for f in self.fields:
            names = [lab.replace("_", " ") for lab in f.labels]
            enc = tokenizer(
                names, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)
            pooled = self.backbone(enc["input_ids"], enc["attention_mask"])
            w = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            heads[f.name].weight.copy_(w)
            heads[f.name].bias.zero_()
            del enc, pooled, w

    @torch.no_grad()
    def decode(
        self, logits: dict[str, torch.Tensor], threshold: float | dict[str, float] = 0.5
    ) -> list[dict]:
        """Logits -> propella-style annotation dicts (string labels)."""
        bs = next(iter(logits.values())).shape[0]
        rows: list[dict] = [{} for _ in range(bs)]
        for f in self.fields:
            lg = logits[f.name].float()
            if f.kind == "single":
                if f.ordinal and self.ordinal_decode == "expected":
                    # decode the ordinal scale in expectation, not by argmax: the metric is
                    # quadratic-weighted kappa, which rewards being close (System-One 'Score')
                    p = lg.softmax(-1)
                    ranks = torch.arange(f.num_labels, device=lg.device, dtype=p.dtype)
                    exp = (p * ranks).sum(-1).round().clamp(0, f.num_labels - 1).long()
                    idx = exp.tolist()
                else:
                    idx = lg.argmax(-1).tolist()
                for r, i in zip(rows, idx, strict=False):
                    r[f.name] = f.labels[i]
            else:
                p = torch.sigmoid(lg)
                top = p.argmax(-1)
                thr = threshold.get(f.name, 0.5) if isinstance(threshold, dict) else threshold
                hot = p >= thr
                hot[torch.arange(bs), top] = True  # propella always emits >=1 value
                for r, h in zip(rows, hot.tolist(), strict=False):
                    r[f.name] = [lab for lab, on in zip(f.labels, h, strict=False) if on]
        return rows

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, tokenizer, extra_config: dict | None = None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path / "tokenizer")
        sd = {k: v.contiguous() for k, v in self.state_dict().items()}
        # tied embeddings would be duplicated; safetensors refuses shared storage
        save_file({k: v.clone() for k, v in sd.items()}, str(path / "model.safetensors"))
        cfg = {
            "fields": fields_to_config(self.fields),
            "rich_length": self.rich_length,
            "ordinal_decode": self.ordinal_decode,
            "use_length_features": self.use_length_features,
            "head_hidden": self.head_hidden,
            "refine": self.refine,
            "backbone_spec": self.backbone.spec(),
            **(extra_config or {}),
        }
        (path / "config.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", attn_implementation: str = "sdpa"):
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        backbone = backbone_from_spec(cfg["backbone_spec"], attn_implementation)
        model = cls(
            backbone,
            fields_from_config(cfg["fields"]),
            use_length_features=cfg["use_length_features"],
            head_hidden=cfg.get("head_hidden"),
            refine=cfg.get("refine", False),
            rich_length=cfg.get("rich_length", False),
            ordinal_decode=cfg.get("ordinal_decode", "expected"),
        )
        missing, unexpected = model.load_state_dict(
            load_file(str(path / "model.safetensors")), strict=False
        )
        missing = [m for m in missing if "rotary" not in m]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        tok = AutoTokenizer.from_pretrained(path / "tokenizer")
        return model.to(device).eval(), tok, cfg


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

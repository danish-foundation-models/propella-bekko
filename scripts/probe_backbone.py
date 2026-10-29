"""Probe a model's backbone interface: forward pass, hidden size, tokenizer specials."""

import sys

import torch
from transformers import AutoModel, AutoTokenizer

name = sys.argv[1] if len(sys.argv) > 1 else "VAGOsolutions/SauerkrautLM-MoEdernColBERT-115M"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True, dtype=torch.float32).cuda().eval()
print("class:", type(model).__name__, "hidden:", getattr(model.config, "hidden_size", None))
print(
    "specials: cls",
    tok.cls_token_id,
    "sep",
    tok.sep_token_id,
    "pad",
    tok.pad_token_id,
    "vocab",
    len(tok),
)
texts = ["Hej verden, dette er en dansk test.", "Det här är en svensk mening om vägarbete."]
enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model(**enc)
print("keys:", list(out.keys()) if hasattr(out, "keys") else type(out))
h = out.last_hidden_state
print("last_hidden_state:", tuple(h.shape), h.dtype, "finite:", bool(torch.isfinite(h).all()))
m = enc["attention_mask"].unsqueeze(-1).float()
pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
print("pooled:", tuple(pooled.shape), "norm:", pooled.norm(dim=-1).tolist())

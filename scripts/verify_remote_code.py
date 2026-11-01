"""Check that the published repo loads through `transformers` and matches the training package.

    uv run python scripts/verify_remote_code.py [repo_id]

Loads the model twice — once through `AutoModel.from_pretrained(..., trust_remote_code=True)`
from the Hub, once through `propella_bekko.PropellaBekko.load` from the same snapshot — and
compares logits and decoded annotations on one long document.
"""

import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from propella_bekko.models import PropellaBekko

REPO = sys.argv[1] if len(sys.argv) > 1 else "danish-foundation-models/propella-bekko-a25m"
DOC = (
    "TECHNISCH EN ECONOMISCH POTENTIEEL VOOR EEN AARDGASVRIJE GEBOUWDE OMGEVING IN DE REGIO "
    "DRECHTSTEDEN. De gemeenten in de regio Drechtsteden hebben de ambitie om voor 2050 een "
    "aardgasvrije gebouwde omgeving te realiseren. In dit rapport wordt het technisch en "
    "economisch potentieel beschreven van verschillende maatregelen: isolatie, warmtepompen, "
    "restwarmte en groengas. "
) * 40

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True, dtype=torch.float32).eval()
print("auto :", type(model).__name__, sum(p.numel() for p in model.parameters()) // 10**6, "M")

path = snapshot_download(REPO)
ref, _, _ = PropellaBekko.load(path, device="cpu")
print("pkg  :", type(ref).__name__, sum(p.numel() for p in ref.parameters()) // 10**6, "M")

out = model.annotate(tok, [DOC])[0]

with torch.inference_mode():
    enc = model._encode(tok, [DOC])
    a = model(**enc)
    b = ref(**enc)
worst = max(float((a[k] - b[k]).abs().max()) for k in a)
print("logits identical:", worst < 1e-4, "| max |diff|:", f"{worst:.2e}")
print("annotations identical:", out == ref.decode(b)[0])
print({k: out[k] for k in list(out)[:5]})

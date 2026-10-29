"""Generate a model card for a trained student.

    uv run python scripts/model_card.py runs/a25m-xl propella-bekko-a25m \
        --docs-per-s 186 --overall 0.811 --tier large

Writes `runs/<run>/MODEL_CARD.md`, which scripts/publish.py uploads as README.md.
"""

import argparse
import json
from pathlib import Path

TIERS = {
    "static": "no Transformer layers — 686 documents/s",
    "small": "8M active parameters (bekko-embedding-v1-a8m) — 348 documents/s",
    "large": "25M active parameters (bekko-embedding-v1-a25m) — 186 documents/s",
}

TEMPLATE = """---
license: mit
language:
- multilingual
- da
- sv
- no
- is
- fo
- nl
- de
- fi
- fr
- it
- es
- pl
- cs
- hu
- ro
- pt
- en
library_name: pytorch
pipeline_tag: text-classification
tags:
- propella
- data-curation
- document-annotation
- distillation
---

# {repo_name}

A fast document annotator for the [propella-1](https://huggingface.co/ellamind/propella-1-4b)
label space: it predicts 17 document properties (content quality, educational value, content
type, business sector, safety, PII, regional and country relevance, ...) in **one forward pass**,
without generating text.

This is the **{tier}** model of the family: {tier_desc}.

* Training data: {n_train} documents annotated by propella-1-4b (Dynawords + FineWeb-2 + FinePDFs)
* Held-out score: **{overall}** (propella-1 overall score: 11/17 QWK + 1/17 PII-F1 + 5/17 IoU),
  against stored propella-1-4b labels on 1,000 held-out documents
  (propella-1-4b itself: 0.875, propella-1-1.7b: 0.808, propella-1-0.6b: 0.763)
* Throughput: **{docs_per_s} documents/second** on one Quadro RTX 8000
* Repository: https://github.com/danish-foundation-models/propella-bekko

## Usage

```python
from huggingface_hub import snapshot_download
from propella_bekko.models import PropellaBekko
from propella_bekko.inference import Annotator

path = snapshot_download("{repo_id}")
model, tokenizer, cfg = PropellaBekko.load(path)
annotator = Annotator(model, tokenizer, cfg["max_length"], cfg["tail_tokens"])

rows = [{{"id": "doc1", "text": "..."}}]
[{ "id": "doc1", "content_quality": "good", "educational_value": "high", ...}]
print(annotator.annotate(rows))
```

Multi-select properties are decoded at a sigmoid threshold of 0.5
(`thresholds.json` in this repo holds thresholds tuned for macro-F1 if you prefer rare-label
recall).

## Not predicted

`one_sentence_description` (free text) is excluded — it cannot be produced by a classifier.

## Citation

See https://github.com/danish-foundation-models/propella-bekko and
[propella-1](https://arxiv.org/abs/2602.12414).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("repo_name")
    ap.add_argument("--docs-per-s", required=True)
    ap.add_argument("--overall", required=True)
    ap.add_argument("--tier", required=True, choices=list(TIERS))
    a = ap.parse_args()

    run = Path(a.run)
    res = json.loads((run / "results.json").read_text())
    cfg = res["config"]
    n_train = "the training pool"
    data = Path(cfg["data"])
    card = TEMPLATE.format(
        repo_name=a.repo_name,
        repo_id=f"danish-foundation-models/{a.repo_name}",
        tier=a.tier,
        tier_desc=TIERS[a.tier],
        n_train=n_train,
        overall=a.overall,
        docs_per_s=a.docs_per_s,
    )
    out = run / "MODEL_CARD.md"
    out.write_text(card)
    print(f"wrote {out} (data: {data.name})")


if __name__ == "__main__":
    main()

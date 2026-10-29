---
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
label space. It predicts the 17 document properties of propella-1 — content quality, information
density, educational value, content type, business sector, safety, PII, regional and country
relevance, and the rest — in **one forward pass**, without generating text.

This is the **{tier}** model of the family: {tier_desc}.

## What it predicts

Seventeen properties, one head each:

| group | properties |
|---|---|
| ordinal, 11 | `content_integrity`, `content_ratio`, `content_length`, `content_quality`, `information_density`, `educational_value`, `reasoning_indicators`, `audience_level`, `commercial_bias`, `time_sensitivity`, `content_safety` |
| multi-label, 5 | `content_type`, `business_sector`, `technical_content`, `regional_relevance`, `country_relevance` |
| binary, 1 | `pii_presence` (scored as F1) |
| not predicted | `one_sentence_description` — free text, which a classifier cannot produce |

## Architecture

* Backbone: {backbone}
* Heads: one per property, softmax for single-select and sigmoid for multi-select, no
  cross-property conditioning — the properties are predicted independently from one shared
  document state.
* Input: {max_length} tokens of the document head plus {tail_tokens} tokens of its tail, so the
  end of a long document is not lost; document-level length features are passed to the heads.
* Decoding: ordinal properties are decoded to the expected level
  (`round(Σ p_i · rank_i)`) rather than by `argmax`, which is worth about +0.013 QWK on its own.

## Training

* Annotations: propella-1-4b labels joined to the Dynaword corpora, FineWeb-2 and FinePDFs.
* Documents: {n_train}
* Time: {train_minutes} on one Quadro RTX 8000 (3 epochs, batch 32).
* Objective: cross-entropy for single-select, binary cross-entropy for multi-select, token-level
  F1 loss for PII. No distillation loss beyond the labels themselves — the labels *are* the
  distillation.

## Results

Agreement with **DeepSeek-V4.1-Flash** and **GLM-5.3-Flash** annotations on the same 300 held-out
documents (macro-F1 over all properties). Chance — always predicting the most frequent label — is
0.123.

| model | params | docs/s | vs DeepSeek | vs GLM-5.3 |
|---|---|---|---|---|
| {repo_name} | {params} | {docs_per_s} | {mf1_deepseek} | {mf1_glm} |
| propella-1-0.6b | 600M | 0.14 | 0.498 | 0.501 |
| propella-1-1.7b | 1.7B | 0.14 | 0.547 | 0.534 |
| propella-1-4b | 4B | 0.06 | 0.571 | 0.549 |
| GLM-5.3-Flash | — | 0.41 | 0.659 | — |
| DeepSeek-V4.1-Flash | — | 0.34 | — | 0.659 |

Overall propella score (11/17 mean QWK + 1/17 PII-F1 + 5/17 mean IoU) against the stored
propella-1-4b labels on 1,000 held-out documents: **{overall}** — for reference, propella-1-4b
re-run scores 0.875 on its own labels, propella-1-1.7b 0.808 and propella-1-0.6b 0.763.

## Usage

```python
from huggingface_hub import snapshot_download
from propella_bekko.models import PropellaBekko
from propella_bekko.inference import Annotator

path = snapshot_download("{repo_id}")
model, tokenizer, cfg = PropellaBekko.load(path)
annotator = Annotator(model, tokenizer, cfg["max_length"], cfg["tail_tokens"])

rows = [{{"id": "doc1", "text": "..."}}]
print(annotator.annotate(rows))
```

Multi-select properties are decoded at a sigmoid threshold of 0.5. `thresholds.json` in this
repository holds thresholds tuned for macro-F1 if you would rather have rare-label recall.

## Limitations

* It reproduces **propella-1-4b's judgement**, including its mistakes: it is a distillation of one
  teacher's labels, not an independent reading of the document. Where the teacher is wrong
  systematically, so is this model.
* Agreement is measured against that teacher. A model can agree with propella-1-4b more closely
  than DeepSeek does and still be the less accurate annotator of the underlying document.
* Property values come from a fixed taxonomy; a document that does not fit the taxonomy gets the
  closest available label rather than a new one.
* Trained on documents of up to {max_length} tokens of head plus {tail_tokens} of tail; a document
  longer than that is truncated, and length-dependent properties are conditioned on the length
  features rather than on the full text.
* Calibration is not trained: the per-property distributions are usable for ranking, but the
  probabilities are not guaranteed to be calibrated against human judgement.

## Licence and citation

MIT for the model and the code. The labels derive from propella-1-4b and the Dynaword corpora; cite
[propella-1](https://arxiv.org/abs/2602.12414) and
[propella-bekko](https://github.com/danish-foundation-models/propella-bekko) if you use it.

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

<p align="center">
  <img src="banner.svg" width="560" alt="a ki-bekko koi with a propeller tail" />
</p>

# {title}

[propella-1](https://huggingface.co/ellamind/propella-1-4b) is a family of document annotators from
ellamind. Given a document, a propella-1 model returns 18 properties of it — how complete and
readable it is, how dense, educational or safe, what kind of document it is, which business sector
and region it belongs to — as JSON, without a task-specific prompt.

**{title}** distils the largest of them, propella-1-4b, into a model small enough to run over
whole corpora: the same label space, predicted in one forward pass, without generating text. It has
{params} parameters and annotates {docs_per_s} documents per second on one Quadro RTX 8000.

<p align="center">
  <img src="pareto_combined.svg" width="780" alt="annotation quality against throughput, against two independent annotators" />
</p>

Agreement with two annotators that are not the teacher — DeepSeek-V4.1-Flash on the left,
GLM-5.3-Flash on the right — against throughput. Both were asked to annotate the same 300 held-out documents with
propella's own prompt and schema.

## What it predicts

One head per property, 17 of them predicted. Values are the label names used by propella-1; the
levels and their definitions are in the
[property reference](https://huggingface.co/ellamind/propella-1-4b/blob/main/property_descriptions.md).

| category | property | type | what it means |
|---|---|---|---|
| Core content | `content_integrity` | ordinal, 4 | completeness and technical quality of the content |
| | `content_ratio` | ordinal, 5 | proportion of content vs. navigation and boilerplate |
| | `content_length` | ordinal, 4 | amount of substantive content |
| Classification | `content_type` | multi-label, 18 | functional purpose of the document |
| | `business_sector` | multi-label, 37 | industry relevance |
| | `technical_content` | multi-label, 7 | kind and intensity of specialised knowledge |
| Quality and value | `content_quality` | ordinal, 5 | overall writing and presentation quality |
| | `information_density` | ordinal, 5 | valuable information relative to redundancy |
| | `educational_value` | ordinal, 5 | potential for teaching and learning |
| | `reasoning_indicators` | ordinal, 5 | presence of logical reasoning and analysis |
| Audience and purpose | `audience_level` | ordinal, 6 | target sophistication level |
| | `commercial_bias` | ordinal, 5 | commercial influence on objectivity |
| | `time_sensitivity` | ordinal, 4 | how quickly the value of the content decays |
| Safety and compliance | `content_safety` | ordinal, 5 | presence of inappropriate or harmful content |
| | `pii_presence` | binary | contains personally identifiable information |
| Geography | `regional_relevance` | multi-label, 14 | primary regional and cultural context |
| | `country_relevance` | multi-label, 749 | specific country relevance |
| Not predicted | `one_sentence_description` | free text | a neutral one-sentence summary, which a classifier cannot produce |

## Architecture

* Backbone: {backbone}
* Heads: one per property, softmax for single-select and sigmoid for multi-select, with no
  cross-property conditioning — the properties are predicted independently from one shared document
  state.
* Input: {max_length} tokens of the document head plus {tail_tokens} tokens of its tail, so the end
  of a long document is not lost; document-level length features are passed to the heads.
* Decoding: ordinal properties are decoded to the expected level (`round(Σ p_i · rank_i)`) rather
  than by `argmax`, which is worth about +0.013 QWK on its own.

## Training

{title} was trained on propella-1-4b labels derived from the
[Dynaword corpora](https://huggingface.co/papers/2508.02271), FineWeb-2 and FinePDFs, covering {n_langs} languages and {n_train}. Training took {train_minutes} on one Quadro
RTX 8000 ({train_setup}), with a cross-entropy objective for the single-select properties, binary
cross-entropy for the multi-select ones and a token-level F1 loss for PII. The annotations are
published in [propella-bekko-data](https://huggingface.co/datasets/danish-foundation-models/propella-bekko-data)
and the training code in [propella-bekko](https://github.com/danish-foundation-models/propella-bekko).

## Results

As a held-out test, we measure agreement with DeepSeek-V4.1-Flash and GLM-5.3-Flash annotations on
the same 300 held-out documents and report macro-F1 across all properties. Chance — always
predicting the most frequent label — is 0.123.

| model | params | docs/s | vs DeepSeek | vs GLM-5.3 |
|---|---|---|---|---|
{student_rows}
| propella-1-0.6b | 600M | 0.14 | 0.498 | 0.501 |
| propella-1-1.7b | 1.7B | 0.14 | 0.547 | 0.534 |
| propella-1-4b | 4B | 0.06 | 0.571 | 0.549 |
| GLM-5.3-Flash | — | 0.41 | 0.659 | — |
| DeepSeek-V4.1-Flash | — | 0.34 | — | 0.659 |

## Usage

To use the model simply run:

```python
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

path = Path(snapshot_download("{repo_id}"))
cfg = json.loads((path / "config.json").read_text())
state = load_file(str(path / "model.safetensors"))

{encoder_block}
hidden = cfg["head_hidden"]
proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim + 3, hidden), nn.GELU(), nn.LayerNorm(hidden))
proj.load_state_dict(
    {k.removeprefix("proj."): v for k, v in state.items() if k.startswith("proj.")}
)
heads = nn.ModuleDict({f["name"]: nn.Linear(hidden, len(f["labels"])) for f in cfg["fields"]})
heads.load_state_dict(
    {k.removeprefix("heads."): v for k, v in state.items() if k.startswith("heads.")}
)

tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
max_length, tail = cfg["max_length"], cfg["tail_tokens"]


def annotate(text: str) -> dict:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    budget = max_length - 2
    cut = len(ids) > budget
    if cut:
        ids = ids[: budget - tail] + ids[-tail:]
    input_ids = torch.tensor([[tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]])
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        pooled = pool_text(input_ids, attention_mask)
        # length features: the encoder cannot see the full document after truncation
        features = torch.tensor(
            [
                [
                    torch.log1p(torch.tensor(float(len(text)))) / 10,
                    torch.log1p(torch.tensor(float(len(text.split())))) / 10,
                    float(cut or "\n[...]\n" in text),
                ]
            ]
        )
        z = proj(torch.cat([pooled, features], -1))
        logits = {name: head(z)[0] for name, head in heads.items()}

    out = {}
    for f in cfg["fields"]:
        name, labels, lg = f["name"], f["labels"], logits[f["name"]]
        if f["kind"] == "multi":
            hot = lg.sigmoid() >= 0.5
            hot[lg.argmax()] = True  # always at least one label
            out[name] = [lab for lab, on in zip(labels, hot) if on]
        elif f.get("ordinal"):
            ranks = torch.arange(len(labels), dtype=torch.float)
            out[name] = labels[int((lg.softmax(-1) * ranks).sum().round())]
        else:
            out[name] = labels[int(lg.argmax())]
    return out


# a 60,000-character Dutch government report, shortened here for the listing
doc = "TECHNISCH EN ECONOMISCH POTENTIEEL VOOR EEN AARDGASVRIJE GEBOUWDE OMGEVING IN DE REGIO DRECHTSTEDEN ..."
print(annotate(doc))
```

On the full document — a Dutch government report on making neighbourhoods gas-free — the model
answers:

```python
{example_output}
```

Multi-select properties are decoded at a sigmoid threshold of 0.5.{thresholds_note}

## Citation

The model and the code are MIT. If you use this model, please cite the
[Dynaword paper](https://huggingface.co/papers/2508.02271) ([arXiv:2508.02271](https://arxiv.org/abs/2508.02271))
and the project:

```
@inproceedings{dynaword,
	title = {Dynaword: {From} {One}-shot to {Continuously} {Developed} {Datasets}},
	booktitle = {Proceedings of the {Language} {Resources} and {Evaluation} {Conference} ({LREC} 2026)},
	author = {Enevoldsen, Kenneth and Jensen, Kristian Nørgaard and Kostkan, Jan and Szabó, Balázs and Kardos, Márton and Vad, Kirsten and Núñez, Andrea Blasi and Barmina, Gianluca and Nielsen, Jacob and Larsen, Rasmus and Vahlstrup, Peter and Dalum, Per Møldrup and Elliott, Desmond and Galke, Lukas and Schneider-Kamp, Peter and Nielbo, Kristoffer},
	year = {2026},
}
@misc{enevoldsenDanishFoundationModels2023,
	title = {Danish {Foundation} {Models}},
	url = {http://arxiv.org/abs/2311.07264},
	publisher = {arXiv},
	author = {Enevoldsen, Kenneth and Hansen, Lasse and Nielsen, Dan S. and Egebæk, Rasmus A. F. and Holm, Søren V. and Nielsen, Martin C. and Bernstorff, Martin and Larsen, Rasmus and Jørgensen, Peter B. and Højmark-Bertelsen, Malte and Vahlstrup, Peter B. and Møldrup-Dalum, Per and Nielbo, Kristoffer},
	year = {2023},
}
```

---

<h3 style="display: flex; align-items: center;">
  <a href="https://www.foundationmodels.dk">
    <img src="./docs/icon.png" width="30" style="margin-right: 10px;" />
  </a>
  A&nbsp;<a href=https://www.foundationmodels.dk>Danish Foundation Models</a>&nbsp;model
</h3>

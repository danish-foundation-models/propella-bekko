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
  <img src="pareto_deepseek-v4.1-flash.svg" width="49%" />
  <img src="pareto_glm-5.3-flash.svg" width="49%" />
</p>

Agreement with two annotators that are not the teacher — DeepSeek-V4.1-Flash on the left,
GLM-5.3-Flash on the right — against throughput. Both were asked to annotate the same 300 held-out
documents with propella's own prompt and schema.

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

{title} was trained on propella-1-4b labels derived from the Dynaword corpora, FineWeb-2 and
FinePDFs, covering {n_langs} languages and {n_train}. Training took {train_minutes} on one Quadro
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

```bash
pip install git+https://github.com/danish-foundation-models/propella-bekko
```

```python
from huggingface_hub import snapshot_download
from propella_bekko.models import PropellaBekko
from propella_bekko.inference import Annotator

path = snapshot_download("{repo_id}")
model, tokenizer, cfg = PropellaBekko.load(path)
annotator = Annotator(model, tokenizer, cfg["max_length"], cfg["tail_tokens"])

doc = "TECHNISCH EN ECONOMISCH POTENTIEEL VOOR EEN AARDGASVRIJE GEBOUWDE OMGEVING IN DE REGIO DRECHTSTEDEN ..."
print(annotator.annotate([{"id": "pbl_282", "text": doc}])[0])
```

That document is a Dutch government report on making neighbourhoods gas-free, and the model answers:

```python
{
    "content_integrity": "complete",
    "content_ratio": "complete_content",
    "content_length": "substantial",
    "content_type": ["analytical", "technical_documentation"],
    "business_sector": ["government_public", "energy_utilities", "environmental_services"],
    "technical_content": ["scientific", "data_heavy"],
    "content_quality": "excellent",
    "information_density": "dense",
    "educational_value": "moderate",
    "reasoning_indicators": "analytical",
    "audience_level": "expert",
    "commercial_bias": "none",
    "time_sensitivity": "slowly_changing",
    "content_safety": "safe",
    "pii_presence": "no_pii",
    "regional_relevance": ["european"],
    "country_relevance": ["netherlands"],
}
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

MIT for the model and the code.

* Paper: [Dynaword: From One-shot to Continuously Developed Datasets](https://huggingface.co/papers/2508.02271)
  ([arXiv:2508.02271](https://arxiv.org/abs/2508.02271))
* Teacher: [propella-1-4b](https://huggingface.co/ellamind/propella-1-4b) ([arXiv:2602.12414](https://arxiv.org/abs/2602.12414))
* Training data: [propella-bekko-data](https://huggingface.co/datasets/danish-foundation-models/propella-bekko-data)
* Code and experiments: [propella-bekko](https://github.com/danish-foundation-models/propella-bekko)

If you use this model, please cite the paper and the project:

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

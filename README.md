<p align="center">
  <img src="docs/assets/banner.svg" alt="propella-bekko: a ki-bekko koi with a propeller tail" width="720">
</p>

# propella-bekko

Small, fast document annotators that predict the
[propella-1](https://huggingface.co/ellamind/propella-1-4b) label space
(content quality, educational value, content type, business sector, safety, PII,
regional relevance, ...) in **a single forward pass**, without generating text.

The goal is to make annotating large pretraining corpora (e.g. the
[Dynawords](https://huggingface.co/collections/danish-foundation-models/dynawords))
cheap enough to run on everything, while staying
[Pareto-optimal](#results) against the three propella-1 models (0.6B / 1.7B / 4B).

Motivation: [swedish-dynaword#79](https://huggingface.co/datasets/danish-foundation-models/swedish-dynaword/discussions/79).

## How it works

* **Teacher labels**: existing propella-1-4b annotations from the annotated
  Dynawords (`data/*/metadata.parquet`) and
  [openeurollm/propella-annotations](https://huggingface.co/datasets/openeurollm/propella-annotations)
  (FineWeb-2, FinePDFs), joined with their source texts.
* **Students**: a backbone + one classification head per property
  (softmax for single-select, sigmoid for multi-select). Backbones, smallest first:
  1. [`bekko-static-embedding-v1`](https://huggingface.co/hotchpotch/bekko-static-embedding-v1) (no Transformer layers)
  2. [`bekko-embedding-v1-a8m`](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m)
  3. [`bekko-embedding-v1-a25m`](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m)
* Full-document length features (characters, words, whether the text was truncated) are passed to the
  heads so length-dependent properties stay learnable under truncation.
* `one_sentence_description` (free text) is not predicted.

## Quickstart

```bash
make install                     # uv sync (Python 3.12, torch cu129)

# held-out evaluation set (languages x domains), then a training set that excludes it
uv run propella-bekko build-data --dynaword all --n-per-subset 100 \
    --propella fineweb-2:all finepdfs:all --n-per-lang 300 --output data/heldout.parquet
uv run propella-bekko build-data --dynaword all --n-per-subset 2000 \
    --propella fineweb-2:all finepdfs:all --n-per-lang 5000 \
    --exclude data/heldout.parquet --output data/train.parquet

uv run propella-bekko train --config configs/static-v0.json
uv run propella-bekko evaluate  --model runs/static-v0/model --data data/heldout.parquet
uv run propella-bekko benchmark --model runs/static-v0/model --data data/heldout.parquet
uv run propella-bekko annotate  --model runs/static-v0/model --input docs.parquet --output annotations.parquet
```

## Results

Agreement with propella-1-4b on the held-out set vs. throughput.
See [`experiments/`](experiments/) for per-run details.

_No results yet. Progress is tracked in the [tracking issue](../../issues?q=label%3Atracking)._

## Repository layout

```
src/propella_bekko/
  schema.py        propella-1 label space (17 predicted properties)
  data/            source registry + held-out / train set builders
  models/          backbones (static, transformer) + multi-head annotator
  encoding.py      tokenisation, head+tail truncation, batching
  train.py         training loop
  metrics.py       agreement metrics (accuracy, macro-F1, QWK, Jaccard, ...)
  inference.py     fast length-bucketed annotation + throughput benchmark
  cli.py           `propella-bekko` command line
configs/           one config per experiment
experiments/       notes + results per experiment
docs/              decisions, evaluation design, assets
scripts/           misc (banner rendering)
```

## Development

```bash
make lint    # ruff format + ruff check --fix + ty
make test    # pytest
make check   # what CI runs (no auto-fixing)
```

Work is tracked in [issues](../../issues), grouped by milestone; the issue currently
being worked on carries the `in progress` label.

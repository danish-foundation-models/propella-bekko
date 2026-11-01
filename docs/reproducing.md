# Reproducing the results

Everything published — the three students, the dataset, the figures and the model cards — is built by
the commands below, in this order. Each step writes into `data/`, `runs/` or `docs/assets/`, all of
which are gitignored: the repository holds code, the artifacts are rebuilt.

Hardware note: the throughput numbers in the published cards and figures were measured on **one
Quadro RTX 8000** in fp16. The reference baselines (`propella-1-*`) additionally have bf16 numbers on
an **RTX A6000** (Ampere), because Turing has no bf16 — see step 5. Every other number is
hardware-independent.

## 1. Environment

```bash
make install        # uv sync: Python 3.12, torch cu129
make check          # ruff + ty + pytest, what CI runs
```

## 2. Data

The builder samples remotely from the source corpora, so it needs network access and a Hugging Face
login for the gated Dynaword subsets (`hf auth login`). Seeds are fixed in the scripts.

```bash
bash scripts/build_small.sh          # data/heldout-small.parquet (9,720 docs), data/train-small.parquet
bash scripts/build_long.sh heldout   # re-extract the held-out texts at 50k + 10k characters
uv run python scripts/make_pareto_sample.py   # data/heldout-sample1000.parquet (the 1,000-doc test set)
bash scripts/build_pool_xl.sh        # data/train-pool-xl.parquet (844,912 docs, 50k + 10k chars)
bash scripts/build_long.sh pool      # data/train-pool-long.parquet (403,281 docs, same re-extract)
```

`build_long.sh heldout` reuses the labels of `heldout-small.parquet` (same seeds, same documents) and
only re-reads the texts at the longer truncation, so the held-out set is identical across both views.

The 1,000-document test set is drawn from `heldout-small-long.parquet` with `random.Random(0)`, and
`build_pool_xl.sh` excludes `heldout-small.parquet` by `(source, id)` and by text hash, so no test
document can enter the training pool. `scripts/check_rebuild.py` and `scripts/check_labels.py` verify
both properties.

## 3. Train the three published students

```bash
uv run propella-bekko train --config configs/static-long-ctx8192.json   #  206M, no attention, 13 min (train-pool-long)
uv run propella-bekko train --config configs/a8m-xl.json                 #  106M, 8M active, 92 min
uv run propella-bekko train --config configs/a25m-xl-3ep.json            #  123M, 25M active, 316 min
```

Each config pins the data file, backbone, truncation and seed, and writes `runs/<name>/model`,
`thresholds.json` and `results.json`. The configs referenced above are the ones behind the published
checkpoints; `configs/` also holds the variants that were tried and rejected (see `experiments/`).

## 4. Independent reference annotations

Both reference annotators use propella's own prompt and JSON schema, temperature 0.0:

```bash
export REFERENCE_API_KEY=...            # any OpenAI-compatible endpoint
uv run python scripts/reference_labels.py --input data/heldout-sample1000.parquet \
    --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --output runs/reference/deepseek-v4.1-flash-1k.parquet
uv run python scripts/reference_labels.py --input data/heldout-sample1000.parquet \
    --model zai-org/GLM-5.3-Flash \
    --output runs/reference/glm-5.3-flash-pareto300.parquet
```

The 300-document comparison subset used in every figure and card is the first 300 rows of the 1,000
(`--limit 300`, the same rows for every model). The 1,000-document GLM run returned empty answers and
is not used.

## 5. Reference baselines: the propella-1 models

`vllm` pins its own torch, so it lives in a separate environment:

```bash
uv venv ~/envs/vllm && VIRTUAL_ENV=~/envs/vllm uv pip install vllm
for m in 0.6b 1.7b 4b; do
  ~/envs/vllm/bin/python scripts/propella_baseline.py \
      --model ellamind/propella-1-$m --input data/heldout-sample1000.parquet \
      --dtype bfloat16 --output runs/baselines/bf16/propella-1-$m.parquet
done
```

Each run writes a parquet with the annotations and a `.json` with documents per second and the number
of failed JSON parses. Use `--dtype float16` on Turing, `bfloat16` on Ampere and later.

The published numbers come from bf16 runs on one **RTX A6000** with `--json-mode free`: propella's
prompt already asks for JSON, and the object is parsed out of the completion. Do not use the
schema-constrained decoder here — with vLLM 0.10.2 it spends nearly all of the runtime inside the
decoder and measures 0.019 docs/s for propella-1-0.6b, against 2.7 docs/s for free-form generation on
the same card.

That environment, if you need to rebuild it: vLLM 0.10.2 with `transformers<5`, `ninja` on `PATH`,
`CUDA_HOME` set, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`. For the
students, a separate venv with the repository's own dependencies (`uv pip install -e .`), because
their tokenizers are saved by transformers 5 and vLLM 0.10 wants transformers 4.

## 6. Student predictions

```bash
uv run python scripts/student_predict.py runs/a25m-xl-3ep/model \
    data/heldout-sample1000.parquet runs/pareto/a25m-xl-3ep.parquet
# ... same for runs/a8m-xl/model and runs/static-long-ctx8192/model
```

## 7. Figures and scores

```bash
uv run python scripts/pareto_vs_llms.py   # runs/pareto_vs_<reference>.json
uv run python scripts/pareto.py           # docs/assets/pareto_*.svg + pareto_combined.svg
```

`pareto_vs_llms.py` scores every model against the same 300 reference annotations; `pareto.py` renders
one figure per reference plus the combined two-panel figure used in the cards.

## 8. Model cards and the Hub

Each published model has a directory here that mirrors its Hub repo — `models/propella-bekko-a25m/`
holds the card as `README.md` plus the banner, the two figures, the property reference and the icon,
so the card is a normal file in this repository: edit it, review it in a diff, and upload the
directory as it stands.

```bash
make per-field   # per-property figure for each student -> docs/assets/per_property_<tier>.svg
make cards       # docs/model_sheet.md + runs/pareto_vs_*.json -> models/<repo>/README.md
make assets      # copy the shared figures into the model directories
make publish     # hf upload of each model directory and its checkpoint
```

`make cards` is optional: the cards are checked in, and regenerating only rewrites the numbers from
`runs/pareto_vs_*.json` and `runs/per_property_<run>.json` (via `src/propella_bekko/release.py`), so
a hand edit and a regeneration touch the same files and show up in the same diff. `make assets` only
copies — the per-property figure is the one thing per model, everything else is shared.

Uploading is plain `hf upload`; each model takes two calls, the card directory and the checkpoint:

```bash
hf upload danish-foundation-models/propella-bekko-a25m models/propella-bekko-a25m .
hf upload danish-foundation-models/propella-bekko-a25m runs/a25m-xl-3ep/model .
```

`docs/property_descriptions.md` is propella-1's own property reference, copied here (Apache-2.0) and
shipped next to each card so the definitions behind the labels stay available even if the original
moves.

## 9. The dataset

```bash
uv run python scripts/make_dataset_release.py --out ../propella-bekko-data
uv run python scripts/verify_dataset_release.py
```

The release carries the document text; see
[danish-foundation-models/propella-bekko-data](https://huggingface.co/datasets/danish-foundation-models/propella-bekko-data)
for the datasheet, which documents the splits, the sources and the licence.

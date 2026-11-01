# Scripts

The entry points that matter are the `propella-bekko` CLI (`build-data`, `train`, `score`,
`annotate`, `benchmark`) and `make data / models / figures / cards`. The scripts here are the pieces
that were run once per release or per experiment; `docs/reproducing.md` puts them in order.

## Data

| script | what it does |
|---|---|
| `build_small.sh` | held-out set (9,720 docs) + small training set |
| `build_long.sh` | re-extract held-out or pool texts at 50k + 10k characters, reusing the labels |
| `build_pool_xl.sh` | the 844,912-document training pool (what the a25m student used) |
| `build_train.sh` | the first, smaller training pool |
| `make_pareto_sample.py` | the seeded 1,000-document comparison sample |
| `make_short200.py` | a 200-document subset of that sample, short documents only |
| `check_rebuild.py` | verify a `--labels-from` rebuild sampled exactly the same documents |
| `check_labels.py` | label sanity checks on a built parquet |

## Annotations

| script | what it does |
|---|---|
| `reference_labels.py` | independent labels from an OpenAI-compatible LLM (DeepSeek, GLM) |
| `propella_baseline.py` | run a propella-1 model with vLLM and measure throughput |
| `student_predict.py` | annotate a parquet with a trained student, keeping `source` and `id` |
| `make_consensus.py` | combine several LLM annotation sets into one reference |
| `ref_as_gold.py` | rewrite a held-out file so its labels come from a reference annotator |

## Scoring and analysis

| script | what it does |
|---|---|
| `collect_pareto.py` | macro-F1 against the propella-1-4b labels + docs/s on the 300-doc comparison set |
| `pareto_vs_llms.py` | the same against an LLM reference annotator, one JSON per reference |
| `pareto.py` | the Pareto figures (one per reference, plus the combined two-panel figure) |
| `per_field.py` | per-property agreement figure (student, teacher and the no-signal floor) |
| `palette.py` | the one palette every figure uses |
| `pareto_two.py` | both Paretos in one figure |
| `mf1_vs_glm.py` | macro-F1 against the 300 documents that have GLM labels |
| `field_compare.py` | per-property comparison against the propella-1 baselines |
| `breakdown.py` | per-field and per-slice breakdown of one prediction file |
| `agreement_matrix.py`, `agreement_short200.py` | every annotation set against every other |
| `error_analysis.py`, `inspect_errors.py`, `inherited_errors.py` | where the students disagree, and whether the teacher disagrees too |
| `dtype_sensitivity.py` | fp16 vs fp32 predictions on the same documents: how much the dtype moves the scores |

## Training experiments

| script | what it does |
|---|---|
| `soup.py` | weight averaging over checkpoints of one run |
| `ensemble.py` | per-field voting across several prediction files |
| `soft_ordinal.py` | expected-value vs argmax decoding for the ordinal properties |
| `prior_correction.py` | logit adjustment towards the training-label prior |
| `clean_pool.py` | the duplicate/label-noise filter that was tried and rejected |
| `probe_backbone.py` | inspect a backbone's interface before training on it |
| `summarize_runs.py` | one table over every run directory |

## Release

| script | what it does |
|---|---|
| `example_output.py` | record one worked example per run, quoted in its model card |
| `update_sections.py` | rewrite the marked regions of the cards and the README from the scores |
| `verify_remote_code.py` | check a published repo loads via `AutoModel` and matches the package |
| `make_dataset_release.py`, `verify_dataset_release.py` | build and check the published dataset |
| `render_banner.js` | render the banner SVG |

The `.sh` runners (`eval_both_refs.sh`, `eval_categories.sh`, `eval_finished.sh`,
`predict_and_score.sh`, `sweep_static_ctx.sh`, `train_after.sh`, `train_when_ready.sh`,
`tune_queue.sh`) are the ad-hoc queues used during the runs; they are kept for provenance, not as
entry points.

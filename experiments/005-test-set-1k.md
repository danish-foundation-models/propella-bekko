# 005 — the 1,000-document test set, two references

Issues: https://github.com/danish-foundation-models/propella-bekko/issues/10,
https://github.com/danish-foundation-models/propella-bekko/issues/18

**Test set**: the seeded 1,000 held-out documents (`data/heldout-sample1000.parquet`),
long texts (up to 60k chars stored; students see 8,192 tokens ≈ 20k chars, propella
baselines see the first 20k chars).

**References**
* `propella`: the stored propella-1-4b labels shipped with the corpora (what the students
  were trained on).
* `llm`: DeepSeek-V4.1-Flash labelled with propella's full annotator prompt + strict JSON
  schema (the paper's protocol), 1,000/1,000 valid. GLM-5.3-Flash is being added as a second
  annotator (consensus of the two).

**Metric**: the propella-1 paper's overall score — 11/17 × mean QWK (ordinal) + 1/17 ×
PII F1 + 5/17 × mean IoU (multi-select).

| model | docs/s | vs stored propella labels | vs LLM (DeepSeek) |
|---|---|---|---|
| propella-1-1.7b | 0.50 | 0.808 | 0.703 |
| propella-1-0.6b | 0.50 | 0.763 | 0.679 |
| **a25m-v1-pool** | 193 | **0.803** | 0.691 |
| a25m-domcap | 183 | 0.777 | 0.669 |
| a8m-v1-pool | 376 | 0.773 | 0.663 |
| static-long-ctx8192 | 686 | 0.737 | 0.655 |
| static-v1-pool | 681 | 0.731 | 0.657 |

**Takeaways**
1. **a25m-v1-pool is level with propella-1-1.7b on the paper score** against the labels it was
   distilled from (0.803 vs 0.808) and clearly above 0.6b (0.763), at ~390× the throughput.
   Against the independent LLM reference it sits between 0.6b and 1.7b (0.691 vs 0.679/0.703) —
   i.e. when judged by a *different* annotator, ~30% of the apparent gap to 1.7b is teacher-specific.
2. **Domain capping helps macro-F1 but hurts the paper score** (0.777 vs 0.803): capping shifts
   label frequencies towards rare labels, which macro-F1 rewards and IoU/QWK penalise. With the
   paper metric as the target, the plain pool is better.
3. **The static model is the cheapest useful point** (686 docs/s, 0.737): ~10% below a8m and 1.8×
   faster, and it holds up on long documents.
4. propella-1-4b re-run on the same documents (fp16, 20k-char inputs) is the practical ceiling;
   its row is filled in when the run finishes.

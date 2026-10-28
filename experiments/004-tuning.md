# 004 — tuning the students (macro-F1 target)

Issue: https://github.com/danish-foundation-models/propella-bekko/issues/19

All runs: a25m/a8m backbone, `train-pool-long` (403k docs, long texts, 1024 tokens),
2 epochs unless noted, evaluated on `heldout-small-long` (9,720 docs) against the stored
propella-1-4b labels.

| run | change | single acc | Jaccard | macro-F1 |
|---|---|---|---|---|
| a25m-v1-pool | baseline (pool) | 0.848 | 0.763 | 0.616 |
| **a25m-domcap** | no domain > 25% of training | **0.848** | 0.702 | **0.632** |
| a8m-v1-pool | baseline (pool) | 0.832 | 0.752 | 0.582 |
| a8m-pw | + inverse-frequency class weights | 0.832 | 0.669 | 0.595 |
| a8m-pw-ls | + class weights + label smoothing 0.05 | 0.830 | 0.673 | 0.585 |
| a25m-v1-pool + tuned thresholds | per-field cuts on val | 0.848 | 0.727 | 0.618 |

**Takeaways**
1. **Domain capping is the strongest lever so far** (+0.016 macro-F1 at no cost in accuracy):
   the pool was 37% PDF/legal-heavy, which biased label frequencies; capping each domain at
   25% removed that.
2. **Class weighting** helps a8m (+0.013) but costs Jaccard; **label smoothing hurts** (0.585 vs 0.595) → dropped.
3. **Per-field thresholds** tuned on the validation split give a free +0.01 macro-F1 (and can be
   applied post-hoc to any checkpoint), at the cost of exact overlap.
4. Both weighting and threshold tuning move the model towards predicting rare labels; that is
   what macro-F1 rewards. Since the target is now macro-F1 (and the paper's QWK/IoU score),
   these are the right knobs.

Pending: label-conditional second pass (`refine`), more epochs, bigger pool (XL), every
combination with domain capping.

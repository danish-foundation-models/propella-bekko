# 006 — what actually moves the score

Issues: https://github.com/danish-foundation-models/propella-bekko/issues/19,
https://github.com/danish-foundation-models/propella-bekko/issues/14

Every model below is scored on the same 1,000 held-out documents with the propella-1
overall score (QWK/F1/IoU, arXiv 2602.12414 §5.2) against the stored propella-1-4b labels.
`docs/s` on one RTX 8000.

| model | training data | change vs its baseline | docs/s | overall score |
|---|---|---|---|---|
| propella-1-4b (re-run) | – | reference | 0.22 | **0.875** |
| propella-1-1.7b | – | – | 0.50 | 0.808 |
| propella-1-0.6b | – | – | 0.50 | 0.763 |
| **a25m-v1-pool** | 403k | – | 193 | **0.803** |
| a25m-langcap | 403k | ≤15% per language | 198 | 0.797 |
| a25m-domcap | 403k | ≤25% per domain | 183 | 0.777 |
| a25m-domcap-pw | 403k | domcap + class weights | 200 | 0.777 |
| a8m-xl | 845k | 2.1× data | 348 | 0.783 |
| a8m-v1-pool | 403k | – | 376 | 0.773 |
| a8m-domcap | 403k | ≤25% per domain | 376 | 0.747 |
| a8m-v0 | 98k | – | 374 | 0.724 |
| static-long-ctx8192 | 403k | – | 686 | 0.737 |
| static-v1-pool | 403k | – | 681 | 0.731 |

Held-out macro-F1 (9,720 docs, 0.5 thresholds) for the same runs: a25m-v1-pool 0.616,
a25m-domcap 0.632, a25m-domcap-pw 0.632, a25m-langcap 0.617 (0.630 with tuned thresholds),
a8m-xl 0.602, a8m-v1-pool 0.582, a8m-domcap 0.589.

## What moved the score

1. **More data helps, and had not saturated**: a8m 98k → 403k → 845k gives 0.724 → 0.773 → 0.783.
2. **Bigger backbone beats more data at equal cost**: a25m on 403k (0.803, 193 docs/s) is better
   than a8m on 845k (0.783, 348 docs/s) — per unit of GPU time they are close, so both are on the
   frontier.

## What did not

3. **Domain capping** improves macro-F1 (+0.016) but *lowers* the paper score (−0.026, driven by
   IoU on the multi-select properties). Macro-F1 and the paper score disagree here: capping shifts
   prediction frequencies towards rare labels.
4. **Class weighting** (pos_weight) on top of capping changes nothing (0.777 both), and alone it
   trades Jaccard for macro-F1.
5. **Language capping** is neutral (−0.006).
6. **Label smoothing** hurt macro-F1 (0.585 vs 0.595 for a8m).
7. **Label-conditional second pass** (refine: every field sees all first-pass distributions):
   neutral (0.616 macro-F1, identical to baseline).
8. **A 3-model ensemble** (static+a8m+a25m) is dominated: 0.788 at 108 docs/s vs a25m alone
   0.803 at 193 docs/s.
9. **Threshold tuning** for IoU picks the same cut-offs as for F1 (0.1–0.35), i.e. the residual gap
   on multi-select properties is capacity/representation, not decoding calibration.

**Conclusion**: the remaining headroom is model capacity and data volume, not label engineering.
Per-property, the students already beat propella-1-1.7b on 9 of 17 properties (integrity, ratio,
length, quality, educational value, commercial bias, time sensitivity, safety, PII) and lose on
the ontology-heavy multi-select ones (business sector 0.637 vs 0.739, country 0.686 vs 0.855,
content type 0.708 vs 0.757).

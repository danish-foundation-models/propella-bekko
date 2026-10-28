# 003 — scaling the students

Issues: https://github.com/danish-foundation-models/propella-bekko/issues/6,
https://github.com/danish-foundation-models/propella-bekko/issues/7,
https://github.com/danish-foundation-models/propella-bekko/issues/8,
https://github.com/danish-foundation-models/propella-bekko/issues/10

**Setup**: same heads/loss/recipe, only the backbone and the amount of training data
change. Data: `train-small` (98k docs, ≤500/dynaword subset, 2k/language) versus the
training pool (403k docs, ≤2k/subset, 10k/language). Everything is scored on the same
300 held-out documents (long texts), agreement with the propella-1-4b labels in the data.

| model | params | data | docs/s | single acc | macro-F1 |
|---|---|---|---|---|---|
| propella-1-4b (re-run) | 4B | – | 0.06 | 0.900 | **0.740** |
| propella-1-1.7b | 1.7B | – | 0.12 | 0.846 | 0.651 |
| propella-1-0.6b | 0.6B | – | 0.15 | 0.813 | 0.586 |
| **a25m-v1-pool** | 118M | 403k | 193 | 0.836 | **0.588** |
| a25m-v0 | 118M | 98k | 199 | 0.831 | 0.545 |
| a8m-v1-pool | 106M | 403k | 376 | 0.825 | 0.567 |
| a8m-v0 | 106M | 98k | 374 | 0.805 | 0.485 |
| static-long-ctx8192 | 206M | 403k | 686 | 0.810 | 0.543 |
| static-v1-pool | 206M | 403k | 681 | 0.805 | 0.536 |

(static-v1-pool was trained on 15k-char texts, static-long-ctx8192 on the long texts.)

**Takeaways**
1. **propella-1-4b agrees with itself at macro-F1 0.740** on these documents (same model
   re-run, fp16 vLLM vs whatever produced the stored labels, plus our text truncation and
   decoding differences). That is the practical ceiling: 1.7b reaches 88% of it, 0.6b 79%,
   a25m-v1-pool 79%.
2. **a25m at 403k docs matches propella-1-0.6b** (0.588 vs 0.586) at ~1,300× the throughput.
   a8m is 29% behind in macro-F1 and 2× faster; the static model is 8% behind a8m and 1.8× faster.
3. **More data helped both** (a8m 0.485 → 0.567, a25m 0.545 → 0.588) — the data curve had not
   flattened, hence the bigger pool (https://github.com/danish-foundation-models/propella-bekko/issues/14).
4. Long texts help slightly (static: 0.536 → 0.543) — see
   https://github.com/danish-foundation-models/propella-bekko/issues/17.

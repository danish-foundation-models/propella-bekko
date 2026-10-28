# 002 — context length (static model)

Issue: https://github.com/danish-foundation-models/propella-bekko/issues/17

**Question**: how much text does the static model need? Longer contexts only matter
if the stored text is longer than the context, so the held-out set and the
training pool were rebuilt with up to 60k chars per document (50k head + 10k tail,
≈ propella-1-4b's own 50k-char view), reusing the existing labels
(`build-data --labels-from`, identical documents and full-text hashes).

**Setup**: static backbone on `train-pool-long` (403k docs, mean 2,811 tokens),
3 epochs, only `max_length` varies (`tail_tokens` fixed at 512).

| max_length | held-out acc | Jaccard | macro-F1 | docs/s | train |
|---|---|---|---|---|---|
| 4,096 | 0.814 | 0.758 | 0.561 | 723 | 9 min |
| **8,192** | **0.815** | **0.761** | **0.569** | 665 | 13 min |
| 16,384 | 0.815 | 0.758 | 0.562 | 621 | 16 min |
| 32,768 | 0.814 | 0.758 | 0.562 | 621 | 18 min |

(Held-out 9,720 docs at 4k–32k; docs/s on the 1,000-doc sample, idle GPU.)

**Takeaway**: flat. The static model saturates around **8k tokens**, which is the
knee: beyond it agreement does not improve (differences ≤0.01 are noise at this
size) while throughput drops ~14% and training takes 40% longer. This is expected
— the pooled mean of static token vectors loses word order, so added context adds
mostly noise, and the median document is ~2.7k tokens anyway.

Use **8,192** (with 512 tail tokens) as the default for the static model. Longer
context only needs reconsidering for a config with a real attention mechanism.

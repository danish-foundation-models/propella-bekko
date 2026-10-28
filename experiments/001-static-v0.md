# 001 — static-v0

Issue: https://github.com/danish-foundation-models/propella-bekko/issues/6

**Goal**: first end-to-end baseline with the smallest backbone.

**Setup**
- Backbone: `static:hotchpotch/bekko-static-embedding-v1` (200k × 1024 static table, mean pooling, fine-tuned), MLP 1024 + per-property heads + length features
- Data: `train-small` (98,049 docs: ≤500 per Dynaword subset, 2,000 per FineWeb-2/FinePDFs language), held-out excluded
- Config: `configs/static-v0.json` (8,192 tokens head+tail, 3 epochs, lr 1e-2 embeddings / 1e-3 heads)
- Training time: ~4 min on 1× Quadro RTX 8000

**Held-out (`heldout-small`, 9,720 docs, 20 languages, 15 domains), agreement with propella-1-4b**

| metric | value |
|---|---|
| mean single-select accuracy | 0.789 (majority baseline 0.620) |
| mean multi-select Jaccard | 0.726 |
| mean macro-F1 | 0.515 |
| throughput (fp16, incl. tokenisation) | 1,040 docs/s (all 9,720); 684 docs/s on the 1,000-doc baseline sample |

Per property (accuracy / majority / macro-F1; ordinal: within-1, QWK):

| property | acc | maj | macro-F1 | within-1 | QWK |
|---|---|---|---|---|---|
| content_integrity | 0.831 | 0.808 | 0.450 | 0.884 | 0.390 |
| content_ratio | 0.889 | 0.885 | 0.442 | 0.978 | 0.432 |
| content_length | 0.712 | 0.349 | 0.722 | 0.992 | 0.845 |
| content_quality | 0.743 | 0.528 | 0.559 | 0.965 | 0.691 |
| information_density | 0.761 | 0.667 | 0.529 | 0.927 | 0.560 |
| educational_value | 0.596 | 0.326 | 0.555 | 0.906 | 0.765 |
| reasoning_indicators | 0.642 | 0.298 | 0.598 | 0.873 | 0.818 |
| audience_level | 0.796 | 0.542 | 0.496 | 0.941 | 0.765 |
| commercial_bias | 0.880 | 0.852 | 0.473 | 0.947 | 0.745 |
| time_sensitivity | 0.731 | 0.320 | 0.622 | 0.854 | 0.658 |
| content_safety | 0.984 | 0.979 | 0.385 | 0.996 | 0.573 |
| pii_presence | 0.907 | 0.880 | 0.745 | | |

Multi-select (Jaccard / exact / macro-F1): content_type 0.587 / 0.372 / 0.538,
business_sector 0.588 / 0.339 / 0.438, technical_content 0.848 / 0.816 / 0.659,
regional_relevance 0.876 / 0.842 / 0.286, country_relevance 0.731 / 0.667 / 0.254.

**Takeaways**
- Clearly above majority on the graded properties (educational value, reasoning, quality, time sensitivity; QWK 0.66–0.82).
- Weak where the signal is subtle or rare: content_integrity / content_ratio (near majority), safety, rare regions/countries, business_sector.
- Web (FineWeb-2) and French/German are the hardest slices; Legal / Report / Norwegian the easiest.
- Next: more data (training pool, https://github.com/danish-foundation-models/propella-bekko/issues/14), then a8m (https://github.com/danish-foundation-models/propella-bekko/issues/7).

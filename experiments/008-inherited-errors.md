# 008 — how much of the student's error is inherited from the teacher?

Script: `scripts/inherited_errors.py`. Three annotation sets on the same 1,000 held-out
documents: the student (`a25m-xl`), the teacher whose labels it was trained on (stored
propella-1-4b), and an independent reference (DeepSeek-V4.1-Flash, propella's full annotator
prompt).

For every property and document:

* **inherited** — student and teacher both disagree with the reference (the student reproduces
  a teacher disagreement);
* **own** — the student disagrees while the teacher agrees;
* **fixed** — the student agrees where the teacher was wrong.

| property | inherited | own | fixed | inherited share of errors |
|---|---|---|---|---|
| content_integrity | 250 | 49 | 41 | 84% |
| content_ratio | 85 | 23 | 36 | 79% |
| content_length | 181 | 101 | 54 | 64% |
| content_type | 341 | 82 | 136 | 81% |
| business_sector | 395 | 135 | 123 | 75% |
| technical_content | 75 | 44 | 43 | 63% |
| content_quality | 342 | 76 | 67 | 82% |
| information_density | 244 | 73 | 73 | 77% |
| educational_value | 269 | 114 | 130 | 70% |
| reasoning_indicators | 257 | 82 | 85 | 76% |
| audience_level | 146 | 48 | 52 | 75% |
| commercial_bias | 55 | 37 | 33 | 60% |
| time_sensitivity | 126 | 71 | 72 | 64% |
| content_safety | 25 | 2 | 7 | 93% |
| pii_presence | 66 | 16 | 19 | 80% |
| regional_relevance | 76 | 50 | 41 | 60% |

**75% of the student's disagreements with the independent reference are inherited from the
teacher** (2,933 vs 1,003 own errors), and the student *fixes* 1,012 teacher disagreements
(it is not simply copying the teacher either).

## Why this matters

1. At 25M active parameters the student is close to the **teacher's own ceiling** on this
   annotation task. A bigger student trained on the same labels has little headroom: the
   errors that remain in the reference comparison are mostly the teacher's.
2. The earlier per-property analysis (`experiments/007`) shows the same picture from the other
   side: the fields where the student trails propella-1-1.7b (business_sector, country and
   content_type) are exactly the ones where the teacher itself disagrees with the reference most.
3. To move the Pareto frontier further, the lever is **better labels** (a stronger or ensemble
   teacher, or agreement-filtered labels), not only more student capacity. That is what
   https://github.com/danish-foundation-models/propella-bekko/issues/11 was queued for.

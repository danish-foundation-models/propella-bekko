# 011 — scoreboard after the decision-model and error-analysis changes

All numbers: paper score (11/17 QWK + 1/17 PII F1 + 5/17 Jaccard), 1,000 held-out documents,
20k-character inputs, expected-value ordinal decoding unless stated.

## Results

| model | params (active) | docs/s | vs propella labels | vs DeepSeek |
|---|---|---|---|---|
| propella-1-4b | 4B | 0.22 | 0.875 | 0.737 |
| propella-1-1.7b | 1.7B | 0.50 | 0.808 | 0.703 |
| propella-1-0.6b | 0.6B | 0.50 | 0.763 | 0.679 |
| **a25m-xl-3ep** | 25M (118M) | **186** | **0.820** | **0.724** |
| a25m-v2 (rich length) | 25M | 193 | 0.819 | — |
| a25m-v3 (rich length + label init) | 25M | 193 | 0.819 | — |
| a25m-xl (2 epochs) | 25M | 193 | 0.819 | 0.712* |
| a25m-soup4 (4-checkpoint soup) | 25M | 193 | 0.820 | — |
| a8m-v1-pool | 8M (106M) | 376 | 0.779 | 0.694 |
| static-v1-pool | 0 (206M) | 686 | 0.747 | 0.678 |

\* argmax decoding, before the change.

## What moved the score

| change | effect |
|---|---|
| expected-value ordinal decoding (no training) | **+0.013 QWK, +0.008 paper score** |
| 3 epochs instead of 2 on the 845k pool | +0.001 paper, +0.004 macro-F1 (held-out), +0.007 after threshold tuning |
| rubric-threshold length features | ±0.000 |
| GLiClass-style label-name head init | ±0.000 on the test set (worse held-out macro-F1: 0.621 vs 0.645) |
| 4-checkpoint soup | ±0.000 (0.820 vs 0.820) |
| trajectory checkpoint averaging (SWA) | −0.002 |
| cleaned pool (16.5% of rows removed) | −0.026 |
| bigger backbone (mmBERT-small, 42M active) | −0.011 |

## Reading

The recipe has plateaued at **~0.82** against the stored labels and **~0.72** against an
independent annotator. Every remaining lever we can pull with this teacher has been measured
and does not move: architecture tweaks, more epochs (+0.001), checkpoint averaging, pooling
variants, data cleaning (negative), bigger backbones (negative — it buys accuracy at the cost
of Jaccard on multi-select fields).

This is the expected shape given `experiments/008`: 75% of the student's disagreements with an
independent annotator are disagreements the *teacher* also has. The student is at the teacher's
ceiling, so more capacity cannot help; only different labels can. Since we have no human
labels, "better labels" is not currently measurable — training on DeepSeek's labels instead
would move agreement, not quality, and would by construction score worse against the propella
labels we can measure.

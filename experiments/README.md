# Experiments

One markdown file per experiment (`NNN-short-name.md`) with: goal, config, data, results (held-out
summary + throughput), takeaways, and the issue it belongs to. Raw `results.json` files live in
`runs/` (not committed); the numbers that matter are copied into the markdown.

| # | experiment | outcome |
|---|---|---|
| 001 | [static-v0](001-static-v0.md) | the static backbone works: 0.72 on the held-out set before any tuning |
| 002 | [context length](002-context-length-static.md) | the static model keeps improving to 8k tokens, then saturates |
| 003 | [scaling](003-scaling.md) | 8M and 25M active parameters: more capacity helps, more heads per property do not |
| 004 | [tuning](004-tuning.md) | class weighting and thresholds; tuned thresholds buy macro-F1, not accuracy |
| 005 | [the 1k test set](005-test-set-1k.md) | the 1,000-document set and the two independent references |
| 006 | [what moves the score](006-what-moves-the-score.md) | decode in expectation (+0.013 QWK), epochs (+0.001), everything else flat |
| 007 | [error analysis](007-error-analysis.md) | where the student is wrong against the teacher, property by property |
| 007b | [error analysis vs the LLM](007b-error-analysis-vs-llm.md) | the same run scored against DeepSeek: 0.708 |
| 008 | [inherited errors](008-inherited-errors.md) | 75% of the student's disagreements with an independent annotator are the teacher's |
| 009 | [inspecting disagreements](009-inspecting-disagreements.md) | what the disagreements look like in the text |
| 010 | [GLiClass and decision models](010-gliclass-and-decision-models.md) | expected-value decoding and label-name initialisation: one works, one does not |
| 011 | [scoreboard](011-scoreboard.md) | every change measured; the recipe plateaus around 0.82 |

# Evaluation design

## Held-out set

Built once with `propella-bekko build-data` and excluded from all training sets
(by `(source, id)` and by normalised text hash).

| Source | Languages | Domain | Size |
|---|---|---|---|
| danish/norwegian/icelandic/faroese/dutch-dynaword | dan, nor, isl, fao, nld | per subset (Legal, News, Books, Social Media, ...) | ≤100 per subset |
| propella-annotations/fineweb-2 | deu, fin, fra, ita, spa, swe | Web | 300 per language |
| propella-annotations/finepdfs | ces, dan, deu, eng, fin, fra, hun, ita, nld, nob, pol, por, ron, spa, swe | PDF | 300 per language |

Swedish Dynaword has no propella annotations yet, so Swedish is covered via
FineWeb-2 and FinePDFs.

## Metrics (student vs propella-1-4b)

Per property:
* single-select: accuracy, macro-F1, majority-class baseline; ordinal properties
  additionally MAE, within-1 accuracy and quadratic-weighted kappa
* multi-select: exact match, Jaccard, micro/macro-F1

Summaries: mean single-select accuracy, mean multi-select Jaccard, mean macro-F1
(primary — accuracy is inflated by majority classes), each broken down by
source, language and domain.

## Throughput

`propella-bekko benchmark` reports docs/s on the held-out documents; the propella-1 baselines are run
with vLLM on the same GPU. The published figures measure every model on one **RTX A6000** in bf16:
the students with their own pipeline, the baselines with vLLM 0.10.2.

Two things matter for the baseline numbers. First, dtype and hardware — Turing has no bf16, so the
RTX 8000 numbers in the first release were fp16. Second, and much larger: schema-constrained decoding.
With xgrammar constraining every step to propella's JSON schema, propella-1-0.6b ran at 0.019 docs/s;
generating free-form and parsing the object (the prompt already asks for JSON, and the reference
annotations were always parsed this way) gives 2.7 docs/s for the same model on the same card. The
schema-constrained runs were measuring the decoder, not the model, and are not used.

Free-form output occasionally names a label outside the taxonomy (a country the training data never
had, a capitalised duplicate). Those are dropped — or counted as `other` where the property has such
a bucket — rather than crashing the scorer.

The agreement numbers are scored from each model's predictions on the held-out documents; for the
students those were generated in fp16, because Turing has no bf16. That dtype is not what the numbers
turn on: `scripts/dtype_sensitivity.py` re-runs `a25m` in fp16 and fp32 over the same 300 documents
and scores both against the same reference — macro-F1 0.5184 (fp16) vs 0.5188 (fp32), with 296 of
300 documents annotated identically. bf16 sits between the two, so a bf16 prediction run would move
the tables by less than the third decimal.

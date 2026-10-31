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

`propella-bekko benchmark` reports docs/s on the held-out documents, on one Quadro RTX 8000 in fp16
for every model, students and propella-1 baselines alike, so the axis compares like with like.
propella-1 baselines are run with vLLM on the same card.

A bf16 rerun on an RTX A6000 was attempted and abandoned: with vLLM 0.10.2 and xgrammar
schema-constrained decoding, propella-1-0.6b ran at 0.019 docs/s (54 s per document) — seven times
slower than the same model in fp16 on the RTX 8000, which is a property of that serving stack and not
of the model. Nothing in this release uses those numbers. The recipe that was tried is in
[`docs/reproducing.md`](reproducing.md#5-reference-baselines-the-propella-1-models); on a newer vLLM
with a faster structured-output backend (or by generating free-form JSON and parsing it) that run
should be revisited, and would replace the fp16 baseline row wholesale.

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

`propella-bekko benchmark` reports docs/s on the held-out documents (same GPU for
all models). propella-1 baselines are run on the same hardware.

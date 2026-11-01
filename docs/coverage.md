# Coverage: the label space and the languages

Two properties are not simply "everything propella-1 can do", and both are visible on the cards:
`country_relevance` predicts 65 values, and the models are trained on 17 languages. This is what
those two limits are, and what they cost.

## `country_relevance`: 65 of the 749 values that occur

propella-1's `country_relevance` is open vocabulary — the prompt asks for standard country names,
"any region worldwide", plus `supranational` and `none`. There is no fixed enumeration to copy, so
the label set has to be built from the training labels:

| | |
|---|---|
| documents in the training pool | 844,912 |
| country annotations in it | 1,053,007 |
| distinct values that occur | 749 |
| countries with ≥ 10,000 annotations | 21 |
| countries with ≥ 1,000 annotations | 60 |
| countries with ≥ 100 annotations | 167 |
| values kept (most frequent 64 + `other`) | 65 |
| share of all annotations inside those 64 | **96.19%** |
| share outside (684 values) | 3.81% |
| documents whose every country is outside the 64 | 0.78% |

So the cut is a data decision, not a modelling one: 64 countries carry 96% of the annotations, and
the remaining 684 values average a few dozen examples each, which is not enough to learn from. The
0.78% of documents that can then only be answered `other` are the price. `TrainConfig.country_top_k`
is a knob — `configs/a25m-xl-c150.json` is the 150-value variant — but the wider label space scores
lower overall (mean macro-F1 0.612 against 0.649), because macro-F1 over more, rarer labels is
arithmetically harsher; the comparison is not clean enough to call the extra 86 values useless.

Both the student and propella-1-4b are scored against the same 65 values, so the gap on that
property (0.29 against 0.52 macro-F1 on the 300-document reference set) is model quality on the same
label space, not the vocabulary.

## Languages: 17, and not all of Europe

The pool was built from the Dynaword corpora, FineWeb-2 and FinePDFs (`scripts/build_pool_xl.sh`).
Those sources contain 17 languages, of which 14 are official languages of the EU:

| language | documents | EU official |
|---|---|---|
| Danish | 167,392 | ✓ |
| Norwegian (Bokmål) | 128,164 | — |
| Dutch | 103,337 | ✓ |
| Icelandic | 56,020 | — |
| German | 40,000 | ✓ |
| Finnish | 40,000 | ✓ |
| French | 40,000 | ✓ |
| Italian | 40,000 | ✓ |
| Spanish | 40,000 | ✓ |
| Swedish | 40,000 | ✓ |
| Faroese | 29,999 | — |
| Czech | 20,000 | ✓ |
| English | 20,000 | ✓ |
| Hungarian | 20,000 | ✓ |
| Polish | 20,000 | ✓ |
| Portuguese | 20,000 | ✓ |
| Romanian | 20,000 | ✓ |

The ten EU official languages that are missing: **Bulgarian, Croatian, Estonian, Greek, Irish,
Latvian, Lithuanian, Maltese, Slovak, Slovenian**. Non-EU European languages missing as well:
Ukrainian, Serbian, Macedonian, Bosnian, Albanian, Turkish, Catalan, Galician, Basque.

That is a data-selection limit, not a teacher one: `openeurollm/propella-annotations` carries all of
them (`hplt-4-unfiltered` has `bul_Cyrl`, `hrv_Latn`, `ell_Grek`, `gle_Latn`, `lvs_Latn`, `lit_Latn`,
`mlt_Latn`, `slk_Latn`, `slv_Latn`, `ukr_Cyrl`, `srp_Cyrl`, `mkd_Cyrl`, `bos_Latn`, `als_Latn`,
`tur_Latn`, `cat_Latn`, `glg_Latn`, `eus_Latn`), while the pool builder only samples `fineweb-2` and
`finepdfs`. Adding those sources and rebuilding the pool is what it takes to cover the rest of
Europe; the student itself needs no architectural change.

Per-language quality on the 1,000-document held-out set is reported in `runs/pareto_<model>.json`
(`by_lang`), but each language has only 11–80 documents there, so treat those numbers as indicative.

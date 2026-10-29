# 009 — inspecting actual disagreements

Scripts: `scripts/inspect_errors.py` (print examples), `scripts/inherited_errors.py`
(split errors into inherited/own/fixed). Reference: DeepSeek-V4.1-Flash labels on the 1k test
set; teacher: stored propella-1-4b labels.

## What the examples show

### 1. `content_length` — the model under-rates long documents

| document | words | teacher / reference | student |
|---|---|---|---|
| Hungarian safety data sheet (30.7k chars) | 3,803 | substantial (2k+) | **moderate** |
| Danish journal article (52.9k chars) | 8,153 | substantial | **moderate** |
| Danish municipal meeting (6.2k chars) | 910 | moderate (500–2k) | **brief** |
| French hostel page (518 chars) | 87 | minimal (<100) | **brief** |

The rubric defines this property **by word count** (100 / 500 / 2k thresholds), yet the model
misplaces documents across thresholds in a consistent direction. The old length features were
`log1p(chars)/10, log1p(words)/10, truncated` — a weak, non-monotone signal for thresholds the
label is literally defined by.

**Fix applied**: `rich_length` — `log10(words)`, `log10(chars)`, `truncated`, plus indicator
features for the rubric thresholds (words > 100 / 500 / 2000 / 5000).

### 2. `business_sector` / `content_type` — the model predicts too few labels

Examples (all "teacher and reference agree, student differs"):

| document | teacher / reference | student |
|---|---|---|
| German product page | retail_commerce + advertising_marketing | retail_commerce |
| German PDF (2.3k words) | academic_research + general_interest | academic_research |
| Danish PDF (3.4k words) | healthcare_medical + government_public + nonprofit_ngo | healthcare_medical |
| Danish book (124k words) | media_entertainment + general_interest | media_entertainment |

The training data is not short of multi-label examples (business_sector: 40% of documents have
two labels, 23% have three or more), so this is a decoding/calibration bias: under a sigmoid at
0.5, predicting fewer labels minimises BCE.

**Fix applied**: a **label-count head** per multi-select property (predicts 1–5 labels), used at
decode time as top-k instead of a fixed threshold (`count_head`, default on).

### 3. Multi-select "errors" that are not errors

Several apparent business_sector disagreements were set-order differences
(`['government_public','education_sector']` vs `['education_sector','government_public']`).
Those are comparison artefacts, now fixed in the inspection script.

## Next

`a25m-v2` = same recipe as `a25m-xl` (845k docs) plus `rich_length` + `count_head`, queued behind
the clean-pool run. Expected to move `content_length` (currently 0.889 QWK) and the multi-select
IoU (0.79), i.e. the two places where inspection found a systematic, explainable error.

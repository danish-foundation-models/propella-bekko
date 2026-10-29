# Error analysis — a25m-xl on the 1,000-doc test set

Student: `runs/a25m-xl` (25M active, 845k training docs). Reference: stored propella-1-4b labels.

Overall **0.811** (QWK 0.812, IoU 0.792, PII-F1 0.904)

## Ordinal properties: dominant confusions

- **content_integrity** (117 of 900 off): fragment→complete (48), mostly_complete→complete (44), complete→fragment (16). Model under-rates.
- **content_ratio** (67 of 900 off): mostly_content→complete_content (34), complete_content→mostly_content (14), mostly_content→mixed_content (4). Model under-rates.
- **content_length** (160 of 900 off): substantial→moderate (62), brief→moderate (33), moderate→brief (23). Model over-rates (predicts a better level than the teacher).
- **content_quality** (171 of 900 off): good→excellent (48), excellent→good (41), adequate→good (24). Model under-rates.
- **information_density** (165 of 900 off): adequate→dense (46), dense→adequate (29), moderate→adequate (26). Model under-rates.
- **educational_value** (283 of 900 off): minimal→none (52), basic→minimal (43), none→minimal (38). Model over-rates (predicts a better level than the teacher).
- **reasoning_indicators** (217 of 900 off): none→minimal (24), minimal→none (24), analytical→explanatory (23). Model balanced.
- **audience_level** (126 of 900 off): expert→advanced (25), advanced→expert (21), general→advanced (20). Model balanced.
- **commercial_bias** (83 of 900 off): minimal→none (20), minimal→moderate (12), heavy→moderate (9). Model balanced.
- **time_sensitivity** (161 of 900 off): evergreen→slowly_changing (29), slowly_changing→evergreen (24), regularly_updating→slowly_changing (22). Model balanced.
- **content_safety** (11 of 900 off): mild_concerns→safe (8), safe→mild_concerns (1), nsfw→safe (1). Model under-rates.

## Multi-select: per-label bias (predicted vs teacher frequency)

- **content_type**: under-predicted procedural (-3.2pp), opinion_editorial (-2.3pp), review_critique (-2.2pp); over-predicted analytical (+2.8pp), legal_document (+2.8pp), creative (+1.7pp)
- **business_sector**: under-predicted nonprofit_ngo (-2.4pp), media_entertainment (-2.4pp), consulting_professional (-1.8pp); over-predicted general_interest (+5.4pp), government_public (+5.2pp), academic_research (+1.8pp)
- **technical_content**: under-predicted basic_technical (-0.7pp), engineering (-0.3pp), data_heavy (-0.2pp); over-predicted non_technical (+10.3pp), scientific (+1.8pp), code_heavy (+0.0pp)
- **regional_relevance**: under-predicted global (-1.7pp), middle_eastern (-1.0pp), north_american (-0.8pp); over-predicted european (+12.1pp), southeast_asian (+0.2pp), latin_american (+0.2pp)
- **country_relevance**: under-predicted other (-3.0pp), united_kingdom (-2.4pp), sweden (-2.1pp); over-predicted denmark (+4.2pp), norway (+3.4pp), none (+1.2pp)

## Slices (mean QWK over ordinal properties / mean IoU over multi-select)


**domain** (worst first, n≥25)
- Report: n=28, overall 0.577 (QWK 0.436, IoU 0.804)
- Conversation: n=68, overall 0.610 (QWK 0.534, IoU 0.732)
- Books: n=98, overall 0.630 (QWK 0.561, IoU 0.811)
- News: n=35, overall 0.696 (QWK 0.627, IoU 0.787)
- Social Media: n=31, overall 0.696 (QWK 0.645, IoU 0.769)
- Other: n=71, overall 0.767 (QWK 0.747, IoU 0.798)
- PDF: n=210, overall 0.771 (QWK 0.760, IoU 0.764)
- Encyclopedic: n=78, overall 0.790 (QWK 0.802, IoU 0.821)
- Legal: n=198, overall 0.791 (QWK 0.744, IoU 0.864)
- Web: n=142, overall 0.797 (QWK 0.825, IoU 0.733)

**language** (worst first, n≥25)
- dan: n=295, overall 0.726 (QWK 0.687, IoU 0.784)
- fao: n=35, overall 0.730 (QWK 0.738, IoU 0.760)
- spa_Latn: n=30, overall 0.776 (QWK 0.800, IoU 0.715)
- deu_Latn: n=28, overall 0.782 (QWK 0.805, IoU 0.704)
- isl: n=92, overall 0.785 (QWK 0.771, IoU 0.817)
- nld: n=102, overall 0.791 (QWK 0.751, IoU 0.847)
- fra_Latn: n=31, overall 0.801 (QWK 0.821, IoU 0.732)
- fin_Latn: n=37, overall 0.817 (QWK 0.829, IoU 0.753)
- nor: n=170, overall 0.836 (QWK 0.828, IoU 0.844)
- ita_Latn: n=29, overall 0.846 (QWK 0.863, IoU 0.806)
- swe_Latn: n=29, overall 0.847 (QWK 0.862, IoU 0.795)

**length** (worst first, n≥25)
- >20k: n=271, overall 0.660 (QWK 0.582, IoU 0.791)
- ≤1k chars: n=258, overall 0.773 (QWK 0.765, IoU 0.759)
- 5k–20k: n=170, overall 0.774 (QWK 0.746, IoU 0.817)
- 1k–5k: n=301, overall 0.786 (QWK 0.764, IoU 0.804)

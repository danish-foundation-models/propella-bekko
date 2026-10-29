# Error analysis — a25m-xl on the 1,000-doc test set

Student: `runs/a25m-xl` (25M active, 845k training docs). Reference: stored propella-1-4b labels.

Overall **0.708** (QWK 0.671, IoU 0.771, PII-F1 0.803)

## Ordinal properties: dominant confusions

- **content_integrity** (332 of 1000 off): mostly_complete→complete (175), fragment→complete (120), complete→fragment (15). Model under-rates.
- **content_ratio** (121 of 1000 off): mostly_content→complete_content (77), complete_content→mostly_content (16), minimal_content→complete_content (7). Model under-rates.
- **content_length** (301 of 1000 off): substantial→moderate (165), moderate→brief (81), brief→minimal (33). Model over-rates (predicts a better level than the teacher).
- **content_quality** (468 of 1000 off): good→excellent (222), adequate→good (131), adequate→excellent (30). Model under-rates.
- **information_density** (351 of 1000 off): adequate→dense (227), thin→dense (33), moderate→dense (16). Model under-rates.
- **educational_value** (435 of 1000 off): minimal→none (190), basic→minimal (56), basic→moderate (49). Model over-rates (predicts a better level than the teacher).
- **reasoning_indicators** (365 of 1000 off): minimal→none (110), basic_reasoning→explanatory (49), basic_reasoning→minimal (38). Model over-rates (predicts a better level than the teacher).
- **audience_level** (214 of 1000 off): advanced→expert (70), general→advanced (52), advanced→general (30). Model under-rates.
- **commercial_bias** (94 of 1000 off): none→moderate (18), minimal→moderate (15), heavy→pure_marketing (14). Model over-rates (predicts a better level than the teacher).
- **time_sensitivity** (219 of 1000 off): time_sensitive→slowly_changing (50), slowly_changing→evergreen (43), regularly_updating→slowly_changing (27). Model under-rates.
- **content_safety** (32 of 1000 off): mild_concerns→safe (23), mild_concerns→nsfw (3), nsfw→safe (2). Model under-rates.

## Multi-select: per-label bias (predicted vs teacher frequency)

- **content_type**: under-predicted reference (-5.8pp), opinion_editorial (-2.6pp), boilerplate (-2.0pp); over-predicted legal_document (+5.5pp), creative (+1.6pp), structured_data (+1.6pp)
- **business_sector**: under-predicted media_entertainment (-4.9pp), education_sector (-3.3pp), legal_services (-3.0pp); over-predicted government_public (+6.4pp), general_interest (+5.8pp), real_estate_construction (+1.3pp)
- **technical_content**: under-predicted basic_technical (-1.7pp), engineering (-0.6pp), math_heavy (-0.1pp); over-predicted scientific (+2.4pp), data_heavy (+0.7pp), non_technical (+0.5pp)
- **regional_relevance**: under-predicted culturally_neutral (-3.5pp), global (-1.7pp), indeterminate (-1.3pp); over-predicted european (+6.0pp), latin_american (+0.4pp), east_asian (+0.2pp)
- **country_relevance**: under-predicted faroe_islands (-3.3pp), supranational (-2.3pp), other (-2.1pp); over-predicted iceland (+3.8pp), denmark (+3.0pp), china (+0.2pp)

## Slices (mean QWK over ordinal properties / mean IoU over multi-select)


**domain** (worst first, n≥25)
- None: n=1000, overall 0.708 (QWK 0.671, IoU 0.771)

**language** (worst first, n≥25)
- None: n=1000, overall 0.708 (QWK 0.671, IoU 0.771)

**length** (worst first, n≥25)
- ≤1k chars: n=1000, overall 0.708 (QWK 0.671, IoU 0.771)

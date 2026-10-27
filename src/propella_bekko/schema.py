"""Label space of the propella-1 annotation framework.

Enum values are taken from the propella-1 system prompt
(https://huggingface.co/ellamind/propella-1-4b/blob/main/propella.py).

`one_sentence_description` is free text and is not distilled.
`country_relevance` is open vocabulary; its label set is built from the
training data (top-k most frequent values + `other`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["single", "multi"]


@dataclass
class Field:
    name: str
    kind: Kind
    labels: list[str]
    # Ordinal fields are ordered best -> worst (as in the propella prompt) and
    # additionally evaluated with MAE / off-by-one accuracy.
    ordinal: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def num_labels(self) -> int:
        return len(self.labels)


def _s(name: str, labels: str, ordinal: bool = False) -> Field:
    return Field(name, "single", labels.split("|"), ordinal=ordinal)


def _m(name: str, labels: str) -> Field:
    return Field(name, "multi", labels.split("|"))


PROPELLA_FIELDS: list[Field] = [
    _s("content_integrity", "complete|mostly_complete|fragment|severely_degraded", True),
    _s(
        "content_ratio",
        "complete_content|mostly_content|mixed_content|mostly_navigation|minimal_content",
        True,
    ),
    _s("content_length", "substantial|moderate|brief|minimal", True),
    _m(
        "content_type",
        "analytical|instructional|reference|procedural|qa_structured|conversational|creative|"
        "transactional|boilerplate|news_report|opinion_editorial|review_critique|"
        "technical_documentation|specification_standard|legal_document|press_release|"
        "structured_data|source_code",
    ),
    _m(
        "business_sector",
        "academic_research|education_sector|technology_software|hardware_electronics|"
        "healthcare_medical|pharmaceutical_biotech|financial_services|legal_services|"
        "government_public|manufacturing_industrial|mining_resources|chemicals_materials|"
        "energy_utilities|retail_commerce|wholesale_distribution|real_estate_construction|"
        "transportation_logistics|automotive_industry|telecommunications|media_entertainment|"
        "advertising_marketing|hospitality_tourism|agriculture_food|environmental_services|"
        "aerospace_defense|insurance_industry|nonprofit_ngo|consulting_professional|"
        "human_resources|security_cyber|gaming_industry|gambling_betting|travel_aviation|"
        "food_beverage_hospitality|consumer_goods|general_interest|other",
    ),
    _m(
        "technical_content",
        "code_heavy|math_heavy|scientific|data_heavy|engineering|basic_technical|non_technical",
    ),
    _s("content_quality", "excellent|good|adequate|poor|unacceptable", True),
    _s("information_density", "dense|adequate|moderate|thin|empty", True),
    _s("educational_value", "high|moderate|basic|minimal|none", True),
    _s(
        "reasoning_indicators",
        "analytical|explanatory|basic_reasoning|minimal|none",
        True,
    ),
    _s("audience_level", "expert|advanced|general|beginner|youth|children", True),
    _s("commercial_bias", "none|minimal|moderate|heavy|pure_marketing", True),
    _s(
        "time_sensitivity",
        "evergreen|slowly_changing|regularly_updating|time_sensitive",
        True,
    ),
    _s("content_safety", "safe|mild_concerns|nsfw|harmful|illegal", True),
    _s("pii_presence", "no_pii|contains_pii"),
    _m(
        "regional_relevance",
        "european|north_american|east_asian|south_asian|southeast_asian|middle_eastern|"
        "sub_saharan_african|latin_american|oceanian|central_asian|russian_sphere|global|"
        "culturally_neutral|indeterminate",
    ),
    # labels filled in by build_country_field()
    Field("country_relevance", "multi", []),
]

FIELD_NAMES = [f.name for f in PROPELLA_FIELDS]
OTHER = "other"


def build_country_field(counts: dict[str, int], top_k: int = 64) -> Field:
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in top if k != OTHER][:top_k]
    return Field("country_relevance", "multi", [*labels, OTHER])


def fields_from_config(cfg: list[dict]) -> list[Field]:
    return [Field(**d) for d in cfg]


def fields_to_config(fields: list[Field]) -> list[dict]:
    return [
        {
            "name": f.name,
            "kind": f.kind,
            "labels": f.labels,
            "ordinal": f.ordinal,
            "extra": f.extra,
        }
        for f in fields
    ]


def encode_labels(row: dict, fields: list[Field]) -> dict[str, list[int] | int]:
    """Map a propella annotation row to integer targets.

    single -> class index (-100 if missing/unknown)
    multi  -> multi-hot list (all -100 if missing)
    """
    out: dict[str, list[int] | int] = {}
    for f in fields:
        v = row.get(f.name)
        idx = {lab: i for i, lab in enumerate(f.labels)}
        if f.kind == "single":
            out[f.name] = idx.get(v, -100) if isinstance(v, str) else -100
        else:
            if v is None or len(v) == 0:
                out[f.name] = [-100] * f.num_labels
                continue
            hot = [0] * f.num_labels
            for item in v:
                j = idx.get(item)
                if j is None and OTHER in idx:
                    j = idx[OTHER]
                if j is not None:
                    hot[j] = 1
            out[f.name] = hot
    return out

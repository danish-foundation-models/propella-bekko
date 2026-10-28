import torch

from propella_bekko.data.sources import TRUNC_MARK, TruncCfg, truncate_text
from propella_bekko.inference import token_budget_batches
from propella_bekko.metrics import field_metrics, summarize
from propella_bekko.models import PropellaBekko, StaticBackbone
from propella_bekko.schema import PROPELLA_FIELDS, build_country_field, encode_labels

FIELDS = [f for f in PROPELLA_FIELDS if f.name != "country_relevance"] + [
    build_country_field({"denmark": 5, "sweden": 3}, top_k=8)
]

ROW = {
    "content_integrity": "complete",
    "content_type": ["news_report", "not_a_label"],
    "country_relevance": ["denmark", "chile"],
    "content_quality": None,
}


def test_encode_labels():
    y = encode_labels(ROW, FIELDS)
    assert y["content_integrity"] == 0
    assert y["content_quality"] == -100
    ct = next(f for f in FIELDS if f.name == "content_type")
    assert sum(y["content_type"]) == 1
    assert y["content_type"][ct.labels.index("news_report")] == 1
    cr = FIELDS[-1]
    assert y["country_relevance"][cr.labels.index("other")] == 1  # chile -> other
    assert (
        y["business_sector"]
        == [-100] * next(f for f in FIELDS if f.name == "business_sector").num_labels
    )


def test_truncate():
    t = "a" * 100
    assert truncate_text(t, TruncCfg(200, 10)) == t
    out = truncate_text(t, TruncCfg(20, 10))
    assert out.startswith("a" * 20) and TRUNC_MARK in out and out.endswith("a" * 10)


def test_token_budget_batches():
    lengths = [5, 100, 7, 50, 3, 99]
    batches = list(token_budget_batches(lengths, max_tokens=200))
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert max(lengths[i] for i in b) * len(b) <= 200 or len(b) == 1


def test_model_roundtrip(tmp_path):
    bb = StaticBackbone(100, 16, pad_id=0)
    model = PropellaBekko(bb, FIELDS, head_hidden=32)
    ids = torch.randint(1, 100, (3, 12))
    mask = torch.ones_like(ids)
    mask[0, 6:] = 0
    feats = dict(
        n_chars=torch.tensor([10, 20, 30]),
        n_words=torch.tensor([2, 4, 6]),
        truncated=torch.tensor([False, True, False]),
    )
    logits = model(ids, mask, **feats)
    rows = model.decode(logits)
    assert len(rows) == 3
    for r in rows:
        for f in FIELDS:
            if f.kind == "single":
                assert r[f.name] in f.labels
            else:
                assert len(r[f.name]) >= 1

    class Tok:  # minimal tokenizer stub
        def save_pretrained(self, p):
            pass

    model.save(tmp_path, Tok())
    import json

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["backbone_spec"]["kind"] == "static"


def test_metrics_perfect():
    gold = [
        {"content_integrity": "complete", "content_type": ["news_report"]},
        {"content_integrity": "fragment", "content_type": ["creative"]},
    ]
    res = field_metrics(gold, gold, FIELDS)
    assert res["content_integrity"]["accuracy"] == 1.0
    assert res["content_type"]["jaccard"] == 1.0
    assert summarize(res)["mean_single_accuracy"] == 1.0


def test_refine_and_pooling_variants():
    for pool in ("mean", "attn"):
        bb = StaticBackbone(100, 16, pad_id=0, pool=pool)
        model = PropellaBekko(bb, FIELDS, head_hidden=32, refine=True)
        ids = torch.randint(1, 100, (2, 10))
        mask = torch.ones_like(ids)
        feats = dict(
            n_chars=torch.tensor([5, 9]),
            n_words=torch.tensor([1, 2]),
            truncated=torch.tensor([False, True]),
        )
        logits = model(ids, mask, **feats)
        assert set(logits) == {f.name for f in FIELDS}
        targets = {f.name: torch.zeros(2, dtype=torch.long) for f in FIELDS if f.kind == "single"}
        targets |= {f.name: torch.zeros(2, f.num_labels) for f in FIELDS if f.kind == "multi"}
        loss, _ = model.loss(logits, targets, ordinal_smoothing=0.1)
        assert torch.isfinite(loss)
        assert len(model.decode(logits)) == 2

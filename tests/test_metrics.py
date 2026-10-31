def test_unknown_multi_labels_are_dropped():
    """Free-form output can name a label the taxonomy does not have."""
    from propella_bekko.metrics import field_metrics
    from propella_bekko.schema import Field

    field = Field("country_relevance", "multi", ["denmark", "norway"])
    preds = [{"country_relevance": ["denmark", "rwanda"]}]
    golds = [{"country_relevance": ["denmark"]}]
    m = field_metrics(preds, golds, [field])
    assert m["country_relevance"]["jaccard"] == 1.0  # rwanda dropped, denmark kept

from ragops.confidence import assess_retrieval_confidence


def test_confidence_gate_passes_strong_rerank_score():
    result = assess_retrieval_confidence(
        [{"id": "doc_1", "rerank_score": 2.5}],
        threshold=-3.0,
    )
    assert result["sufficient_evidence"] is True
    assert result["gate"] == "pass"


def test_confidence_gate_abstains_on_weak_rerank_score():
    result = assess_retrieval_confidence(
        [{"id": "doc_1", "rerank_score": -3.8}],
        threshold=-3.0,
    )
    assert result["sufficient_evidence"] is False
    assert result["gate"] == "abstain"

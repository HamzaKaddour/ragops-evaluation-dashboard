from ragops.citations import extract_citations, validate_citations


def test_extract_and_validate_citations():
    answer = "Use hybrid retrieval [doc_06], then rerank [doc_09]. Ignore [fake_1]."
    assert extract_citations(answer) == ["doc_06", "doc_09", "fake_1"]
    result = validate_citations(answer, {"doc_06", "doc_09"})
    assert result["valid_citation_count"] == 2
    assert result["invalid_citation_count"] == 1
    assert result["invalid_citations"] == ["fake_1"]
    assert result["citation_validity"] == 2 / 3


def test_zero_citations_are_not_perfect_validity():
    result = validate_citations("A factual answer without a citation.", {"doc_01"})
    assert result["citation_count"] == 0
    assert result["citation_validity"] is None
    assert result["citation_coverage"] == 0.0


def test_abstention_citation_metrics_are_not_applicable():
    result = validate_citations(
        "The available evidence is insufficient to answer this question.",
        {"doc_01"},
        abstained=True,
    )
    assert result["citation_validity"] is None
    assert result["citation_coverage"] is None
    assert result["citation_status"] == "not_applicable"

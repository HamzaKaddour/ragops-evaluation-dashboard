import pytest

from ragops.citations import extract_citations, validate_citations


def test_extract_and_validate_citations():
    answer = "Use hybrid retrieval [doc_06], then rerank [doc_09]. Ignore [fake_1]."
    assert extract_citations(answer) == ["doc_06", "doc_09", "fake_1"]

    result = validate_citations(answer, {"doc_06", "doc_09"})

    assert result["valid_citation_count"] == 2
    assert result["invalid_citation_count"] == 1
    assert result["invalid_citations"] == ["fake_1"]
    assert result["citation_validity"] == pytest.approx(2 / 3, abs=1e-3)


def test_no_citations_returns_null_validity():
    result = validate_citations("No citations in this answer.", {"doc_01"})

    assert result["citation_count"] == 0
    assert result["valid_citation_count"] == 0
    assert result["invalid_citation_count"] == 0
    assert result["citation_validity"] is None

from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def extract_citations(text: str) -> list[str]:
    """Return citation identifiers in appearance order."""
    return CITATION_RE.findall(text)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if len(s.strip()) > 3]


def citation_coverage(answer: str, *, abstained: bool = False) -> float | None:
    """Estimate sentence-level citation coverage.

    For an abstention there are no factual answer claims to cover, so the metric is
    not applicable and returns ``None``. For a normal answer, every non-empty
    sentence is treated as a claim and is considered covered if it contains at
    least one citation marker.
    """
    if abstained:
        return None
    sentences = _sentences(answer)
    if not sentences:
        return 0.0
    covered = sum(bool(extract_citations(sentence)) for sentence in sentences)
    return round(covered / len(sentences), 3)


def validate_citations(answer: str, allowed_doc_ids: set[str], *, abstained: bool = False) -> dict:
    """Validate citation IDs and report both validity and coverage.

    ``citation_validity`` is ``None`` when no citations are present; a response
    with zero citations is therefore not incorrectly scored as perfectly valid.
    """
    citations = extract_citations(answer)
    unique = list(dict.fromkeys(citations))
    valid = sorted({c for c in unique if c in allowed_doc_ids})
    invalid = sorted({c for c in unique if c not in allowed_doc_ids})

    validity = None if not unique else round(len(valid) / len(unique), 3)
    coverage = citation_coverage(answer, abstained=abstained)

    return {
        "citations": unique,
        "citation_count": len(unique),
        "valid_citation_count": len(valid),
        "invalid_citation_count": len(invalid),
        "invalid_citations": invalid,
        "citation_validity": validity,
        "citation_coverage": coverage,
        "citation_status": "not_applicable" if abstained else "evaluated",
    }

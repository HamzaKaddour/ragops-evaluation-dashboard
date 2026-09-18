from __future__ import annotations

import os

DEFAULT_RERANK_THRESHOLD = float(os.getenv("RAGOPS_RERANK_THRESHOLD", "-3.0"))

ABSTENTION_ANSWER = (
    "The available evidence is insufficient to answer this question."
)


def top_rerank_score(sources: list[dict]) -> float | None:
    """Return the strongest CrossEncoder score from retrieved sources."""
    scores = [
        float(source["rerank_score"])
        for source in sources
        if source.get("rerank_score") is not None
    ]
    return max(scores) if scores else None


def assess_retrieval_confidence(
    sources: list[dict],
    threshold: float | None = None,
) -> dict:
    """Apply a simple, configurable retrieval-confidence gate.

    The default threshold is calibrated to this repository's v1 benchmark and
    should be re-evaluated if the corpus, reranker, or evaluation set changes.
    """
    threshold = DEFAULT_RERANK_THRESHOLD if threshold is None else float(threshold)
    score = top_rerank_score(sources)
    sufficient = score is not None and score >= threshold
    return {
        "sufficient_evidence": sufficient,
        "top_rerank_score": None if score is None else round(score, 4),
        "rerank_threshold": threshold,
        "gate": "pass" if sufficient else "abstain",
    }

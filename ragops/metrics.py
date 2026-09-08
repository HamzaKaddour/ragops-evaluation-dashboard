from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(1 for doc_id in top if doc_id in relevant_ids) / len(top)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    gains = [1 if doc_id in relevant_ids else 0 for doc_id in ranked_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> dict[str, float]:
    return {
        f"recall_at_{k}": recall_at_k(ranked_ids, relevant_ids, k),
        f"precision_at_{k}": precision_at_k(ranked_ids, relevant_ids, k),
        "mrr": reciprocal_rank(ranked_ids, relevant_ids),
        f"ndcg_at_{k}": ndcg_at_k(ranked_ids, relevant_ids, k),
    }

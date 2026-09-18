from __future__ import annotations

import math


def recall_at_k(results: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(results[:k]) & relevant) / len(relevant)


def precision_at_k(results: list[str], relevant: set[str], k: int) -> float:
    top = results[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def reciprocal_rank(results: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(results, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[str], relevant: set[str], k: int) -> float:
    gains = [1 if item in relevant else 0 for item in results[:k]]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg

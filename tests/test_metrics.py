from ragops.metrics import evaluate_ranking, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_perfect_ranking_metrics():
    ranked = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert precision_at_k(ranked, relevant, 2) == 1.0
    assert reciprocal_rank(ranked, relevant) == 1.0
    assert ndcg_at_k(ranked, relevant, 2) == 1.0


def test_partial_ranking_metrics():
    metrics = evaluate_ranking(["x", "a", "y"], {"a", "b"}, 3)
    assert metrics["recall_at_3"] == 0.5
    assert round(metrics["precision_at_3"], 4) == 0.3333
    assert metrics["mrr"] == 0.5
    assert 0.0 < metrics["ndcg_at_3"] < 1.0


def test_empty_relevance_is_safe():
    metrics = evaluate_ranking(["a"], set(), 5)
    assert metrics["recall_at_5"] == 0.0
    assert metrics["mrr"] == 0.0

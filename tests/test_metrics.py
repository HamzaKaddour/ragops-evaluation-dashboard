from ragops.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_ranking_metrics():
    results = ["a", "b", "c"]
    relevant = {"b", "c"}
    assert recall_at_k(results, relevant, 3) == 1.0
    assert precision_at_k(results, relevant, 3) == 2 / 3
    assert reciprocal_rank(results, relevant) == 0.5
    assert 0 < ndcg_at_k(results, relevant, 3) <= 1

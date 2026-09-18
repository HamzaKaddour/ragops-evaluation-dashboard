"""Run retrieval evaluation over the labeled sample query set.

This script evaluates BM25, dense FAISS, hybrid, and hybrid+CrossEncoder retrieval.
It does not call the local LLM, so it is useful for testing retrieval independently.
"""
from __future__ import annotations

import json
from pathlib import Path

from ragops.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from ragops.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


def evaluate() -> dict:
    queries = json.loads((ROOT / "data" / "evaluation_queries.json").read_text(encoding="utf-8"))
    retriever = HybridRetriever(ROOT / "data" / "corpus.json")
    methods = {
        "bm25": lambda q: retriever.bm25_search(q, k=5),
        "dense_faiss": lambda q: retriever.dense_search(q, k=5),
        "hybrid": lambda q: retriever.hybrid_search(q, k=5),
        "hybrid_rerank": lambda q: retriever.hybrid_rerank(q, k=5),
    }
    output = {"pipelines": {}}
    for name, fn in methods.items():
        rows = []
        for q in queries:
            results = fn(q["query"])
            ids = [r["id"] for r in results]
            relevant = set(q["relevant_doc_ids"])
            rows.append({
                "query_id": q["id"],
                "query": q["query"],
                "retrieved_ids": ids,
                "recall_at_5": round(recall_at_k(ids, relevant, 5), 3),
                "precision_at_5": round(precision_at_k(ids, relevant, 5), 3),
                "mrr": round(reciprocal_rank(ids, relevant), 3),
                "ndcg_at_5": round(ndcg_at_k(ids, relevant, 5), 3),
            })
        output["pipelines"][name] = {
            "queries": rows,
            "mean_recall_at_5": round(sum(r["recall_at_5"] for r in rows) / len(rows), 3),
            "mean_precision_at_5": round(sum(r["precision_at_5"] for r in rows) / len(rows), 3),
            "mrr": round(sum(r["mrr"] for r in rows) / len(rows), 3),
            "mean_ndcg_at_5": round(sum(r["ndcg_at_5"] for r in rows) / len(rows), 3),
        }
    return output


if __name__ == "__main__":
    result = evaluate()
    path = ROOT / "data" / "real_evaluation_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {path}")

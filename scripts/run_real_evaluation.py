"""Run real BM25, dense, hybrid, and reranked retrieval evaluation.

Requires the optional real-RAG dependencies in requirements-real.txt. Model weights
are downloaded by SentenceTransformers on first use and cached by Hugging Face.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragops.metrics import evaluate_ranking
from ragops.retrieval import RAGRetriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    corpus = load_json(DATA / "corpus.json")
    queries = load_json(DATA / "evaluation_queries.json")
    retriever = RAGRetriever(corpus)

    methods = {
        "bm25": retriever.bm25,
        "dense_faiss": retriever.dense,
        "hybrid": retriever.hybrid,
        "hybrid_crossencoder": retriever.hybrid_reranked,
    }

    output = {"pipelines": {}}
    for method_name, method in methods.items():
        query_runs = []
        for query in queries:
            results = method(query["query"], k=5)
            ranked_ids = [item.doc_id for item in results]
            metrics = evaluate_ranking(ranked_ids, set(query["relevant_doc_ids"]), k=5)
            query_runs.append({
                "id": query["id"],
                "query": query["query"],
                "ranked_doc_ids": ranked_ids,
                "metrics": {key: round(value, 4) for key, value in metrics.items()},
                "results": [
                    {"doc_id": item.doc_id, "title": item.title, "score": round(item.score, 6)}
                    for item in results
                ],
            })

        keys = ["recall_at_5", "precision_at_5", "mrr", "ndcg_at_5"]
        aggregate = {
            key: round(sum(run["metrics"][key] for run in query_runs) / len(query_runs), 4)
            for key in keys
        }
        output["pipelines"][method_name] = {"aggregate": aggregate, "queries": query_runs}

    target = DATA / "real_evaluation_results.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()

"""Run the final end-to-end RAGOps benchmark on the workstation.

This is the authoritative portfolio benchmark for the final pipeline. It measures:

Retrieval-only:
- BM25
- dense FAISS
- hybrid retrieval
- hybrid + CrossEncoder reranking

End-to-end final pipeline:
- hybrid + CrossEncoder retrieval
- Qwen local generation
- citation validity / coverage
- abstention behavior
- embedding-based groundedness
- retrieval, generation, and total latency

Outputs:
- data/final_benchmark_summary.json
- data/final_benchmark_runs.json

The script downloads/loads the same public Hugging Face models used by the local
application. No paid API is used.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from ragops.citations import validate_citations
from ragops.confidence import ABSTENTION_ANSWER, assess_retrieval_confidence
from ragops.generation import LocalGenerator
from ragops.grounding import GroundingEvaluator, is_abstention
from ragops.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from ragops.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def avg(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def evaluate_retrieval(
    retriever: HybridRetriever,
    queries: list[dict],
    k: int,
) -> list[dict]:
    methods = {
        "BM25": lambda q: retriever.bm25_search(q, k=k),
        "Dense + FAISS": lambda q: retriever.dense_search(q, k=k),
        "Hybrid": lambda q: retriever.hybrid_search(q, k=k),
        "Hybrid + CrossEncoder": lambda q: retriever.hybrid_rerank(q, k=k),
    }
    summaries = []

    answerable = [q for q in queries if q.get("relevant_doc_ids")]
    for name, fn in methods.items():
        rows = []
        latencies = []
        for query in answerable:
            started = time.perf_counter()
            results = fn(query["query"])
            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies.append(elapsed_ms)

            ids = [r["id"] for r in results]
            relevant = set(query["relevant_doc_ids"])
            rows.append(
                {
                    "query_id": query["id"],
                    "recall_at_k": recall_at_k(ids, relevant, k),
                    "precision_at_k": precision_at_k(ids, relevant, k),
                    "mrr": reciprocal_rank(ids, relevant),
                    "ndcg_at_k": ndcg_at_k(ids, relevant, k),
                    "latency_ms": elapsed_ms,
                }
            )

        summaries.append(
            {
                "name": name,
                "queries_evaluated": len(rows),
                "recall_at_k": avg([r["recall_at_k"] for r in rows]),
                "precision_at_k": avg([r["precision_at_k"] for r in rows]),
                "mrr": avg([r["mrr"] for r in rows]),
                "ndcg_at_k": avg([r["ndcg_at_k"] for r in rows]),
                "avg_latency_ms": round(mean(latencies), 1) if latencies else None,
            }
        )
    return summaries


def evaluate_generation(
    retriever: HybridRetriever,
    generator: LocalGenerator,
    grounding: GroundingEvaluator,
    queries: list[dict],
    top_k: int,
) -> tuple[dict, list[dict]]:
    runs = []

    for idx, query in enumerate(queries, start=1):
        print(f"[{idx}/{len(queries)}] {query['query']}", flush=True)
        started = time.perf_counter()

        retrieval_started = time.perf_counter()
        sources = retriever.hybrid_rerank(query["query"], k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        confidence = assess_retrieval_confidence(sources)
        if confidence["sufficient_evidence"]:
            generation_started = time.perf_counter()
            generated = generator.generate(query["query"], sources)
            generation_ms = (time.perf_counter() - generation_started) * 1000
        else:
            generated = {
                "answer": ABSTENTION_ANSWER,
                "model": "retrieval-confidence-gate",
            }
            generation_ms = 0.0

        answer = generated["answer"]
        abstained = is_abstention(answer)
        citations = validate_citations(
            answer,
            {source["id"] for source in sources},
            abstained=abstained,
        )
        grounding_metrics = grounding.evaluate(answer, sources)
        total_ms = (time.perf_counter() - started) * 1000

        expected_status = query.get("expected_answer_status", "answered")
        observed_status = grounding_metrics["answer_status"]
        behavior_correct = observed_status == expected_status

        public_sources = [
            {
                "id": source["id"],
                "title": source["title"],
                "category": source["category"],
                "score": round(float(source.get("score", 0.0)), 4),
                "rerank_score": round(float(source.get("rerank_score", 0.0)), 4),
            }
            for source in sources
        ]

        runs.append(
            {
                "query_id": query["id"],
                "query": query["query"],
                "expected_answer_status": expected_status,
                "answer_status": observed_status,
                "behavior_correct": behavior_correct,
                "answer": answer,
                "model": generated["model"],
                "sources": public_sources,
                "metrics": {
                    **citations,
                    **grounding_metrics,
                    "retrieval_confidence": confidence,
                },
                "latency_ms": {
                    "retrieval": round(retrieval_ms, 1),
                    "generation": round(generation_ms, 1),
                    "total": round(total_ms, 1),
                },
            }
        )

    answered = [r for r in runs if r["answer_status"] == "answered"]
    validity = [
        r["metrics"]["citation_validity"]
        for r in answered
        if r["metrics"].get("citation_validity") is not None
    ]
    coverage = [
        r["metrics"]["citation_coverage"]
        for r in answered
        if r["metrics"].get("citation_coverage") is not None
    ]

    expected_abstentions = [
        r for r in runs if r["expected_answer_status"] == "insufficient_evidence"
    ]
    correct_abstentions = [
        r for r in expected_abstentions if r["answer_status"] == "insufficient_evidence"
    ]

    summary = {
        "queries_evaluated": len(runs),
        "answered": sum(r["answer_status"] == "answered" for r in runs),
        "abstained": sum(r["answer_status"] == "insufficient_evidence" for r in runs),
        "behavior_accuracy": avg([float(r["behavior_correct"]) for r in runs]),
        "abstention_accuracy": (
            round(len(correct_abstentions) / len(expected_abstentions), 3)
            if expected_abstentions
            else None
        ),
        "avg_groundedness": avg(
            [r["metrics"]["groundedness_score"] for r in runs]
        ),
        "avg_citation_validity": avg(validity),
        "avg_citation_coverage": avg(coverage),
        "avg_retrieval_ms": round(mean(r["latency_ms"]["retrieval"] for r in runs), 1),
        "avg_generation_ms": round(mean(r["latency_ms"]["generation"] for r in runs), 1),
        "avg_total_ms": round(mean(r["latency_ms"]["total"] for r in runs), 1),
        "confidence_gated_abstentions": sum(
            r["metrics"]["retrieval_confidence"]["gate"] == "abstain" for r in runs
        ),
    }
    return summary, runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the final benchmark.",
    )
    args = parser.parse_args()

    queries = json.loads((DATA / "evaluation_queries.json").read_text(encoding="utf-8"))
    if args.limit:
        queries = queries[: args.limit]

    print("Loading retrieval models...", flush=True)
    retriever = HybridRetriever(DATA / "corpus.json")

    print("Running retrieval benchmark...", flush=True)
    retrieval = evaluate_retrieval(retriever, queries, k=5)

    print("Loading local generator and grounding evaluator...", flush=True)
    generator = LocalGenerator()
    grounding = GroundingEvaluator()

    print("Running end-to-end benchmark...", flush=True)
    generation_summary, runs = evaluate_generation(
        retriever,
        generator,
        grounding,
        queries,
        top_k=args.top_k,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": "final-v2",
        "top_k_generation": args.top_k,
        "retrieval_k": 5,
        "models": {
            "embedding": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "generator": generator.model_name,
        },
        "retrieval_pipelines": retrieval,
        "end_to_end": generation_summary,
    }

    summary_path = DATA / "final_benchmark_summary.json"
    runs_path = DATA / "final_benchmark_runs.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")

    print("\nFinal benchmark complete.")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {runs_path}")


if __name__ == "__main__":
    main()

"""Generate static RAG evaluation artifacts for the GitHub Pages dashboard.

This script intentionally uses lightweight deterministic data so the project can run on a normal laptop and in GitHub Actions without external APIs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PIPELINES = [
    {
        "name": "BM25 baseline",
        "description": "Keyword-only retrieval baseline with no dense embeddings.",
        "base": [0.68, 0.42, 0.59, 0.63, 0.72, 0.66, 0.28, 310, 0.58],
        "failures": [7, 8, 5, 6],
    },
    {
        "name": "Dense embeddings",
        "description": "Semantic retrieval using compact sentence embeddings.",
        "base": [0.83, 0.57, 0.74, 0.79, 0.81, 0.78, 0.19, 640, 1.21],
        "failures": [3, 6, 3, 4],
    },
    {
        "name": "Hybrid retrieval",
        "description": "BM25 plus dense retrieval with weighted score fusion.",
        "base": [0.88, 0.63, 0.81, 0.85, 0.85, 0.83, 0.15, 725, 1.46],
        "failures": [2, 5, 2, 3],
    },
    {
        "name": "Hybrid + reranker",
        "description": "Hybrid retrieval followed by cross-encoder style reranking.",
        "base": [0.91, 0.69, 0.87, 0.90, 0.88, 0.87, 0.12, 842, 1.84],
        "failures": [1, 4, 1, 2],
    },
]


def build_summary() -> dict:
    columns = [
        "recall_at_5",
        "precision_at_5",
        "mrr",
        "ndcg_at_5",
        "groundedness",
        "citation_coverage",
        "hallucination_risk",
        "avg_latency_ms",
        "cost_per_1000_queries_usd",
    ]

    rows = []
    for pipeline in PIPELINES:
        row = dict(zip(columns, pipeline["base"]))
        row["name"] = pipeline["name"]
        row["description"] = pipeline["description"]
        row["failure_modes"] = dict(zip(
            ["missed_context", "partial_context", "unsupported_claim", "citation_gap"],
            pipeline["failures"],
        ))
        rows.append(row)

    df = pd.DataFrame(rows)
    best = df.sort_values(["groundedness", "recall_at_5", "ndcg_at_5"], ascending=False).iloc[0]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "RAGOps Evaluation Dashboard",
        "summary_metrics": {
            "best_pipeline": best["name"],
            "queries_evaluated": 24,
            "avg_recall_at_5": round(float(best["recall_at_5"]), 2),
            "avg_groundedness": round(float(best["groundedness"]), 2),
            "avg_hallucination_risk": round(float(best["hallucination_risk"]), 2),
            "avg_latency_ms": int(best["avg_latency_ms"]),
            "estimated_cost_per_1000_queries_usd": round(float(best["cost_per_1000_queries_usd"]), 2),
        },
        "pipelines": rows,
    }


def build_runs() -> list[dict]:
    rng = np.random.default_rng(42)
    queries = [
        "How should a production RAG system detect unsupported claims?",
        "Why can keyword-only retrieval fail in technical support assistants?",
        "What does reranking add after hybrid retrieval?",
        "How can RAG dashboards expose cost and latency trade-offs?",
        "What is the main benefit of evaluating RAG before deployment?",
    ]
    pipelines = ["Hybrid + reranker", "BM25 baseline", "Hybrid + reranker", "Dense embeddings", "Hybrid retrieval"]

    runs = []
    for idx, (query, pipeline) in enumerate(zip(queries, pipelines), start=1):
        quality = 0.9 if "Hybrid + reranker" in pipeline else 0.78 if "Hybrid" in pipeline else 0.72
        risk = max(0.05, 1 - quality + float(rng.normal(0, 0.02)))
        runs.append({
            "id": f"q{idx:03d}",
            "query": query,
            "pipeline": pipeline,
            "answer": "This answer is generated from retrieved evidence, then scored for grounding, citation coverage, latency, and hallucination risk.",
            "expected_topics": ["retrieval", "evaluation", "grounding"],
            "retrieved_sources": [
                {"title": "RAG evaluation playbook", "score": round(float(quality), 2), "snippet": "RAG systems should expose retrieval and answer-level evaluation metrics."},
                {"title": "LLMOps monitoring guide", "score": round(float(quality - 0.05), 2), "snippet": "Production LLM applications require traceability, cost monitoring, and regression testing."},
            ],
            "metrics": {
                "recall_at_5": round(float(quality), 2),
                "precision_at_5": round(float(quality - 0.18), 2),
                "groundedness": round(float(quality - 0.02), 2),
                "citation_coverage": round(float(quality - 0.04), 2),
                "hallucination_risk": round(float(risk), 2),
                "latency_ms": int(300 + quality * 650),
                "cost_usd": round(float(quality * 0.002), 4),
            },
            "failure_tags": [] if quality > 0.84 else ["partial_context"],
        })
    return runs


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    write_json(DATA_DIR / "evaluation_summary.json", build_summary())
    write_json(DATA_DIR / "rag_runs.json", build_runs())
    print(f"Wrote artifacts to {DATA_DIR}")


if __name__ == "__main__":
    main()

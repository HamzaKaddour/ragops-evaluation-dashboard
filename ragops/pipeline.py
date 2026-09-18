from __future__ import annotations

import time
from pathlib import Path

from .citations import validate_citations
from .confidence import ABSTENTION_ANSWER, assess_retrieval_confidence
from .generation import LocalGenerator
from .grounding import GroundingEvaluator, is_abstention
from .retrieval import HybridRetriever
from .tracing import TraceStore


class RAGPipeline:
    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        self.retriever = HybridRetriever(root / "data" / "corpus.json")
        self.generator = LocalGenerator()
        self.grounding = GroundingEvaluator()
        self.traces = TraceStore(root / "data" / "ragops.db")

    def query(self, question: str, top_k: int = 3) -> dict:
        started = time.perf_counter()

        r0 = time.perf_counter()
        sources = self.retriever.hybrid_rerank(question, k=top_k)
        retrieval_ms = (time.perf_counter() - r0) * 1000

        confidence = assess_retrieval_confidence(sources)

        if confidence["sufficient_evidence"]:
            g0 = time.perf_counter()
            generated = self.generator.generate(question, sources)
            generation_ms = (time.perf_counter() - g0) * 1000
        else:
            generated = {
                "answer": ABSTENTION_ANSWER,
                "model": "retrieval-confidence-gate",
            }
            generation_ms = 0.0

        abstained = is_abstention(generated["answer"])
        citation_metrics = validate_citations(
            generated["answer"],
            {s["id"] for s in sources},
            abstained=abstained,
        )
        grounding_metrics = self.grounding.evaluate(generated["answer"], sources)
        metrics = {
            **citation_metrics,
            **grounding_metrics,
            "retrieval_confidence": confidence,
        }
        total_ms = (time.perf_counter() - started) * 1000

        public_sources = [
            {
                "id": s["id"],
                "title": s["title"],
                "category": s["category"],
                "score": round(float(s.get("score", 0.0)), 4),
                "rerank_score": round(float(s.get("rerank_score", 0.0)), 4),
                "text": s["text"],
            }
            for s in sources
        ]

        self.traces.add(
            query=question,
            answer=generated["answer"],
            sources=public_sources,
            metrics=metrics,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

        return {
            "query": question,
            "answer_status": metrics["answer_status"],
            "answer": generated["answer"],
            "model": generated["model"],
            "sources": public_sources,
            "metrics": metrics,
            "latency_ms": {
                "retrieval": round(retrieval_ms, 1),
                "generation": round(generation_ms, 1),
                "total": round(total_ms, 1),
            },
        }

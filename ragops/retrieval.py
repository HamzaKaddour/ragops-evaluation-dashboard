from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if abs(hi - lo) < 1e-12:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


class HybridRetriever:
    def __init__(
        self,
        corpus_path: str | Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.documents = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        self.texts = [f"{d['title']}\n{d['category']}\n{d['text']}" for d in self.documents]
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.reranker = CrossEncoder(reranker_model, device=device)
        self.embeddings = self.embedder.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def bm25_search(self, query: str, k: int = 5) -> list[dict]:
        scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=float)
        order = np.argsort(scores)[::-1][:k]
        return [self._row(int(i), float(scores[i]), "bm25") for i in order]

    def dense_search(self, query: str, k: int = 5) -> list[dict]:
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q, min(k, len(self.documents)))
        return [self._row(int(i), float(s), "dense") for s, i in zip(scores[0], indices[0])]

    def hybrid_search(self, query: str, k: int = 5, candidate_k: int = 8, alpha: float = 0.5) -> list[dict]:
        bm25_scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=float)
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        dense_scores = (self.embeddings @ q[0]).astype(float)
        fused = alpha * _minmax(bm25_scores) + (1 - alpha) * _minmax(dense_scores)
        order = np.argsort(fused)[::-1][: min(candidate_k, len(self.documents))]
        rows = [self._row(int(i), float(fused[i]), "hybrid") for i in order]
        return rows[:k]

    def hybrid_rerank(self, query: str, k: int = 3, candidate_k: int = 8, alpha: float = 0.5) -> list[dict]:
        candidates = self.hybrid_search(query, k=candidate_k, candidate_k=candidate_k, alpha=alpha)
        pairs = [(query, f"{c['title']}\n{c['text']}") for c in candidates]
        scores = self.reranker.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:k]

    def _row(self, idx: int, score: float, method: str) -> dict:
        d = self.documents[idx]
        return {
            "id": d["id"],
            "title": d["title"],
            "category": d["category"],
            "text": d["text"],
            "score": score,
            "method": method,
        }

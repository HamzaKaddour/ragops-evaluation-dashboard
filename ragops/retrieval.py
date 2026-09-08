from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RetrievedDocument:
    doc_id: str
    title: str
    text: str
    score: float


def _doc_text(doc: dict) -> str:
    return f"{doc.get('title', '')} {doc.get('category', '')} {doc.get('text', '')}".strip()


class RAGRetriever:
    """Real retrieval stack using BM25, SentenceTransformers, FAISS and CrossEncoder.

    Heavy dependencies are imported lazily so the static dashboard can still be
    generated in lightweight CI environments without downloading model weights.
    """

    def __init__(
        self,
        corpus: Sequence[dict],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.corpus = list(corpus)
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self._bm25 = None
        self._embedder = None
        self._reranker = None
        self._index = None
        self._doc_embeddings = None

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi
        tokenized = [_doc_text(doc).lower().split() for doc in self.corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _ensure_dense(self) -> None:
        if self._index is not None:
            return
        import faiss
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedding_model_name)
        texts = [_doc_text(doc) for doc in self.corpus]
        embeddings = self._embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._doc_embeddings = embeddings
        self._index = index

    def _ensure_reranker(self) -> None:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model_name)

    def bm25(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        self._ensure_bm25()
        scores = self._bm25.get_scores(query.lower().split())
        order = np.argsort(scores)[::-1][:k]
        return [
            RetrievedDocument(self.corpus[i]["id"], self.corpus[i]["title"], self.corpus[i]["text"], float(scores[i]))
            for i in order
        ]

    def dense(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        self._ensure_dense()
        query_embedding = self._embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")
        scores, indices = self._index.search(query_embedding, min(k, len(self.corpus)))
        return [
            RetrievedDocument(self.corpus[i]["id"], self.corpus[i]["title"], self.corpus[i]["text"], float(score))
            for score, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def hybrid(self, query: str, k: int = 5, candidate_k: int = 10, alpha: float = 0.5) -> list[RetrievedDocument]:
        lexical = self.bm25(query, min(candidate_k, len(self.corpus)))
        dense = self.dense(query, min(candidate_k, len(self.corpus)))

        def normalize(items: list[RetrievedDocument]) -> dict[str, float]:
            if not items:
                return {}
            values = [item.score for item in items]
            lo, hi = min(values), max(values)
            if hi == lo:
                return {item.doc_id: 1.0 for item in items}
            return {item.doc_id: (item.score - lo) / (hi - lo) for item in items}

        lex_scores = normalize(lexical)
        dense_scores = normalize(dense)
        by_id = {doc["id"]: doc for doc in self.corpus}
        candidate_ids = set(lex_scores) | set(dense_scores)
        scored = []
        for doc_id in candidate_ids:
            score = alpha * lex_scores.get(doc_id, 0.0) + (1.0 - alpha) * dense_scores.get(doc_id, 0.0)
            doc = by_id[doc_id]
            scored.append(RetrievedDocument(doc_id, doc["title"], doc["text"], score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]

    def hybrid_reranked(self, query: str, k: int = 5, candidate_k: int = 10) -> list[RetrievedDocument]:
        candidates = self.hybrid(query, k=min(candidate_k, len(self.corpus)), candidate_k=candidate_k)
        self._ensure_reranker()
        pairs = [[query, f"{item.title} {item.text}"] for item in candidates]
        scores = self._reranker.predict(pairs)
        reranked = [
            RetrievedDocument(item.doc_id, item.title, item.text, float(score))
            for item, score in zip(candidates, scores)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:k]

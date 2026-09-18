from __future__ import annotations

import re

import numpy as np

ABSTENTION_PATTERNS = (
    "available evidence is insufficient",
    "provided context is insufficient",
    "context is insufficient",
    "not enough evidence",
    "insufficient evidence",
    "cannot be answered based on the provided context",
    "cannot answer based on the provided context",
    "retrieved documents do not support",
    "retrieved context does not support",
)

CITATION_RE = re.compile(r"\[[A-Za-z0-9_-]+\]")
CITATION_ONLY_RE = re.compile(r"^(?:\s*\[[A-Za-z0-9_-]+\]\s*)+$")


def _sentences(text: str) -> list[str]:
    candidates = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 3
    ]
    return [s for s in candidates if not CITATION_ONLY_RE.fullmatch(s)]


def _clean_for_embedding(sentence: str) -> str:
    """Remove citation markers so they do not distort semantic similarity."""
    return re.sub(r"\s+", " ", CITATION_RE.sub("", sentence)).strip()


def is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in ABSTENTION_PATTERNS)


def _is_abstention_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(pattern in lowered for pattern in ABSTENTION_PATTERNS)


class GroundingEvaluator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "GroundingEvaluator requires sentence-transformers. "
                "Install the full runtime with: pip install -r requirements.txt"
            ) from exc

        self.model = SentenceTransformer(model_name, device=device)

    def attach_source_citations(
        self,
        answer: str,
        contexts: list[dict],
        min_similarity: float = 0.45,
    ) -> str:
        """Attach one best-supported retrieved source citation per factual sentence.

        This is a deterministic post-generation citation layer. It only cites a
        retrieved source when semantic support clears the same weak-support
        threshold used by groundedness evaluation. Abstentions are returned
        unchanged.
        """
        if is_abstention(answer) or not contexts:
            return answer

        sentences = _sentences(answer)
        if not sentences:
            return answer

        context_texts = [c["text"] for c in contexts]
        context_embeddings = self.model.encode(
            context_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        cited_sentences: list[str] = []
        for sentence in sentences:
            clean = _clean_for_embedding(sentence)
            if not clean:
                continue

            emb = self.model.encode(
                [clean],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            sims = context_embeddings @ emb
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            # Remove any model-emitted citations first, then attach the
            # strongest retrieved source only when semantic support exists.
            base = CITATION_RE.sub("", sentence).strip()
            if best_score >= min_similarity:
                citation = f"[{contexts[best_idx]['id']}]"
                if base and base[-1] in ".!?":
                    base = f"{base[:-1].rstrip()} {citation}{base[-1]}"
                else:
                    base = f"{base} {citation}"
            cited_sentences.append(base)

        return " ".join(cited_sentences).strip()

    def evaluate(
        self,
        answer: str,
        contexts: list[dict],
        supported_threshold: float = 0.65,
        weak_threshold: float = 0.45,
    ) -> dict:
        sentences = _sentences(answer)
        abstained = is_abstention(answer)

        if not sentences:
            return {
                "answer_status": "insufficient_evidence" if abstained else "answered",
                "abstained": abstained,
                "groundedness_score": 1.0 if abstained else 0.0,
                "supported_sentences": 0,
                "weak_sentences": 0,
                "unsupported_sentences": 0,
                "abstention_sentences": 0,
                "sentences": [],
            }

        context_texts = [c["text"] for c in contexts]
        if not context_texts:
            return {
                "answer_status": "insufficient_evidence",
                "abstained": True,
                "groundedness_score": 1.0 if abstained else 0.0,
                "supported_sentences": 0,
                "weak_sentences": 0,
                "unsupported_sentences": 0 if abstained else len(sentences),
                "abstention_sentences": len(sentences) if abstained else 0,
                "sentences": [
                    {
                        "sentence": s,
                        "max_similarity": None,
                        "support": "abstention" if abstained else "unsupported",
                    }
                    for s in sentences
                ],
            }

        context_embeddings = self.model.encode(
            context_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        semantic_sentences = [_clean_for_embedding(s) for s in sentences]
        sentence_embeddings = self.model.encode(
            semantic_sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        rows = []
        support_values: list[float] = []
        for sentence, emb in zip(sentences, sentence_embeddings):
            if _is_abstention_sentence(sentence):
                rows.append(
                    {
                        "sentence": sentence,
                        "max_similarity": None,
                        "support": "abstention",
                    }
                )
                support_values.append(1.0)
                continue

            sims = context_embeddings @ emb
            best = float(np.max(sims)) if len(sims) else 0.0
            if best >= supported_threshold:
                label, value = "supported", 1.0
            elif best >= weak_threshold:
                label, value = "weak", 0.5
            else:
                label, value = "unsupported", 0.0

            rows.append(
                {
                    "sentence": sentence,
                    "max_similarity": round(best, 3),
                    "support": label,
                }
            )
            support_values.append(value)

        return {
            "answer_status": "insufficient_evidence" if abstained else "answered",
            "abstained": abstained,
            "groundedness_score": round(
                sum(support_values) / len(support_values),
                3,
            ),
            "supported_sentences": sum(r["support"] == "supported" for r in rows),
            "weak_sentences": sum(r["support"] == "weak" for r in rows),
            "unsupported_sentences": sum(r["support"] == "unsupported" for r in rows),
            "abstention_sentences": sum(r["support"] == "abstention" for r in rows),
            "sentences": rows,
        }

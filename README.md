# RAGOps Evaluation Dashboard

A production-oriented RAG evaluation and observability project for measuring retrieval quality, answer support, failure modes, latency, and cost across multiple Retrieval-Augmented Generation pipeline variants.

The repository contains two execution paths:

1. **Lightweight dashboard mode** for deterministic GitHub Pages deployment and CI.
2. **Real retrieval mode** using BM25, SentenceTransformers, FAISS vector search, hybrid score fusion, and CrossEncoder reranking.

## Live application

```text
https://hamzakaddour.github.io/ragops-evaluation-dashboard/
```

## Why this project exists

A useful RAG system needs more than a chat interface. Before deployment, the engineering team should be able to answer:

- Did the retriever find the correct evidence?
- Did it rank useful evidence near the top?
- Do dense and hybrid retrieval outperform lexical search?
- Does reranking improve relevance enough to justify added latency?
- Which queries fail because of missed context or weak ranking?
- What quality/latency trade-offs appear across pipeline variants?

## Implemented retrieval stack

### Lexical retrieval
- BM25 using `rank-bm25`

### Dense retrieval
- SentenceTransformer embeddings
- default model: `sentence-transformers/all-MiniLM-L6-v2`
- normalized embeddings
- FAISS `IndexFlatIP` vector search

### Hybrid retrieval
- BM25 lexical scores
- dense semantic scores
- score normalization
- weighted lexical + dense fusion

### Reranking
- CrossEncoder reranking
- default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Retrieval and reranking do **not** use evaluation labels when assigning document scores.

## Evaluation metrics

The reusable `ragops.metrics` package implements:

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- nDCG@K

The dashboard also shows heuristic demo indicators for groundedness, citation coverage, hallucination risk, latency, and failure tags. These are visualization-oriented heuristics, not LLM-judge scores.

## Architecture

```text
corpus.json
   |--------------------> BM25 ---------------------+
   |                                               |
   +--> SentenceTransformer --> FAISS ------------+--> Hybrid Fusion
                                                   |        |
                                                   |        v
                                                   |   CrossEncoder
                                                   |      Reranker
                                                   |        |
evaluation_queries.json ---------------------------+--------+
                                                            |
                                                            v
                                                   Ranked Documents
                                                            |
                                                            v
                                           Recall / Precision / MRR / nDCG
                                                            |
                                      +---------------------+------------------+
                                      |                                        |
                                      v                                        v
                           real_evaluation_results.json              Dashboard artifacts
```

## Repository structure

```text
ragops/
  metrics.py                     Ranking metrics
  retrieval.py                   BM25, dense, hybrid and reranked retrieval

scripts/
  generate_demo_artifacts.py     Lightweight deterministic Pages artifacts
  run_real_evaluation.py         Real retrieval evaluation runner

data/
  corpus.json                    Source corpus
  evaluation_queries.json        Labeled evaluation queries
  evaluation_summary.json        Dashboard aggregate metrics
  rag_runs.json                  Dashboard query traces

tests/
  test_metrics.py                Metric tests

requirements.txt                Lightweight Pages / CI dependencies
requirements-real.txt           SentenceTransformers + FAISS + BM25 stack
pyproject.toml                  Installable Python package
.github/workflows/ci.yml        Automated tests
.github/workflows/pages.yml     Static dashboard deployment
```

## Run the dashboard locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_demo_artifacts.py
python -m http.server 8000
```

Open `http://localhost:8000`.

## Run the real RAG retrieval evaluation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-real.txt
pip install -e .
python scripts/run_real_evaluation.py
```

The first run downloads the embedding and reranker model weights through Hugging Face. Results are written to:

```text
data/real_evaluation_results.json
```

The real evaluator compares:

```text
BM25
Dense + FAISS
Hybrid BM25 + Dense
Hybrid + CrossEncoder reranker
```

## CI strategy

GitHub Actions intentionally uses the lightweight dependency set. CI validates package installation, dashboard artifact generation, and ranking-metric tests without downloading large model weights on every Pages deployment.

## Current scope and limitations

Implemented:

- real BM25 retrieval
- real SentenceTransformer embeddings
- real FAISS vector search
- hybrid lexical/semantic score fusion
- real CrossEncoder reranking
- reusable ranking metrics
- automated tests and CI
- static evaluation dashboard

Not yet implemented:

- live LLM answer generation
- LLM-as-judge groundedness/faithfulness scoring
- prompt/version registry
- persistent vector database service such as Qdrant or OpenSearch
- OpenTelemetry tracing
- production authentication
- online feedback loops

## Technology stack

`Python` · `SentenceTransformers` · `Hugging Face` · `FAISS` · `BM25` · `CrossEncoder` · `NumPy` · `GitHub Actions` · `GitHub Pages`

## Repository topics

```text
rag, llm, llmops, retrieval-augmented-generation, sentence-transformers,
faiss, semantic-search, reranking, ai-evaluation, mlops, observability, python
```

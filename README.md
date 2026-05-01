# RAGOps Evaluation Dashboard

A RAG evaluation and observability dashboard for analyzing retrieval quality, answer grounding, hallucination risk, latency, and cost across multiple Retrieval-Augmented Generation pipeline variants.

The project separates the evaluation layer from the user-facing interface. A lightweight Python pipeline reads a document corpus and labeled evaluation queries, computes retrieval metrics, exports structured JSON artifacts, and renders them through an interactive dashboard.

## Live application

```text
https://hamzakaddour.github.io/ragops-evaluation-dashboard/
```

## Problem

Retrieval-Augmented Generation systems are often demonstrated through a chat interface, but the main engineering challenge is reliability. A production RAG workflow must answer several questions before deployment:

- Did the retriever find the right evidence?
- Was the most relevant evidence ranked near the top?
- Is the generated answer grounded in the retrieved context?
- Are citations or supporting passages sufficient?
- What failure modes appear across the evaluation set?
- How much quality is gained or lost when latency and cost change?

This project focuses on the evaluation and observability layer needed to answer those questions.

## Core capabilities

- RAG pipeline comparison across keyword, dense-style, hybrid, and reranked retrieval variants.
- Retrieval quality metrics: Recall@K, Precision@K, MRR, and nDCG.
- Answer reliability metrics: groundedness, citation coverage, hallucination risk, and faithfulness-style indicators.
- Query-level trace inspection with retrieved source snippets and evidence scores.
- Operational metrics: latency, estimated token cost, and failure-mode distribution.
- Reproducible artifact generation using lightweight Python scripts.

## System architecture

```text
data/
  corpus.json                 Source document corpus
  evaluation_queries.json     Labeled queries with relevant document IDs
  evaluation_summary.json     Generated aggregate metrics
  rag_runs.json               Generated query-level traces

scripts/
  generate_demo_artifacts.py  Evaluation artifact generator

css/
  styles.css                  Dashboard styling

js/
  app.js                      Interactive dashboard logic

index.html                    Dashboard entry point
.github/workflows/pages.yml   Deployment workflow
```

## Evaluation methodology

### 1. Labeled corpus and query set

The evaluation starts from a compact corpus in `data/corpus.json` and a labeled query set in `data/evaluation_queries.json`. Each query includes relevant document IDs and expected answer topics.

### 2. Retrieval variants

The artifact generator compares four retrieval configurations:

- **BM25 baseline**: keyword-style retrieval.
- **Dense embeddings**: semantic-style retrieval approximated with lightweight term-set similarity.
- **Hybrid retrieval**: score fusion between keyword and semantic-style retrieval.
- **Hybrid + reranker**: hybrid retrieval with an additional relevance-aware reranking bonus.

The current implementation is intentionally lightweight and deterministic. It does not require external APIs or a live vector database, but it preserves the structure of a real RAG evaluation workflow.

### 3. Ranking metrics

The script computes:

- **Recall@5**: whether relevant evidence appears in the top five retrieved results.
- **Precision@5**: how much of the retrieved top five is relevant.
- **MRR**: how high the first relevant result appears.
- **nDCG@5**: whether relevant documents are ranked near the top.

### 4. Reliability metrics

The dashboard estimates answer reliability using retrieval coverage and precision:

- **Groundedness**: whether the answer is supported by retrieved evidence.
- **Citation coverage**: whether the response has enough source support.
- **Hallucination risk**: the inverse risk signal derived from grounding weakness.

### 5. Operational metrics

Each pipeline stores latency and cost estimates. This allows quality to be compared against serving constraints, which is important when a more accurate retrieval stack is slower or more expensive.

### 6. Trace-level debugging

Each query-level run stores the user query, retrieved source snippets, ranking scores, generated answer, metrics, and failure tags. This supports debugging beyond aggregate scores.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_demo_artifacts.py
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## Deployment

The GitHub Actions workflow regenerates evaluation artifacts before deploying the static dashboard. The deployed application reads the generated JSON files from the `data/` directory.

## Production extensions

A production version could replace the lightweight retrieval simulation with:

- real embedding models,
- FAISS, Chroma, Qdrant, Weaviate, or OpenSearch,
- LLM-based groundedness scoring,
- prompt/version regression tests,
- OpenTelemetry-style tracing,
- authentication and dashboard-level access controls,
- scheduled evaluation runs over real documents.

## Repository topics

```text
rag, llmops, retrieval-augmented-generation, semantic-search, ai-evaluation, embeddings, mlops, observability, github-pages, python
```

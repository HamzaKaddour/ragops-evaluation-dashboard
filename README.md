# RAGOps Evaluation Dashboard

A RAG evaluation and observability dashboard for analyzing retrieval quality, answer grounding, hallucination risk, latency, and cost across multiple Retrieval-Augmented Generation pipeline variants.

The project separates the evaluation layer from the user-facing interface. Python scripts generate structured evaluation artifacts, and the dashboard visualizes those artifacts through an interactive static application.

## Live application

```text
https://hamzakaddour.github.io/ragops-evaluation-dashboard/
```

## Core capabilities

- RAG pipeline comparison across keyword, dense, hybrid, and reranked retrieval variants.
- Retrieval quality metrics: Recall@K, Precision@K, MRR, and nDCG.
- Answer reliability metrics: groundedness, citation coverage, hallucination risk, and faithfulness.
- Query-level trace inspection with retrieved source snippets and evidence scores.
- Operational metrics: latency, estimated token cost, and failure-mode distribution.
- Reproducible artifact generation using lightweight Python scripts.

## System architecture

```text
scripts/                 Evaluation artifact generation
  generate_demo_artifacts.py

data/                    Structured RAG metrics and traces
  evaluation_summary.json
  rag_runs.json

css/                     Dashboard styling
  styles.css

js/                      Interactive dashboard logic
  app.js

index.html               Dashboard entry point
.github/workflows/       Deployment workflow
```

The evaluation pipeline produces JSON artifacts that can be inspected directly or rendered through the dashboard. This mirrors a common production pattern where batch evaluation jobs generate reliability reports that are later consumed by monitoring or review interfaces.

## Evaluation dimensions

### Retrieval diagnostics

The system compares retrieval strategies using ranking metrics. This helps identify whether poor answers are caused by missing evidence, weak ranking, or insufficient context coverage.

### Grounding analysis

Generated answers are assessed against retrieved passages to estimate whether the response is supported by evidence. The dashboard tracks citation coverage, hallucination risk, and faithfulness-style indicators.

### Operational monitoring

The dashboard includes latency and cost estimates to make pipeline comparison practical. A higher-quality RAG pipeline may not be suitable for deployment if it introduces unacceptable latency or cost.

### Trace-level inspection

Each evaluated query includes retrieved sources, answer text, metrics, and failure tags. This supports debugging beyond aggregate scores.

## Local preview

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## Regenerate artifacts

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_demo_artifacts.py
```

## Repository topics

```text
rag, llmops, retrieval-augmented-generation, semantic-search, ai-evaluation, embeddings, mlops, observability, github-pages, python
```

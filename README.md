# RAGOps Evaluation Dashboard

A production-style static dashboard for evaluating Retrieval-Augmented Generation (RAG) systems. The project is designed for GitHub Pages: no paid backend, no always-on server, and no API keys required for the public demo.

The dashboard demonstrates practical LLMOps skills that are valuable in AI engineering roles: retrieval evaluation, grounding analysis, hallucination risk scoring, chunking comparison, cost/latency awareness, and trace-level debugging.

## Live demo

After GitHub Pages is enabled, the app will be available at:

```text
https://hamzakaddour.github.io/ragops-evaluation-dashboard/
```

## What this project demonstrates

- RAG pipeline design: query, retrieval, reranking, generation, and evaluation.
- Retrieval quality metrics: Recall@K, Precision@K, MRR, and nDCG.
- Answer quality metrics: groundedness, citation coverage, unsupported claim risk, and faithfulness.
- LLMOps thinking: trace inspection, prompt/version comparison, latency, and token-cost awareness.
- Static deployment architecture suitable for GitHub Pages.
- Reproducible offline artifact generation through Python scripts.

## Architecture

```text
scripts/                 Offline artifact generation
  generate_demo_artifacts.py

data/                    Precomputed RAG traces and metrics
  evaluation_summary.json
  rag_runs.json

css/                     Styling
  styles.css

js/                      Frontend dashboard logic
  app.js

index.html               GitHub Pages application
.github/workflows/       Optional Pages deployment workflow
```

The public app is static. The heavier work is done offline or through GitHub Actions, then exported to JSON files consumed by the dashboard.

## Why this design is practical

Many RAG demos stop at "chat with PDF." This project focuses on the part companies care about in production: whether the system retrieves the right context, cites its sources, avoids hallucination, stays within latency/cost targets, and exposes failure cases clearly.

## Local preview

Clone the repository and run a simple static server:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Regenerate demo artifacts

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_demo_artifacts.py
```

## Suggested GitHub repository topics

```text
rag, llmops, retrieval-augmented-generation, semantic-search, ai-evaluation, embeddings, mlops, observability, github-pages, python
```

## Recruiter-facing summary

This project simulates the evaluation layer of a production RAG system. It is intentionally backend-free for the deployed demo, but the repository includes an artifact generation workflow that mirrors how AI teams separate offline evaluation pipelines from frontend monitoring dashboards.

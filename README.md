# RAGOps Evaluation Dashboard

A local-first RAG evaluation and observability platform built with BM25, SentenceTransformers, FAISS, CrossEncoder reranking, a local Qwen LLM, FastAPI, citation validation, abstention-aware groundedness checks, SQLite query tracing, tests, Docker, GitHub Actions, and GitHub Pages.

## Current version

`v0.3.0`

This version fixes three important evaluation issues:

1. zero citations no longer receive a perfect citation-validity score;
2. citation validity and citation coverage are separate metrics;
3. correct "insufficient evidence" responses are classified as abstentions instead of hallucinations.

The sample corpus is also expanded so common RAG questions, including "What is retrieval-augmented generation?", are answerable from retrieved evidence.

## Architecture

```text
Public / always online

GitHub repository
      |
      v
GitHub Actions
      |
      v
GitHub Pages static portfolio dashboard

Local / on demand

corpus.json
   |--------------------|
   v                    v
 BM25           SentenceTransformers
                         |
                         v
                       FAISS
   |--------------------|
             |
             v
       Hybrid retrieval
             |
             v
        CrossEncoder
             |
             v
        Top passages
             |
             v
       Qwen local LLM
             |
             v
      Answer + citations
        /           \
       v             v
Citation checks   Grounding + abstention
       \             /
        v           v
         SQLite traces
              |
              v
           FastAPI
              |
              v
     Local interactive UI
```

## Cost

The default architecture is designed to cost $0 beyond electricity and the GitHub usage already included with your account.

No paid LLM API, vector database, cloud GPU, or hosted backend is required.

## Workstation installation

```bash
git clone https://github.com/HamzaKaddour/ragops-evaluation-dashboard.git
cd ragops-evaluation-dashboard

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

The first real run downloads public Hugging Face model weights. They are cached locally for later runs.

## 1. Retrieval benchmark

```bash
python scripts/run_real_evaluation.py
```

This evaluates BM25, dense FAISS retrieval, hybrid retrieval, and hybrid + CrossEncoder reranking. Results are written to:

```text
data/real_evaluation_results.json
```

## 2. Local LLM smoke test

```bash
python scripts/test_local_llm.py
```

## 3. CLI

```bash
python scripts/query_cli.py
```

Type `exit` to quit.

## 4. FastAPI + local web app

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

The root now redirects to the interactive local UI:

```text
http://127.0.0.1:8000/app/
```

Swagger API documentation remains available at:

```text
http://127.0.0.1:8000/docs
```

## API endpoints

- `GET /health`
- `POST /query`
- `GET /traces?limit=20`
- `GET /` → redirects to `/app/`

Example request:

```json
{
  "query": "What is retrieval augmented generation?",
  "top_k": 3
}
```

A normal supported answer should contain citations such as `[doc_01]`. If the corpus does not contain sufficient evidence, the response should return:

```json
{
  "answer_status": "insufficient_evidence"
}
```

rather than inventing an answer.

## Evaluation semantics

### Citation validity

Measures whether citations that are present point to retrieved source IDs.

If an answer has no citations, validity is `null`, not `1.0`.

### Citation coverage

Estimates whether factual answer sentences are accompanied by citations.

### Abstention

A correct refusal such as "the available evidence is insufficient" is classified as an abstention. Abstention language is not counted as an unsupported factual claim.

### Groundedness

The current groundedness score is an embedding-similarity heuristic using SentenceTransformers. It is useful for portfolio observability but is not presented as human verification or an LLM-judge ground truth.

## Tests

Run:

```bash
pytest -q
```

The GitHub CI workflow intentionally runs only lightweight tests and does not download the Qwen, embedding, or reranker model weights.

## GitHub deployment

The repository has two deployment modes:

### GitHub Pages — public static site

`.github/workflows/pages.yml` publishes only `static/`.

The Pages site is always online and free within your GitHub plan. It intentionally does not run Qwen or FastAPI because GitHub Pages is static hosting.

### Workstation — full RAG backend

The complete FastAPI + Qwen + FAISS service is started on demand with:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Your workstation does not need to remain on after you finish testing or demonstrating the backend.

## Updating the existing GitHub repository

After replacing the changed files and testing locally:

```bash
git status
git add .
git commit -m "feat: complete grounded RAG evaluation workflow"
git push origin main
```

Then verify:

1. GitHub → **Actions** → `CI` is green.
2. GitHub → **Actions** → `Deploy static dashboard` is green.
3. GitHub → **Settings → Pages** shows **GitHub Actions** as the source.
4. Open `https://hamzakaddour.github.io/ragops-evaluation-dashboard/`.

The public Pages site will be static. The local interactive API remains available only when FastAPI is running on the workstation.

## Requirements

All runtime dependencies are listed in `requirements.txt`. No paid API credentials are required.

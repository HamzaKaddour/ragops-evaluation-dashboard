# RAGOps Evaluation Dashboard

[![CI](https://github.com/HamzaKaddour/ragops-evaluation-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/HamzaKaddour/ragops-evaluation-dashboard/actions/workflows/ci.yml)
[![Pages](https://github.com/HamzaKaddour/ragops-evaluation-dashboard/actions/workflows/pages.yml/badge.svg)](https://github.com/HamzaKaddour/ragops-evaluation-dashboard/actions/workflows/pages.yml)
[![Release](https://img.shields.io/github/v/release/HamzaKaddour/ragops-evaluation-dashboard?label=release)](https://github.com/HamzaKaddour/ragops-evaluation-dashboard/releases/latest)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/HamzaKaddour/ragops-evaluation-dashboard?quickstart=1)

[**Portfolio case study → https://hamzakaddour.github.io/case-studies/ragops.html**](https://hamzakaddour.github.io/case-studies/ragops.html)

A local-first RAG evaluation and observability platform built with BM25, SentenceTransformers, FAISS, CrossEncoder reranking, a local Qwen LLM, FastAPI, citation validation, abstention-aware groundedness checks, SQLite query tracing, tests, Docker, GitHub Actions, and GitHub Pages.

## Current version

`v0.6.0`

This version adds a measured reliability iteration on top of the v1 benchmark:

1. a retrieval-confidence gate blocks generation when CrossEncoder evidence is too weak;
2. the default gate threshold is `-3.0`, calibrated against the repository's v1 benchmark and configurable with `RAGOPS_RERANK_THRESHOLD`;
3. Qwen is prompted to attach citations to every factual sentence;
4. groundedness ignores standalone citation fragments and removes citation tokens before semantic comparison;
5. the current published benchmark is versioned as `final-v3`.

The sample corpus is also expanded so common RAG questions, including "What is retrieval-augmented generation?", are answerable from retrieved evidence.


## Final benchmark results

The current published `final-v3` benchmark uses a small purpose-built 12-query regression set: 10 answerable questions and 2 intentionally unanswerable questions. These metrics are intended to demonstrate observability, failure analysis, and measured iteration; they are not claims of broad production performance.

| Metric | final-v1 | final-v2 | final-v3 |
| --- | ---: | ---: | ---: |
| Behavior accuracy | 91.7% | 91.7% | **100%** |
| Groundedness | 58.0% | 84.7% | **91.7%** |
| Citation validity | 100%* | N/A | **100%** |
| Citation coverage | 18.9% | 0.0% | **95.0%** |
| Abstention accuracy | 50.0% | **100%** | **100%** |
| Avg. total latency | 10.0 s | 5.5 s | **5.5 s** |

\* In v1, citation validity was computed only for responses that emitted citations; low citation coverage showed that citations were frequently missing.

The main progression was:

- **v1:** exposed weak grounding, citation coverage, and abstention behavior.
- **v2:** added a CrossEncoder confidence gate, improving abstention and latency.
- **v3:** added deterministic sentence-level source assignment and citation-aware groundedness scoring, substantially improving coverage while preserving valid citations.

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

## 1. Final end-to-end benchmark

Run the authoritative workstation benchmark after pulling the latest code:

```bash
python scripts/run_final_benchmark.py
```

It evaluates BM25, dense FAISS, hybrid, and hybrid + CrossEncoder retrieval, then runs the complete Hybrid + CrossEncoder + Qwen generation pipeline over the labeled evaluation set, including explicit unanswerable cases for abstention testing.

Outputs:

```text
data/final_benchmark_summary.json
data/final_benchmark_runs.json
```

These two files are the artifacts used by the public GitHub Pages dashboard after you commit them.

For a quick smoke test before the full run:

```bash
python scripts/run_final_benchmark.py --limit 2
```

Do not commit the limited smoke-test outputs as final benchmark results.

## 2. Retrieval-only benchmark

```bash
python scripts/run_real_evaluation.py
```

This evaluates BM25, dense FAISS retrieval, hybrid retrieval, and hybrid + CrossEncoder reranking. Results are written to:

```text
data/real_evaluation_results.json
```

## 3. Local LLM smoke test

```bash
python scripts/test_local_llm.py
```

## 4. CLI

```bash
python scripts/query_cli.py
```

Type `exit` to quit.

## 5. FastAPI + local web app

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

`.github/workflows/pages.yml` publishes `static/` plus committed benchmark JSON artifacts from `data/`.

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


## Publishing the final benchmark

After the full workstation benchmark completes, inspect the two generated JSON files and then publish them:

```bash
git status
git add data/final_benchmark_summary.json data/final_benchmark_runs.json
git commit -m "bench: publish final workstation RAG benchmark"
git push origin main
```

GitHub Pages will automatically redeploy. The dashboard prefers the final benchmark artifacts when they exist and falls back to the legacy demo summary only when they are absent.


## Reliability v2

The v1 benchmark exposed three generation-side weaknesses despite strong retrieval:
- average groundedness: 58%
- average citation coverage: 18.9%
- abstention accuracy: 50%

The v2 pipeline addresses these directly.

### Retrieval-confidence gate

Before Qwen runs, the strongest CrossEncoder score is checked. The default threshold is:

```text
-3.0
```

If the best reranked passage is below that value, the system returns a deterministic insufficient-evidence response and skips generation. The threshold was calibrated from this repository's small v1 benchmark and is intentionally configurable:

```bash
export RAGOPS_RERANK_THRESHOLD=-3.0
```

It is not presented as a universal CrossEncoder threshold.

### Citation contract

Normal generated answers are constrained to one to three concise factual sentences. Every sentence must end with one or more retrieved source IDs, for example:

```text
BM25 is a lexical retrieval method [doc_04].
```

### Re-running v2

After pulling the v2 code:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pytest -q
python scripts/run_final_benchmark.py
```

The generated benchmark JSON files overwrite the previous final artifacts. Inspect them before committing.


## Reliability v3

v3 keeps the retrieval-confidence gate from v2 and adds deterministic citation assignment after generation. Each factual sentence is compared against retrieved passages with the same SentenceTransformer model used for groundedness. A source citation is attached only when semantic similarity clears the weak-support threshold (`0.45` by default).

This separates two responsibilities:
- Qwen generates a concise grounded answer.
- the evaluation layer verifies sentence support and attaches a retrieved source ID only when support is strong enough.

The final benchmark output is versioned as `final-v3`.

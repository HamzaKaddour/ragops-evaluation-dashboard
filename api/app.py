from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"

app = FastAPI(title="RAGOps Evaluation API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=8)


@lru_cache(maxsize=1)
def get_pipeline():
    # Lazy import keeps health/root/tests usable without loading ML models.
    from ragops.pipeline import RAGPipeline
    return RAGPipeline(ROOT)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ragops-evaluation-dashboard", "version": "0.3.0"}


@app.post("/query")
def query(payload: QueryRequest) -> dict:
    return get_pipeline().query(payload.query, top_k=payload.top_k)


@app.get("/traces")
def traces(limit: int = 20) -> dict:
    return {"traces": get_pipeline().traces.recent(limit=min(max(limit, 1), 100))}


# Mounted last so API routes above retain priority. The same files are also
# deployed independently by GitHub Pages.
app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="static-app")

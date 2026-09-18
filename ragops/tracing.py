from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class TraceStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    retrieval_ms REAL NOT NULL,
                    generation_ms REAL NOT NULL,
                    total_ms REAL NOT NULL
                )
                """
            )

    def add(self, *, query: str, answer: str, sources: list[dict], metrics: dict, retrieval_ms: float, generation_ms: float, total_ms: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO traces (created_at, query, answer, sources_json, metrics_json, retrieval_ms, generation_ms, total_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    query,
                    answer,
                    json.dumps(sources),
                    json.dumps(metrics),
                    retrieval_ms,
                    generation_ms,
                    total_ms,
                ),
            )

    def recent(self, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, created_at, query, answer, sources_json, metrics_json, retrieval_ms, generation_ms, total_ms FROM traces ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "created_at": r[1],
                "query": r[2],
                "answer": r[3],
                "sources": json.loads(r[4]),
                "metrics": json.loads(r[5]),
                "retrieval_ms": r[6],
                "generation_ms": r[7],
                "total_ms": r[8],
            }
            for r in rows
        ]

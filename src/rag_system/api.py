from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_system.evaluation import evaluate_pipeline, load_eval_cases
from rag_system.pipeline import (
    RAGPipeline,
    answer_to_dict,
    build_index,
    index_is_complete,
    read_index_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
INDEX_PATH = Path(os.getenv("RAG_INDEX_PATH", "indexes/demo"))
CACHE_PATH = Path(os.getenv("RAG_CACHE_PATH", "cache/query_cache.json"))

app = FastAPI(title="Enterprise Knowledge Assistant", version="0.1.0")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=12)
    filters: dict[str, str] = Field(default_factory=dict)
    use_cache: bool = True


class IngestRequest(BaseModel):
    docs_path: str = "data/sample_docs"
    index_path: str = str(INDEX_PATH)
    chunk_size: int = 140
    overlap: int = 32


class EvalRequest(BaseModel):
    eval_file: str = "eval/queries.jsonl"
    top_k: int = 5


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_exists": index_is_complete(INDEX_PATH),
        "index": read_index_manifest(INDEX_PATH),
    }


@app.post("/api/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    chunks = build_index(
        request.docs_path,
        request.index_path,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )
    return {"chunks": len(chunks), "index_path": request.index_path}


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, Any]:
    _ensure_index()
    pipeline = RAGPipeline.load(INDEX_PATH, cache_path=CACHE_PATH)
    answer = pipeline.ask(
        request.question,
        top_k=request.top_k,
        filters=request.filters,
        use_cache=request.use_cache,
    )
    return answer_to_dict(answer)


@app.post("/api/evaluate")
def evaluate(request: EvalRequest) -> dict[str, Any]:
    _ensure_index()
    pipeline = RAGPipeline.load(INDEX_PATH)
    return evaluate_pipeline(pipeline, load_eval_cases(request.eval_file), top_k=request.top_k)


def _ensure_index() -> None:
    if not index_is_complete(INDEX_PATH):
        build_index("data/sample_docs", INDEX_PATH)

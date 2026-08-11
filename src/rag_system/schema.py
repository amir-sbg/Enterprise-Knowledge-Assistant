from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]


@dataclass
class Document:
    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    start_token: int = 0
    end_token: int = 0


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class Citation:
    chunk_id: str
    document_id: str
    source: str
    quote: str
    score: float


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    retrieval: list[RetrievedChunk]
    metrics: Metadata = field(default_factory=dict)


@dataclass
class EvalCase:
    id: str
    question: str
    expected_doc_ids: list[str]
    answer_keywords: list[str]
    filters: Metadata = field(default_factory=dict)

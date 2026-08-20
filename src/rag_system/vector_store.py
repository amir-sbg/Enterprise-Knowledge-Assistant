from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from rag_system.embeddings import cosine_similarity
from rag_system.schema import Chunk, RetrievedChunk


class InMemoryVectorStore:
    def __init__(self, chunks: list[Chunk] | None = None, vectors: np.ndarray | None = None) -> None:
        self.chunks = chunks or []
        self.vectors = vectors if vectors is not None else np.zeros((0, 384), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk and vector counts do not match")
        if not self.chunks:
            self.chunks = list(chunks)
            self.vectors = vectors.astype(np.float32)
            return
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 8,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        filters = normalize_filters(filters)
        scores = cosine_similarity(self.vectors, query_vector)
        candidates: list[RetrievedChunk] = []
        for idx, score in enumerate(scores.tolist()):
            chunk = self.chunks[idx]
            if not _metadata_matches(chunk.metadata, filters):
                continue
            candidates.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    semantic_score=float(score),
                )
            )
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "vector_store.pkl").open("wb") as f:
            pickle.dump({"chunks": self.chunks, "vectors": self.vectors}, f)

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryVectorStore":
        with (Path(path) / "vector_store.pkl").open("rb") as f:
            payload = pickle.load(f)
        return cls(chunks=payload["chunks"], vectors=payload["vectors"])


def normalize_filters(filters: dict[str, object] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in (filters or {}).items():
        clean_key = str(key).strip()
        if not clean_key:
            continue
        normalized[clean_key] = _normalize_filter_value(value)
    return normalized


def _metadata_matches(metadata: dict[str, object], filters: dict[str, object]) -> bool:
    for key, value in normalize_filters(filters).items():
        if key not in metadata:
            return False
        if _normalize_filter_value(metadata[key]) != value:
            return False
    return True


def _normalize_filter_value(value: object) -> str:
    return str(value).strip().casefold()

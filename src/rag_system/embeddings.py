from __future__ import annotations

import hashlib
import math

import numpy as np

from rag_system.text import tokenize


class HashEmbeddingModel:
    """Small deterministic embedding model for local testing and demos."""

    def __init__(self, dim: int = 384) -> None:
        if dim < 1:
            raise ValueError("embedding dimension must be positive")
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                idx, sign = self._hash_token(token)
                vectors[row, idx] += sign
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _hash_token(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        idx = value % self.dim
        sign = 1.0 if (value >> 9) & 1 else -1.0
        return idx, sign


def cosine_similarity(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if matrix.ndim != 2 or query.ndim != 1:
        raise ValueError("matrix must be 2-D and query must be 1-D")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError("matrix and query dimensions must match")
    if not np.isfinite(matrix).all() or not np.isfinite(query).all():
        raise ValueError("embeddings must contain only finite values")
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return matrix @ (query / query_norm)


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    peak = max(scores)
    exps = [math.exp(score - peak) for score in scores]
    denom = sum(exps) or 1.0
    return [value / denom for value in exps]

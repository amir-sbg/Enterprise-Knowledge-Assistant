from __future__ import annotations

import hashlib
import math

import numpy as np

from rag_system.text import tokenize


class HashEmbeddingModel:
    """Small deterministic embedding model for local testing and demos."""

    def __init__(self, dim: int = 384) -> None:
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


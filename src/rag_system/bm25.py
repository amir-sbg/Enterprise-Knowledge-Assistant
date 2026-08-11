from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from rag_system.schema import Chunk, RetrievedChunk
from rag_system.text import tokenize
from rag_system.vector_store import _metadata_matches


class BM25Index:
    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks or []
        self.doc_freq: dict[str, int] = {}
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        if chunks:
            self.build(chunks)

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.term_freqs = []
        self.doc_lengths = []
        doc_freq: defaultdict[str, int] = defaultdict(int)

        for chunk in chunks:
            terms = tokenize(chunk.text)
            counts = Counter(terms)
            self.term_freqs.append(counts)
            self.doc_lengths.append(len(terms))
            for term in counts:
                doc_freq[term] += 1

        self.doc_freq = dict(doc_freq)
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

    def search(
        self,
        query: str,
        top_k: int = 8,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        filters = filters or {}
        query_terms = tokenize(query)
        scored: list[RetrievedChunk] = []
        for idx, chunk in enumerate(self.chunks):
            if not _metadata_matches(chunk.metadata, filters):
                continue
            score = self._score(query_terms, idx)
            if score <= 0:
                continue
            scored.append(RetrievedChunk(chunk=chunk, score=score, bm25_score=score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _score(self, query_terms: list[str], idx: int) -> float:
        score = 0.0
        total_docs = len(self.chunks)
        doc_len = self.doc_lengths[idx] or 1
        counts = self.term_freqs[idx]
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_length or 1))
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "bm25.pkl").open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with (Path(path) / "bm25.pkl").open("rb") as f:
            return pickle.load(f)


from __future__ import annotations

from rag_system.schema import RetrievedChunk
from rag_system.text import token_set, tokenize


class LightweightReranker:
    def score(self, query: str, item: RetrievedChunk) -> float:
        q_terms = token_set(query)
        c_terms = token_set(item.chunk.text)
        overlap = len(q_terms & c_terms) / max(len(q_terms), 1)

        phrase_bonus = 0.0
        lowered = item.chunk.text.lower()
        for term in tokenize(query):
            if len(term) > 4 and term in lowered:
                phrase_bonus += 0.015

        source_bonus = 0.03 if item.chunk.metadata.get("source") else 0.0
        return 0.58 * item.semantic_score + 0.32 * item.bm25_score + 0.10 * overlap + phrase_bonus + source_bonus

    def rerank(self, query: str, items: list[RetrievedChunk], top_k: int = 6) -> list[RetrievedChunk]:
        reranked: list[RetrievedChunk] = []
        for item in items:
            item.rerank_score = self.score(query, item)
            item.score = item.rerank_score
            reranked.append(item)
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]


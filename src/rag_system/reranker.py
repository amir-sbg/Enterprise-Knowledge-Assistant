from __future__ import annotations

from rag_system.schema import RetrievedChunk
from rag_system.text import content_terms, token_set, tokenize


class LightweightReranker:
    def score(
        self,
        query: str,
        item: RetrievedChunk,
        semantic_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> float:
        q_terms = content_terms(query) or token_set(query)
        c_terms = token_set(item.chunk.text)
        overlap = len(q_terms & c_terms) / max(len(q_terms), 1)

        phrase_bonus = 0.0
        lowered = item.chunk.text.lower()
        for term in tokenize(query):
            if len(term) > 4 and term in lowered:
                phrase_bonus += 0.015

        source_bonus = 0.03 if item.chunk.metadata.get("source") else 0.0
        weight_total = semantic_weight + bm25_weight
        if semantic_weight < 0 or bm25_weight < 0 or weight_total <= 0:
            raise ValueError("reranker retrieval weights must be non-negative and sum to a positive value")
        retrieval_score = (
            semantic_weight * item.semantic_score + bm25_weight * item.bm25_score
        ) / weight_total
        return 0.90 * retrieval_score + 0.10 * overlap + phrase_bonus + source_bonus

    def rerank(
        self,
        query: str,
        items: list[RetrievedChunk],
        top_k: int = 6,
        semantic_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[RetrievedChunk]:
        reranked: list[RetrievedChunk] = []
        for item in items:
            item.rerank_score = self.score(
                query,
                item,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
            )
            item.score = item.rerank_score
            reranked.append(item)
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]

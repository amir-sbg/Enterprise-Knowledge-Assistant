from __future__ import annotations

from collections import OrderedDict

from rag_system.bm25 import BM25Index
from rag_system.embeddings import HashEmbeddingModel
from rag_system.query import QueryRewriter
from rag_system.reranker import LightweightReranker
from rag_system.schema import RetrievedChunk
from rag_system.vector_store import InMemoryVectorStore


class HybridRetriever:
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        bm25: BM25Index,
        embedding_model: HashEmbeddingModel | None = None,
        rewriter: QueryRewriter | None = None,
        reranker: LightweightReranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.rewriter = rewriter or QueryRewriter()
        self.reranker = reranker or LightweightReranker()

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 16,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        rewritten = self.rewriter.rewrite(query)
        query_vector = self.embedding_model.embed_query(rewritten)
        semantic = self.vector_store.search(query_vector, top_k=candidate_k, filters=filters)
        lexical = self.bm25.search(rewritten, top_k=candidate_k, filters=filters)
        fused = self._merge(semantic, lexical)
        return self.reranker.rerank(rewritten, fused, top_k=top_k)

    def _merge(
        self,
        semantic: list[RetrievedChunk],
        lexical: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        merged: OrderedDict[str, RetrievedChunk] = OrderedDict()

        for rank, item in enumerate(semantic, start=1):
            key = item.chunk.id
            item.semantic_score = _rank_score(rank)
            item.score = item.semantic_score
            merged[key] = item

        for rank, item in enumerate(lexical, start=1):
            key = item.chunk.id
            bm25_score = _rank_score(rank)
            if key in merged:
                merged[key].bm25_score = bm25_score
                merged[key].score += bm25_score
            else:
                item.bm25_score = bm25_score
                item.score = bm25_score
                merged[key] = item

        return list(merged.values())


def _rank_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


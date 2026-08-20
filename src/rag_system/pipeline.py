from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rag_system.bm25 import BM25Index
from rag_system.cache import JsonQueryCache
from rag_system.chunking import TokenChunker
from rag_system.citations import CitationVerifier
from rag_system.embeddings import HashEmbeddingModel
from rag_system.generation import ExtractiveAnswerer
from rag_system.parsing import load_documents
from rag_system.retriever import HybridRetriever
from rag_system.schema import Answer, Chunk
from rag_system.tracking import CostModel, estimate_tokens, latency_timer
from rag_system.vector_store import InMemoryVectorStore

INDEX_MANIFEST = "manifest.json"


class RAGPipeline:
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        bm25: BM25Index,
        embedding_model: HashEmbeddingModel | None = None,
        cache: JsonQueryCache | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.retriever = HybridRetriever(vector_store, bm25, self.embedding_model)
        self.answerer = ExtractiveAnswerer()
        self.verifier = CitationVerifier()
        self.cache = cache
        self.cost_model = cost_model or CostModel()

    def ask(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> Answer:
        filters = filters or {}
        cache_key = {"question": question, "top_k": top_k, "filters": filters}
        if self.cache and use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return _answer_from_dict(cached)

        with latency_timer() as timing:
            retrieved = self.retriever.retrieve(question, top_k=top_k, filters=filters)
            answer = self.answerer.answer(question, retrieved)
            report = self.verifier.verify(answer)

        input_tokens = estimate_tokens(question + " " + " ".join(item.chunk.text for item in retrieved))
        output_tokens = estimate_tokens(answer.text)
        answer.metrics = {
            "latency_ms": round(timing["latency_ms"], 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(self.cost_model.estimate(input_tokens, output_tokens), 8),
            "citation_accuracy": round(report.citation_accuracy, 4),
            "faithfulness": round(report.faithfulness, 4),
            "unsupported_claims": report.unsupported_claims,
        }

        if self.cache and use_cache:
            self.cache.set(cache_key, answer_to_dict(answer))
        return answer

    @classmethod
    def load(cls, index_path: str | Path, cache_path: str | Path | None = None) -> "RAGPipeline":
        path = Path(index_path)
        embedding_model = HashEmbeddingModel()
        vector_store = InMemoryVectorStore.load(path)
        bm25 = BM25Index.load(path)
        cache = JsonQueryCache(cache_path) if cache_path else None
        return cls(vector_store, bm25, embedding_model=embedding_model, cache=cache)


def build_index(
    docs_path: str | Path,
    index_path: str | Path,
    chunk_size: int = 140,
    overlap: int = 32,
) -> list[Chunk]:
    index_path = Path(index_path)
    documents = load_documents(docs_path)
    chunks = TokenChunker(chunk_size=chunk_size, overlap=overlap).split(documents)
    embedding_model = HashEmbeddingModel()
    vectors = embedding_model.embed([chunk.text for chunk in chunks])

    vector_store = InMemoryVectorStore()
    vector_store.add(chunks, vectors)
    vector_store.save(index_path)

    bm25 = BM25Index(chunks)
    bm25.save(index_path)
    _write_manifest(index_path, docs_path, documents, chunks, chunk_size, overlap, embedding_model.dim)
    return chunks


def read_index_manifest(index_path: str | Path) -> dict:
    manifest_path = Path(index_path) / INDEX_MANIFEST
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def index_is_complete(index_path: str | Path) -> bool:
    path = Path(index_path)
    return (
        (path / "vector_store.pkl").exists()
        and (path / "bm25.pkl").exists()
        and (path / INDEX_MANIFEST).exists()
    )


def _write_manifest(
    index_path: Path,
    docs_path: str | Path,
    documents: list,
    chunks: list[Chunk],
    chunk_size: int,
    overlap: int,
    embedding_dim: int,
) -> None:
    sources = sorted({str(doc.metadata.get("source", doc.id)) for doc in documents})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "docs_path": str(docs_path),
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dim": embedding_dim,
        "sources": sources,
    }
    index_path.mkdir(parents=True, exist_ok=True)
    (index_path / INDEX_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def answer_to_dict(answer: Answer) -> dict:
    return {
        "question": answer.question,
        "text": answer.text,
        "citations": [asdict(citation) for citation in answer.citations],
        "retrieval": [
            {
                "chunk": asdict(item.chunk),
                "score": item.score,
                "semantic_score": item.semantic_score,
                "bm25_score": item.bm25_score,
                "rerank_score": item.rerank_score,
            }
            for item in answer.retrieval
        ],
        "metrics": answer.metrics,
    }


def retrieval_trace(answer: Answer, preview_tokens: int = 28) -> list[dict]:
    trace = []
    for rank, item in enumerate(answer.retrieval, start=1):
        trace.append(
            {
                "rank": rank,
                "chunk_id": item.chunk.id,
                "document_id": item.chunk.document_id,
                "source": item.chunk.metadata.get("source", item.chunk.document_id),
                "score": round(item.score, 6),
                "semantic_score": round(item.semantic_score, 6),
                "bm25_score": round(item.bm25_score, 6),
                "rerank_score": round(item.rerank_score, 6),
                "preview": _preview(item.chunk.text, preview_tokens),
            }
        )
    return trace


def _preview(text: str, token_limit: int) -> str:
    tokens = text.split()
    if len(tokens) <= token_limit:
        return " ".join(tokens)
    return " ".join(tokens[:token_limit]) + " ..."


def _answer_from_dict(payload: dict) -> Answer:
    from rag_system.schema import Citation, RetrievedChunk

    citations = [Citation(**item) for item in payload.get("citations", [])]
    retrieval = []
    for item in payload.get("retrieval", []):
        retrieval.append(
            RetrievedChunk(
                chunk=Chunk(**item["chunk"]),
                score=item["score"],
                semantic_score=item.get("semantic_score", 0.0),
                bm25_score=item.get("bm25_score", 0.0),
                rerank_score=item.get("rerank_score", 0.0),
            )
        )
    return Answer(
        question=payload["question"],
        text=payload["text"],
        citations=citations,
        retrieval=retrieval,
        metrics=payload.get("metrics", {}),
    )

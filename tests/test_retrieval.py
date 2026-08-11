from rag_system.bm25 import BM25Index
from rag_system.chunking import TokenChunker
from rag_system.embeddings import HashEmbeddingModel
from rag_system.retriever import HybridRetriever
from rag_system.schema import Document
from rag_system.vector_store import InMemoryVectorStore


def _retriever():
    docs = [
        Document(id="security", text="restricted data requires encryption and a security ticket"),
        Document(id="travel", text="conference travel requests include event name and estimated cost"),
    ]
    chunks = TokenChunker(chunk_size=30, overlap=4).split(docs)
    model = HashEmbeddingModel(dim=128)
    vectors = model.embed([chunk.text for chunk in chunks])
    store = InMemoryVectorStore()
    store.add(chunks, vectors)
    return HybridRetriever(store, BM25Index(chunks), embedding_model=model)


def test_hybrid_retriever_finds_policy_evidence():
    results = _retriever().retrieve("How do I send restricted data?", top_k=2)
    assert results
    assert results[0].chunk.document_id == "security"


def test_query_rewriter_expands_known_acronym():
    results = _retriever().retrieve("What is needed for PII transfer?", top_k=2)
    assert results[0].chunk.document_id == "security"


def test_metadata_filter_excludes_other_departments():
    docs = [
        Document(
            id="security",
            text="restricted data requires encryption and a security ticket",
            metadata={"department": "security"},
        ),
        Document(
            id="travel",
            text="conference travel requests include event name and estimated cost",
            metadata={"department": "finance"},
        ),
    ]
    chunks = TokenChunker(chunk_size=30, overlap=4).split(docs)
    model = HashEmbeddingModel(dim=128)
    store = InMemoryVectorStore()
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))
    results = HybridRetriever(store, BM25Index(chunks), embedding_model=model).retrieve(
        "conference travel", filters={"department": "security"}
    )

    assert all(item.chunk.metadata["department"] == "security" for item in results)

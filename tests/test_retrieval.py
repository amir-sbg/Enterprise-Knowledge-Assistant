import pytest

from rag_system.bm25 import BM25Index
from rag_system.chunking import TokenChunker
from rag_system.diversity import diversify_results
from rag_system.embeddings import HashEmbeddingModel
from rag_system.retriever import HybridRetriever
from rag_system.schema import Chunk, Document, RetrievedChunk
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


def test_metadata_filters_ignore_case_and_padding():
    docs = [
        Document(
            id="ops",
            text="on call handoffs must include open incidents and owners",
            metadata={"department": "Operations"},
        ),
        Document(
            id="finance",
            text="travel requests include the conference budget",
            metadata={"department": "Finance"},
        ),
    ]
    chunks = TokenChunker(chunk_size=30, overlap=4).split(docs)
    model = HashEmbeddingModel(dim=128)
    store = InMemoryVectorStore()
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))

    results = HybridRetriever(store, BM25Index(chunks), embedding_model=model).retrieve(
        "handoff incidents",
        filters={"department": " operations "},
    )

    assert results
    assert {item.chunk.document_id for item in results} == {"ops"}


def test_retriever_can_weight_lexical_matches_more_heavily():
    docs = [
        Document(id="semantic", text="people operations onboarding transfer request"),
        Document(id="exact", text="pii pii encryption ticket approval"),
    ]
    chunks = TokenChunker(chunk_size=30, overlap=4).split(docs)
    model = HashEmbeddingModel(dim=128)
    store = InMemoryVectorStore()
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))

    results = HybridRetriever(
        store,
        BM25Index(chunks),
        embedding_model=model,
        semantic_weight=0.0,
        bm25_weight=2.0,
    ).retrieve("pii encryption", top_k=2)

    assert results[0].chunk.document_id == "exact"
    assert results[0].score > 0


def test_retriever_validates_fusion_weights():
    retriever = _retriever()
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(
            retriever.vector_store,
            retriever.bm25,
            semantic_weight=-0.1,
        )
    with pytest.raises(ValueError, match="retrieval weight"):
        HybridRetriever(
            retriever.vector_store,
            retriever.bm25,
            semantic_weight=0.0,
            bm25_weight=0.0,
        )


def test_retriever_validates_query_and_candidate_budget():
    retriever = _retriever()

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("  ")
    with pytest.raises(ValueError, match="positive"):
        retriever.retrieve("security", top_k=0)
    with pytest.raises(ValueError, match="candidate_k"):
        retriever.retrieve("security", top_k=3, candidate_k=2)


def test_diversifier_reduces_repeated_context():
    items = [
        RetrievedChunk(
            Chunk("a1", "security", "restricted data encryption approval ticket"),
            score=0.90,
        ),
        RetrievedChunk(
            Chunk("a2", "security", "restricted data encryption approval ticket"),
            score=0.88,
        ),
        RetrievedChunk(
            Chunk("b1", "governance", "model review drift monitoring escalation"),
            score=0.70,
        ),
    ]

    selected = diversify_results(items, top_k=2, diversity_weight=0.75)

    assert [item.chunk.id for item in selected] == ["a1", "b1"]


def test_diversifier_validates_settings():
    with pytest.raises(ValueError, match="top_k"):
        diversify_results([], top_k=0)
    with pytest.raises(ValueError, match="diversity_weight"):
        diversify_results([], top_k=1, diversity_weight=1.5)

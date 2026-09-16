import numpy as np
import pytest

from rag_system.embeddings import HashEmbeddingModel, cosine_similarity
from rag_system.schema import Chunk
from rag_system.vector_store import InMemoryVectorStore


def test_vector_store_rejects_dimension_changes() -> None:
    store = InMemoryVectorStore()
    store.add([Chunk("a", "doc", "text")], np.ones((1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="embedding dimension"):
        store.add([Chunk("b", "doc", "more")], np.ones((1, 3), dtype=np.float32))


def test_cosine_similarity_checks_shapes_and_finite_values() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity(np.ones((2, 4)), np.ones(3))

    with pytest.raises(ValueError, match="finite"):
        cosine_similarity(np.array([[np.nan, 0.0]]), np.ones(2))


def test_embedding_dimension_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        HashEmbeddingModel(dim=0)

from rag_system.chunking import TokenChunker
from rag_system.schema import Document


def test_token_chunker_keeps_overlap_offsets():
    doc = Document(id="doc", text=" ".join(f"token{i}" for i in range(20)))
    chunks = TokenChunker(chunk_size=8, overlap=2).split([doc])

    assert len(chunks) == 3
    assert chunks[0].start_token == 0
    assert chunks[1].start_token == 6
    assert chunks[2].end_token == 20


def test_token_chunker_rejects_bad_overlap():
    try:
        TokenChunker(chunk_size=8, overlap=8)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected invalid overlap to fail")


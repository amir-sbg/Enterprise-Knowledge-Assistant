from __future__ import annotations

from rag_system.schema import Chunk, Document
from rag_system.text import batched, normalize_text, tokenize


class TokenChunker:
    def __init__(self, chunk_size: int = 140, overlap: int = 32) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            tokens = tokenize(doc.text)
            if not tokens:
                continue
            stride = self.chunk_size - self.overlap
            for chunk_idx, (start, end, token_window) in enumerate(
                batched(tokens, self.chunk_size, stride)
            ):
                chunk_text = self._recover_chunk_text(doc.text, token_window)
                chunks.append(
                    Chunk(
                        id=f"{doc.id}::chunk-{chunk_idx:03d}",
                        document_id=doc.id,
                        text=chunk_text,
                        metadata=dict(doc.metadata),
                        start_token=start,
                        end_token=end,
                    )
                )
        return chunks

    def _recover_chunk_text(self, original: str, tokens: list[str]) -> str:
        # For a local demo this is enough; a production parser would keep character offsets.
        return normalize_text(" ".join(tokens))


from __future__ import annotations

from rag_system.schema import RetrievedChunk
from rag_system.tracking import estimate_tokens


def pack_context(
    items: list[RetrievedChunk],
    token_budget: int | None = None,
    max_chunks_per_document: int | None = None,
) -> tuple[list[RetrievedChunk], dict[str, int | None]]:
    if token_budget is not None and token_budget < 1:
        raise ValueError("token_budget must be positive")
    if max_chunks_per_document is not None and max_chunks_per_document < 1:
        raise ValueError("max_chunks_per_document must be positive")

    selected: list[RetrievedChunk] = []
    used_tokens = 0
    document_counts: dict[str, int] = {}
    dropped_for_budget = 0
    dropped_for_source_cap = 0
    for item in items:
        document_id = item.chunk.document_id
        if (
            max_chunks_per_document is not None
            and document_counts.get(document_id, 0) >= max_chunks_per_document
        ):
            dropped_for_source_cap += 1
            continue
        chunk_tokens = estimate_tokens(item.chunk.text)
        if token_budget is not None and used_tokens + chunk_tokens > token_budget:
            dropped_for_budget += 1
            continue
        selected.append(item)
        used_tokens += chunk_tokens
        document_counts[document_id] = document_counts.get(document_id, 0) + 1

    return selected, {
        "context_token_budget": token_budget,
        "context_tokens_used": used_tokens,
        "context_chunks_dropped": len(items) - len(selected),
        "context_chunks_dropped_for_budget": dropped_for_budget,
        "context_chunks_dropped_for_source_cap": dropped_for_source_cap,
        "max_chunks_per_document": max_chunks_per_document,
    }

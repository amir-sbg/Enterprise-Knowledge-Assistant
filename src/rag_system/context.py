from __future__ import annotations

from rag_system.schema import RetrievedChunk
from rag_system.tracking import estimate_tokens


def pack_context(
    items: list[RetrievedChunk],
    token_budget: int,
) -> tuple[list[RetrievedChunk], dict[str, int]]:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")

    selected: list[RetrievedChunk] = []
    used_tokens = 0
    for item in items:
        chunk_tokens = estimate_tokens(item.chunk.text)
        if used_tokens + chunk_tokens > token_budget:
            continue
        selected.append(item)
        used_tokens += chunk_tokens

    return selected, {
        "context_token_budget": token_budget,
        "context_tokens_used": used_tokens,
        "context_chunks_dropped": len(items) - len(selected),
    }

from __future__ import annotations

from rag_system.schema import RetrievedChunk
from rag_system.text import token_set


def diversify_results(
    items: list[RetrievedChunk],
    top_k: int,
    diversity_weight: float = 0.15,
) -> list[RetrievedChunk]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be between 0 and 1")
    if len(items) <= top_k:
        return list(items)

    candidates = list(items)
    normalized_scores = _normalized_scores(candidates)
    selected: list[RetrievedChunk] = []

    while candidates and len(selected) < top_k:
        if not selected:
            choice = max(candidates, key=lambda item: normalized_scores[item.chunk.id])
        else:
            choice = max(
                candidates,
                key=lambda item: _mmr_score(
                    item,
                    selected,
                    normalized_scores[item.chunk.id],
                    diversity_weight,
                ),
            )
        selected.append(choice)
        candidates.remove(choice)

    return selected


def _normalized_scores(items: list[RetrievedChunk]) -> dict[str, float]:
    scores = [float(item.score) for item in items]
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return {item.chunk.id: 1.0 for item in items}
    return {item.chunk.id: (float(item.score) - lo) / (hi - lo) for item in items}


def _mmr_score(
    item: RetrievedChunk,
    selected: list[RetrievedChunk],
    normalized_score: float,
    diversity_weight: float,
) -> float:
    redundancy = max(_token_jaccard(item, other) for other in selected)
    return (1.0 - diversity_weight) * normalized_score - diversity_weight * redundancy


def _token_jaccard(first: RetrievedChunk, second: RetrievedChunk) -> float:
    a = token_set(first.chunk.text)
    b = token_set(second.chunk.text)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

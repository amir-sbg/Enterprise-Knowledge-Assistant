from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from rag_system.text import tokenize


@dataclass(slots=True)
class CostModel:
    input_per_1k: float = 0.00015
    output_per_1k: float = 0.00060

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000) * self.input_per_1k + (output_tokens / 1000) * self.output_per_1k


def estimate_tokens(text: str) -> int:
    return max(1, len(tokenize(text)))


@contextmanager
def latency_timer() -> Iterator[dict[str, float]]:
    start = time.perf_counter()
    result: dict[str, float] = {}
    try:
        yield result
    finally:
        result["latency_ms"] = (time.perf_counter() - start) * 1000


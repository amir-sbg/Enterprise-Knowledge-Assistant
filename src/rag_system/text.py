from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-./]*")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def batched(items: list[str], size: int, stride: int) -> Iterable[tuple[int, int, list[str]]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    start = 0
    while start < len(items):
        end = min(start + size, len(items))
        yield start, end, items[start:end]
        if end == len(items):
            break
        start += stride


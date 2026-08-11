from __future__ import annotations

import json
from pathlib import Path

from rag_system.schema import Document
from rag_system.text import normalize_text


SUPPORTED_SUFFIXES = {".md", ".txt", ".jsonl"}


def load_documents(path: str | Path) -> list[Document]:
    root = Path(path)
    if root.is_file():
        return _load_file(root)

    docs: list[Document] = []
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.extend(_load_file(file_path))
    return docs


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)

    text = normalize_text(path.read_text(encoding="utf-8"))
    return [
        Document(
            id=path.stem,
            text=text,
            metadata={
                "source": str(path),
                "title": path.stem.replace("-", " ").replace("_", " ").title(),
                "file_type": suffix.lstrip("."),
            },
        )
    ]


def _load_jsonl(path: Path) -> list[Document]:
    docs: list[Document] = []
    for row_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        text = normalize_text(record.get("text", ""))
        metadata = dict(record.get("metadata", {}))
        metadata.setdefault("source", str(path))
        metadata.setdefault("row", row_num)
        docs.append(
            Document(
                id=str(record.get("id", f"{path.stem}-{row_num}")),
                text=text,
                metadata=metadata,
            )
        )
    return docs


from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class JsonQueryCache:
    def __init__(self, path: str | Path = "cache/query_cache.json", ttl_seconds: int = 3600) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items = self._load()

    def get(self, key_parts: dict[str, Any]) -> dict[str, Any] | None:
        key = self._key(key_parts)
        item = self._items.get(key)
        if not item:
            return None
        if time.time() - item["created_at"] > self.ttl_seconds:
            self._items.pop(key, None)
            self.flush()
            return None
        return item["value"]

    def set(self, key_parts: dict[str, Any], value: dict[str, Any]) -> None:
        key = self._key(key_parts)
        self._items[key] = {"created_at": time.time(), "value": value}
        self.flush()

    def flush(self) -> None:
        self.path.write_text(json.dumps(self._items, indent=2, sort_keys=True), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _key(self, key_parts: dict[str, Any]) -> str:
        blob = json.dumps(key_parts, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


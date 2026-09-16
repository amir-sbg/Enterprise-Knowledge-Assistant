import pytest

from rag_system.cache import JsonQueryCache


def test_cache_recovers_from_corrupt_file(tmp_path):
    cache_path = tmp_path / "query_cache.json"
    cache_path.write_text("{not valid json", encoding="utf-8")

    cache = JsonQueryCache(cache_path)
    cache.set({"q": "hello"}, {"answer": "world"})

    assert cache.get({"q": "hello"}) == {"answer": "world"}
    assert (tmp_path / "query_cache.json.bad").exists()


def test_cache_rejects_non_positive_ttl(tmp_path):
    with pytest.raises(ValueError, match="ttl_seconds"):
        JsonQueryCache(tmp_path / "query_cache.json", ttl_seconds=0)


def test_cache_flush_writes_valid_json(tmp_path):
    cache = JsonQueryCache(tmp_path / "query_cache.json")
    cache.set({"q": "hello"}, {"answer": "world"})

    assert cache.path.read_text(encoding="utf-8").startswith("{")
    assert list(tmp_path.glob("*.tmp")) == []

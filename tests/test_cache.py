from rag_system.cache import JsonQueryCache


def test_cache_recovers_from_corrupt_file(tmp_path):
    cache_path = tmp_path / "query_cache.json"
    cache_path.write_text("{not valid json", encoding="utf-8")

    cache = JsonQueryCache(cache_path)
    cache.set({"q": "hello"}, {"answer": "world"})

    assert cache.get({"q": "hello"}) == {"answer": "world"}
    assert (tmp_path / "query_cache.json.bad").exists()


from pathlib import Path

from rag_system.evaluation import evaluate_pipeline, load_eval_cases
from rag_system.pipeline import RAGPipeline, build_index, index_is_complete, read_index_manifest


def test_pipeline_returns_cited_answer(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path)

    answer = pipeline.ask("What should employees do before sending restricted data?", top_k=3)

    assert "restricted" in answer.text.lower()
    assert answer.citations
    assert answer.citations[0].document_id == "security-policy"
    assert answer.metrics["citation_accuracy"] > 0
    assert answer.metrics["latency_ms"] >= 0


def test_eval_suite_reports_retrieval_metrics(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path)

    report = evaluate_pipeline(pipeline, load_eval_cases("eval/queries.jsonl"), top_k=5)

    assert report["summary"]["cases"] == 4
    assert report["summary"]["recall_at_k"] >= 0.75
    assert "hallucination_rate" in report["summary"]
    assert "cost_per_query_avg" in report["summary"]


def test_index_manifest_tracks_build_settings(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path, chunk_size=80, overlap=10)

    manifest = read_index_manifest(index_path)

    assert index_is_complete(index_path)
    assert manifest["chunk_size"] == 80
    assert manifest["overlap"] == 10
    assert manifest["documents"] == 4
    assert manifest["chunks"] >= 4

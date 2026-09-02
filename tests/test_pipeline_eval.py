from pathlib import Path

from rag_system.evaluation import (
    _context_precision_at_k,
    _ndcg_at_k,
    evaluate_pipeline,
    evaluate_retrieval_modes,
    load_eval_cases,
)
from rag_system.pipeline import (
    RAGPipeline,
    build_index,
    index_is_complete,
    metadata_facets,
    read_index_manifest,
    retrieval_coverage_metrics,
    retrieval_trace,
)
from rag_system.schema import Chunk, RetrievedChunk


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
    assert answer.metrics["retrieved_chunks"] == 3
    assert answer.metrics["unique_retrieved_documents"] >= 1


def test_eval_suite_reports_retrieval_metrics(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path)

    report = evaluate_pipeline(pipeline, load_eval_cases("eval/queries.jsonl"), top_k=5)

    assert report["summary"]["cases"] == 4
    assert report["summary"]["recall_at_k"] >= 0.75
    assert "ndcg_at_k" in report["summary"]
    assert "context_precision_at_k" in report["summary"]
    assert "hallucination_rate" in report["summary"]
    assert "cost_per_query_avg" in report["summary"]


def test_retrieval_ablation_compares_search_modes(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path)
    cases = load_eval_cases("eval/queries.jsonl")

    report = evaluate_retrieval_modes(pipeline, cases, top_k=3)

    assert {row["mode"] for row in report["summary"]} == {"hybrid", "semantic", "bm25"}
    assert len(report["rows"]) == 12
    assert all("context_precision_at_k" in row for row in report["summary"])


def test_ranking_metrics_discount_late_and_duplicate_hits():
    expected = {"security", "governance"}

    strong = _ndcg_at_k(["security", "governance", "security"], expected, top_k=3)
    late = _ndcg_at_k(["travel", "security", "security"], expected, top_k=3)
    precision = _context_precision_at_k(["travel", "security", "security"], expected, top_k=3)

    assert strong == 1.0
    assert late < strong
    assert precision == 2 / 3


def test_index_manifest_tracks_build_settings(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path, chunk_size=80, overlap=10)

    manifest = read_index_manifest(index_path)

    assert index_is_complete(index_path)
    assert manifest["chunk_size"] == 80
    assert manifest["overlap"] == 10
    assert manifest["documents"] == 4
    assert manifest["chunks"] >= 4
    assert manifest["metadata_facets"]["department"]["security"] >= 1
    assert manifest["metadata_facets"]["sensitivity"]["internal"] >= 1


def test_retrieval_trace_summarizes_answer_evidence(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path)

    answer = pipeline.ask("How are vendor security reviews handled?", top_k=2)
    trace = retrieval_trace(answer, preview_tokens=8)

    assert len(trace) == len(answer.retrieval)
    assert trace[0]["rank"] == 1
    assert trace[0]["document_id"]
    assert "score" in trace[0]
    assert len(trace[0]["preview"].split()) <= 9


def test_retrieval_coverage_metrics_track_duplicate_context() -> None:
    retrieved = [
        RetrievedChunk(Chunk("a", "security", "restricted data"), score=0.7),
        RetrievedChunk(Chunk("b", "security", "encryption approval"), score=0.5),
        RetrievedChunk(Chunk("c", "governance", "model review"), score=0.3),
    ]

    metrics = retrieval_coverage_metrics(retrieved)

    assert metrics["retrieved_chunks"] == 3
    assert metrics["unique_retrieved_documents"] == 2
    assert metrics["duplicate_document_rate"] == 0.3333


def test_metadata_facets_counts_chunk_metadata() -> None:
    chunks = [
        Chunk("a", "security", "restricted data", metadata={"department": "security"}),
        Chunk("b", "security", "encryption", metadata={"department": "security"}),
        Chunk("c", "finance", "travel", metadata={"department": "finance"}),
    ]

    assert metadata_facets(chunks)["department"] == {"finance": 1, "security": 2}

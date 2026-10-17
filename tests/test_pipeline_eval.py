from pathlib import Path

import pytest

from rag_system.evaluation import (
    _bootstrap_mean_interval,
    _context_precision_at_k,
    _citation_metrics,
    _ndcg_at_k,
    _recall_at_k,
    evaluate_pipeline,
    evaluate_retrieval_modes,
    load_eval_cases,
    paired_bootstrap_difference,
    paired_sign_test,
    retrieval_confidence_calibration,
    retrieval_failure_summary,
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


def test_pipeline_can_abstain_below_score_floor(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path)

    answer = pipeline.ask(
        "What should employees do before sending restricted data?",
        top_k=3,
        score_floor=1.0,
    )

    assert "could not find enough evidence" in answer.text.lower()
    assert answer.metrics["retrieved_chunks"] == 0
    assert answer.metrics["low_evidence_answer"] == 1
    assert answer.metrics["retrieval_score_floor"] == 1.0


def test_pipeline_rejects_negative_score_floor(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path)

    with pytest.raises(ValueError, match="score_floor"):
        pipeline.ask("hello", score_floor=-0.1)


def test_pipeline_normalizes_questions_and_reports_cache_hits(tmp_path: Path):
    index_path = tmp_path / "idx"
    cache_path = tmp_path / "cache.json"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path, cache_path=cache_path)

    first = pipeline.ask("  What is SSO?  ", top_k=2)
    second = pipeline.ask("What   is SSO?", top_k=2)

    assert first.metrics["cache_hit"] == 0
    assert second.metrics["cache_hit"] == 1
    assert second.text == first.text


def test_pipeline_rejects_empty_question(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path)

    with pytest.raises(ValueError, match="question"):
        pipeline.ask(" ")


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
    assert "citation_f1" in report["summary"]
    assert "confidence_intervals" in report["summary"]
    assert "recall_at_k" in report["summary"]["confidence_intervals"]
    assert "cost_per_query_avg" in report["summary"]
    assert "retrieval_confidence_calibration" in report["summary"]
    assert report["summary"]["retrieval_confidence_calibration"]["rows"] == 4
    assert report["summary"]["failure_analysis"]["cases"] == 4


def test_retrieval_confidence_calibration_reports_reliability() -> None:
    report = retrieval_confidence_calibration(
        confidences=[0.1, 0.3, 0.8, 0.9],
        outcomes=[0, 0, 1, 1],
        bins=2,
    )

    assert report["rows"] == 4
    assert report["brier_score"] == pytest.approx(0.0375)
    assert report["ece"] == pytest.approx(0.175)
    assert len(report["bins"]) == 2


def test_retrieval_failure_summary_separates_failure_modes() -> None:
    report = retrieval_failure_summary(
        [
            {"id": "ok", "recall_at_k": 1.0, "faithfulness": 1.0, "citation_f1": 1.0},
            {"id": "partial", "recall_at_k": 0.5, "faithfulness": 1.0, "citation_f1": 0.5},
            {"id": "miss", "recall_at_k": 0.0, "faithfulness": 0.5, "citation_f1": 0.0},
        ]
    )

    assert report["complete_retrieval_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["partial_retrieval_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["no_retrieval_hit_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["case_ids"]["no_retrieval_hit"] == ["miss"]


def test_retrieval_ablation_compares_search_modes(tmp_path: Path):
    index_path = tmp_path / "idx"
    build_index("data/sample_docs", index_path)
    pipeline = RAGPipeline.load(index_path)
    cases = load_eval_cases("eval/queries.jsonl")

    report = evaluate_retrieval_modes(pipeline, cases, top_k=3)

    assert {row["mode"] for row in report["summary"]} == {"hybrid", "semantic", "bm25"}
    assert len(report["rows"]) == 12
    assert all("context_precision_at_k" in row for row in report["summary"])
    assert len(report["comparisons"]) == 6
    assert {row["comparison"] for row in report["comparisons"]} == {"semantic", "bm25"}


def test_paired_bootstrap_reports_direction_and_interval() -> None:
    report = paired_bootstrap_difference(
        [1.0, 0.8, 0.9, 0.7],
        [0.4, 0.5, 0.3, 0.2],
        samples=400,
        seed=7,
    )

    assert report["mean_difference"] > 0
    assert report["confidence_interval"]["low"] > 0
    assert report["probability_reference_better"] == 1.0


def test_paired_bootstrap_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_difference([1.0], [1.0, 0.0])


def test_paired_sign_test_counts_query_level_wins() -> None:
    report = paired_sign_test(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.1],
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.2, 0.2],
    )

    assert report["reference_wins"] == 7
    assert report["comparison_wins"] == 1
    assert report["ties"] == 0
    assert report["two_sided_p_value"] == pytest.approx(0.070312)


def test_ranking_metrics_discount_late_and_duplicate_hits():
    expected = {"security", "governance"}

    strong = _ndcg_at_k(["security", "governance", "security"], expected, top_k=3)
    late = _ndcg_at_k(["travel", "security", "security"], expected, top_k=3)
    precision = _context_precision_at_k(["travel", "security", "security"], expected, top_k=3)

    assert strong == 1.0
    assert late < strong
    assert precision == 2 / 3


def test_recall_at_k_counts_all_expected_documents() -> None:
    expected = {"security", "governance"}

    assert _recall_at_k(["security", "travel"], expected, top_k=2) == 0.5
    assert _recall_at_k(["security", "security", "governance"], expected, top_k=3) == 1.0


def test_citation_metrics_penalize_extra_and_missing_sources() -> None:
    metrics = _citation_metrics(
        {"security", "travel"},
        {"security", "governance"},
    )

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_bootstrap_interval_is_reproducible_and_contains_mean() -> None:
    first = _bootstrap_mean_interval([0.0, 0.5, 1.0], samples=300, seed=8)
    second = _bootstrap_mean_interval([0.0, 0.5, 1.0], samples=300, seed=8)

    assert first == second
    assert first["low"] <= 0.5 <= first["high"]


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
    assert metrics["top_score_margin"] == pytest.approx(0.2)
    assert 0.0 <= metrics["score_concentration"] <= 1.0


def test_retrieval_score_concentration_increases_with_clear_winner() -> None:
    flat = [
        RetrievedChunk(Chunk("a", "a", "a"), score=0.5),
        RetrievedChunk(Chunk("b", "b", "b"), score=0.5),
    ]
    decisive = [
        RetrievedChunk(Chunk("a", "a", "a"), score=5.0),
        RetrievedChunk(Chunk("b", "b", "b"), score=0.0),
    ]

    assert retrieval_coverage_metrics(decisive)["score_concentration"] > retrieval_coverage_metrics(flat)[
        "score_concentration"
    ]


def test_metadata_facets_counts_chunk_metadata() -> None:
    chunks = [
        Chunk("a", "security", "restricted data", metadata={"department": "security"}),
        Chunk("b", "security", "encryption", metadata={"department": "security"}),
        Chunk("c", "finance", "travel", metadata={"department": "finance"}),
    ]

    assert metadata_facets(chunks)["department"] == {"finance": 1, "security": 2}

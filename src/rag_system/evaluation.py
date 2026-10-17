from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from rag_system.pipeline import RAGPipeline
from rag_system.schema import EvalCase
from rag_system.text import token_set


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            EvalCase(
                id=row["id"],
                question=row["question"],
                expected_doc_ids=list(row["expected_doc_ids"]),
                answer_keywords=list(row.get("answer_keywords", [])),
                filters=dict(row.get("filters", {})),
            )
        )
    return cases


def evaluate_pipeline(
    pipeline: RAGPipeline,
    cases: list[EvalCase],
    top_k: int = 5,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        answer = pipeline.ask(case.question, top_k=top_k, filters=case.filters, use_cache=False)
        retrieved_doc_ids = [item.chunk.document_id for item in answer.retrieval[:top_k]]
        expected = set(case.expected_doc_ids)

        recall = _recall_at_k(retrieved_doc_ids, expected, top_k=top_k)
        reciprocal_rank = _reciprocal_rank(retrieved_doc_ids, expected)
        ndcg = _ndcg_at_k(retrieved_doc_ids, expected, top_k=top_k)
        context_precision = _context_precision_at_k(retrieved_doc_ids, expected, top_k=top_k)
        correctness = _answer_correctness(answer.text, case.answer_keywords)
        cited_doc_ids = {citation.document_id for citation in answer.citations}
        citation_metrics = _citation_metrics(cited_doc_ids, expected)

        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "retrieved_doc_ids": retrieved_doc_ids,
                "expected_doc_ids": case.expected_doc_ids,
                "recall_at_k": recall,
                "mrr": reciprocal_rank,
                "ndcg_at_k": ndcg,
                "context_precision_at_k": context_precision,
                "answer_correctness": correctness,
                "faithfulness": answer.metrics.get("faithfulness", 0.0),
                "hallucination_rate": 1.0 - answer.metrics.get("faithfulness", 0.0),
                "citation_accuracy": citation_metrics["recall"],
                "citation_precision": citation_metrics["precision"],
                "citation_recall": citation_metrics["recall"],
                "citation_f1": citation_metrics["f1"],
                "latency_ms": answer.metrics.get("latency_ms", 0.0),
                "estimated_cost_usd": answer.metrics.get("estimated_cost_usd", 0.0),
                "retrieval_confidence": answer.metrics.get("score_concentration", 0.0),
                "retrieval_hit": float(recall > 0.0),
            }
        )

    summary = {
        "cases": len(rows),
        "recall_at_k": round(mean([row["recall_at_k"] for row in rows]), 4) if rows else 0.0,
        "mrr": round(mean([row["mrr"] for row in rows]), 4) if rows else 0.0,
        "ndcg_at_k": round(mean([row["ndcg_at_k"] for row in rows]), 4) if rows else 0.0,
        "context_precision_at_k": round(
            mean([row["context_precision_at_k"] for row in rows]), 4
        )
        if rows
        else 0.0,
        "answer_correctness": round(mean([row["answer_correctness"] for row in rows]), 4)
        if rows
        else 0.0,
        "faithfulness": round(mean([row["faithfulness"] for row in rows]), 4) if rows else 0.0,
        "hallucination_rate": round(mean([row["hallucination_rate"] for row in rows]), 4)
        if rows
        else 0.0,
        "citation_accuracy": round(mean([row["citation_accuracy"] for row in rows]), 4)
        if rows
        else 0.0,
        "citation_precision": round(mean([row["citation_precision"] for row in rows]), 4)
        if rows
        else 0.0,
        "citation_recall": round(mean([row["citation_recall"] for row in rows]), 4)
        if rows
        else 0.0,
        "citation_f1": round(mean([row["citation_f1"] for row in rows]), 4)
        if rows
        else 0.0,
        "latency_ms_avg": round(mean([row["latency_ms"] for row in rows]), 2) if rows else 0.0,
        "cost_per_query_avg": round(mean([row["estimated_cost_usd"] for row in rows]), 8)
        if rows
        else 0.0,
    }
    interval_metrics = ("recall_at_k", "ndcg_at_k", "faithfulness", "citation_f1")
    summary["confidence_intervals"] = {
        metric: _bootstrap_mean_interval(
            [float(row[metric]) for row in rows],
            seed=42 + index,
        )
        for index, metric in enumerate(interval_metrics)
    }
    summary["retrieval_confidence_calibration"] = retrieval_confidence_calibration(
        [float(row["retrieval_confidence"]) for row in rows],
        [int(row["retrieval_hit"]) for row in rows],
    )
    summary["failure_analysis"] = retrieval_failure_summary(rows)
    return {
        "summary": summary,
        "rows": rows,
    }


def retrieval_confidence_calibration(
    confidences: list[float],
    outcomes: list[int],
    *,
    bins: int = 5,
) -> dict[str, Any]:
    if len(confidences) != len(outcomes):
        raise ValueError("confidence and outcome rows must match")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if any(not 0.0 <= value <= 1.0 for value in confidences):
        raise ValueError("retrieval confidences must be in [0, 1]")
    if any(value not in {0, 1} for value in outcomes):
        raise ValueError("retrieval outcomes must be binary")
    if not confidences:
        return {"rows": 0, "ece": 0.0, "brier_score": 0.0, "bins": []}

    calibration_rows = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            row
            for row, confidence in enumerate(confidences)
            if confidence >= lower
            and (confidence < upper if index < bins - 1 else confidence <= upper)
        ]
        if not selected:
            continue
        mean_confidence = mean(confidences[row] for row in selected)
        hit_rate = mean(outcomes[row] for row in selected)
        gap = abs(mean_confidence - hit_rate)
        ece += len(selected) / len(confidences) * gap
        calibration_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "rows": len(selected),
                "mean_confidence": round(mean_confidence, 4),
                "hit_rate": round(hit_rate, 4),
                "absolute_gap": round(gap, 4),
            }
        )
    brier = mean(
        (confidence - outcome) ** 2
        for confidence, outcome in zip(confidences, outcomes)
    )
    return {
        "rows": len(confidences),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "bins": calibration_rows,
    }


def retrieval_failure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "complete_retrieval_rate": 0.0,
            "partial_retrieval_rate": 0.0,
            "no_retrieval_hit_rate": 0.0,
            "unfaithful_answer_rate": 0.0,
            "citation_failure_rate": 0.0,
            "case_ids": {},
        }
    required = {"id", "recall_at_k", "faithfulness", "citation_f1"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"evaluation row is missing fields: {sorted(missing)}")

    complete = [row["id"] for row in rows if float(row["recall_at_k"]) >= 1.0]
    partial = [
        row["id"]
        for row in rows
        if 0.0 < float(row["recall_at_k"]) < 1.0
    ]
    no_hit = [row["id"] for row in rows if float(row["recall_at_k"]) == 0.0]
    unfaithful = [row["id"] for row in rows if float(row["faithfulness"]) < 1.0]
    citation_failure = [row["id"] for row in rows if float(row["citation_f1"]) < 1.0]
    count = len(rows)
    return {
        "cases": count,
        "complete_retrieval_rate": round(len(complete) / count, 4),
        "partial_retrieval_rate": round(len(partial) / count, 4),
        "no_retrieval_hit_rate": round(len(no_hit) / count, 4),
        "unfaithful_answer_rate": round(len(unfaithful) / count, 4),
        "citation_failure_rate": round(len(citation_failure) / count, 4),
        "case_ids": {
            "partial_retrieval": partial,
            "no_retrieval_hit": no_hit,
            "unfaithful_answer": unfaithful,
            "citation_failure": citation_failure,
        },
    }


def evaluate_retrieval_modes(
    pipeline: RAGPipeline,
    cases: list[EvalCase],
    top_k: int = 5,
    modes: tuple[str, ...] = ("hybrid", "semantic", "bm25"),
) -> dict[str, Any]:
    rows = []
    for mode in modes:
        mode_rows = []
        for case in cases:
            retrieved = _retrieve_for_mode(pipeline, case, top_k=top_k, mode=mode)
            retrieved_doc_ids = [item.chunk.document_id for item in retrieved[:top_k]]
            expected = set(case.expected_doc_ids)
            mode_rows.append(
                {
                    "id": case.id,
                    "mode": mode,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "recall_at_k": _recall_at_k(retrieved_doc_ids, expected, top_k=top_k),
                    "mrr": _reciprocal_rank(retrieved_doc_ids, expected),
                    "ndcg_at_k": _ndcg_at_k(retrieved_doc_ids, expected, top_k=top_k),
                    "context_precision_at_k": _context_precision_at_k(
                        retrieved_doc_ids,
                        expected,
                        top_k=top_k,
                    ),
                }
            )
        rows.extend(mode_rows)

    summary = []
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        summary.append(
            {
                "mode": mode,
                "cases": len(mode_rows),
                "recall_at_k": round(mean([row["recall_at_k"] for row in mode_rows]), 4)
                if mode_rows
                else 0.0,
                "mrr": round(mean([row["mrr"] for row in mode_rows]), 4)
                if mode_rows
                else 0.0,
                "ndcg_at_k": round(mean([row["ndcg_at_k"] for row in mode_rows]), 4)
                if mode_rows
                else 0.0,
                "context_precision_at_k": round(
                    mean([row["context_precision_at_k"] for row in mode_rows]),
                    4,
                )
                if mode_rows
                else 0.0,
            }
        )
    comparisons = []
    if modes:
        reference = modes[0]
        reference_rows = {row["id"]: row for row in rows if row["mode"] == reference}
        for mode_index, mode in enumerate(modes[1:], start=1):
            candidate_rows = {row["id"]: row for row in rows if row["mode"] == mode}
            shared_ids = sorted(reference_rows.keys() & candidate_rows.keys())
            for metric_index, metric in enumerate(("recall_at_k", "mrr", "ndcg_at_k")):
                comparison = paired_bootstrap_difference(
                    [float(reference_rows[case_id][metric]) for case_id in shared_ids],
                    [float(candidate_rows[case_id][metric]) for case_id in shared_ids],
                    seed=100 + mode_index * 10 + metric_index,
                )
                comparisons.append(
                    {
                        "reference": reference,
                        "comparison": mode,
                        "metric": metric,
                        "cases": len(shared_ids),
                        **comparison,
                    }
                )
    return {"summary": summary, "comparisons": comparisons, "rows": rows}


def paired_bootstrap_difference(
    reference: list[float],
    comparison: list[float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    if len(reference) != len(comparison):
        raise ValueError("paired samples must have the same length")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not reference:
        return {
            "mean_difference": 0.0,
            "confidence_interval": {"low": 0.0, "high": 0.0},
            "probability_reference_better": 0.0,
        }

    differences = [left - right for left, right in zip(reference, comparison)]
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_difference": round(mean(differences), 4),
        "confidence_interval": {
            "low": round(_quantile(estimates, tail), 4),
            "high": round(_quantile(estimates, 1.0 - tail), 4),
        },
        "probability_reference_better": round(
            sum(estimate > 0.0 for estimate in estimates) / len(estimates),
            4,
        ),
    }


def _retrieve_for_mode(
    pipeline: RAGPipeline,
    case: EvalCase,
    top_k: int,
    mode: str,
):
    retriever = pipeline.retriever
    rewritten = retriever.rewriter.rewrite(case.question)
    if mode == "hybrid":
        return retriever.retrieve(case.question, top_k=top_k, filters=case.filters)
    if mode == "semantic":
        query_vector = retriever.embedding_model.embed_query(rewritten)
        return retriever.vector_store.search(
            query_vector,
            top_k=top_k,
            filters=case.filters,
        )
    if mode in {"bm25", "lexical"}:
        return retriever.bm25.search(rewritten, top_k=top_k, filters=case.filters)
    raise ValueError(f"unknown retrieval mode: {mode}")


def _reciprocal_rank(retrieved_doc_ids: list[str], expected: set[str]) -> float:
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / idx
    return 0.0


def _recall_at_k(retrieved_doc_ids: list[str], expected: set[str], top_k: int) -> float:
    if top_k < 1 or not expected:
        return 0.0
    retrieved = set(retrieved_doc_ids[:top_k])
    return len(retrieved & expected) / len(expected)


def _dcg(relevance: list[int]) -> float:
    return sum(gain / _log2(rank + 1) for rank, gain in enumerate(relevance, start=1))


def _log2(value: int) -> float:
    from math import log2

    return log2(value)


def _deduped_relevance(retrieved_doc_ids: list[str], expected: set[str], top_k: int) -> list[int]:
    seen: set[str] = set()
    relevance = []
    for doc_id in retrieved_doc_ids[:top_k]:
        if doc_id in expected and doc_id not in seen:
            relevance.append(1)
            seen.add(doc_id)
        else:
            relevance.append(0)
    return relevance


def _ndcg_at_k(retrieved_doc_ids: list[str], expected: set[str], top_k: int) -> float:
    if top_k < 1 or not retrieved_doc_ids or not expected:
        return 0.0
    relevance = _deduped_relevance(retrieved_doc_ids, expected, top_k)
    ideal_hits = min(len(expected), top_k, len(retrieved_doc_ids))
    ideal = [1] * ideal_hits + [0] * max(0, min(top_k, len(retrieved_doc_ids)) - ideal_hits)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(relevance) / ideal_dcg


def _context_precision_at_k(
    retrieved_doc_ids: list[str],
    expected: set[str],
    top_k: int,
) -> float:
    if top_k < 1 or not retrieved_doc_ids:
        return 0.0
    window = retrieved_doc_ids[:top_k]
    return sum(1 for doc_id in window if doc_id in expected) / len(window)


def _answer_correctness(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer_terms = token_set(answer)
    hits = sum(1 for keyword in keywords if token_set(keyword) & answer_terms)
    return hits / len(keywords)


def _citation_metrics(cited: set[str], expected: set[str]) -> dict[str, float]:
    overlap = len(cited & expected)
    precision = overlap / len(cited) if cited else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _bootstrap_mean_interval(
    values: list[float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not values:
        return {"low": 0.0, "high": 0.0}
    if any(not isinstance(value, (int, float)) for value in values):
        raise ValueError("bootstrap values must be numeric")

    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "low": round(_quantile(estimates, tail), 4),
        "high": round(_quantile(estimates, 1.0 - tail), 4),
    }


def _quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

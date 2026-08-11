from __future__ import annotations

import json
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

        recall = int(bool(expected & set(retrieved_doc_ids)))
        reciprocal_rank = _reciprocal_rank(retrieved_doc_ids, expected)
        correctness = _answer_correctness(answer.text, case.answer_keywords)
        cited_doc_ids = {citation.document_id for citation in answer.citations}
        citation_accuracy = len(expected & cited_doc_ids) / max(len(expected), 1)

        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "retrieved_doc_ids": retrieved_doc_ids,
                "expected_doc_ids": case.expected_doc_ids,
                "recall_at_k": recall,
                "mrr": reciprocal_rank,
                "answer_correctness": correctness,
                "faithfulness": answer.metrics.get("faithfulness", 0.0),
                "citation_accuracy": citation_accuracy,
                "latency_ms": answer.metrics.get("latency_ms", 0.0),
                "estimated_cost_usd": answer.metrics.get("estimated_cost_usd", 0.0),
            }
        )

    return {
        "summary": {
            "cases": len(rows),
            "recall_at_k": round(mean([row["recall_at_k"] for row in rows]), 4) if rows else 0.0,
            "mrr": round(mean([row["mrr"] for row in rows]), 4) if rows else 0.0,
            "answer_correctness": round(mean([row["answer_correctness"] for row in rows]), 4)
            if rows
            else 0.0,
            "faithfulness": round(mean([row["faithfulness"] for row in rows]), 4) if rows else 0.0,
            "citation_accuracy": round(mean([row["citation_accuracy"] for row in rows]), 4)
            if rows
            else 0.0,
            "latency_ms_avg": round(mean([row["latency_ms"] for row in rows]), 2) if rows else 0.0,
            "cost_per_query_avg": round(mean([row["estimated_cost_usd"] for row in rows]), 8)
            if rows
            else 0.0,
        },
        "rows": rows,
    }


def _reciprocal_rank(retrieved_doc_ids: list[str], expected: set[str]) -> float:
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / idx
    return 0.0


def _answer_correctness(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer_terms = token_set(answer)
    hits = sum(1 for keyword in keywords if token_set(keyword) & answer_terms)
    return hits / len(keywords)


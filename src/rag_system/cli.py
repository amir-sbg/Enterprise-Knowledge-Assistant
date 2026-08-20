from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_system.evaluation import evaluate_pipeline, load_eval_cases
from rag_system.pipeline import (
    RAGPipeline,
    answer_to_dict,
    build_index,
    read_index_manifest,
    retrieval_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-system")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Build a local hybrid retrieval index")
    ingest.add_argument("--docs", default="data/sample_docs")
    ingest.add_argument("--index", default="indexes/demo")
    ingest.add_argument("--chunk-size", type=int, default=140)
    ingest.add_argument("--overlap", type=int, default=32)

    ask = subcommands.add_parser("ask", help="Ask a question against a built index")
    ask.add_argument("question")
    ask.add_argument("--index", default="indexes/demo")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--filter", action="append", default=[], help="metadata filter as key=value")
    ask.add_argument("--no-cache", action="store_true")
    ask.add_argument("--trace", action="store_true", help="include retrieval score details")

    evaluate = subcommands.add_parser("evaluate", help="Run the evaluation suite")
    evaluate.add_argument("--eval-file", default="eval/queries.jsonl")
    evaluate.add_argument("--index", default="indexes/demo")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--output", default="")

    status = subcommands.add_parser("status", help="Show index metadata")
    status.add_argument("--index", default="indexes/demo")

    args = parser.parse_args()
    if args.command == "ingest":
        chunks = build_index(args.docs, args.index, chunk_size=args.chunk_size, overlap=args.overlap)
        print(json.dumps({"chunks": len(chunks), "index": args.index}, indent=2))
    elif args.command == "ask":
        pipeline = RAGPipeline.load(args.index, cache_path="cache/query_cache.json")
        answer = pipeline.ask(
            args.question,
            top_k=args.top_k,
            filters=_parse_filters(args.filter),
            use_cache=not args.no_cache,
        )
        payload = answer_to_dict(answer)
        if args.trace:
            payload["retrieval_trace"] = retrieval_trace(answer)
        print(json.dumps(payload, indent=2))
    elif args.command == "evaluate":
        pipeline = RAGPipeline.load(args.index)
        report = evaluate_pipeline(pipeline, load_eval_cases(args.eval_file), top_k=args.top_k)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
    elif args.command == "status":
        print(json.dumps(read_index_manifest(args.index), indent=2))


def _parse_filters(items: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid filter {item!r}; expected key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid filter {item!r}; filter key cannot be empty")
        filters[key] = value.strip()
    return filters


if __name__ == "__main__":
    main()

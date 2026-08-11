.PHONY: test api eval ingest

ingest:
	python -m rag_system.cli ingest --docs data/sample_docs --index indexes/demo

eval:
	python -m rag_system.cli evaluate --eval-file eval/queries.jsonl --index indexes/demo

api:
	uvicorn rag_system.api:app --reload

test:
	python -m pytest -q


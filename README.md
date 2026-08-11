# Enterprise Knowledge Assistant

Production-style Retrieval-Augmented Generation system with hybrid retrieval, reranking, citation checks, and evaluation.

This project is intentionally not a “chat with a PDF” demo. It is a compact implementation of the moving parts usually needed in a production RAG service:

```text
Documents -> Parsing -> Chunking -> Embeddings -> Vector DB
          -> Hybrid Retrieval -> Reranking -> LLM
          -> Citation Verification -> Evaluation
```

The default implementation runs locally with deterministic embeddings and an extractive answerer so the whole pipeline can be tested without API keys. The same interfaces are structured so hosted embedding/LLM providers can be swapped in later.

## What is included

- Document parsing for Markdown, text, and JSONL records
- Token-aware chunking with overlap
- Deterministic embedding model for reproducible local retrieval
- In-memory vector index with cosine search
- BM25 lexical retrieval
- Hybrid retrieval with metadata filters
- Lightweight reranker
- Query rewriting
- Citation-aware answer generation
- Citation verification and faithfulness checks
- Retrieval and answer evaluation suite
- Latency, token, and estimated cost tracking
- Query-result caching
- FastAPI backend
- Simple web frontend
- Docker setup

## Architecture notes

The retrieval path combines lexical and semantic evidence instead of relying on one search mode:

- **BM25** handles exact policy terms, acronyms, and rare keywords.
- **Hash embeddings** provide deterministic semantic search for local development and CI.
- **Hybrid fusion** merges BM25 and vector candidates before reranking.
- **Reranking** prioritizes chunks that match the rewritten query and have source metadata.
- **Citation verification** checks whether generated claims are supported by cited chunks.

This keeps the project runnable on a laptop while still reflecting the same interfaces used with production embedding models, vector databases, and hosted LLMs.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"

python -m rag_system.cli ingest --docs data/sample_docs --index indexes/demo
python -m rag_system.cli ask "What should employees do before sending restricted data?"
python -m rag_system.cli evaluate --eval-file eval/queries.jsonl --index indexes/demo
```

Run the API:

```bash
uvicorn rag_system.api:app --reload
```

Open `http://localhost:8000`.

Useful API routes:

```text
POST /api/ingest     build or rebuild the local index
POST /api/query      retrieve, rerank, answer, cite, and score one question
POST /api/evaluate   run the evaluation set and return aggregate metrics
GET  /api/health     check whether the service and index are available
```

## Docker

```bash
docker build -t enterprise-rag .
docker run -p 8000:8000 enterprise-rag
```

## Repository layout

```text
src/rag_system/     core RAG pipeline
frontend/           small browser UI served by FastAPI
data/sample_docs/   sample enterprise knowledge base
eval/               evaluation questions and expected evidence
tests/              unit tests for retrieval, citations, and evaluation
```

## Evaluation

The evaluation runner reports:

- Retrieval Recall@K
- MRR
- Answer correctness
- Faithfulness
- Hallucination rate
- Citation accuracy
- Latency per query
- Estimated token/cost per query

The goal is not to hide behind a polished chat interface; it is to show whether the retrieval and answer pipeline can be measured and trusted.

Example summary from the bundled sample data:

```json
{
  "recall_at_k": 1.0,
  "mrr": 1.0,
  "faithfulness": 1.0,
  "citation_accuracy": 1.0,
  "hallucination_rate": 0.0
}
```

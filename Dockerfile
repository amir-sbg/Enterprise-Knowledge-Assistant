FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_INDEX_PATH=/app/indexes/demo \
    RAG_CACHE_PATH=/app/cache/query_cache.json

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY frontend ./frontend
COPY data ./data
COPY eval ./eval

RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000

CMD ["uvicorn", "rag_system.api:app", "--host", "0.0.0.0", "--port", "8000"]


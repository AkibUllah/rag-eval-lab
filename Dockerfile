# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /srv/rag

# Dependencies first so the layer is cached across source changes.
# psycopg (pgvector store) and pypdf (PDF upload) are included in the image.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt "psycopg[binary]>=3.1" "pypdf>=4.0"

COPY pyproject.toml README.md ./
COPY rag_eval_lab ./rag_eval_lab
COPY data ./data
COPY web ./web
RUN pip install --no-deps . \
    && addgroup --system rag && adduser --system --ingroup rag rag \
    && mkdir -p /srv/rag/runs && chown -R rag:rag /srv/rag

USER rag
ENV RAG_CORPUS_DIR=/srv/rag/data/corpus \
    RAG_GOLDEN_PATH=/srv/rag/data/golden.jsonl \
    RAG_RUNS_DIR=/srv/rag/runs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=2).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn rag_eval_lab.api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]

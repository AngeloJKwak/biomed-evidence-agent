# --- Frontend build ---------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- Python app -------------------------------------------------------------
FROM python:3.12-slim AS app
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    BIOMED_CHROMA_DIR=/app/.data/chroma \
    BIOMED_RUNS_DIR=/app/.data/runs
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra tracing
COPY src/ src/
RUN uv sync --frozen --no-dev --extra tracing

# Bake the embedding model into the image so containers start without a download.
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY --from=web /web/dist web/dist

RUN useradd --create-home app && mkdir -p /app/.data && chown -R app /app/.data /app/.cache
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uv", "run", "--no-sync", "uvicorn", "biomed_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]

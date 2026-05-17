FROM python:3.11.9-slim
# Pin to a digest for full supply-chain integrity:
# FROM python:3.11.9-slim@sha256:<run: docker inspect python:3.11.9-slim --format='{{index .RepoDigests 0}}'>

WORKDIR /app

COPY requirements.txt .

# Install build deps + curl (kept for HEALTHCHECK), compile wheels, then purge build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api.py .
COPY artifacts/ ./artifacts/

RUN groupadd --system appuser && useradd --system --gid appuser appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

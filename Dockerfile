# syntax=docker/dockerfile:1

# ---- Base build stage: install dependencies into a venv --------------
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage: copy only the venv + app code, run as non-root ---
FROM python:3.11-slim AS runtime

# libgomp1 is required at runtime by LightGBM (OpenMP), not just at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=appuser:appuser . .

RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/logs \
    && chown -R appuser:appuser /app/data /app/models /app/logs

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

ENTRYPOINT ["/app/scripts/entrypoint.sh"]

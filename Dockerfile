# syntax=docker/dockerfile:1.6
# CPU variant — Docling backend (torch CPU). For local dev on Mac/Linux.

# -------- Stage 1: build wheels --------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# CPU torch wheels from PyTorch's CPU index (smaller than the CUDA wheels).
RUN pip wheel --wheel-dir=/wheels \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
        -r requirements.txt


# -------- Stage 2: runtime --------
FROM python:3.12-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    OCR_LANGUAGE=ch \
    OMP_NUM_THREADS=4 \
    HF_HOME=/tmp/.cache/huggingface
# Note: DOCLING_ARTIFACTS_PATH intentionally NOT set — see Dockerfile.gpu.

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
        redis-server \
        supervisor \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels torch torchvision -r requirements.txt \
    && rm -rf /wheels \
    && find /usr/local/lib -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib -type d -name 'tests' -path '*/site-packages/*' -prune -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib -type f -name '*.pyc' -delete 2>/dev/null || true \
    && rm -rf /root/.cache /tmp/pip-*

COPY ocr_settings.json supervisord.conf ./
COPY app ./app

ARG PRECACHE_MODELS=0
RUN mkdir -p /tmp/.docling /tmp/.cache/huggingface && \
    chmod -R 0777 /tmp/.docling /tmp/.cache /tmp && \
    if [ "$PRECACHE_MODELS" = "1" ]; then \
      python -c "from docling.utils.model_downloader import download_models; download_models(); print('docling models cached')" \
        || echo "WARN: precache failed; first boot will download models"; \
    else echo "skipping model pre-cache (PRECACHE_MODELS=0)"; fi && \
    chmod -R 0777 /tmp/.docling /tmp/.cache /tmp

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["supervisord", "-n", "-c", "/srv/supervisord.conf"]

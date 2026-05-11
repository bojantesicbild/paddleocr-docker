# syntax=docker/dockerfile:1.6
# -------- Stage 1: build wheels --------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir=/wheels \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
        paddlepaddle==3.3.0 \
        -r requirements.txt


# -------- Stage 2: runtime --------
FROM python:3.12-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_CACHE_HOME=/tmp/.paddlex \
    HOME=/tmp \
    OCR_LANGUAGE=ch \
    OMP_NUM_THREADS=4

# Runtime system libraries required by opencv + paddlepaddle, plus redis+supervisor
# for the single-container queue (supervisord runs redis + api + rq worker).
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY --from=builder /wheels /wheels
COPY requirements.txt .
# Install, then strip caches / bytecode / package test dirs (~500 MB savings
# on CPU image, no behavioral impact).
RUN pip install --no-index --find-links=/wheels paddlepaddle==3.3.0 -r requirements.txt \
    && rm -rf /wheels \
    && pip cache purge \
    && find /usr/local/lib -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib -type d -name 'tests' -path '*/site-packages/*' -prune -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib -type f -name '*.pyc' -delete 2>/dev/null || true \
    && rm -rf /root/.cache /tmp/pip-*

COPY ocr_settings.json supervisord.conf ./
COPY app ./app

# Pre-download PaddleOCR-VL-1.5 weights so first request doesn't pay the
# download cost. Off by default on CPU builds because loading the VLM at
# build time needs ~10 GB RAM. Enable with --build-arg PRECACHE_MODELS=1
# on hosts with more RAM (the GPU image always sets it).
ARG PRECACHE_MODELS=0
RUN mkdir -p /tmp/.paddlex && chmod -R 0777 /tmp/.paddlex && \
    if [ "$PRECACHE_MODELS" = "1" ]; then \
      python -c "import os; os.environ['OCR_LANGUAGE']='${OCR_LANGUAGE}'; \
from app.ocr_service import get_pipeline; get_pipeline(); print('models cached')"; \
    else echo "skipping model pre-cache (PRECACHE_MODELS=0)"; fi && \
    chmod -R 0777 /tmp/.paddlex

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["supervisord", "-n", "-c", "/srv/supervisord.conf"]

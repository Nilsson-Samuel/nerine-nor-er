# syntax=docker/dockerfile:1.7
# Rough single-stage image for the NERINE pipeline + Streamlit HITL app.
# CPU-only. Pre-downloads HuggingFace models at build time for offline use.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_OFFLINE=0 \
    NERINE_DATA_DIR=/app/data/processed

# libgomp1: required by lightgbm, faiss, and torch (OpenMP).
# git: occasionally pulled in by HF tooling for snapshot fetches.
# ca-certificates + curl: HTTPS to huggingface.co at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches across source edits.
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements-docker.txt

# Pre-download the NER and SBERT models into HF_HOME. This bakes ~1.5 GB into
# the image so the running container does not need network access.
COPY docker/preload_models.py /app/docker/preload_models.py
RUN mkdir -p "${HF_HOME}" \
    && python /app/docker/preload_models.py

# Project source. Tests, docs, and large data dirs are excluded via .dockerignore.
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY data/ /app/data/

# Bake the tuned LightGBM reranker into the runtime location the pipeline
# reads from. This makes the image fully self-contained for offline deployment
# (Kripos handoff): no separate model transfer step is needed.
# Bind-mounting ./data over /app/data at runtime will hide these — that's
# fine, because in that case the host already provides the model.
COPY data/lightgbm_kripos_conservative_trial128_20260511/reranker_model.txt          /app/data/processed/reranker_model.txt
COPY data/lightgbm_kripos_conservative_trial128_20260511/reranker_model_metadata.json /app/data/processed/reranker_model_metadata.json

# Streamlit HITL port. The pipeline service does not expose anything.
EXPOSE 8501

# Default command shows the pipeline help. docker-compose overrides per service.
CMD ["python", "-m", "src.pipeline", "--help"]

# Build context: frozen uv.lock plus dependencies/ copied from a uv-synced
# CPython 3.10 environment. Project source is a separate Concourse input.
FROM python:3.10.20-slim-bookworm@sha256:019e31cc1f52e4139d338bca44f27b71339a9da261738fd80fab9cf4595152fc
ARG LOCK_SHA256
LABEL org.opencontainers.image.title="ncm-fewnerd-probe" \
      org.opencontainers.image.version="cu130-${LOCK_SHA256}"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY dependencies/ /usr/local/lib/python3.10/site-packages/
COPY uv.lock /opt/ncm-image/uv.lock
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# GPU exposure is requested by the task, never by the build or preflight.
RUN CUDA_VISIBLE_DEVICES="" python -c "import torch, transformers, pyarrow, hydra, boto3; print(torch.__version__)"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Offline, CPU-only quality evaluation. GPU training remains Concourse-only.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export PYTHONPATH="$PWD/third_party/ncm:$PWD/src"
export LC_ALL=C.UTF-8 TZ=UTC
exec uv run --offline --frozen python scripts/autoresearch_fewnerd_quality.py

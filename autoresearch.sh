#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# CPU-only: no host GPU use, network access, or dynamically resolved dependencies.
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/third_party/ncm:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --offline python scripts/autoresearch_cache_benchmark.py

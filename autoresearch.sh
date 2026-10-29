#!/usr/bin/env bash
set -euo pipefail

# Deterministic benchmark entrypoint. NCM_TASK selects the task under measurement;
# `real_extraction` scores Re-DocRED dev documents, which are never trained on.
export NCM_TASK="${NCM_TASK:-real_extraction}"

exec uv run python -m ncm_graphrag.benchmark

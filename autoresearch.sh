#!/usr/bin/env bash
set -euo pipefail

# Deterministic benchmark entrypoint. NCM_TASK selects the task under measurement;
# `fewnerd_extraction` scores Few-NERD validation sentences, which are never trained on.
export NCM_TASK="${NCM_TASK:-fewnerd_extraction}"

exec uv run python -m ncm_graphrag.benchmark

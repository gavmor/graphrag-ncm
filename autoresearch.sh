#!/usr/bin/env bash
set -euo pipefail

# Deterministic benchmark entrypoint. NCM_TASK selects the task under measurement;
# `report_multinews` scores a seeded sample of Multi-News *validation* (never trained on): ROUGE-1 vs the reference sentence plus source grounding.
# The corpora are fetched into a sibling worktree, so point at them explicitly.
export NCM_TASK="${NCM_TASK:-report_multinews}"
export NCM_CORPORA_ROOT="${NCM_CORPORA_ROOT:-/home/user/Desktop/gavmor/ncm-graphrag-worktrees/training-corpora/corpora}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec uv run python -m ncm_graphrag.benchmark

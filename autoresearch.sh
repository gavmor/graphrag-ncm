#!/usr/bin/env bash
set -euo pipefail

# Deterministic benchmark entrypoint. NCM_TASK selects the task under measurement;
# `map_stage_hotpot` scores HotpotQA *validation* questions (never trained on) by score AUC.
# The corpora are fetched into a sibling worktree, so point at them explicitly.
export NCM_TASK="${NCM_TASK:-map_stage_hotpot}"
export NCM_CORPORA_ROOT="${NCM_CORPORA_ROOT:-/home/user/Desktop/gavmor/ncm-graphrag-worktrees/training-corpora/corpora}"

exec uv run python -m ncm_graphrag.benchmark

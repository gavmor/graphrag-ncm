#!/usr/bin/env bash
set -euo pipefail

# Deterministic benchmark entrypoint. NCM_TASK selects the task under measurement;
# `relation_tacred` scores a seeded random sample of TACRED *validation* (never trained on) by micro-F1 over non-NA relations.
# The corpora are fetched into a sibling worktree, so point at them explicitly.
export NCM_TASK="${NCM_TASK:-relation_tacred}"
export NCM_CORPORA_ROOT="${NCM_CORPORA_ROOT:-/home/user/Desktop/gavmor/ncm-graphrag-worktrees/training-corpora/corpora}"

exec uv run python -m ncm_graphrag.benchmark

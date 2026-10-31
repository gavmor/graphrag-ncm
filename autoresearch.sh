#!/usr/bin/env bash
set -euo pipefail

# GPU HYGIENE GUARD:
# Bare host GPU execution is prohibited on this machine to avoid contention with
# Concourse workloads and local services. ALL GPU workloads must run inside Concourse
# or under gpu-lock.
if [ -z "${ALLOW_BARE_GPU:-}" ]; then
  echo "ERROR: autoresearch.sh ran bare on host without gpu-lock or Concourse!" >&2
  echo "Run inside Concourse or set ALLOW_BARE_GPU=1 if gpu-lock is held explicitly." >&2
  exit 1
fi

# Deterministic benchmark entrypoint. NCM_TASK selects the task under measurement;
# `report_multinews` scores a seeded sample of Multi-News *validation* (never trained on): ROUGE-1 vs the reference sentence plus source grounding.
# The corpora are fetched into a sibling worktree, so point at them explicitly.
export NCM_TASK="${NCM_TASK:-report_multinews}"
export NCM_CORPORA_ROOT="${NCM_CORPORA_ROOT:-/home/user/Desktop/gavmor/ncm-graphrag-worktrees/training-corpora/corpora}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec uv run python -m ncm_graphrag.benchmark

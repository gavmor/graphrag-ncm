#!/usr/bin/env bash
set -euo pipefail

# Deterministic benchmark entrypoint for NCM GraphRAG extraction workload
exec uv run python -m ncm_graphrag.benchmark

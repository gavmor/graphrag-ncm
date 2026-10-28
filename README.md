# ncm-graphrag

Exploring [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (Next-Concept
Modeling: parallel, coarse-to-fine, non-autoregressive text generation) as an
accelerated backend for one bounded-output stage of
[microsoft/graphrag](https://github.com/microsoft/graphrag): the Global Search
map-stage relevance scoring call.

**Start here:** [`docs/constraints.md`](docs/constraints.md) — the two hard
constraints that shape everything else (no pretrained NCM checkpoint fits any
GraphRAG task, and no training code is released for NCM at all — this project trains
its own tokenizer+generator pair from scratch against the released model code).
Then [`docs/architecture.md`](docs/architecture.md) for the integration seam,
pipeline diagrams, and actual (not projected) run results.

## What's here

- `src/ncm_graphrag/data/map_stage.py` — synthetic GraphRAG map-stage dataset
  (community report + query → `{"points":[{"description","score"}]}` JSON).
- `src/ncm_graphrag/ncm/` — from-scratch NCM tokenizer + generator trainer
  (`train_tokenizer.py`, `train_generator.py`) and inference (`generate.py`,
  best-of-N JSON-validity reranking).
- `src/ncm_graphrag/serve/` — litellm `CustomLLM` adapter serving the trained
  checkpoint as an OpenAI-compatible endpoint.
- `src/ncm_graphrag/graphrag_integration/` — `HybridCompletion` (NCM for the map
  stage, a standard AR LLM for reduce/final-answer synthesis) and an end-to-end demo
  against graphrag's real `GlobalSearch` class.
- `third_party/{ncm,graphrag}` — upstream repos as git submodules (code only; see
  `docs/constraints.md` #5 for why checkpoints aren't duplicated here).

## Two environments, on purpose

- **This project's venv** (`uv sync`, Python 3.10, pinned to match the reference NCM
  setup at `/home/user/Desktop/ncm/repo`): torch/omegaconf/transformers for training
  and serving NCM, plus litellm for the OpenAI-compatible proxy.
- **`third_party/graphrag`'s own venv** (`uv sync` inside that directory, Python
  3.11+, its own lockfile): graphrag requires Python ≥3.11 and a heavy
  spacy/pandas/pyarrow/graspologic dependency closure incompatible with the NCM/torch
  stack's pin. The two talk over plain HTTP (the litellm proxy's OpenAI-compatible
  endpoint) — exactly how a real deployment would separate an inference service from
  a GraphRAG client, so this isn't a workaround, it's the realistic shape.

## Reproduce

```bash
# 1. Train the tiny NCM tokenizer + generator (this project's venv, ~3 min on a GPU)
uv sync
uv run python -m ncm_graphrag.ncm.train_tokenizer
uv run python -m ncm_graphrag.ncm.train_generator

# 2. Serve it as an OpenAI-compatible endpoint
uv run litellm --config src/ncm_graphrag/serve/litellm_config.yaml --port 4010

# 3. Run the real-graphrag end-to-end demo (graphrag's own venv, separate terminal)
cd third_party/graphrag && uv sync
NCM_LITELLM_BASE=http://127.0.0.1:4010/v1 \
  PYTHONPATH=../../src \
  ANTHROPIC_API_KEY=... ANTHROPIC_WORKSPACE_ID=... \
  uv run --package graphrag python ../../src/ncm_graphrag/graphrag_integration/run_map_stage_demo.py
```

## Project tooling

`uv` (project/venv/deps), `ruff` (lint + format), `ty` (type check), `lefthook`
(pre-commit: parallel ruff check/format + ty check on staged `*.py`).

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check .
```

## Status / next steps

One vertical slice end-to-end: trained-from-scratch NCM tokenizer+generator → served
over HTTP → wired into graphrag's real `GlobalSearch` as the map-stage backend via
pure config (`HybridCompletion` routes map calls to NCM, reduce calls to Claude).
See `docs/architecture.md` for what the real run surfaced (NCM's single-report-per-call
training scope vs. graphrag's real multi-report batching) and the three other
brief ideas (parallel entity/claim extraction, hierarchical community-report drafting,
drift-search routing) that would each need their own synthetic dataset + trained pair
following the same recipe.

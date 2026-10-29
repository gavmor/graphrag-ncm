# ncm-graphrag

Exploring [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (Next-Concept
Modeling: parallel, coarse-to-fine, non-autoregressive text generation) as an
accelerated backend for bounded-output stages of
[microsoft/graphrag](https://github.com/microsoft/graphrag): Global Search map-stage
relevance scoring, and entity extraction.

**Start here:** [`docs/constraints.md`](docs/constraints.md) — the hard constraints
that shape everything else (no pretrained NCM checkpoint fits any GraphRAG task, no
training code is released for NCM at all, so this project trains its own
tokenizer+generator pairs from scratch against the released model code) and the real
measured findings from building a second slice (a decode-recipe bug found and fixed,
structural-validity-vs-content-correctness gap). Then
[`docs/architecture.md`](docs/architecture.md) for the integration seams, pipeline
diagrams, and actual (not projected) run results for both slices.
[`docs/NCM Training for GraphRAG.md`](docs/NCM%20Training%20for%20GraphRAG.md) is
unvetted AI-generated background reading (see its own header), not a design this
project strictly follows — where they disagree, `constraints.md`/`architecture.md`
describe what was actually built and measured.

## What's here

- `src/ncm_graphrag/data/{map_stage,extraction}.py` — synthetic datasets for the two
  slices: community report + query → `{"points":[{"description","score"}]}` JSON; text
  unit → graphrag's real entity delimiter-tuple grammar (`("entity"<|>...)##...##<|COMPLETE|>`).
- `src/ncm_graphrag/ncm/` — from-scratch NCM tokenizer + generator trainer
  (`train_tokenizer.py --task {mapstage,extraction}`, `train_generator.py --task ...`),
  a task registry (`task_registry.py`) so both slices share one trainer/server
  implementation, and inference (`generate.py`'s `NCMTaskGenerator`, with Minimum
  Bayes Risk best-of-N reranking).
- `src/ncm_graphrag/serve/` — litellm `CustomLLM` adapters serving both trained
  checkpoints as OpenAI-compatible endpoints from one proxy process.
- `src/ncm_graphrag/graphrag_integration/` — `HybridCompletion` (NCM for the map
  stage, a standard AR LLM for reduce/final-answer synthesis) plus two end-to-end demos
  against graphrag's real, unmodified `GlobalSearch` and `GraphExtractor` classes.
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
# 1. Train both tiny NCM tokenizer+generator pairs (this project's venv, ~10 min total on a GPU)
uv sync
uv run python -m ncm_graphrag.ncm.train_tokenizer --task mapstage
uv run python -m ncm_graphrag.ncm.train_generator --task mapstage
uv run python -m ncm_graphrag.ncm.train_tokenizer --task extraction
uv run python -m ncm_graphrag.ncm.train_generator --task extraction

# 2. Serve both as OpenAI-compatible endpoints (one proxy, two models)
uv run litellm --config src/ncm_graphrag/serve/litellm_config.yaml --port 4010

# 3. Run the real-graphrag end-to-end demos (graphrag's own venv, separate terminal)
cd third_party/graphrag && uv sync
NCM_LITELLM_BASE=http://127.0.0.1:4010/v1 PYTHONPATH=../../src \
  ANTHROPIC_API_KEY=... ANTHROPIC_WORKSPACE_ID=... \
  uv run --package graphrag python ../../src/ncm_graphrag/graphrag_integration/run_map_stage_demo.py

NCM_LITELLM_BASE=http://127.0.0.1:4010/v1 PYTHONPATH=../../src \
  ANTHROPIC_API_KEY=... ANTHROPIC_WORKSPACE_ID=... \
  uv run --package graphrag python ../../src/ncm_graphrag/graphrag_integration/run_extraction_demo.py
```

## Project tooling

`uv` (project/venv/deps), `ruff` (lint + format), `ty` (type check), `lefthook`
(pre-commit: parallel ruff check/format + ty check on staged `*.py`).

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check .
```

## Status / next steps

Two vertical slices end-to-end: trained-from-scratch NCM tokenizer+generator pairs →
served over HTTP → wired into graphrag's real, unmodified `GlobalSearch` and
`GraphExtractor` classes via pure config (no graphrag code changes).

- **Map stage**: `HybridCompletion` routes map calls to NCM, reduce calls to Claude.
  Real run surfaced NCM's single-report-per-call training scope vs. graphrag's real
  multi-report batching (see `docs/architecture.md`).
- **Entity extraction**: entity-only (not entity+relationship — a first attempt at the
  fuller grammar degraded badly; see `docs/constraints.md` #6 for the measured before/
  after). Building it surfaced and fixed a real bug: the decode-penalty recipe copied
  from the released xsum/wmt14 checkpoints actively suppressed this grammar's
  legitimate repeated structure (0/30 well-formed → 30/30 once disabled for this task).
  Structural validity is solid; content correctness is still rough on the second
  extracted entity, consistent with this generator being ~143x smaller than the scale
  the background doc itself prescribes.

**Real-data extraction was then measured, and it does not transfer.** The same
architecture on two real annotated corpora (Re-DocRED; Few-NERD) scores **1-5% entity
F1**, not the synthetic slice's structural-validity numbers. The failure is localised: the
model has no span-copy mechanism, so recall is flat across name lengths (1-token 20%,
5+ tokens 1.3%) and 93% of remaining errors are wholly invented names. One real defect was
found and fixed along the way — targets upper-cased entity names, forcing a BPE-level case
transform while copying — worth **1.56 → 4.56 F1**. Ceiling, capture, and every rejected
in-budget lever are recorded in `docs/constraints.md` §7 so they are not re-run.

**Real map-stage scoring was also measured** (HotpotQA human-annotated relevance, held-out
score AUC). The raw-text NCM scores 61.3, barely above BM25 (59.1); handing it five cheap lexical
features raises that to 72.6, above a linear classifier on the same features (66.7) but below a
lookup table over them (75.2). Against Claude haiku-5-5 on the same 300 held-out examples, Claude
scores 90.9 AUC at ~2 s/call versus the NCM's 71.9 at 12 ms/call (~160x faster, ~19 AUC lower; not
like-for-like: network+reasoning vs in-process). NCM emits valid bounded map-stage JSON fast but
has not been shown to *judge* relevance from text. Label leaks found and fixed
along the way, and the control baselines to report, are in `docs/constraints.md` §8.

**Closed-vocabulary relation classification (TACRED, Lead 2) was measured too**, because a fixed
label set removes the string-copying bottleneck. A from-scratch 0.83M NCM reaches **25.8 micro-F1**
(3-seed mean, sd 0.5; every output a valid label; ~7 ms/call): on par with naive Bayes (23.0) and
well below a bag-of-words softmax regression (44.7) and Claude haiku-5-5 (61.8, ~900 ms/call).
What did and did not move it, and the traps hit along the way (sorted val file, leaky train
holdout, single-seed deltas that reversed on replication), are in `docs/constraints.md` §9.

**At the paper's own scale the picture is more favourable, with the same caveat.** The *released*
88M XSum checkpoint (no training here) gets ROUGE-1/2/L of 34.1/11.8/26.3 on 150 XSum test articles,
against 17.4/1.4/12.6 for the lead sentence and 28.3/7.4/18.8 for zero-shot Claude haiku-5-5, at
~102 ms per summary in-process. That is not evidence it beats Claude (it was trained on XSum; ROUGE
rewards house style), and it still garbles proper nouns (`Guingamp` -> `Guernan Guineamp`), the same
weakness that limits extraction here. See `docs/constraints.md` §10.

**That ROUGE result overstated the summaries.** Judged blind at claim level, the same checkpoint's
output on its *own* XSum articles has only 27% of claims supported (fluency 2.1/5), and on GraphRAG
benchmark text (podcasts, earnings calls, Wikipedia) 3-8%: no better than a summary of the wrong
chunk (0-2%), against 88-94% for Claude. So the released XSum NCM does not transfer zero-shot, and the
"NCM suits bounded summarisation" hypothesis is untested rather than supported. `docs/constraints.md` §11.

Natural next slices following the same recipe: relationship extraction (decoupled
from entity extraction, per the background doc's own suggestion), community-report
generation, and — for extraction to become production-useful — either the ~120M-parameter
regime the background doc prescribes or constrained decoding to source spans
(`docs/constraints.md` §7 #18).

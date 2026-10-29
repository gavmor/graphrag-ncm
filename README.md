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
- `corpora/` — training & benchmark corpora fetched per `GraphRAG Training Dataset Discovery.md`
  (DocRED, Few-NERD, SciRIFF, SciERC, TACRED, REBEL, SREDFM, HotpotQA, MuSiQue, KILT,
  GraphRAG-Bench, WildGraphBench, SQuALITY, QMSum, GovReport, Multi-News, Microsoft GraphRAG benchmarks),
  with programmatic loader utilities in `src/ncm_graphrag/data/corpora.py` and fetcher `scripts/fetch_corpora.py`.

## Training Corpora & Benchmarks

Real-world GraphRAG training requires datasets spanning document-level relation extraction, multi-hop reasoning, and query-focused summarization. All datasets identified in `GraphRAG Training Dataset Discovery.md` are fetched locally into `corpora/` with metadata recorded in `corpora/manifest.json`.

### Acquired Datasets

| Category | Dataset | Primary Source | Local Size | Role in GraphRAG / NAR Modeling |
| :--- | :--- | :--- | :--- | :--- |
| **Information Extraction** | DocRED | `thunlp/docred` | 91.2 MB | Multi-sentence document-level relation extraction (96 Wikidata relations) |
| | Re-DocRED | `tonytan48/Re-DocRED` | 23.9 MB | Re-annotated DocRED resolving false negatives; dense relational supervision |
| | Few-NERD | `DFKI-SLT/few-nerd` | 60.7 MB | Fine-grained entity typing (8 coarse, 66 fine types over 188.2k sentences) |
| | SciRIFF | `allenai/SciRIFF` | 1.33 GB | Scientific literature instruction-tuning and claim extraction |
| | SciERC | UW / AllenAI (`zj88zj/SCIERC`) | 1.40 GB | Scientific abstracts with 6 entity types and 7 directional relations |
| | TACRED | `DFKI-SLT/tacred` & `xiaobendanyn/tacred` | 44.1 MB | 106k sentence relation extraction instances across 41 relation classes |
| | Re-TACRED | `gstoica27/Re-TACRED` | 4.7 MB | Relabeled TACRED resolving labeling inconsistencies and error drift |
| | REBEL | `Babelscape/rebel-dataset` | 11.1 GB | End-to-end relation extraction seq2seq dataset (triples from Wikipedia) |
| | SREDFM | `Babelscape/sredfm` | 119.0 MB | Silver relation extraction evaluation splits for filtered Wikidata relations |
| **Multi-Hop Reasoning** | HotpotQA | `hotpotqa/hotpot_qa` | 712.1 MB | Multi-hop reasoning with sentence-level supporting fact supervision |
| | MuSiQue | `dgslibisey/MuSiQue` | 258.9 MB | Multi-hop QA with decomposed hops and up to 20 distractor paragraphs |
| | KILT Tasks | `facebook/kilt_tasks` | 1.00 GB | 11 standardized knowledge-intensive benchmarks with Wikipedia provenance |
| | KILT Wikipedia | `facebook/kilt_wikipedia` | 17.4 KB | Schema definition and knowledge source snapshot loader script |
| | GraphRAG-Bench | `GraphRAG-Bench/GraphRAG-Bench` | 14.0 MB | Medical, algorithmic, and novel multi-constraint graph reasoning benchmark |
| | WildGraphBench | `Bstwpy/WildGraphBench` | 211.1 MB | Complex semi-structured Wikipedia graph topology multi-fact benchmark |
| **Query-Focused Summarization** | SQuALITY | `GEM/squality` | 104.4 KB | Multi-perspective story summarization (4 human reference summaries per query) |
| | ZeroSCROLLS | `tau/zero_scrolls` | 211.7 MB | Long-text evaluation suite (contains QMSum, SQuALITY, GovReport, MuSiQue) |
| | QMSum (Cleaned) | `pszemraj/qmsum-cleaned` | 46.8 MB | Query-based multi-domain meeting summarization (academic, corporate, policy) |
| | GovReport | `ccdv/govreport-summarization` | 483.2 MB | Long institutional legislative and policy reports paired with executive summaries |
| | Multi-News | `alexfabbri/multi_news` | 721.8 MB | Multi-document news synthesis into community-style narrative conclusions |
| | Microsoft GraphRAG | `microsoft/graphrag-benchmarking-datasets` | 66.3 MB | Official benchmarks: Kevin Scott Podcasts (125 Qs), MSFT Earnings, Filtered HotPotQA (5,491 Qs) |
| **Additional / Graph Traversal** | text2cypher | `neo4j/text2cypher` | 7.8 MB | Natural language questions paired with Cypher property graph query statements |

### Programmatic Access in Python

Load any corpus or query the manifest using `ncm_graphrag.data.corpora`:

```python
from ncm_graphrag.data.corpora import (
    get_corpus_path,
    list_corpora,
    load_docred,
    load_graphrag_bench,
    load_microsoft_graphrag,
    load_musique,
    load_re_docred,
)

# Query manifest of all 22 fetched corpora
corpora = list_corpora(category="information_extraction")
print(f"Found {len(corpora)} IE corpora")

# Direct path resolution
hotpot_dir = get_corpus_path("hotpotqa")

# Specialized dataset loaders
docred_dev = load_docred("dev")
re_docred_dev = load_re_docred("dev")
musique_dev = load_musique("dev")
graphrag_bench_med = load_graphrag_bench("medical")
msft_podcasts = load_microsoft_graphrag("podcasts")
msft_hotpot = load_microsoft_graphrag("hotpotqa")
```

### Fetching & Updating Corpora

To re-fetch or selectively update datasets:

```bash
# Fetch everything
uv run python scripts/fetch_corpora.py --all

# Or fetch by category:
uv run python scripts/fetch_corpora.py --category ie
uv run python scripts/fetch_corpora.py --category multi_hop
uv run python scripts/fetch_corpora.py --category summarization
uv run python scripts/fetch_corpora.py --category additional

# Or fetch an individual dataset:
uv run python scripts/fetch_corpora.py --dataset microsoft_graphrag
```

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

Natural next slices following the same recipe: relationship extraction (decoupled
from entity extraction, per the background doc's own suggestion), community-report
generation, and training on real graphrag-produced data instead of closed-vocabulary
synthetic templates (the biggest remaining gap for either slice to be production-useful).

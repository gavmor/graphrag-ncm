# ncm-graphrag

> [!CAUTION]
> **GPU HYGIENE / CONCURRENCY NOTICE**
> Never run benchmarks or GPU-intensive scripts (such as `autoresearch.sh`) bare on the host. The local RTX 3090 is shared with Concourse CI pipelines and local services. Bare execution causes contention and process eviction (e.g. OOM / exit 137). All GPU workloads must be scheduled through Concourse or run under the `gpu-lock` mutex.

Exploring [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (Next-Concept
Modeling: parallel, coarse-to-fine, non-autoregressive text generation) as an
accelerated backend for bounded-output stages of
[microsoft/graphrag](https://github.com/microsoft/graphrag): Global Search map-stage
relevance scoring, and entity extraction.

**Final report (start here):** [`docs/findings.md`](docs/findings.md) -- what was measured, what it
does and does not show, and the limits to read before quoting any number. Across 0.83M to 92.8M
parameters, NCM produced fast, valid output for bounded and structured schemas, but did not match
standard baselines on open-vocabulary content extraction. Trained at paper scale (92.8M parameters)
on 100,000 REBEL examples against an equal-size autoregressive control (92.1M non-embedding parameters),
the AR control achieved 100% validity, **9.50% Entity F1**, and transferred to gold DocRED (1.59% F1).
NCM reached 79% validity, **0.53% Entity F1**, and hallucinated entity names, confirming that NCM's
discrete residual VQ-bottleneck without a span copy mechanism is an architectural bottleneck for
open-vocabulary extraction.

**Then:** [`docs/constraints.md`](docs/constraints.md) — the hard constraints
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
- `scripts/fetch_corpora.py` and `src/ncm_graphrag/data/corpora.py` — download and access
  corpus data outside Git. Set `NCM_CORPORA_ROOT` to use a shared directory; otherwise
  `corpora/` in this checkout is used. Corpora and checkpoints are not committed.

## Training Corpora & Benchmarks

The fetcher can download the datasets below into `NCM_CORPORA_ROOT` (or `corpora/` by
default), with metadata in that directory's `manifest.json`. The listed sizes reflect one
local snapshot, not files included in the repository. Check each dataset's license before use.

### Available datasets

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

# Query fetched corpora from the configured manifest
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

**The released 88M XSum checkpoint tests paper-scale parameters, but not GraphRAG training.** With
no training here, it gets ROUGE-1/2/L of 34.1/11.8/26.3 on 150 XSum test articles,
against 17.4/1.4/12.6 for the lead sentence and 28.3/7.4/18.8 for zero-shot Claude haiku-5-5, at
~102 ms per summary in-process. That is not evidence it beats Claude (it was trained on XSum; ROUGE
rewards house style), and it still garbles proper nouns (`Guingamp` -> `Guernan Guineamp`), the same
weakness that limits extraction here. See `docs/constraints.md` §10.

**That ROUGE result overstated the summaries.** Judged blind at claim level, the same checkpoint's
output on its *own* XSum articles has only 27% of claims supported (fluency 2.1/5), and on GraphRAG
benchmark text (podcasts, earnings calls, Wikipedia) 3-8%: no better than a summary of the wrong
chunk (0-2%), against 88-94% for Claude. So the released XSum NCM does not transfer zero-shot, and the
"NCM suits bounded summarisation" hypothesis is untested rather than supported. `docs/constraints.md` §11.

**The trained community-report-style drafter also failed at this scale.** The baseline NCM scored
ROUGE-1 4.14 versus 13.9 for lead-1; blind claim-level judging found 10% of claims supported, 5%
fully faithful, and fluency 1.0/5. NCM variants remained at 9-16% supported, within the judge's
observed 7-11% repeatability range on identical output. Widening the generator from 384 to 512
(19.3M parameters) scored 9% / 3% / 1.0. A matched-size, early-stopped autoregressive control scored
11% / 2% / 1.1 and also looped. At roughly 11M non-embedding parameters and 38k training pairs,
neither architecture learned usable drafting; this does not establish an architectural winner.

The remaining useful drafting experiment is a larger NCM and autoregressive control trained on at
least 100k real GraphRAG-shaped pairs, with an independent validation set. The currently staged
corpora do not provide that data budget, so scaling the model alone would mostly increase the risk
of memorisation. Full results, judge anchors, rejected variants, and selection caveats are in
[`docs/findings.md`](docs/findings.md) and [`docs/constraints.md` §12](docs/constraints.md#12-trained-community-report-drafter-multi-news-the-last-experiment).

**Paper-scale extraction on 100,000 REBEL examples isolates the architecture gap.**
A 92.8M NCM generator and an equal-size AR control (92.1M non-embedding parameters) were trained on
100k REBEL text-to-GraphRAG tuple examples. On held-out REBEL validation, AR reached 100% validity,
**9.50% Entity F1**, and 0.97% Relation F1, successfully extracting in-context entities. NCM scored
79% validity, **0.53% Entity F1**, and 0.00% Relation F1, frequently repeating tokens and garbling
delimiters. On gold transfer to Re-DocRED and DocRED dev sets, AR transferred 1.33% and 1.59% Entity F1,
whereas NCM scored 0.14% and 0.00%. Full details and checkpoints are documented in
[`docs/findings.md`](docs/findings.md) and [`docs/constraints.md` §13](docs/constraints.md#13-paper-scale-entity-and-relationship-extraction-rebel-928m-ncm-vs-matched-ar-control).

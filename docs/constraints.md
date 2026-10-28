> `docs/NCM Training for GraphRAG.md` in this same directory is unvetted AI-generated
> background reading (see its own header disclaimer) — a second, more speculative take
> on the same problem (targets entity/relation extraction instead of the map stage,
> assumes a 2M-sample/8xH100-cluster scale this project never attempted). It is kept
> for context, not as a design this project follows; where the two disagree, this file
> and `docs/architecture.md` describe what was actually built and verified.

# Constraints and scope decisions

## 1. No pretrained checkpoint fits GraphRAG's tasks

HazyResearch's released NCM checkpoints (`hazyresearch/ncm-checkpoints` on HF) are
trained only for XSum, WMT14, LM1B, OpenWebText, TinyStories, and Sudoku, each with a
fixed output length and a task-specific tokenizer (BERT/GPT-2/Qwen vocab, depending on
task). There is no "entity extraction" or "community report" or "relevance scoring"
checkpoint. Pointing graphrag's model config at an NCM checkpoint is not a drop-in
swap for any of the brief's four integration ideas — using NCM for GraphRAG requires
training a new NCM tokenizer+generator pair on GraphRAG-shaped data.

## 2. No training code is released for NCM at all (discovered during this project)

This is a second, more severe constraint the brief didn't anticipate. HazyResearch's
`ncm` repo ships only `evaluate.py` + `scripts/eval_shard.py`/`eval_merge.py` — inference
and scoring, not training. There is no `train.py`, no optimizer loop, no training
recipe or hyperparameters published anywhere in the repo. This is confirmed openly:
[github.com/HazyResearch/ncm/issues/2](https://github.com/HazyResearch/ncm/issues/2)
("Request for training codes", opened 2026-10-15, still open, no maintainer response
as of this project).

**What *is* available**: the model architecture itself (`models/vqvae.py`,
`models/quant.py`, `models/next_concept.py`, `models/vae.py`, `models/common.py`) is
fully implemented and trainable as a library — `MultiscaleResidualVectorQuantizer`
already computes its own commitment/reconstruction loss and EMA-updates its codebook
internally whenever `self.training=True` (`models/quant.py:448-549`); `VQVAE.forward()`
returns `(x_recon, vq_loss, ...)` ready for a standard VQ-VAE training loop. What's
missing is the *training script* (data pipeline, optimizer loop, teacher-forcing input
construction for `NextConceptTransformer.forward()`) — which this project had to
reverse-engineer from the model code and the one place its exact input contract is
exercised correctly: `NextConceptTransformer.generate()`.

This project is therefore not "fine-tune a released checkpoint" but "reimplement an
unreleased training script against released model code, then train a small model on
synthetic data" — a materially larger scope than the brief assumed. See
`src/ncm_graphrag/ncm/train_tokenizer.py` and `train_generator.py` for the
reverse-engineered training contract (documented inline with exact `models/*.py` line
references).

## 3. Scope: two vertical slices (map-stage, then extraction), not all four ideas

Per the brief's own hybrid table and the scout research into graphrag's actual LLM
seam (`docs/architecture.md`), **Global Search map-stage scoring** was the best-fitting
first slice:
- Single-shot structured JSON output (`{"points":[{"description":...,"score":...}]}`),
  no multi-turn "gleaning" loop.
- `graphrag`'s model-provider abstraction (`graphrag-llm` package, LiteLLM-backed) makes
  swapping backends a pure config change for any OpenAI-compatible HTTP endpoint — no
  graphrag code changes needed to point it at a locally served NCM checkpoint.

A second slice, **entity/relationship extraction**, was added afterward (see
`docs/NCM Training for GraphRAG.md`'s background inspiration and §6 below) once
`src/ncm_graphrag/ncm/` was generalized to a task registry
(`src/ncm_graphrag/ncm/task_registry.py`) so a second task didn't mean duplicating the
trainer/serving code. Unlike the map stage, extraction's real call site
(`graph_extractor.py:_process_document`) *does* use a multi-round "gleaning" loop —
this project's NCM extraction model only covers the single first-round call
(`max_gleanings=0`), not the gleaning continuation rounds (see §6).

Community-report generation remains a natural follow-on slice (same training+serving
recipe, different synthetic dataset and output schema) but is out of scope for this pass.

## 4. Known scope cuts in the map-stage slice

- **One community report per map call.** GraphRAG's real map stage packs as many
  community reports as fit in an 8000-token budget into one call and expects multiple
  `points` back (`community_context.py:build_community_context`). This project trains
  and serves a single-report-in, single-point-out model. Scaling to N-reports-per-call
  would need either a larger bounded output (N points) or client-side fan-out across N
  parallel single-report NCM calls — a real design decision, not implemented here.
- **Closed-vocabulary synthetic data.** The training set (`src/ncm_graphrag/data/map_stage.py`)
  is a closed set of ~12 domains and ~12 entity names filled into fixed templates, not
  open-domain text. This proves the training+serving+graphrag-wiring pipeline works
  end-to-end on a from-scratch NCM pair; it does not prove NCM's abstractive relevance
  judgment quality on real community reports. A real deployment needs a tokenizer+
  generator trained on actual graphrag-produced community reports and queries.
- **Reused inference tricks, didn't re-derive them.** Greedy argmax decoding is
  degenerate on this task too (same failure mode HazyResearch's released configs guard
  against); `src/ncm_graphrag/ncm/generate.py` reuses the frequency-penalty +
  adjacency-penalty decode loop from `scripts/eval_shard.py`'s `pick_tokens`, not a
  from-scratch decoding strategy. The eval_shard.py EOS-off-by-one bug noted in the
  task brief is specific to LM1B-style *unconditional* decoding seeded with EOS as a
  position-0 BOS token (`predict_level0` path); this project's model is conditional
  (`query_level0=True`, like the released xsum checkpoint) and does not hit that code
  path — verified by inspecting `NextConceptTransformer.generate()`'s branching, not
  assumed.

## 5. Environment reuse

A working NCM environment (Python 3.10, uv-managed venv, cloned repo, downloaded
`lm1b-large`/`xsum` checkpoints) already exists at `/home/user/Desktop/ncm/repo`. This
project does **not** duplicate that download: `third_party/ncm` is a fresh git
submodule (code only, ~tens of MB) and this project trains its own tiny
map-stage-specific checkpoints from scratch rather than reusing `lm1b`/`xsum` weights
(their tokenizers/vocab/output-length don't match the map-stage task — see §1). The
reference venv at `/home/user/Desktop/ncm/repo/.venv` was used only to inspect installed
package versions (torch build) while setting up this project's own `uv`-managed venv.

## 6. Known scope cuts in the extraction slice, and a real measured bug found while building it

- **No gleaning loop.** GraphRAG's real extraction call (`graph_extractor.py:_process_document`)
  issues a first call, then up to `max_gleanings` more rounds (`CONTINUE_PROMPT`/
  `LOOP_PROMPT`) asking the model whether it missed anything — a variable-length,
  multi-turn dialog. NCM produces one bounded-length output per call; this project's
  extraction model and `run_extraction_demo.py` both run with `max_gleanings=0`
  (single-shot only). Covering the gleaning rounds would need either training NCM on
  the continuation-turn shape too (a different, shorter bounded task: "what did you
  miss, if anything") or accepting that NCM only replaces the first-pass call and an
  AR LLM still handles any gleaning.
- **Entity-only, not entity+relationship.** A first version targeted 2 entities + 1
  relationship (`<|>`/`##`/`<|COMPLETE|>` grammar, ~140 gpt2 tokens, 6 VQ levels / 284
  total codes). Its free-running generation degraded badly past roughly the first
  entity — **0/30 well-formed, even with best-of-32 reranking.** Shrinking to entity-only
  (2 entities, ~87 tokens, 5 levels / 156 codes — still larger than map-stage's 46) did
  **not** fix it on its own (still 0/30). The actual cause, found by disabling pieces
  one at a time: `src/ncm_graphrag/ncm/generate.py`'s frequency/adjacency-penalty decode
  loop (copied from `eval_shard.py`, tuned for the released xsum/wmt14 checkpoints'
  low-repetition natural-language targets) **actively suppresses the legitimate
  cross-record repetition** in this grammar (two entity records share "ORGANIZATION",
  "operates in `<domain>` and was named in the text unit."). Turning the penalty off
  for this task alone: **28/30 well-formed greedy, 30/30 with best-of-4+** (measured
  directly, see `docs/architecture.md`). This is why `build_extraction_generation_cfg`
  sets `frequency_penalty=0`/`adjacency_penalty=""` while `build_generation_cfg`
  (map-stage) keeps them on — the "reuse the released checkpoints' decode recipe"
  default is actively wrong for a grammar with legitimate repeated structure, and this
  was found empirically, not assumed. Relationship extraction (and the fuller
  entity+relationship grammar) is a follow-on slice at a larger model/token budget —
  directly matching the Gemini background doc's own suggestion to "decouple extraction
  into two dedicated, parallel models."
- **Structural validity ≠ content correctness.** Even after the penalty fix and with
  MBR-style consensus reranking (`generate_best_of_n` now picks the well-formed
  candidate with highest word-overlap similarity to the others, not just the first
  valid one — a cheap Minimum Bayes Risk approximation, per the Gemini doc's own
  recommendation), **0/10 sampled examples exactly matched their ground truth.** The
  first entity per text unit is reliably correct; the second entity's name/description
  frequently has substitution or duplication noise (e.g. "NILE LABS" → "NILELELENICS",
  "operates in" → "operates operates in"). This is consistent with (not a surprise
  given) the scale gap: this project's extraction generator is 0.84M non-embedding
  parameters trained on 8,000 synthetic examples; the Gemini doc's own target is a
  120M-parameter model (~143x larger) trained on ~1-2 billion tokens. MBR/best-of-N
  reranking can only select among a small model's imperfect candidates — it cannot
  manufacture the capacity a 143x-smaller model doesn't have.
- **Closed-vocabulary, fixed-cardinality synthetic data.** `src/ncm_graphrag/data/extraction.py`
  generates exactly 2 entities per text unit from the same closed ~12-domain/~12-entity
  vocabulary as the map-stage dataset — not the variable 0-N entities real text
  produces. The target format is graphrag's *exact* delimiter-tuple grammar, so a real
  `GraphExtractor` parses this project's output with zero custom parsing code — but the
  fixed cardinality is a training-time simplification, not something graphrag's format
  itself requires.
- **Nested codebook dropout reused, not newly implemented.** The Gemini background doc
  (`docs/NCM Training for GraphRAG.md` §3) independently describes "nested codebook
  dropout" as a mitigation for early-level collapse; this is the same mechanism as
  `models/quant.py`'s `fine_drop` (lines 262-268, `aux.fine_drop.{prob,min_keep}` in
  both tasks' tokenizer configs) — already present upstream and already used by the
  map-stage tokenizer too, not something built newly for extraction.

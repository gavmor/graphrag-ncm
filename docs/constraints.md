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

## 3. Scope: one vertical slice, not all four ideas

Per the brief's own hybrid table and the scout research into graphrag's actual LLM
seam (`docs/architecture.md`), **Global Search map-stage scoring** is the best-fitting
first slice:
- Single-shot structured JSON output (`{"points":[{"description":...,"score":...}]}`),
  no multi-turn "gleaning" loop (unlike entity/claim extraction, which requires
  variable-length multi-round generation — a real prompt/parsing redesign, not just a
  backend swap).
- `graphrag`'s model-provider abstraction (`graphrag-llm` package, LiteLLM-backed) makes
  swapping backends a pure config change for any OpenAI-compatible HTTP endpoint — no
  graphrag code changes needed to point it at a locally served NCM checkpoint.

Community-report generation and entity/claim extraction are documented as natural
follow-on slices (same training+serving recipe, different synthetic dataset and output
schema) but are out of scope for this pass.

## 4. Known scope cuts in the vertical slice itself

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

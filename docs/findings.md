# Findings: can Next-Concept Modeling (NCM) accelerate GraphRAG?

Status: research closed out for review. Everything below is measured in this repository; the
per-experiment detail, run history and traps are in [`constraints.md`](constraints.md) §7-§12.

## One-paragraph answer

At the scale this project could train (1-11M parameters from scratch, ~40 minutes per run), **NCM
reliably produces valid, bounded, fast output but did not match simple non-neural baselines on
content quality on any GraphRAG task tried.** It emitted well-formed output 97-100% of the time at
6-13 ms per call, and was typically 100-200x faster than a Claude call (not like-for-like). It did not
reproduce entity names, did not beat a bag-of-words linear classifier on relation typing, did not
beat a feature lookup table on map-stage relevance, and a drafter trained on multi-document news
did not use its source text. **None of this tests the architecture at the paper's scale** (~120M
parameters, 1-2B tokens): the models here are more than 10x smaller and trained on ~40,000x less data.
The released 88M XSum checkpoint, the only paper-scale model available, also failed on GraphRAG text.

## Scope and honest limits (read before quoting any number)

- Models trained here: 0.83M-11M generators (+6-15M VQ-VAE tokenizers), single GPU, 12k steps.
- The NCM training code is not released; this project re-implemented training against the released
  model code. Treat the training recipe as unverified against the authors'.
- Most training-quality deltas are single-seed. Where replicated, seed noise was large (TACRED sd 0.5
  at one config and 3.9 at another), so deltas under ~8 F1 points are not interpretable.
- The faithfulness judge is Claude (`claude-haiku-5-5`), so Claude's own rows may be flattered. NCM
  conclusions rest on the NCM rows plus two calibration anchors, not on Claude's scores.
- Sample sizes are small (n=60-400 per cell). Multi-News is a stand-in for community reports, not
  real GraphRAG community reports; no GraphRAG-produced training data was used.
- Latency comparisons to Claude are not like-for-like (in-process batch 1 vs a network call with
  hidden reasoning tokens).

## Results by task (held-out data throughout)

| Task | NCM | Simple baseline | Claude haiku-5-5 | NCM speed |
|---|---|---|---|---|
| Entity extraction, Few-NERD, strict name+type F1 | **5.9** (11M) | not measured | not measured | ~12 ms |
| Map-stage relevance, HotpotQA, AUC | **72.6** (0.84M, fed 5 lexical features) | 75.2 lookup table on those features; 70.5 7-feature logistic regression | 90.9 | ~12 ms |
| Relation typing, TACRED, micro-F1 | **25.75 +- 0.5** (0.83M, 3 seeds) | 23.0 naive Bayes; **44.7** softmax regression | 61.8 | ~7 ms |
| XSum summarisation, released 88M, ROUGE-1 | 34.1 | lead-1 17.4 | 28.3 | ~102 ms |
| XSum, blind claim support, in-domain | **27%** of claims supported, 0% fully faithful, fluency 2.1/5 | human refs 56% | - | - |
| GraphRAG-shaped text, released XSum NCM, zero-shot | **3-8%** claims supported | wrong-chunk control 0-2% | 88-94% | ~100 ms |
| Multi-News drafter, trained here (11M), blind claim support | **10%** supported, 5% fully faithful, fluency 1.0 | verbatim lead-1 96% | - | ~11 ms |

### What each result means

1. **Extraction (5.9 F1).** Real corpora behave very differently from synthetic: a 12-name synthetic
   benchmark scored 97% F1 but 0.00% on names it had never seen (memorisation). On real text the model
   recalls 0-6% of entity names regardless of name length, and 93% of errors are wholly invented
   names. Teacher-forced accuracy is 78.5% on name tokens vs 99.1% elsewhere. NCM predicts every
   position of a level in parallel through a discrete VQ bottleneck and has no copy mechanism.
   One real bug in the data pipeline was worth 3x (upper-cased targets forced a case transform while
   copying; verbatim names raised F1 from 1.56 to 4.56). Every model-side lever (capacity, depth,
   codebook size/placement, dropout, schedule, decode-side reranking) was swept and rejected.
2. **Map stage (72.6 AUC).** Labels are human-annotated supporting facts (HotpotQA). Raw-text NCM
   scores 61.3, barely above BM25 (59.1). It reaches 72.6 only when handed cheap lexical features
   computed in code, and a lookup table over those same features scores 75.2. So the NCM is a fast
   emitter of valid map-stage JSON; it was not shown to judge relevance from text.
3. **Relations (25.75 F1).** Closed label set (no string copying) did not make NCM competitive: it
   is on par with naive Bayes and ~19 points under a bag-of-words linear model. Only rebalancing the
   `NA` class (5 -> 18) and adding entity markers (-> 22) exceeded noise. Single-token numeric label ids
   made every output a valid label (valid rate 0.962 -> 1.000).
4. **XSum probe.** At 88M the released model beats lead-1 and zero-shot Claude on ROUGE, but a blind
   claim-level judge finds only 27% of its claims supported even on its own domain. **ROUGE hid this**:
   it measures overlap with XSum's house style, not support. An earlier statement in this project that
   the model writes "fluent abstractive summaries" was withdrawn after the judge result.
5. **Zero-shot GraphRAG text.** On podcast, earnings-call and Wikipedia text the released XSum model
   was statistically indistinguishable from a summary of the *wrong* chunk. It invents entities.
6. **Trained drafter (final experiment).** Multi-News (2-5 articles -> first summary sentence,
   38k pairs). Baseline: ROUGE-1 4.14 (lead-1 13.9), 10% of claims supported, fluency 1.0. Held-out
   ROUGE is ~equal in-sample and held-out (4.0 vs 3.8), which I first read as "not memorising"; that was
   wrong. Teacher-forced loss says it **memorises** (in-sample loss 1.50 with the correct prefix vs 2.46
   shuffled; held-out 3.88 vs 4.01), so it barely generalises its use of the source, and free-running
   decoding fails even on memorised pairs, which hid the gap from ROUGE. It emits topical word salad
   (`A alert singer recruiting Sol Biden hostage...`). Remedies tried:
   a bidirectional prefix encoder (no gain), and condition dropout + a train-time pad mask +
   classifier-free guidance. Guidance raised proxies slightly (entity/number grounding 41.7 -> 47.3 at
   scale 1) and the judge moved from 9% to 15% claims supported, but fully faithful summaries fell
   5% -> 2% and fluency stayed 1.0: noise at n=60, far below the 27% floor.

## Where NCM is and is not a fit (evidence-ordered)

- **Not a fit (measured):** anything that must reproduce input strings exactly (entity names), and,
  at the scales tried, content-quality tasks generally.
- **Supported:** structural validity and low latency for bounded outputs (JSON, label ids).
- **Open, not tested:** whether a model near the paper's scale, trained on real GraphRAG-produced
  data, closes the quality gap. This is the natural next experiment and the only one that would
  change the conclusion. A decision-useful alternative is to keep an autoregressive model for content
  and use a tiny classifier or lookup for the bounded stages, which matched or beat the NCM here.

## Measurement traps worth reusing (each nearly produced a wrong conclusion)

- ROUGE is not faithfulness: 34 ROUGE-1 coexisted with 27% claim support.
- Calibrate any LLM judge with anchors: verbatim lead-1 must score ~100% and a wrong-source summary
  ~0%. Both held throughout (96-99% / 0%).
- Always report a trivial baseline beside the model (always-NA, BM25, lookup table, lead-1). Several
  early "gains" were label leaks (chunk length, truncated queries) that a length-only control exposed.
- Single-seed deltas reversed on replication (TACRED dropout 29.7 vs 24.0/22.2); replicate x3.
- Teacher-forced accuracy did not predict generation quality in five separate experiments.
- Benchmark files can be sorted (TACRED val), and train-split holdouts can leak (60.3 vs 44.7).
- `_optimized_generate` silently ignored `cfg_scale`, temperature and top-k; any decode-knob sweep
  must first verify the knob changes outputs. Fixed and checked token-identical to the vendored path.
- Harness logging records whichever task `autoresearch.sh` defaults to; confirm it matches the task
  just trained. A discard resets the worktree and silently reverts in-progress edits.

## Reproducing

Corpora are fetched into a separate worktree; set `NCM_CORPORA_ROOT` (see `autoresearch.sh`).
Task names: `fewnerd_extraction`, `map_stage_hotpot`, `relation_tacred`, `report_multinews`.
Train with `uv run python -m ncm_graphrag.ncm.train_tokenizer --task T` then `train_generator --task T`;
score with `NCM_TASK=T bash autoresearch.sh`. Faithfulness: `scripts/judge_drafter.py` then
`scripts/claim_judge.py` (needs `ANTHROPIC_API_KEY`; the script fails if its anchors miss calibration).
Training knobs: `NCM_STEPS`, `NCM_BATCH_SIZE`, `NCM_LR`, `NCM_SEED`, `NCM_NA_FRACTION`,
`NCM_COND_DROP`; decode knob `NCM_CFG_SCALE` (valid only for a model trained with `NCM_COND_DROP>0`).
Checkpoints are not committed (`artifacts/` is ignored); the numbers above regenerate from the scripts.

## Addendum (after the first publication of this report)

- **Correction:** item 6 originally said the drafter "does not condition on the source" and "is not
  memorising". Teacher-forced loss shows it **memorises** (in-sample 1.50 with the correct prefix vs 2.46
  shuffled; held-out 3.88 vs 4.01). See `constraints.md` §42. The main conclusion (no content-quality
  advantage at this scale) is unchanged.
- **Further drafter attempts, all judge-controlled and none adopted** (`constraints.md` §43-§45):
  dropout 0.3 (16% claims supported vs 10% baseline, within noise), decode penalties (13%), grounded
  training targets (14%). Fluency stayed 1.0 throughout.
- **Where the failure shows up:** free-running output has 17.3% adjacent repeated words vs 0.1% in
  human references and 0.3% in the tokenizer's own reconstruction, so the degeneration is in generation,
  not the tokenizer. Untested explanation: parallel per-position decoding.
- **Not tested, and what would test it:** an autoregressive model of equal size and data on the same
  task. That would separate "NCM is worse" from "11M parameters on 38k pairs is too little for anything".

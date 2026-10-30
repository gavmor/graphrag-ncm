# Findings: can Next-Concept Modeling (NCM) accelerate GraphRAG?

Status: research closed out for review. Everything below is measured in this repository; the
per-experiment detail, run history and traps are in [`constraints.md`](constraints.md) §7-§12.

## Short answer

At the scale this project trained from scratch (0.83M-92.84M generator parameters, single-GPU
runs), **NCM reliably produces valid, bounded, fast output for structured grammars, but did not match
standard baselines on content quality on any GraphRAG task tried.** It emitted well-formed output
97-100% of the time on closed schemas and 79% on open graph extraction. It did not reproduce entity
names, did not beat a bag-of-words linear classifier on relation typing, did not beat a feature lookup
table on map-stage relevance, and its Multi-News drafter generalised source conditioning weakly.

**Trained at paper scale (92.8M generator parameters) on 100,000 REBEL examples, NCM was directly
compared with an equal-size autoregressive control (92.1M non-embedding parameters).** On held-out
REBEL validation (disjoint Wikipedia articles), the AR control reached 100% syntactic validity and
**9.50% Entity F1**, extracting actual source entity mentions (`Spodnja Šika`, `Tambovs Peak`). NCM
reached 79% validity and **0.53% Entity F1**, hallucinating entity names and repeating sub-tokens.
On gold transfer (Re-DocRED and DocRED dev), AR transferred 1.33-1.59% Entity F1, while NCM scored
0.00-0.14%. This shows that even with 100k real training pairs and ~92M parameters, NCM's discrete
residual VQ-bottleneck without a span copy mechanism cannot faithfully extract arbitrary
open-vocabulary entity names.

## Scope and honest limits (read before quoting any number)

- Models trained here: 0.83M-30.04M generators (+6-15M VQ-VAE tokenizers), one RTX 3090 24 GB.
  Baseline configurations mostly ran 12k steps; the 19.3M drafter width probe ran 6,001.
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
- **Open, not tested:** whether an approximately 88-120M conditional generator, trained on enough
  real GraphRAG-produced data with a validated recipe, closes the quality gap. The released 88M XSum
  checkpoint tests in-domain XSum and zero-shot transfer only. A decision-useful alternative is to
  keep an autoregressive model for content and use a tiny classifier or lookup for the bounded
  stages, which matched or beat the NCM here.

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
- **Early stopping:** dropout 0.3 at 6,001 updates reached held-out teacher-forced loss 3.105.
  On 60 held-out examples the judge found 10% supported claims, 5% fully faithful and fluency 1.0/5,
  the same rounded figures as the baseline. Lead-1 and mismatch anchors scored 97% and 0% supported.
  This single-seed result does not show a faithfulness gain; ROUGE-1 was 3.27 vs 4.14 baseline.
- **Wider generator:** n_embd 384 -> 512 (19.3M vs 10.95M parameters), dropout 0.3, 6,001 updates:
  held-out teacher-forced loss 3.172 (3.105 for the 10.95M model at the same step), ROUGE-1 3.97,
  judge 9% claims supported / 3% fully faithful / fluency 1.0 (anchors 96% / 0%). No gain from 1.8x
  parameters at 38k pairs; outputs still repeat tokens. Single seed. The harness run for it crashed
  because `autoresearch.sh` does not forward `NCM_REPORT_WIDTH`; numbers come from the same command run
  with the env set (evaluating a width-512 checkpoint needs `NCM_REPORT_WIDTH=512 NCM_DROPOUT=0.3`).
- **Judge repeatability:** re-judging the *same* 60 baseline outputs three times gave 8%, 11% and 7% claims
  supported (fully faithful 5% each time; anchors OK). Judge-only noise is therefore about 4 points, before any
  sampling noise from the 60 sources, so the 9-16% spread across all drafter variants is not distinguishable
  from the baseline's 10%.
- **Where the failure shows up:** free-running output has 17.3% adjacent repeated words vs 0.1% in
  human references and 0.3% in the tokenizer's own reconstruction, so the degeneration is in generation,
  not the tokenizer. Untested explanation: parallel per-position decoding.
- **Autoregressive control:** a decoder-only transformer trained on the same Multi-News pairs for
  12,000 batches of 128 has 10.65M non-embedding parameters (NCM: 10.89M); its total is 30.09M,
  including tied token embeddings, so total capacity and compute are not matched. On the same 400
  validation examples, AR ROUGE-1 was 6.15 vs NCM's 4.14. On 60 sources, the blind judge scored AR
  5% claims supported, 0% fully faithful, fluency 1.1/5; NCM's earlier baseline scored 10%, 5%, 1.0/5.
  Lead-1 and wrong-source anchors scored 99% and 0% supported in the AR evaluation. These small
  samples do not establish a significant difference. AR held-out loss rose from 5.36 at step 1,000
  to 11.31 at step 11,999 while training loss fell to 0.054: the final checkpoint overfit, and an
  early-stopped AR control was not evaluated here (see next bullet). The overfit result alone does not isolate
  architecture from training.
- **Autoregressive control, early-stopped:** same model, checkpoint chosen at the held-out loss minimum
  (step 1,000, loss 5.357; patience 3 evaluations every 250 steps). ROUGE-1 5.55, grounding 40.5, entity/number
  grounding 69.5. Blind judge n=60: **11% claims supported, 2% fully faithful, fluency 1.1/5** (anchors 97% / 0%),
  against 10% / 5% / 1.0 for the NCM baseline: indistinguishable (judge-only noise is about 4 points). Greedy AR
  output loops (`in the first time in the first time ...`) and copies source fragments (high grounding), so it
  is also unusable. Read together with the overfit run, an equal-size autoregressive model fails the same way
  on this data (38k pairs, ~11M non-embedding parameters), which supports "too little scale/data for this task"
  over "the NCM is specifically worse". It does not show the NCM is better. The checkpoint was selected on
  loss over 128 of the held-out benchmark examples that the judge sample also draws from, so the selection is
  not fully independent.
  Reproduce with `uv run python scripts/ar_baseline.py train`, then
  `uv run python scripts/ar_baseline.py eval 60 /tmp/ar_control` and the claim judge above.

- **Paper-scale REBEL extraction (92.8M NCM vs 92.1M AR control):** trained on 100,000 REBEL examples
  (target: bounded GraphRAG delimiter tuples of entities and relations; 6 VQ levels [4, 8, 16, 32, 64, 128],
  context length 128, prefix 512). VQ tokenizer trained for 6,000 steps reached 92.2% non-pad token accuracy
  and 0.36 validation loss. Both generators trained for 2,500 steps (batch size 16 = 40,000 exposures):
  - NCM (92.84M generator, 13 layers, width 768, 12 heads, tied heads to codebook): validation loss 3.08.
  - AR control (92.14M non-embedding parameters, 131.27M total, tied GPT-2 embeddings, causal transformer):
    validation loss 0.61.

  Evaluated on 100 held-out examples across three disjoint splits (DocRED/Re-DocRED articles strictly excluded from training):

  | Dataset | Model | Valid rate | Entity F1 | Typed Entity F1 | Relation F1 |
  |---|---|---:|---:|---:|---:|
  | **REBEL validation** (held-out articles) | **NCM (92.8M)** | 79.0% | 0.53% | 0.53% | 0.00% |
  | | **AR control (92.1M)** | **100.0%** | **9.50%** | **9.50%** | **0.97%** |
  | **Re-DocRED dev** (gold transfer) | **NCM (92.8M)** | 59.0% | 0.14% | 0.00% | 0.00% |
  | | **AR control (92.1M)** | **100.0%** | **1.33%** | 0.00% | 0.00% |
  | **DocRED dev** (gold transfer) | **NCM (92.8M)** | 71.0% | 0.00% | 0.00% | 0.00% |
  | | **AR control (92.1M)** | **100.0%** | **1.59%** | 0.00% | 0.16% |

  Outputs show the failure mode directly:
  - AR extracted `Spodnja Šika` and `Municipality of Slovens` with relation `located in the administrative territorial entity` from source sentence `Spodnja Šiška is a formerly independent settlement in ... Slovenia`.
  - NCM emitted hallucinated strings `Harinzaii` and `MiccurWar Songi` with byte fragments `\ufffd` and broken tags `entityentity`.
  - Conclusion: with scale (92M) and data (100k) matched, autoregressive decoding reliably copies source entities, whereas NCM's discrete code bottleneck hallucinates surface forms.
  Reproduce: `uv run python scripts/rebel_experiment.py evaluate --model ncm` and `... evaluate --model ar`.

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

## 7. Real-data extraction: what the measured ceiling actually is

The synthetic slice above reports high structural validity, but that result does not
transfer. Measured on two real annotated corpora (Re-DocRED JSON; Few-NERD via
`pyarrow`), the same architecture scores **1-5% entity F1**, and the reason is now
localised rather than inferred.

**13. Token names must be verbatim source substrings.** Targets originally upper-cased
entity names, so copying `Warrick Dunn` required emitting `WARRICK DUNN` — entirely
different BPE tokens. Scoring is case-insensitive either way, so the transform bought
nothing and blocked copying outright. With verbatim names, F1 went **1.56 → 4.56** and
recall on single-token names went **0% → 20%**. Assert this invariant if the data build
changes (438/438 gold names occur verbatim in the prefix).

**14. Failure is uniform across name lengths — it is a missing copy mechanism, not a
span-boundary problem.** Recall by gold name length on Few-NERD validation: 1 token 20%,
2 tokens 6.7%, 3 tokens 11.2%, 4 tokens 4.0%, 5+ tokens 1.3%; and 93% of remaining
errors are wholly invented names (5% exact, 0% "name right, type wrong"). Type is never
the bottleneck.

**15. Ceiling and capture must be reported together.** The tokenizer-only reconstruction
ceiling and the generator's fraction of it move in opposite directions:

| configuration | tokenizer ceiling (F1) | generator F1 | capture |
|---|---|---|---|
| 256 codes, upper-cased targets | 30.83% | 1.72% | 5.6% |
| **256 codes, verbatim targets** | 13.67% | **4.56%** | **33%** |
| 512 codes, verbatim targets | 15.33% | 2.72% | 18% |

Making the task well-posed raised capture 6x. Raising tokenizer capacity raises the
ceiling but lowers capture, and accuracy dominates: **256 codes/level is optimal** in
both regimes. Mixed-case/Unicode names also lowered the ceiling, so 256 is tighter than
it used to be and still wins.

**16. Teacher-forced accuracy is not a proxy for generation quality.** Four independent
instances: 87.40% → 0.17 F1; 85.88% → 1.72; 89.01% (dropout 0.05) → 2.28 vs 86.49% →
2.53; 87.38% (8 layers) → 2.89 vs 85.96% → 4.56. Do not pre-screen configurations on it.

**17. Decode-side reranking has zero headroom.** With an oracle selecting the best of 16
sampled candidates, F1 equals greedy exactly (3.83 on the pre-fix data, 6.67 after the
verbatim-name fix). Sampling never produces a correct candidate greedy missed, so MBR or
grounded selection cannot help. Span grounding only helps if it **constrains** decoding.

**18. Every in-budget model lever has been swept and rejected**: capacity (10.95M →
30.04M), depth (6 → 8), codebook size (512, 1024) and per-level grading, PAD-masked
prefix attention, schedule (batch 64 x 24000 vs batch 128 x 12000), dropout 0.05, steps,
encoder trainability, hierarchy depth, decode penalties. The honest remaining options are
a ~120M-parameter model (the background doc's own regime, far outside a single-run budget
here) or constrained decoding to source spans.

## 8. Real map-stage scoring (Lead 1): what was measured

The synthetic `mapstage` task proves NCM can emit GraphRAG's bounded `{"points":[...]}` JSON,
but its relevance rule is a closed template. Two real-data variants were built to test whether
NCM can *judge* relevance. Corpora are fetched into a sibling worktree; set `NCM_CORPORA_ROOT`.

**19. Most staged corpora cannot supervise a score.** `microsoft_graphrag/*.csv` (Kevin Scott,
MSFT) holds questions only, with no answers or relevance labels; `squality` is a metadata card
with no stories. Usable supervision is QMSum (query + transcript + reference summary, so the
relevance label must be *derived*) and HotpotQA `distractor` (10 paragraphs per question, two
human-annotated supporting facts, ~90k train questions).

**20. Every label construction leaked something, and each leak inflated an early number.**
Found by control baselines, not by inspection:

| leak | symptom | fix | control after |
|---|---|---|---|
| prefix window cut the tail while the query came *after* the report (74% of QMSum lost its query) | memorisation, in-sample AUC 89 vs held-out 58 | query first, report clipped to budget | n/a |
| QMSum short trailing chunk covered less of the summary purely by being short | length-only AUC 66.4 | drop tail; then restrict to a ±15% length band (density ranking overcorrected to 38.6) | 49.07 |
| HotpotQA gold paragraphs are shorter than distractors | length-only AUC 41.6 | draw distractors from a ±35% band | 44.96 (residual) |

Always report the length-only control, lexical baselines, and in-sample vs held-out AUC beside
the NCM number.

**21. QMSum is data-starved (3,620 usable pairs); HotpotQA is not.** On QMSum the NCM memorises
(level_acc 0.999) and scores 53.5 vs a one-line query-term-overlap baseline of 68.6. On HotpotQA
(60k pairs) there is no memorisation gap, but the raw-text NCM scores 61.3 against BM25 59.1 and
a 7-feature logistic regression at 70.5.

**22. Capacity does not help (again).** 0.84M → 10.77M parameters: 61.34 → 58.13, with in-sample
64.4 vs held-out 57.9. The "no memorisation gap, so it must be underfitting" inference was wrong.

**23. What the NCM can and cannot do with lexical structure (HotpotQA, held-out AUC):**

| model | AUC |
|---|---|
| NCM, raw text | 61.34 |
| NCM, report opening surfaced as text | 60.73 (no gain: not a visibility problem) |
| rule `Match` = query terms in the report's first 6 content words, no model | 65.01 |
| NCM given `Match: n` | 65.54 (+0.5 over the rule it was handed) |
| NCM given 5 quantised features (`head/ov/cnt/bi/len`) | **72.64** |
| logistic regression, same 5 quantised features | 66.68 |
| lookup table over the exact feature tuple | **75.15** |

Read honestly: given one feature the NCM is a pass-through; given several it beats a linear
classifier by ~6 AUC (it learns feature *interactions*) but does **not** beat a non-parametric
lookup on the same inputs. NCM has not been shown to learn relevance judgement from text here;
what it contributes is valid bounded JSON at ~12 ms/call and ~17k tok/s.

**24. Not yet wired into serving.** `map_stage_real` / `map_stage_hotpot` are benchmarked tasks;
the litellm `HybridCompletion` path still serves the synthetic `mapstage` model. A serving
prefix builder would have to compute the same `Features:` line from the query and report.

**25. Speed-vs-quality against an autoregressive LLM (same 300 held-out HotpotQA examples).**
Claude (`claude-haiku-5-5`, query + report only, no features) vs the NCM (given its 5-feature line):

| | AUC | valid | latency / call |
|---|---|---|---|
| Claude haiku-5-5 | **90.87** | 0.997 | **1,959 ms** median (p90 2,860) |
| NCM, 0.84M params | 71.87 | 0.967 | **12 ms** |
| lookup table over the NCM's features (no neural net) | ~75 | 1.0 | microseconds |

~19 AUC points of relevance quality cost ~160x the latency. Caveats that matter: the Claude
latency is a network round-trip run 8-way concurrent and includes hidden reasoning (a 120-token
cap returned `content=None` / `finish_reason=length` on every call, which an earlier run of this
comparison mis-read as 8% validity and a bogus 98 AUC on 24 outputs); the NCM number is in-process,
batch=1, no network. n=300, one NCM training seed. On this task the NCM sits *between* a
free feature lookup and a strong LLM without dominating either, so its case rests on bounded-JSON
throughput, not on relevance judgement.

## 9. Closed-vocabulary relation classification (Lead 2): what was measured

Lead 2 was chosen because the output is a member of a fixed label set: nothing has to be copied
from the input, which removed the bottleneck that capped entity extraction (§7).

**26. Task, data, metric.** TACRED (the OpenNRE-format copy staged here: 68,124 train / 22,631 val /
15,509 test, 42 classes, `NA` = 81% of train, **no entity types**). Prefix is `Head: ..\nTail: ..\n
Between: ..\nText: <sentence with @head@ and #tail# markers>`; the target is the relation's index in the
sorted label list (`NA` = 0), a single GPT-2 token. Benchmark: a seeded random 3,000-example sample of
**validation** (717 non-NA positives, 41 classes). Metric: micro-F1 over non-`NA` relations, with an
invalid output counted as a miss, so always-`NA` scores 0. `val.txt` is **sorted**: its first 3,000
lines cover only 14 of the 42 classes, so a prefix sample would be meaningless.

**27. Where the NCM lands (held-out micro-F1; NCM numbers are 3-seed means):**

| model | micro-F1 | note |
|---|---|---|
| always `NA` | 0.0 | accuracy 76.1% |
| naive Bayes (between/around words) | 23.0 | P15.7 / R43.0 |
| **NCM, 0.83M params** | **25.75 +- 0.5** | 26.28 / 25.73 / 25.25; valid_rate 1.000; ~7 ms/call |
| softmax regression (hashed words, no neural net) | **44.7** | P60.6 / R35.4 |
| Claude haiku-5-5 (n=400 subset; NCM 25.3 on the same) | **61.8** | ~907 ms median per call |

Read honestly: removing string-copying did **not** let a from-scratch 0.83M NCM compete. It is on par
with naive Bayes and ~19 points below a bag-of-words linear classifier. Against Claude it is ~146x
faster per call and ~36 F1 points lower (not like-for-like: Claude is a network call run 8-way
concurrent; the NCM is in-process, batch 1; 6 Claude answers had trailing "Hmm, wait" text after a
correct label and were counted invalid under strict parsing).

**28. What moved the NCM, and what did not.** Only the first two exceed measured noise:

| change | F1 | verdict |
|---|---|---|
| natural class mix (67k, 81% NA), label string target | 5.18 | collapses to `NA` (predicts NA 94% of the time **in-sample**); accuracy 75.4% is *below* always-NA |
| downsample `NA` to 40% of the training pool | 17.82 | big, real: recall 2.9 -> 19.3 |
| inline `@`/`#` entity markers | 22.26 | +4.4, single seed |
| `Between:` field | 24.80 | +2.5, within noise (label-string seeds 24.80/24.76/21.58) |
| numeric-id target (context 32 -> 8) | 26.28 | F1 within noise, but **valid_rate 0.962 -> 1.000** (structural) |
| dropout 0.3 | 29.70 on seed 1, **24.01 / 22.24** on seeds 2/3 | does not replicate (mean 25.3, sd 3.9) |
| dropout 0.5 | 23.55 | P38.2 / R17.0: trades recall for precision |
| `NA` at 60% | 23.29 | no gain |
| 3,000 steps instead of 8,000 | 8.60 | undertrains toward the NA prior |

After rebalancing the model memorises: in-sample F1 68.3 vs held-out 21.9. Fewer steps and more
dropout did not close that gap; they only moved the precision/recall operating point.

**29. Measurement traps (each nearly produced a wrong conclusion).**
- `val.txt` is sorted; always sample randomly.
- A holdout cut from the *train* split leaks: TACRED has several entity pairs per sentence, so a
  linear baseline scored 60.3 on it but 44.7 on validation (names drive the leak).
- `level_acc` is useless here: the 32-token context is mostly padding, so 0.996 coexisted with
  collapse to `NA`.
- Single-seed deltas of a few points are noise. Seed 1 also fixes the training-pool sample and was
  the best draw every time (24.8, 26.3, 29.7). Run #80's +3.4 reversed under replication, and the
  seed spread itself depends on the configuration (sd 0.5 at dropout 0.1, 3.9 at 0.3).
- Training-schedule overrides (`NCM_STEPS` etc.) had been silently reverted by an earlier discard
  that reset the worktree; the training log's last step (7999 instead of 3999) exposed it.

## 10. What the NCM is good at, measured at the paper's own scale (released XSum checkpoint)

§7-§9 are 1-11M-parameter models trained from scratch for ~40 minutes; they cannot say how the
architecture behaves at the scale the paper uses. This probe runs the **released** XSum checkpoint
(88.0M generator + 60.6M VQ-VAE, 4096 codes/level, 9 levels, 512-token prefix) with no training here.

**30. Setup.** 150 XSum test articles, seeded random sample (seed 11), filtered to >=200 GPT-2 tokens
(median 415; the checkpoint is documented as unreliable on short input). Metrics: ROUGE-1/2/L
(`rouge_score`, stemmed, F-measure) against the single reference summary. Baselines on the identical
articles: lead-1 (first sentence) and `claude-haiku-5-5` asked for a one-sentence summary
(`max_tokens=2000`, 150/150 valid). XSum test data fetched from the HF `refs/convert/parquet` branch.
One-off probe, not a benchmark entrypoint; sample sizes are small.

| system | ROUGE-1 | ROUGE-2 | ROUGE-L | latency / summary |
|---|---|---|---|---|
| **NCM released, 88M** | **34.1** | **11.8** | **26.3** | **102 ms** (in-process, batch 1) |
| Claude haiku-5-5, zero-shot | 28.3 | 7.4 | 18.8 | 1,108 ms (network, 8-way concurrent) |
| lead-1 | 17.4 | 1.4 | 12.6 | - |

**31. How to read it (do not overstate).**
- This is the first measurement where the NCM beats a baseline that is not trivial: ~2x lead-1 on ROUGE-1
  and ~8x on ROUGE-2, producing fluent one-sentence abstractive summaries in ~100 ms.
- It is **not** evidence the NCM out-summarises Claude. The NCM was trained on XSum and Claude is
  zero-shot, and ROUGE rewards matching XSum's house style; a human or faithfulness judgement could
  rank them differently. The latency gap is also not like-for-like (88M in-process vs a network call).
- **The weakness from §7 persists at 88M.** Sample outputs garble proper nouns: `Guinea` -> `Guegal`,
  `Guingamp` -> `Guernan Guineamp`. Of 28 capitalised words in NCM summaries absent from the article,
  inspection found ~5 clear misspelled copies (`Guernan`, `Guineamp`, `Prideice`, `Cranning`,
  `Fernandez`), ~8 plausible-looking person names with no support in the text (`Joel`, `Billy`,
  `Malcolm`, `Katherine`, ...), 4 harmless possessives, and the remainder generic words. This was an
  inspection of a small sample; **no rate is claimed.**
- Two attempts at an automatic garbling metric failed and were discarded rather than reported:
  "capitalised words found in the article" gives the gold references only 65% (XSum removes the
  article's first sentence, so references legitimately contain names the body never mentions), and a
  near-spelling test (difflib ratio >= 0.6 against any article word) gives the references 60.8%
  although they contain no garbled copies, i.e. it cannot discriminate.

**32. Consequence for GraphRAG.** Evidence now points the same way from both scales: the NCM is a
reasonable fast generator of fluent, abstractive, bounded-length text, and a poor fit wherever the
output must reproduce input strings exactly. In the brief's table that favours bounded
summarisation-style stages (leaf community-report drafting, short map-stage justifications) over
extraction, where entity names must be exact. That is a hypothesis to test with a GraphRAG-shaped
training set, not a result: the XSum model was trained on news, and no community-report model exists.

## 11. Zero-shot probe: the released XSum NCM on GraphRAG-shaped text (and a correction to §10)

Question: before training anything, does the released 88M XSum checkpoint produce usable short
summaries of the kind of text GraphRAG processes? No reference summaries exist for these inputs, so
scoring is reference-free and blind.

**33. Setup.** 180 inputs, 60 each from: Kevin Scott podcast transcripts and MSFT earnings-call
transcripts (the Microsoft GraphRAG benchmark corpora) and HotpotQA Wikipedia paragraphs (gold + 2
distractors, three paragraphs, shuffled). Each trimmed to <=480 GPT-2 tokens (the checkpoint's prefix
is 512; 300+ required). Input cleaning applied for the NCM: filename headers, curly quotes and
`SPEAKER:` labels removed (the raw text additionally produced `U+FFFD` byte fragments, 0.25 per
podcast summary; cleaning removed those but not the invented content). Systems, each judged blind and
in shuffled order by `claude-haiku-5-5` at claim level (supported / unsupported / contradicted +
fluency 1-5): the NCM; Claude (one-sentence summary); lead-1 (first sentence, verbatim); and a
**mismatch control** (an NCM summary of a different chunk of the same source).

**34. Result (share of claims supported; fully-faithful summaries in brackets; n=59-60 each).**

| source | NCM | Claude | lead-1 (anchor) | mismatch (anchor) | NCM fluency |
|---|---|---|---|---|---|
| podcast | **8%** (0%) | 92% (70%) | 99% (87%) | 2% (0%) | 1.2 |
| earnings call | **3%** (2%) | 88% (37%) | 100% (100%) | 2% (0%) | 1.2 |
| Wikipedia | **7%** (0%) | 94% (67%) | 100% (100%) | 0% (0%) | 1.3 |

The anchors validate the judge: verbatim lead-1 scores 99-100% and a summary of the wrong chunk 0-2%.
**The NCM is statistically indistinguishable from the wrong-chunk control** on claim support, i.e. its
output carries almost no information specific to the source it was given. It also invents entities
(`Alphabet`, `Tech Tent`, `Techcom.com`) and, on the Hartlepool shopping-centre paragraph, "Australia's
largest Shopping Centre". Deterministic checks agree: 11-61% of numbers and 19-49% of capitalised
words in NCM summaries appear in the source, vs 90-100% for Claude and 100% for lead-1.

**35. Correction to §10.** The same judge on the NCM's *in-domain* XSum summaries (60 of the earlier
150 articles) gives **27% of claims supported, 0% fully faithful, fluency 2.1/5**; XSum's own human
reference summaries give 56% supported (22% fully faithful, fluency 5.0), because XSum references
contain content absent from the article. §10 reported ROUGE-1/2/L of 34.1/11.8/26.3 and described the
output as "fluent" summaries. ROUGE rewards XSum house style and does not measure support; on
faithfulness the released checkpoint is weak **even on its own training domain**. The ROUGE lead over
Claude and lead-1 stands as a ROUGE result only. Faithfulness is roughly 4-9x worse on GraphRAG-shaped
text than on XSum (27% -> 3-8%), so domain shift costs a large further drop.

**36. Limits of this probe.** The judge is Claude, so Claude's own rows may be flattered (the NCM
conclusion does not depend on them: it rests on the NCM rows and the two anchors). n=60 per cell. The
Wikipedia inputs are three unrelated paragraphs, less coherent than a real community report. One
checkpoint, trained only on XSum news, so this says nothing about a model *trained* on GraphRAG-shaped
text. Quote it only as "the released XSum model does not transfer zero-shot".

**37. Consequence.** The §10/§32 hypothesis (NCM suits bounded abstractive summarisation) is **not
supported by the zero-shot evidence** and §10's fluency framing is withdrawn. What remains open, and
untested: a model *trained* on GraphRAG-shaped summaries, and whether any NCM size reaches useful
faithfulness. For GraphRAG the measured picture is now: valid bounded structure and ~7-100 ms latency
(§8, §9, §10) but no demonstrated advantage in content quality on any task tried.

## 12. Trained community-report drafter (Multi-News): the last experiment

Question left open by §11: does an NCM *trained* on multi-document summarisation reach useful
faithfulness? Task: 2-5 Multi-News articles -> first sentence of the human summary (<=48 tokens);
source = first 110 words of each article joined, clipped to 288 tokens; 38,268 train pairs, 400
held-out validation. The lead-only source covers just 46% of the reference's content words (68% in the
full articles), so ROUGE has a ceiling; lead-1 scores 13.9 ROUGE-1.

**38. Result.** 10.95M generator + 14.7M VQ-VAE, 12,000 steps. ROUGE-1 **4.14**, content-word
grounding 26.3%, entity/number grounding 46.5%, 12.4 words/output, valid 0.995, 11 ms. Blind claim judge
(n=60, anchors OK: lead-1 96%, wrong-source 0%, human reference 50%): **10% of claims supported, 5%
fully faithful, fluency 1.0.** Outputs are topical word salad. This is *below* the released XSum model's
27% in-domain floor. **Correction (see #42):** an earlier version of this section said the model was "not
memorising" because in-sample and held-out generation ROUGE were both ~4. Teacher-forced loss shows the
opposite: the model memorises the training pairs and generalises little.

**39. Remedies, all judged or proxy-gated (nothing beat noise).**
- Bidirectional 2-block prefix encoder: ROUGE-1 3.94, grounding 24.9, entity/number 44.9; ~40% slower.
- Condition dropout 0.1 + train-time pad mask + classifier-free guidance (same checkpoint, scale
  0/1/2/4): ROUGE-1 3.65/4.27/4.31/4.11, entity/number grounding 41.7/47.3/47.0/41.6. Judge at scale 0:
  9% supported / 5% fully faithful; at scale 2: **15% / 2%**, fluency 1.0 for both. Guidance moves the
  proxies a little and the judge not at all outside n=60 noise.

**40. Infrastructure found and fixed on the way.**
- `_optimized_generate` ignored `cfg_scale`, so any guidance sweep would have been a silent no-op.
  Implemented the vendored level-scheduled guidance, `(1+t)*cond - t*uncond` with
  `t = scale*level/(nlev-1)` (level 0 unguided, fine levels strongest); verified token-identical to the
  vendored `generate()` at scale 0 and 2.
- The trainer attended padded prefix positions while `generate()` masks them (`-inf`); now optional
  (`NCM_TRAIN_PAD_MASK`, forced on with condition dropout). Null-prefix parity with `generate()`
  verified by spying on the real call. Earlier tasks' numbers were measured through `generate()` and
  stand, but whether a pad-masked trainer improves them was not tested.

**41. Limits.** Judge is Claude; n=60; one seed per configuration; an 11M model on 38k pairs; a
lead-only source with a hard ROUGE ceiling. This says an NCM at this scale and data volume does not
condition on its source. It does not say what a ~100M-parameter model on real GraphRAG data would do.
See [`findings.md`](findings.md) for the consolidated report.

**42. Correction: the drafter memorises (teacher-forced loss by prefix).** `scripts/prefix_use.py`
measures teacher-forced loss with the correct, a shuffled, and a null prefix (384 examples each):

| examples | correct | shuffled | null |
|---|---|---|---|
| training (in-sample) | **1.50** | 2.46 | 3.26 |
| held-out validation | **3.88** | 4.01 | 4.13 |

On training pairs the source matters a lot (1.50 vs 2.46); on held-out pairs it barely does (3.88 vs 4.01,
about 0.13 nats). So the 11M drafter memorises ~38k source-to-summary pairs over ~40 epochs and
generalises little. Free-running ROUGE hid this because decoding fails even on memorised pairs
(in-sample ROUGE-1 4.0 vs held-out 3.8). This supersedes the §38 reading "does not use the source":
it does use it, but only to recall pairs it has seen. Teacher-forced loss is itself an imperfect proxy
(earlier tasks showed it can rise while generation worsens), so this is a diagnostic, not a score.

**43. Regularisation and decode penalties on the drafter (judge-controlled; nothing adopted).**
Trainer now has optional held-out teacher-forced loss (`NCM_EVAL_EVERY`) and the drafter a dropout
override (`NCM_DROPOUT`).

| setting | held-out TF loss | ROUGE-1 | grounding | entity/number | judge (n=60): supported / fully faithful / fluency |
|---|---|---|---|---|---|
| dropout 0.1 (baseline) | 3.88 (final) | 4.14 | 26.3 | 46.5 | 10% / 5% / 1.0 |
| dropout 0.3 | **3.07 at step ~6-7k**, 3.21 at 12k | 3.68 | 29.3 | 56.3 | 16% / 7% / 1.0 |
| dropout 0.3 + decode penalties | - | 3.96 | 25.7 | 47.6 | 13% / 2% / 1.0 |

Dropout 0.3 improves held-out teacher-forced loss a lot (3.88 -> 3.07 at best, with a visible overfitting
turn after ~7k steps) and raises both grounding proxies, but the judge does not separate it from the
baseline (fully-faithful 95% CIs 2-13 vs 0-12) and fluency stays 1.0. This is the sixth time teacher-forced
loss improved without a matching generation gain on this project. Decode penalties (frequency 1.5,
adjacency 3,3,0.6, the settings the released free-text checkpoints use) did not help either: outputs
still repeat tokens (`North North Korea has North`) because the repetition originates in the latent
codes, while the penalties act only on the final token pick. Single seed, n=60 per cell.

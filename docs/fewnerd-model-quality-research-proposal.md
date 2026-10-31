# Research proposal: recover exact entity copying before scaling Few-NERD generation

Status: proposed work, not completed experiments. Prepared after local merge `05cfe2f` into `main` on 2026-10-31. Owner: the next research agent.

## Decision and research question

Cache correctness, reproducible CUDA execution, and the negative experiment report are merged. Model-quality approval is not implied. The 1,500-step tokenizer / 3,000-step generator probe is finished; do not resume it as though training were incomplete.

**Question:** can the current bounded Few-NERD tokenizer reconstruct entity names reliably enough to justify training a conditional generator, and, if so, does that generator outperform a cheap grounded baseline or compete with an exposure-matched autoregressive control?

Start with data/scoring controls and tokenizer reconstruction. Do not begin with a larger generator, a broad hyperparameter sweep, or an unconditional 12,000-step run. Longer training is a hypothesis to test, not an established remedy.

The hypotheses and numerical continuation gates below are proposed experimental decisions. They are neither measured results nor production acceptance criteria.

## 1. Verified starting point

The cache branch tip `2d051f4` was merged while preserving `main`'s sampling fix `ed5e93c`. Merge verification passed 15 combined regression tests, 7/7 CPU cache acceptance checks, pipeline validation, and a real-model CPU smoke test: optimized and vendored generation produced identical codes for greedy and stochastic sampling, both with and without classifier-free guidance.

A post-merge audit added a fail-closed requirement: saved-run evaluation must reject a missing persisted training cache rather than build a new cache and validate that replacement. The regression failed before the fix and passed afterward; the combined suite now has 16 passing tests. A real-model CPU smoke accepted a persisted cache with zero error, then rejected its removal without creating a replacement.

The historical CUDA probe used 6,000 tokenizer-training examples, 50,000 generator-training examples, batch 128, learning rate 3e-4, tokenizer seed 0, and generator seed 1. Its architecture is defined by `build_fewnerd_extraction_*_cfg` in [config.py](../src/ncm_graphrag/ncm/config.py): tokenizer embedding width 128, five 256-entry codebooks, levels `[2, 5, 10, 20, 80]`, target context 80; generator six layers, width 384, six heads, prefix 64, raw non-contextual prefix, tied codebook heads, conditioning noise 0.

| Observed measurement | Value | Interpretation limit |
| --- | ---: | --- |
| Registered validation mean typed entity F1, 300 examples | 0.3333% | 28 exact prefixes also occur in generator training |
| Mean typed entity F1 on the 272 unseen-prefix examples | 0.3676% | Post-hoc overlap audit of the same fixed predictions; not a new untouched test |
| Registered validity, all 300 | 98.6667% | Existing validity predicate; not exact extraction or a strict full-record parser guarantee |
| Tokenizer reconstruction entity F1, all 300 | 0.6667% | Diagnostic of this checkpoint, not a mathematical upper bound on generator F1 |
| Tokenizer non-PAD token accuracy, all 300 | 88.9026% | Includes EOS and boilerplate; does not measure exact name copying |
| Teacher-forced code accuracy, all 300 | 81.5071% | Does not establish correct free-running names |
| Corrected CUDA cache comparison | 0 maximum embedding error over 272 rows | Original batch geometry and construction policy; seven rebatched representative code rows also matched exactly |

Only one of the 300 generated entity sets matched exactly. These measurements do not demonstrate useful extraction.

### Canonical evidence

Read [architecture.md](architecture.md) for the detailed run history and [findings.md](findings.md) / [constraints.md](constraints.md) for prior negative results and measurement traps.

Private MinIO bucket: `ncm-fewnerd-probes`.

- `fewnerd/results/fewnerd-probe-20261031T165937Z.tar.gz`: completed training checkpoints, validated cache, manifest, and training log; nine intermediate model/optimizer/RNG snapshots were also uploaded under `fewnerd/runs/20261031-probe1/checkpoints/`.
- `fewnerd/evaluations/fewnerd-evaluation-20261031T173056Z.tar.gz`: current corrected evaluation, `cache-validation.json`, manifest, and log from Concourse `evaluate-saved-probe/3`.
- `fewnerd/final/fewnerd-probe-20261031T171103Z-report.tar.gz`: earlier evaluation plus CPU split audit. Use its overlap findings, not its superseded cache-diagnostic schema, as current acceptance evidence.
- `fewnerd/inputs/probe-inputs-1.0.3.tar.gz`: immutable historical source/data input. Its checksum and dependency-image digest are in [fewnerd-probe-pins.yml](../concourse/fewnerd-probe-pins.yml).

Small local reports are in `artifacts/fewnerd-cache-validation-corrected/` and `artifacts/fewnerd-probe-evaluation/` **in the cache-validation worktree**, not necessarily in another checkout. MinIO is the durable source. Check checkpoint hashes against the report; do not assume files named `artifacts/fewnerd_extraction/tokenizer.pt` or `generator.pt` are these checkpoints.

## 2. Scope and falsifiable hypotheses

| Hypothesis | Controlled test | Evidence against it / next decision |
| --- | --- | --- |
| Data conversion, scoring, or target alignment is broken | Gold serialization round-trip and a tiny-set VQ overfit test | If either fails, stop model comparisons and repair that concrete defect first |
| The 1,500-step tokenizer is undertrained | Same architecture, data, optimizer, and initialization; inspect predeclared 1,500/3,000/6,000-step checkpoints | If exact reconstruction remains poor, more generator updates cannot diagnose the tokenizer problem |
| Reconstruction learning is diluted by PAD/boilerplate | Name-specific metrics and one controlled non-PAD reconstruction-loss arm | If exact names do not improve, do not interpret overall token-accuracy gains as progress |
| A usable tokenizer still leaves a conditional-generation problem | Freeze a qualifying tokenizer; compare NCM with a grounded dictionary and an AR control | If AR copies names but NCM does not, investigate the NCM conditioning/teacher-forcing/free-running transition rather than tokenizer capacity |

A larger architecture, new codebook hierarchy, new target grammar, pretrained encoder, or different corpus is outside the first experiment packet. Prior codebook/capacity sweeps and a separate paper-scale REBEL comparison already produced negative results; they are not a justification to repeat those changes without a new hypothesis. Do not transfer REBEL or Multi-News scores to Few-NERD.

## 3. Phase A: freeze an honest evaluation protocol before GPU work

1. Audit [fewnerd_extraction.py](../src/ncm_graphrag/data/fewnerd_extraction.py) and representative training/tuning records. Current eligibility is one or two entities, prefixes fitting 64 GPT-2 tokens, and targets including EOS fitting 80. Preserve these bounds for the controlled study. Check span grouping, punctuation, Unicode, entity order, and gold serialization; do not silently change annotations to improve scores.
2. Record that the current prefix supplies `Entity_types` derived from the gold types present in each sentence. This is **bounded extraction with oracle candidate types**, not unrestricted NER or full GraphRAG extraction. Give every control the same information. Predicting types or removing that information is a separate experiment.
3. Form a canonical input identity from `text_unit`: Unicode NFC plus collapsed whitespace, preserving case. Hash it independently of gold type labels and targets. The existing full-prefix overlap audit is not sufficient protection when the same text has different annotations. Deduplicate within each split; quarantine groups with conflicting gold annotations and report counts.
4. Freeze official-train groups for training. Exclude every training group from validation. Exclude all official-train and official-validation groups from the reserved official-test evaluation. If article IDs are available, group by article too; otherwise call the result sentence-disjoint, not document-disjoint. Report the remaining near-duplicate limitation.
5. Select and store a fixed training manifest: generator pool up to 50,000 unique eligible train groups, deterministic selection seed 0; tokenizer subset 6,000 of those same groups. If fewer than 50,000 are available after deduplication, use all remaining groups and declare the actual pool size before training. Fewer than 6,000 requires revising the protocol, not silently reducing the tokenizer study.
6. Freeze 1,000 clean validation groups for tuning with selection seed 42 and 1,000 clean official-test groups with selection seed 43. If either pool is too small, stop and revise the protocol before training. Confirm prior research has not already used the proposed test sample for selection. Do not inspect test predictions or metrics until configuration and checkpoint-selection rules are frozen. Earlier 300-example results are development evidence only.
7. Save row/group IDs, split membership, exclusions, seeds, raw corpus hashes, pre-tokenizer identity, tokenized prefix/target hashes, bounds, code revision, and resolved configs. Verify the union of tokenizer/generator training groups is disjoint from tuning and final testing; tuning and final testing must also be disjoint. Do not exclude shared entity names or shared target strings alone: names recurring in different sentences are normal NER supervision.
8. Separate data-selection seeds from model-initialization/batch-sampling seeds before replication. Current trainers reuse `SEED` for both pool selection and model initialization; changing it today also changes the data. Replications must use identical stored rows and order.
9. After freezing dictionary construction, normalization, overlap and tie rules, compute and record its score on the frozen tuning set before Phase C. Record the dictionary training pool and predictions for the paired comparison; do not use held-out gold to construct entries. Do not inspect reserved-test metrics.

The old 272-example subset is useful as an audit, not as the final test after repeated inspection. Keep the original 300-example report unchanged for historical comparability.

## 4. Metrics and controls

Keep the historical primary definition explicit: **mean per-example typed entity-set F1**, in percentage points, after the existing name/type normalization. Also report aggregate typed micro precision/recall/F1, exact entity-set match, full target-string reconstruction, registered validity, and a separately specified strict full-record grammar/completion check. Do not relabel a new parser metric as the old validity score.

For tokenizer diagnosis, report exact entity-name reconstruction, name-token accuracy, type accuracy, EOS behavior, and non-PAD token accuracy separately from fixed grammar/description tokens. Implement and check name-token alignment against the actual GPT-2 serialization, including Unicode and punctuation; do not assume names occupy fixed token positions. Report train and tuning metrics, one- versus two-entity cases, name length, and names seen versus unseen in tokenizer training. Define seen names using the same frozen name normalization as scoring, independently of type; also report subword-token exposure in training targets. Report counts and denominators for each subgroup, reconstruction and VQ loss terms separately, plus per-level utilization.

Required controls:

- **Scorer oracle:** score the serialized gold target against itself. Entity F1 and entity-set exact match must be 100%; verify parser round-trip on eligible examples. This validates measurement, not a feasible inference baseline.
- **Tiny-set overfit:** same full VQ architecture on 32 fixed train targets, batch 32, at most 2,000 updates. Include multi-subword entity names and report their count and exact-name reconstruction, as well as exact entity-set reconstruction, on those same targets deterministically. Proposed gate: at least 95% exact entity-set reconstruction and at least 95% exact-name reconstruction among the multi-subword names. Failure triggers an implementation/optimization diagnosis, not an unlimited training extension. Passing demonstrates fit on those examples only, not generalization or lossless representation of arbitrary names.
- **Train-only grounded dictionary:** build mention/type entries from frozen training data, find matching spans in each input, use a fixed longest-span-first overlap rule, and respect the same supplied candidate types. Freeze normalization/ties before tuning. No held-out gold may populate the dictionary.
- **Conditional AR control, later:** adapt the existing AR implementation used by [rebel_experiment.py](../scripts/rebel_experiment.py) to the exact Few-NERD manifests and grammar. That runner is not currently a drop-in Few-NERD experiment. Match generator training exposures and approximately 10.9M non-embedding parameters; report total parameters, measured GPU time, memory, and tokenizer-pretraining cost separately. Equal update counts are not equal FLOPs or equal end-to-end cost.

Evaluate quality using deterministic greedy decoding first. Any later best-of-N or sampling comparison must name its sampling budget, test that knobs affect outputs, and charge all candidates to its compute budget.

## 5. Phase B: bounded tokenizer experiment packet

First agent budget: Phase A plus at most **14,000 tokenizer updates**: 2,000 tiny-set updates, 6,000 baseline updates, and one conditional 6,000-update loss ablation. No generator updates in this packet.

### T0: schedule control

After the overfit and scorer controls pass, train from scratch on the frozen 6,000-row tokenizer subset for 6,000 steps, batch 128, learning rate 3e-4, existing architecture/loss/EMA settings, initialization and sampling seed 0. Inspect the predeclared 1,500, 3,000, and 6,000 checkpoints on the tuning set. Preserve all snapshots and curves. Do not choose extra checkpoints after seeing outcomes.

Use the highest tuning reconstruction entity F1 among those checkpoints, breaking ties in favor of the earlier checkpoint. Compare 1,500 versus 6,000 within this new fixed-data run; the historical probe used a different evaluation protocol and is not a controlled treatment arm.

Proposed tokenizer continuation gate: tuning reconstruction entity F1 at least 90% and full serialized-target exact match at least 80%. These deliberately conservative gates prevent generator spending on a tokenizer that barely reconstructs names; they are not mathematical bounds on generator quality.
For both T0 and T1, report the gate metrics on training and tuning, and split name reconstruction and typed entity F1 by names seen versus unseen in tokenizer training, with subgroup counts and subword exposure. The overall tuning gate remains unchanged. If seen training names exceed 90% reconstruction F1 but tuning fails, examine unseen-name and rare-subword errors separately; a concentration of errors there is evidence consistent with limited vocabulary coverage or a rate-distortion constraint, not proof of a hard capacity ceiling. Distinguish this from optimization and PAD dilution with the available diagnostics before proposing the next experiment. The finest codebook has 8 bits per position, but coarser levels jointly contribute capacity, so that figure alone does not establish a per-token impossibility.

### T1: one loss ablation, only if T0 misses the gate

Keep T0's rows, initialization seed, architecture, steps, optimizer, and checkpoint rule. Replace reconstruction CE over all positions with CE averaged over non-PAD positions, retaining EOS and leaving VQ loss unchanged. Continue reporting losses separately: this changes the reconstruction/VQ balance as well as removing PAD supervision, so a gain would not isolate a single mechanism. Log pre-quantization encoder latent norms and, at a declared sampling cadence, encoder gradient norms from reconstruction and VQ losses separately and their ratio; record the measurement method and handle zero VQ gradients explicitly. These diagnostics can identify a gradient-scale shift, but the ratio cannot by itself establish instability or causation. Do not assume the gradient increases by the inverse non-PAD fraction: PAD gradients need not be negligible. No name weighting, codebook change, encoder change, or decoder change in this arm.

If neither T0 nor T1 meets the gate, stop before generator training. Deliver a negative result with name-level errors and a proposal for one new diagnostic (for example, a separately trained continuous-autoencoder control). Do not claim that this one failed packet proves NCM impossible, or spend on a larger generator to conceal the reconstruction failure.

If an arm passes, freeze that arm and its checkpoint-selection rule. Replicate it with initialization/sampling seeds 1 and 2 on the exact same data before making a generalization claim. Replication is a separately recorded bounded packet, not permission for an open-ended sweep.

## 6. Phase C: conditional-generation study, only after the tokenizer gate

Freeze a qualifying tokenizer. Build a new provenance-validated code/embedding cache for every distinct checkpoint and tokenized training manifest. Never reuse a cache merely because tensor shapes or filenames match.

Run an initial 3,000-update NCM pilot and a 3,000-update AR control, batch 128, on identical generator examples: 384,000 example exposures per model. Keep the NCM architecture, zero condition dropout, zero conditioning noise, and greedy decode as the reference configuration. Record and test the prefix-padding-mask contract: legacy training is unmasked while generation masks prefix PAD keys. Evaluate the same reference checkpoint and tuning examples with inference prefix PAD masking both enabled and disabled, holding all other decoding settings fixed; record both results and verify the masks actually applied. This zero-training comparison tests sensitivity to the train/inference mask mismatch, not whether it alone caused the historical collapse. Any change to training masking is an explicit separate arm rather than a silent change to the reference.

Evaluate each model's predeclared 500/1,000/1,500/2,000/2,500/3,000-update snapshots on tuning data; select highest mean typed entity F1, breaking ties toward the earlier checkpoint. Apply the same frozen rule to all seeds. Never use the reserved test sample for arm or checkpoint selection. Proposed continuation gate for a longer generator campaign requires both NCM mean typed entity F1 at least 10% on tuning and a paired 95% interval for its improvement over the grounded dictionary entirely above zero. The dictionary comparison is the primary baseline hurdle; 10% is an absolute floor, not a substitute for beating the baseline. Use the Phase A dictionary tuning score as the known reference. This is a research-spending gate, not deployment approval. If it fails, diagnose source conditioning (correct versus shuffled/null prefix), teacher-forced versus free-running errors, and train/evaluation prefix-mask parity using the zero-training inference comparison; do not automatically run the registered 12,000-step default.

For a comparative claim, freeze the selected recipe, run seeds 0/1/2 with the data fixed, and evaluate once on the protected test sample. Report each seed and its mean/spread. Use paired sentence-group bootstrap intervals with a fixed resampling seed, and do not count seeds as thousands of independent examples. Report the dictionary and AR controls beside NCM. A proposed architecture-competitiveness criterion is beating the grounded dictionary and not trailing AR by more than a predeclared 2 percentage points; distinguish an observed difference from statistical evidence for that margin. This is still not an application-specific production quality target.

If a good tokenizer plus AR succeeds but NCM fails, the next question is NCM conditioning/coarse-to-fine exposure bias. If both generators fail, do not attribute failure specifically to NCM. If neither learned model beats the grounded dictionary, stop promoting a learned extraction pipeline on these data.

## 7. Execution instructions and traps for the next agent

Start a new worktree from merged `main`; preserve the cache and sampling fixes. Work from current source, not the historical archive. Read the files linked above before editing.

### Existing entrypoints are historical, not a new research runner

- `bash autoresearch.sh` currently triggers `ncm-fewnerd-cache-probe/evaluate-saved-probe`. It replays saved checkpoints from the old source/data protocol; it does **not** train the checked-out code or honor `NCM_TASK=T` as a generic task selector.
- The pinned `1.0.3` input predates `main`'s sampling fix and the post-merge missing-cache guard. Replaying it does not validate those changes on merged `main`. Create a new immutable source/data archive containing the merged revision and new tests, record its SHA-256, and update the resource/file references and pins together.
- [fewnerd_probe.py](../scripts/fewnerd_probe.py) hardcodes 1,500/3,000 steps, seeds, batch/LR, checkpoint expectations, and the old evaluation callback. It overwrites `NCM_STEPS`. The ordinary tokenizer trainer has no `NCM_STEPS` override. Implement explicit experiment configuration and frozen-manifest inputs before attempting the proposed schedules; do not invent CLI flags or assume an exported variable changed the run.
- The registry's Few-NERD defaults are 6,000 tokenizer / 12,000 generator steps, not what the short probe ran. Changing a default must not silently change historical replay.
- The pipeline's `full-probe` job hardcodes `fewnerd/runs/20261031-probe1` for checkpoint uploads. **Do not use that prefix for a new experiment.** Give each arm/seed a unique immutable artifact namespace, source archive, manifest, and output directory. Do not overwrite the existing checkpoints or pins to make a new result look like the old one.
- Snapshots contain model, optimizer, completed-step count, and Torch CPU/CUDA RNG state; there is no implemented general resume CLI. Verify the complete resume/data-order contract before calling a continued run an exact continuation.

### GPU and storage contract

Adapt [fewnerd-probe.yml](../concourse/fewnerd-probe.yml) using the [single-GPU workload template](https://github.com/gavmor/genops-pipeline/blob/main/concourse/gpu-workload-template.yml) as the operational reference; do not replace the probe's provenance, checkpoint, cache, or MinIO requirements with the generic template. Record the template revision used with each experiment.

1. Before acquiring the lock, package current source, fixed manifests, cached pre-tokenizer files, corpus identities, and resolved configs; exclude credentials. Fetch/checksum and validate inputs, expected RAM, disk, and storage. Prepare dependencies and any required ahead-of-time CUDA extensions outside the lock; validate imports there. The template's `TORCH_CUDA_ARCH_LIST=8.6` specifies a build target but does not itself compile extensions.
2. Use the purpose-built dependency image pinned by digest in both preparation and execution. Reuse it only if dependencies are unchanged; otherwise build/version/pin a new image before GPU work. If transferring a prepared Python environment as in the template, install non-editably, recreate the same `/tmp/workload-pybase` binding in both tasks, and verify imports from the transferred volume. Do not install dependencies under the lock. Record the image digest, source commit/archive hash, corpus hashes, resolved dependencies, precision, and relevant backend settings for every run.
3. Run CUDA as an ordinary foreground Concourse task on the existing untagged GPU worker, with `NVIDIA_VISIBLE_DEVICES=all`, holding the same Git-backed `gpu-lock` as other participating jobs. Validate CUDA visibility before launching the workload and preserve its exit status through logging. The lock coordinates participating jobs only. Do not run host-shell CUDA, detached containers, worker-namespace workloads, or ComfyUI calls. Do not use `fly execute` as though it acquired the lock.
4. Retain explicit RAM/VRAM budgeting. The current task requests 32 GiB host RAM; preflight requires 24 GiB available host RAM and 20 GiB free task disk. These numbers do not reserve VRAM; measure peaks before expanding models. Capture wall time, GNU time peak RSS, GPU utilization and sampled VRAM as in the template; identify the sampling interval and treat sampled peak VRAM as a lower bound on the actual peak. Coordinate a fleet-wide migration before introducing GPU worker tags.
5. Persist model/optimizer/RNG snapshots during training, not only after success. Declare task outputs and preserve failures, raw predictions, preparation logs, and telemetry. Release the lock first through sequential `ensure`, then package and explicitly publish results to a unique private MinIO destination; configure the artifact resource and its file parameter for the actual resource type. Cleanup and publication are best effort: a failed release stops the template's sequence, and worker or backend loss can prevent upload. Inspect and recover a failed release before retrying; on worker loss, verify process death before any manual stranded-lock release. Periodic durable checkpoints remain necessary because post-run publication cannot protect against worker loss.
6. Keep cache verification meaningful: exact codes on representative rebatching; code/embedding checks across complete original 64-row batches with construction's `cudnn.benchmark=False` policy; embeddings retain `rtol=1e-5, atol=1e-6`. Restore decoder settings afterward. Do not restore the invalid cross-shape floating-point equality check or loosen tolerances to hide a real mismatch. Rebuild provenance-free/stale caches.

When adapting template shell or comments, avoid literal double parentheses that Concourse would interpolate as pipeline variables. Use an insecure image registry only when it is trusted, and validate the adapted pipeline before scheduling.

On this workstation, existing scheduler/storage variables are under `/home/user/code/genops-pipelines/concourse/vars.default.yml` and `.secrets/vars.yml`. Use them through the established tooling; never print or commit credential values. Do not push code, publish a model, deploy, or merge a model-quality claim without the corresponding explicit approval.

Safe CPU/preflight checks, from the new checkout:

```bash
uv sync --offline --frozen
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=third_party/ncm:src \
  uv run --offline python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=third_party/ncm:src uv run --offline python scripts/autoresearch_cache_benchmark.py
fly validate-pipeline -c concourse/fewnerd-probe.yml
```

These validate engineering behavior; they are not new quality measurements. GPU commands must be scheduled through the lock-protected task after the new experiment configuration is implemented and validated.

## 8. Required handoff and stop conditions

The next agent should first deliver Phase A's immutable manifests, scorer controls, explicit runner configuration, and the bounded Phase B tokenizer report. Include every attempted arm and failure, not only a best checkpoint. For each run publish:

- hypothesis and single intended intervention; source/image/data/checkpoint identities;
- selection versus initialization seeds, sample counts, token bounds, loss definition, masks, backend policy, steps, batch size, exposures, and measured compute/memory;
- train/tuning curves, selected-checkpoint rule, name-specific errors, raw reconstructions/predictions, controls, and uncertainty;
- cache acceptance evidence and storage locations; exact run/Concourse status;
- a go/no-go decision for the next phase, limitations, and unperformed work.

Stop immediately on scoring inconsistency, input-group leakage, cache mismatch, non-finite optimization, or inability to persist checkpoints. Stop the tokenizer packet at its budget if the gate fails. Proceed to generator work only on the declared gate, not because a cache microbenchmark or teacher-forced accuracy passed. Reserve a later proposal for scaling, target redesign, or corpus changes.

A successful outcome of this handoff can be a well-supported negative result and a decision to prefer a simpler extractor. It must not be another claim of successful extraction based only on valid-looking tuples, code accuracy, or a completed training job.

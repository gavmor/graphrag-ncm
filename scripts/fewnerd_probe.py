"""Full Few-NERD probe; run only as a gpu-lock-protected Concourse task.

--evaluate-only reuses the durable 1,500/3,000-step checkpoints without retraining.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import boto3
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ncm_graphrag.data.extraction import parse_records
from ncm_graphrag.ncm import train_generator, train_tokenizer
from ncm_graphrag.ncm.generate import NCMTaskGenerator
from ncm_graphrag.ncm.generator_cache import load_or_build_cache
from ncm_graphrag.ncm.task_registry import TASKS

TASK = "fewnerd_extraction"


def entity_scores(gold_text: str, predicted_text: str) -> tuple[float, bool]:
    def entities(text):
        return {(e["name"].strip().upper(), e["type"].strip().upper()) for e in parse_records(text)["entities"]}

    gold, predicted = entities(gold_text), entities(predicted_text)
    return 2 * len(gold & predicted) / max(len(gold) + len(predicted), 1), gold == predicted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_probe(root, spec, tokenizer_cfg, model_cfg):
    bucket = os.environ["NCM_CHECKPOINT_BUCKET"]
    prefix = os.environ["NCM_CHECKPOINT_PREFIX"]
    storage = boto3.client("s3", endpoint_url=os.environ["NCM_S3_ENDPOINT"])
    storage.head_bucket(Bucket=bucket)
    active: dict[str, Any] = {}

    class SnapshotTokenizer(train_tokenizer.VQVAE):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            active["model"] = self

    class SnapshotGenerator(train_generator.NextConceptTransformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            active["model"] = self

    class SnapshotAdamW(torch.optim.AdamW):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.completed_steps = 0
            model, stage = active["model"], active["stage"]

            def save_step(optimizer, args, kwargs):
                optimizer.completed_steps += 1
                if optimizer.completed_steps % 500:
                    return
                checkpoint = root / f"{stage}-resume.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "completed_steps": optimizer.completed_steps,
                        "stage": stage,
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    },
                    checkpoint,
                )
                key = f"{prefix}/checkpoints/{stage}-{optimizer.completed_steps:06d}.pt"
                storage.upload_file(str(checkpoint), bucket, key)
                print(f"Persisted checkpoint: s3://{bucket}/{key}", flush=True)

            self.register_step_post_hook(save_step)

    setattr(train_tokenizer, "VQVAE", SnapshotTokenizer)
    setattr(train_generator, "NextConceptTransformer", SnapshotGenerator)
    setattr(torch.optim, "AdamW", SnapshotAdamW)
    manifest = {
        "task": TASK,
        "tokenizer_steps": 1500,
        "generator_steps": 3000,
        "tokenizer_examples": spec.tokenizer_n_examples,
        "generator_examples": spec.generator_n_examples,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "tokenizer_seed": 0,
        "generator_seed": 1,
        "evaluation_seed": 42,
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "tokenizer_config": OmegaConf.to_container(tokenizer_cfg, resolve=True),
        "model_config": OmegaConf.to_container(model_cfg, resolve=True),
        "generation_config": OmegaConf.to_container(spec.build_generation_cfg(), resolve=True),
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__),
                Path(train_tokenizer.__file__),
                Path(train_generator.__file__),
                Path("src/ncm_graphrag/ncm/generator_cache.py"),
                Path("src/ncm_graphrag/ncm/config.py"),
                Path("third_party/ncm/models/quant.py"),
            )
        },
        "corpus_sha256": {
            split: sha256(
                Path(os.environ["NCM_CORPORA_ROOT"])
                / "information_extraction"
                / "few_nerd"
                / "supervised"
                / f"{split}-00000-of-00001.parquet"
            )
            for split in ("train", "validation")
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sys.argv = [sys.argv[0], "--task", TASK]
    start = time.monotonic()
    print("STAGE tokenizer: 1500 steps", flush=True)
    active["stage"] = "tokenizer"
    train_tokenizer.main()
    storage.upload_file(tokenizer_cfg.save_path, bucket, f"{prefix}/tokenizer.pt")
    active.clear()
    print("STAGE generator: 3000 steps with validated cache", flush=True)
    active["stage"] = "generator"
    train_generator.main()
    storage.upload_file(model_cfg.save_path, bucket, f"{prefix}/generator.pt")
    active.clear()
    return time.monotonic() - start


@torch.no_grad()
def cache_diagnostics(generator, spec, tokenizer_cfg, model_cfg, root):
    prefix, target = train_generator._build_dataset(
        spec, model_cfg.prefix_len, tokenizer_cfg.vqvae.context_length, generator.pre_tok
    )
    # This exercises the production loader against the exact complete dataset,
    # checkpoint bytes, model state and saved tensor checksums before comparing.
    cache = load_or_build_cache(generator.vqvae, tokenizer_cfg.save_path, prefix, target, root / "generator.cache.pt")
    selected = torch.tensor([0, 1, 63, 64, 1000, len(target) // 2, len(target) - 1])
    reference_idx, reference_rest = cache["idx"][selected], cache["rest_embed"][selected]
    diagnostics: dict[str, Any] = {}
    matching = None
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        for benchmark in (False, True):
            torch.backends.cudnn.benchmark = benchmark
            for geometry in ("original_64", "selected_7"):
                if geometry == "selected_7":
                    idx = generator.vqvae.encode(target[selected].to(generator.device))
                    rest = generator.vqvae.vq.idx_to_ncm_input(idx)
                else:
                    indices, embeddings = [], []
                    # Reconstruct each original cache batch, then select its row.
                    for row in selected.tolist():
                        start = (row // 64) * 64
                        batch_idx = generator.vqvae.encode(target[start : start + 64].to(generator.device))
                        batch_rest = generator.vqvae.vq.idx_to_ncm_input(batch_idx)
                        indices.append(batch_idx[row - start].cpu())
                        embeddings.append(batch_rest[row - start].cpu())
                    idx, rest = torch.stack(indices), torch.stack(embeddings)
                idx, rest = idx.cpu(), rest.cpu()
                delta = rest - reference_rest
                name = f"benchmark_{benchmark}_{geometry}"
                diagnostics[name] = {
                    "codes_equal": torch.equal(idx, reference_idx),
                    "embedding_max_abs_error": float(delta.abs().max()),
                    "embedding_relative_l2_error": float(delta.norm() / reference_rest.norm()),
                    "embedding_strict_close": torch.allclose(rest, reference_rest, rtol=1e-5, atol=1e-6),
                }
                if not benchmark and geometry == "original_64":
                    matching = (idx, rest)
                if benchmark and geometry == "selected_7":
                    p = prefix[selected].to(generator.device)
                    z = generator.vqvae.encoder.wte(p)
                    query = generator.net.build_l0_query_input(len(selected), dtype=rest.dtype)
                    cached_logits = generator.net(torch.cat((z, query, reference_rest.to(generator.device)), dim=1))
                    fresh_logits = generator.net(torch.cat((z, query, rest.to(generator.device)), dim=1))
                    labels = idx.to(generator.device).reshape(-1)
                    diagnostics["generator_consumer"] = {
                        "max_abs_logit_delta": float((cached_logits - fresh_logits).abs().max()),
                        "predicted_codes_equal": torch.equal(cached_logits.argmax(-1), fresh_logits.argmax(-1)),
                        "cross_entropy_delta": float(
                            abs(
                                F.cross_entropy(cached_logits.reshape(-1, cached_logits.shape[-1]), labels)
                                - F.cross_entropy(fresh_logits.reshape(-1, fresh_logits.shape[-1]), labels)
                            )
                        ),
                    }
    finally:
        torch.backends.cudnn.benchmark = original_benchmark
    # Persist diagnostics before asserting: a failed check must not lose evidence.
    (root / "cache-diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print("CACHE_DIAGNOSTICS " + json.dumps(diagnostics), flush=True)
    assert matching is not None
    torch.testing.assert_close(matching[0], reference_idx, rtol=0, atol=0)
    torch.testing.assert_close(matching[1], reference_rest, rtol=1e-5, atol=1e-6)
    return diagnostics


def evaluate_probe(root, spec, tokenizer_cfg, model_cfg, training_seconds):
    print("STAGE held-out validation: registered 300 examples, seed 42", flush=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    generator = NCMTaskGenerator(task=TASK, compile_blocks=False)
    assert spec.benchmark_examples is not None
    examples = spec.benchmark_examples()
    rows: list[dict[str, Any]] = []
    token_correct = token_count = code_correct = code_count = 0
    with torch.inference_mode():
        for i, example in enumerate(examples):
            target_ids = generator.pre_tok.encode(example.target_text) + [generator.eos_id]
            padded_ids = target_ids + [generator.pad_id] * (int(tokenizer_cfg.vqvae.context_length) - len(target_ids))
            target = torch.tensor([padded_ids], device=generator.device)
            idx = generator.vqvae.encode(target)
            rest = generator.vqvae.vq.idx_to_ncm_input(idx)
            recon_ids = generator.vqvae.decode(idx).argmax(-1)
            token_correct += int((recon_ids[0, : len(target_ids)] == target[0, : len(target_ids)]).sum())
            token_count += len(target_ids)
            recon = generator._dec(recon_ids[0].tolist())
            prefix = generator._prefix_tensor(example.prefix_text)
            z = generator.vqvae.encoder.wte(prefix)
            x = torch.cat((z, generator.net.build_l0_query_input(1, dtype=rest.dtype), rest), dim=1)
            code_correct += int((generator.net(x).argmax(-1) == idx).sum())
            code_count += idx.numel()
            output = generator.generate(example.prefix_text)
            f1, exact = entity_scores(example.target_text, output)
            recon_f1, _ = entity_scores(example.target_text, recon)
            rows.append(
                {
                    "prefix": example.prefix_text,
                    "gold": example.target_text,
                    "prediction": output,
                    "tokenizer_reconstruction": recon,
                    "entity_f1": f1,
                    "exact_match": exact,
                    "valid": generator.is_well_formed(output),
                    "tokenizer_entity_f1": recon_f1,
                }
            )
            if (i + 1) % 50 == 0:
                print(f"evaluated {i + 1}/{len(examples)}", flush=True)
    count = len(rows)
    metrics = {
        "entity_f1": 100 * sum(r["entity_f1"] for r in rows) / count,
        "exact_match_rate": 100 * sum(r["exact_match"] for r in rows) / count,
        "valid_rate": sum(r["valid"] for r in rows) / count,
        "tokenizer_entity_f1": 100 * sum(r["tokenizer_entity_f1"] for r in rows) / count,
        "tokenizer_nonpad_accuracy": token_correct / token_count,
        "heldout_code_accuracy": code_correct / code_count,
    }
    if training_seconds is not None:
        metrics["training_seconds"] = training_seconds
    report = {
        "metrics": metrics,
        "n_validation": count,
        "split": "validation",
        "seed": 42,
        "evaluation_source_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": {name: sha256(root / name) for name in ("tokenizer.pt", "generator.pt")},
        "examples": rows,
    }
    (root / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    # Do not lose the evaluation if a separate cache diagnostic fails.
    report["cache_diagnostics"] = cache_diagnostics(generator, spec, tokenizer_cfg, model_cfg, root)
    metrics["representative_cache_parity"] = 1
    metrics["probe_completed"] = 1
    (root / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, value in metrics.items():
        print(f"METRIC {name}={value:.8f}", flush=True)
    print(f"Artifacts: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("NCM_GPU_LOCK_HELD") != "1" or not torch.cuda.is_available():
        raise RuntimeError("Run this probe in a CUDA Concourse task holding gpu-lock")
    root = args.output.resolve()
    if args.evaluate_only:
        # The post-step snapshots prove the requested training actually finished.
        for stage, expected in (("tokenizer", 1500), ("generator", 3000)):
            snapshot = torch.load(root / f"{stage}-resume.pt", map_location="cpu", weights_only=True)
            assert snapshot["completed_steps"] == expected
            del snapshot
    else:
        root.mkdir(parents=True, exist_ok=False)
    original = TASKS[TASK]
    if args.evaluate_only:
        manifest = json.loads((root / "manifest.json").read_text())
        tokenizer_cfg = OmegaConf.create(manifest["tokenizer_config"])
        model_cfg = OmegaConf.create(manifest["model_config"])
    else:
        tokenizer_cfg, model_cfg = original.build_tokenizer_cfg(), original.build_model_cfg()
    tokenizer_cfg.save_path = str(root / "tokenizer.pt")
    model_cfg.save_path = str(root / "generator.pt")
    spec = dataclasses.replace(
        original,
        tokenizer_steps=1500,
        generator_steps=3000,
        build_tokenizer_cfg=lambda: tokenizer_cfg,
        build_model_cfg=lambda: model_cfg,
    )
    TASKS[TASK] = spec
    train_generator.BATCH_SIZE = 128
    train_generator.LR = 3e-4
    train_generator.COND_DROP = 0.0
    train_generator.TRAIN_PAD_MASK = False
    train_generator.SEED = 1
    os.environ["NCM_STEPS"] = "3000"
    os.environ["NCM_EVAL_EVERY"] = "1000"
    os.environ["NCM_GENERATOR_CACHE"] = str(root / "generator.cache.pt")
    training_seconds = None if args.evaluate_only else train_probe(root, spec, tokenizer_cfg, model_cfg)
    evaluate_probe(root, spec, tokenizer_cfg, model_cfg, training_seconds)


if __name__ == "__main__":
    main()

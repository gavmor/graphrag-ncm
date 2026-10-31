"""Trains the multiscale residual-VQ tokenizer on a task's target text (bounded JSON
for "mapstage", graphrag's delimiter-tuple grammar for "extraction" -- see
src/ncm_graphrag/ncm/task_registry.py).

No official training script exists upstream for this repo (see docs/constraints.md) --
this reimplements the standard VQ-VAE-with-EMA-codebook recipe directly against
models.vqvae.VQVAE: reconstruction cross-entropy + the commitment/codebook loss the
module already computes in `forward()`. The codebook itself is EMA-updated inside
`MultiscaleResidualVectorQuantizer.forward()` when `self.training` is True (see
models/quant.py:485-500) -- the optimizer only needs to step the encoder/decoder/
quant_resi parameters (codebook embeddings have `requires_grad=False`).
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from models.vqvae import VQVAE
from omegaconf import DictConfig, OmegaConf
from tokenization.hf_tokenizer import HFTokenizer

from ncm_graphrag.data.fewnerd_quality_metrics import entities, name_spans
from ncm_graphrag.ncm.task_registry import TASKS, TaskSpec

BATCH_SIZE = 128
LR = 3e-4
SEED = 0


def _build_dataset(task: TaskSpec, cfg: DictConfig, pre_tok) -> torch.Tensor:
    examples = task.generate_examples(task.tokenizer_n_examples, SEED)
    context_length = cfg.vqvae.context_length
    rows = []
    for ex in examples:
        ids = pre_tok.encode(ex.target_text) + [pre_tok.eos_token_id]
        if len(ids) > context_length:
            raise ValueError(f"target exceeds context_length={context_length} ({len(ids)} tokens): {ex.target_text!r}")
        ids = ids + [pre_tok.pad_token_id] * (context_length - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def _overfit_inputs(root: Path):
    """Resolve the fixed control without sampling or inspecting test predictions."""
    manifest = json.loads((root / "frozen.json").read_text())
    for filename, expected in manifest["files"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen input checksum mismatch: {filename}")
    pools = {
        split: [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        for split in ("generator-train", "tokenizer-train", "tuning")
    }
    groups = {split: {row["group_id"] for row in rows} for split, rows in pools.items()}
    if any(len(groups[split]) != len(rows) for split, rows in pools.items()):
        raise ValueError("Duplicate groups in frozen inputs")
    if not groups["tokenizer-train"] <= groups["generator-train"]:
        raise ValueError("Tokenizer groups are not a subset of generator training")
    if groups["generator-train"] & groups["tuning"]:
        raise ValueError("Training/tuning leakage")
    if len(pools["tokenizer-train"]) != 6000 or len(pools["tuning"]) != 1000:
        raise ValueError("The study requires 6000 tokenizer and 1000 tuning groups")
    config = TASKS["fewnerd_extraction"].build_tokenizer_cfg()
    if OmegaConf.to_container(config.vqvae, resolve=True) != manifest["resolved_configs"]["vqvae"]:
        raise ValueError("Tokenizer architecture differs from the frozen protocol")
    tokenizer = HFTokenizer(model_id=str(root / "tokenizer"), pad_token="<|pad|>", local_files_only=True)
    backend = tokenizer.hf_tokenizer
    if backend is None:
        raise RuntimeError("Local GPT-2 tokenizer could not be loaded")
    rows = pools["tokenizer-train"][:32]
    multi_subword = 0
    for row in rows:
        encoded = backend(row["target_text"], return_offsets_mapping=True)
        if encoded["input_ids"] + [tokenizer.eos_token_id] != row["target_ids"]:
            raise ValueError("Target IDs do not match the actual serialized target")
        if not 1 <= len(row["target_ids"]) <= 80:
            raise ValueError("Overfit target violates the target80 bound")
        for _, _, start, end in name_spans(row["target_text"]):
            positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
            if not positions:
                raise ValueError("A serialized name has no aligned GPT-2 tokens")
            multi_subword += len(positions) > 1
    if multi_subword == 0:
        raise ValueError("Tiny-set control requires multi-subword names")
    issues = []
    reserved = manifest["selection"]["reserved_test"]
    if reserved.get("status") != "frozen" or reserved.get("history_clearance") != "confirmed":
        issues.append("Phase A protected-test selection and prior-use clearance are incomplete")
    else:
        filename = "reserved-test.jsonl"
        if filename not in manifest["files"]:
            raise ValueError("Frozen protected-test manifest has no pinned file")
        test_rows = [json.loads(line) for line in (root / filename).read_text().splitlines()]
        test_groups = {row["group_id"] for row in test_rows}
        if len(test_rows) != 1000 or len(test_groups) != 1000:
            raise ValueError("Protected-test manifest must have 1000 unique groups")
        if test_groups & (groups["generator-train"] | groups["tuning"]):
            raise ValueError("Protected-test input-group leakage")
    resolved = {
        "arm": "tiny-overfit",
        "steps": 2000,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "initialization_seed": 0,
        "sampling": "all 32 rows, in stored order, every update",
        "selection": "first 32 frozen tokenizer rows; no new data selection",
        "loss": "mean CE over all 80 positions plus unchanged VQ loss",
        "checkpoint_every": 100,
        "multi_subword_names": multi_subword,
        "training_manifest_sha256": manifest["files"]["tokenizer-train.jsonl"],
        "selected_group_ids": [row["group_id"] for row in rows],
        "selected_targets_sha256": hashlib.sha256(
            json.dumps([row["target_ids"] for row in rows], separators=(",", ":")).encode()
        ).hexdigest(),
        "vqvae_config": OmegaConf.to_container(config.vqvae, resolve=True),
        "training_ready": not issues,
        "protocol_issues": issues,
    }
    return rows, tokenizer, config, resolved


def _overfit_update(model, optimizer, batch):
    logits, vq_loss, _, utilization, _ = model(batch)
    reconstruction = F.cross_entropy(logits.flatten(0, 1), batch.flatten())
    loss = reconstruction + vq_loss
    if not torch.isfinite(loss):
        raise ValueError("Non-finite tiny-set optimization loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {
        "reconstruction_ce_all_positions": float(reconstruction.detach()),
        "vq_loss": float(vq_loss.detach()),
        "gradient_norm_before_clip": float(gradient_norm),
        "global_utilization": utilization,
    }


@torch.inference_mode()
def _overfit_reconstruction(model, rows, tokenizer):
    model.eval()
    device = next(model.parameters()).device
    target = torch.tensor(
        [row["target_ids"] + [tokenizer.pad_token_id] * (80 - len(row["target_ids"])) for row in rows],
        device=device,
    )
    logits = model.decode(model.encode(target))
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite tiny-set reconstruction")
    counts = {
        "examples": len(rows),
        "entity_set_exact": 0,
        "names": 0,
        "names_exact": 0,
        "multi_subword_names": 0,
        "multi_subword_names_exact": 0,
    }
    predictions = []
    for row, ids in zip(rows, logits.argmax(-1).tolist(), strict=True):
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id)]
        prediction = tokenizer.decode([i for i in ids if i != tokenizer.pad_token_id], skip_special_tokens=True)
        predicted = entities(prediction)
        predicted_names = {name for name, _ in predicted}
        counts["entity_set_exact"] += predicted == entities(row["target_text"])
        encoded = tokenizer.hf_tokenizer(row["target_text"], return_offsets_mapping=True)
        for name, _, start, end in name_spans(row["target_text"]):
            n_tokens = sum(a < end and b > start for a, b in encoded["offset_mapping"])
            exact = name.strip().upper() in predicted_names
            counts["names"] += 1
            counts["names_exact"] += exact
            counts["multi_subword_names"] += n_tokens > 1
            counts["multi_subword_names_exact"] += exact and n_tokens > 1
        predictions.append({"group_id": row["group_id"], "gold": row["target_text"], "prediction": prediction})
    exact_sets = 100 * counts["entity_set_exact"] / counts["examples"]
    exact_multi = 100 * counts["multi_subword_names_exact"] / counts["multi_subword_names"]
    return {
        "counts": counts,
        "entity_set_exact_percent": exact_sets,
        "multi_subword_exact_name_recall_percent": exact_multi,
        "gate_passed": exact_sets >= 95 and exact_multi >= 95,
        "predictions": predictions,
        "limitation": "Fit on these same 32 targets only; not generalization or arbitrary-name copying.",
    }


def _run_overfit(root: Path, output: Path | None, preflight: bool):
    rows, tokenizer, config, resolved = _overfit_inputs(root)
    print("OVERFIT_CONFIG " + json.dumps(resolved, sort_keys=True), flush=True)
    if preflight:
        return
    if not resolved["training_ready"]:
        raise RuntimeError("; ".join(resolved["protocol_issues"]))
    if os.environ.get("NCM_GPU_LOCK_HELD") != "1" or not torch.cuda.is_available():
        raise RuntimeError("Run the research control in a foreground CUDA Concourse task holding gpu-lock")
    if output is None:
        raise ValueError("--output-dir must identify a new, unique experiment directory")
    # Durable uploads are mandatory; validate the new namespace before CUDA allocation.
    import boto3

    bucket = os.environ["NCM_CHECKPOINT_BUCKET"]
    prefix = os.environ["NCM_CHECKPOINT_PREFIX"].strip("/")
    if not prefix or prefix.startswith("fewnerd/runs/20261031-probe1"):
        raise ValueError("The research control requires a new artifact namespace")
    store = boto3.client("s3", endpoint_url=os.environ["NCM_S3_ENDPOINT"])
    store.head_bucket(Bucket=bucket)
    if store.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1).get("KeyCount", 0):
        raise FileExistsError("Refusing to overwrite an existing artifact namespace")
    output.mkdir(parents=True, exist_ok=False)
    resolved["image_digest"] = os.environ["NCM_IMAGE_DIGEST"]
    resolved["input_archive_sha256"] = os.environ["NCM_INPUT_SHA256"]
    resolved["torch"] = str(torch.__version__)
    (output / "manifest.json").write_text(json.dumps(resolved, indent=2) + "\n")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = VQVAE(**config.vqvae).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    batch = torch.tensor(
        [row["target_ids"] + [tokenizer.pad_token_id] * (80 - len(row["target_ids"])) for row in rows],
        device="cuda",
    )

    def snapshot(completed_steps):
        path = output / f"tokenizer-{completed_steps:06d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "completed_steps": completed_steps,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "resolved_config": resolved,
            },
            temporary,
        )
        temporary.replace(path)
        store.upload_file(str(path), bucket, f"{prefix}/checkpoints/{path.name}")
        print(f"Persisted checkpoint: s3://{bucket}/{prefix}/checkpoints/{path.name}", flush=True)

    snapshot(0)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (output / "curve.jsonl").open("x") as curve:
        for step in range(1, 2001):
            diagnostics = _overfit_update(model, optimizer, batch)
            curve.write(json.dumps({"completed_steps": step, **diagnostics}) + "\n")
            curve.flush()
            if step % 100 == 0:
                snapshot(step)
                print(json.dumps({"completed_steps": step, **diagnostics}), flush=True)
    torch.cuda.synchronize()
    training_wall_seconds = time.monotonic() - started
    report = _overfit_reconstruction(model, rows, tokenizer)
    report["training_wall_seconds_including_checkpoints"] = training_wall_seconds
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["completed_steps"] = 2000
    (output / "overfit-report.json").write_text(json.dumps(report, indent=2) + "\n")
    for filename in ("manifest.json", "curve.jsonl", "overfit-report.json"):
        store.upload_file(str(output / filename), bucket, f"{prefix}/{filename}")
    if not report["gate_passed"]:
        raise RuntimeError("Tiny-set reconstruction gate failed; diagnose before T0 or any generator training")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="mapstage")
    ap.add_argument("--overfit-control", action="store_true", help="Run the fixed Few-NERD 32-target control")
    ap.add_argument("--frozen-inputs", type=Path, help="Directory containing the pinned study manifests")
    ap.add_argument("--output-dir", type=Path, help="New unique directory for control outputs")
    ap.add_argument("--preflight", action="store_true", help="Resolve the control and report protocol readiness only")
    args = ap.parse_args()
    if args.overfit_control:
        if args.task != "fewnerd_extraction" or args.frozen_inputs is None:
            ap.error("--overfit-control requires --task fewnerd_extraction and --frozen-inputs")
        _run_overfit(args.frozen_inputs, args.output_dir, args.preflight)
        return
    if args.frozen_inputs is not None or args.output_dir is not None or args.preflight:
        ap.error("Research input/output/preflight options require --overfit-control")
    task = TASKS[args.task]

    torch.manual_seed(SEED)
    cfg = task.build_tokenizer_cfg()
    pre_tok = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")

    data = _build_dataset(task, cfg, pre_tok)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqvae = VQVAE(**cfg.vqvae).to(device)
    vqvae.train()

    opt = torch.optim.AdamW(vqvae.parameters(), lr=LR, weight_decay=0.01)
    n = data.shape[0]

    for step in range(task.tokenizer_steps):
        batch_idx = torch.randint(0, n, (BATCH_SIZE,))
        batch = data[batch_idx].to(device)

        x_recon, vq_loss, _per_level_util, global_util, _ppl = vqvae(batch)
        recon_loss = F.cross_entropy(x_recon.reshape(-1, x_recon.shape[-1]), batch.reshape(-1))
        loss = recon_loss + vq_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == task.tokenizer_steps - 1:
            acc = (x_recon.argmax(-1) == batch).float().mean().item()
            print(
                f"[{args.task}] step {step:5d}  loss {loss.item():.4f}  recon {recon_loss.item():.4f}  "
                f"vq {vq_loss.item():.4f}  token_acc {acc:.4f}  codebook_util {global_util:.3f}",
                flush=True,
            )

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    torch.save({"model": vqvae.state_dict()}, cfg.save_path)
    print(f"saved {cfg.save_path}")


if __name__ == "__main__":
    main()

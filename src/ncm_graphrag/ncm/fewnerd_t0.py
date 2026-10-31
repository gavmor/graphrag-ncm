"""One frozen T0 schedule; GPU training and post-lock CPU selection are separate."""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from models.vqvae import VQVAE

from ncm_graphrag.data.fewnerd_quality_metrics import entities, name_spans, score_rows
from ncm_graphrag.ncm.fewnerd_quality import reconstruct
from ncm_graphrag.ncm.train_tokenizer import _overfit_inputs, _overfit_update

EVALUATION_STEPS = (1500, 3000, 6000)
FIXED_RECIPE = {
    "schema_version": 1,
    "arm": "T0",
    "steps": 6000,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "initialization_seed": 0,
    "sampling_seed": 0,
    "checkpoint_every": 500,
    "evaluation_steps": list(EVALUATION_STEPS),
    "evaluation_device": "cpu",
    "evaluation_batch_size": 8,
    "evaluation_threads_per_worker": 1,
    "evaluation_workers": 3,
    "checkpoint_selection": "Highest tuning mean per-example typed entity-set F1; ties earlier",
    "continuation_entity_f1_percent": 90,
    "continuation_target_exact_percent": 80,
    "sampling_rng": "Independent CPU torch.Generator; randint with replacement; stored row order",
    "loss": "Mean reconstruction CE over all80 positions plus unchanged VQ loss",
    "precision": "float32; TF32 disabled; deterministic algorithms; cuDNN benchmark false",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_protocol(path):
    path = Path(path).resolve()
    protocol = json.loads(path.read_text())
    for key, expected in FIXED_RECIPE.items():
        if protocol.get(key) != expected:
            raise ValueError(f"T0 fixed protocol mismatch: {key}")
    root = path.parent / protocol["frozen_inputs"]
    _, tokenizer, config, control = _overfit_inputs(root)
    if not control["training_ready"]:
        raise ValueError("; ".join(control["protocol_issues"]))
    frozen = json.loads((root / "frozen.json").read_text())
    for field, filename in (
        ("training_manifest_sha256", "tokenizer-train.jsonl"),
        ("tuning_manifest_sha256", "tuning.jsonl"),
    ):
        if protocol[field] != frozen["files"][filename]:
            raise ValueError(f"T0 data identity mismatch: {field}")
    receipt_path = path.parent / protocol["tiny_control_receipt"]
    if sha256(receipt_path) != protocol["tiny_control_sha256"]:
        raise ValueError("Tiny-control prerequisite checksum mismatch")
    receipt = json.loads(receipt_path.read_text())
    counts = receipt["counts"]
    if (
        receipt["completed_steps"] != 2000
        or counts["examples"] != 32
        or not counts["multi_subword_names"]
        or 100 * counts["entity_set_exact"] / counts["examples"] < 95
        or 100 * counts["multi_subword_names_exact"] / counts["multi_subword_names"] < 95
    ):
        raise ValueError("Tiny-control prerequisite does not pass both declared gates")
    train = [json.loads(line) for line in (root / "tokenizer-train.jsonl").read_text().splitlines()]
    tuning = [json.loads(line) for line in (root / "tuning.jsonl").read_text().splitlines()]
    if [row["group_id"] for row in receipt["predictions"]] != [row["group_id"] for row in train[:32]]:
        raise ValueError("Tiny-control receipt targets differ from the frozen control")
    for row in train + tuning:
        expected = {(name.strip().upper(), kind.strip().upper()) for name, kind, _, _ in name_spans(row["target_text"])}
        if entities(row["target_text"]) != expected:
            raise ValueError("Gold serialization/parser round-trip is inconsistent")
        if tokenizer.encode(row["target_text"]) + [tokenizer.eos_token_id] != row["target_ids"]:
            raise ValueError("Frozen T0 target token IDs differ from actual serialization")
        if len(row["prefix_ids"]) > 64 or not 1 <= len(row["target_ids"]) <= 80:
            raise ValueError("T0 row violates the frozen token bounds")
    oracle = score_rows([row["target_text"] for row in tuning], [row["target_text"] for row in tuning])
    if any(oracle[key] != 100 for key in ("entity_f1", "entity_set_exact", "strict_validity")):
        raise ValueError("T0 scorer oracle failed")
    resolved = {
        **protocol,
        "protocol_sha256": sha256(path),
        "frozen_manifest_sha256": sha256(root / "frozen.json"),
        "training_rows": len(train),
        "tuning_rows": len(tuning),
        "example_exposures": 6000 * 128,
        "vqvae_config": control["vqvae_config"],
        "training_ready": True,
        "protocol_issues": [],
        "scorer_oracle": oracle,
        "evaluation_is_after_gpu_lock_release": True,
        "batch_logits_bytes_float32": 128 * 80 * int(config.vqvae.in_vocab_size) * 4,
    }
    return resolved, train, tuning, tokenizer, config


def train(protocol_path, output):
    resolved, rows, _, tokenizer, config = load_protocol(protocol_path)
    if os.environ.get("NCM_GPU_LOCK_HELD") != "1" or not torch.cuda.is_available():
        raise RuntimeError("T0 research training requires a foreground Concourse CUDA task holding gpu-lock")
    import boto3

    bucket = os.environ["NCM_CHECKPOINT_BUCKET"]
    prefix = os.environ["NCM_CHECKPOINT_PREFIX"].strip("/")
    if not prefix.startswith("fewnerd/quality/runs/t0-"):
        raise ValueError("T0 requires a new, dedicated quality/runs/t0- namespace")
    store = boto3.client("s3", endpoint_url=os.environ["NCM_S3_ENDPOINT"])
    store.head_bucket(Bucket=bucket)
    if store.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1).get("KeyCount", 0):
        raise FileExistsError("Refusing to reuse an existing T0 artifact namespace")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    resolved.update(
        image_digest=os.environ["NCM_IMAGE_DIGEST"],
        input_archive_sha256=os.environ["NCM_INPUT_SHA256"],
        torch=str(torch.__version__),
    )
    _json(output / "manifest.json", resolved)
    store.upload_file(str(output / "manifest.json"), bucket, prefix + "/manifest.json")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = VQVAE(**config.vqvae).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sampling_rng = torch.Generator(device="cpu").manual_seed(0)
    data = torch.tensor(
        [row["target_ids"] + [tokenizer.pad_token_id] * (80 - len(row["target_ids"])) for row in rows], dtype=torch.long
    )
    checkpoints = []

    def snapshot(step):
        path = output / f"tokenizer-{step:06d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "completed_steps": step,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "sampling_rng": sampling_rng.get_state(),
                "resolved_config": resolved,
            },
            temporary,
        )
        temporary.replace(path)
        identity = {"step": step, "filename": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        store.upload_file(str(path), bucket, f"{prefix}/checkpoints/{path.name}")
        checkpoints.append(identity)
        _json(output / "checkpoints.json", checkpoints)
        store.upload_file(str(output / "checkpoints.json"), bucket, prefix + "/checkpoints.json")
        print("T0_CHECKPOINT " + json.dumps(identity), flush=True)

    snapshot(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (output / "curve.jsonl").open("x") as curve:
        for step in range(1, 6001):
            indices = torch.randint(0, len(data), (128,), generator=sampling_rng)
            diagnostics = _overfit_update(model, optimizer, data[indices].cuda())
            curve.write(json.dumps({"completed_steps": step, **diagnostics}, allow_nan=False) + "\n")
            curve.flush()
            if step % 500 == 0:
                snapshot(step)
                store.upload_file(str(output / "curve.jsonl"), bucket, prefix + "/curve.jsonl")
                print("T0_UPDATE " + json.dumps({"completed_steps": step, **diagnostics}), flush=True)
    torch.cuda.synchronize()
    report = {
        "completed_steps": 6000,
        "example_exposures": 768000,
        "training_wall_seconds_including_checkpoints": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "quality_selection": "Not performed here; CPU evaluation occurs after GPU lock release",
        "gpu_kernel_only_time_measured": False,
    }
    _json(output / "training.json", report)
    store.upload_file(str(output / "training.json"), bucket, prefix + "/training.json")
    print("T0_TRAINING " + json.dumps(report), flush=True)


def _validate_snapshot(snapshot, expected, step):
    if snapshot["completed_steps"] != step:
        raise ValueError("Checkpoint completed-step count differs from declared evaluation point")
    recorded = snapshot["resolved_config"]
    for key in ("protocol_sha256", "training_manifest_sha256", "tuning_manifest_sha256", "vqvae_config"):
        if recorded[key] != expected[key]:
            raise ValueError(f"T0 checkpoint provenance mismatch: {key}")
    for key in ("optimizer", "cpu_rng", "cuda_rng", "sampling_rng"):
        if key not in snapshot:
            raise ValueError(f"T0 checkpoint lacks persisted state: {key}")
    for key in ("image_digest", "input_archive_sha256"):
        env = "NCM_IMAGE_DIGEST" if key == "image_digest" else "NCM_INPUT_SHA256"
        if os.environ.get(env) and recorded[key] != os.environ[env]:
            raise ValueError(f"T0 checkpoint execution identity mismatch: {key}")


def _evaluate_checkpoint(protocol_path, checkpoint_path, step, destination):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    resolved, rows, tuning, tokenizer, config = load_protocol(protocol_path)
    backend = tokenizer.hf_tokenizer
    if backend is None:
        raise RuntimeError("Pinned GPT-2 tokenizer did not load")
    snapshot = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_snapshot(snapshot, resolved, step)
    model = VQVAE(**config.vqvae).cpu().eval()
    model.load_state_dict(snapshot["model"], strict=True)
    del snapshot
    names = {name for row in rows for name, _ in entities(row["target_text"])}
    exposure = Counter(token for row in rows for token in row["target_ids"][:-1])
    started = time.monotonic()
    result: dict[str, Any] = {
        "step": step,
        "checkpoint_sha256": sha256(checkpoint_path),
        "protocol_sha256": resolved["protocol_sha256"],
        "device": "cpu",
        "threads": 1,
        "batch_size": 8,
    }
    for split, examples in (("train", rows), ("tuning", tuning)):
        metrics, details = reconstruct(model, examples, backend, names, exposure)
        result[split] = {"metrics": metrics, **details}
    result["evaluation_wall_seconds"] = time.monotonic() - started
    _json(destination, result)
    print(
        "T0_EVALUATED "
        + json.dumps({"step": step, "train": result["train"]["metrics"], "tuning": result["tuning"]["metrics"]}),
        flush=True,
    )
    return {
        "step": step,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "train": result["train"]["metrics"],
        "tuning": result["tuning"]["metrics"],
        "report": str(Path(destination).name),
        "evaluation_wall_seconds": result["evaluation_wall_seconds"],
    }


def select_checkpoint(results):
    if sorted(result["step"] for result in results) != list(EVALUATION_STEPS):
        raise ValueError("Selection requires exactly the predeclared 1500/3000/6000 results")
    for result in results:
        for key in ("entity_f1", "target_exact"):
            value = result["tuning"][key]
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"Invalid tuning selection metric: {key}")
    return max(results, key=lambda result: (result["tuning"]["entity_f1"], -result["step"]))


def export_candidate(checkpoint_path, protocol_path, destination, *, role):
    resolved, _, _, _, _ = load_protocol(protocol_path)
    snapshot = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    step = snapshot["completed_steps"]
    if step not in EVALUATION_STEPS:
        raise ValueError("Candidate is not a predeclared T0 checkpoint")
    _validate_snapshot(snapshot, resolved, step)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    candidate = destination / "candidate.pt"
    torch.save({"model": snapshot["model"]}, candidate)
    identity = {
        "kind": "controlled_tokenizer",
        "arm": "T0",
        "completed_steps": step,
        "initialization_seed": 0,
        "sampling_seed": 0,
        "training_rows": 6000,
        "batch_size": 128,
        "training_manifest_sha256": resolved["training_manifest_sha256"],
        "tuning_manifest_sha256": resolved["tuning_manifest_sha256"],
        "vqvae_config": resolved["vqvae_config"],
        "checkpoint_sha256": sha256(candidate),
        "training_snapshot_sha256": sha256(checkpoint_path),
        "protocol_sha256": resolved["protocol_sha256"],
        "input_archive_sha256": snapshot["resolved_config"]["input_archive_sha256"],
        "image_digest": snapshot["resolved_config"]["image_digest"],
        "evaluation_role": role,
        "selection_rule": resolved["checkpoint_selection"],
    }
    _json(destination / "candidate.provenance.json", identity)
    return identity


def evaluate(protocol_path, results_path, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("T0 checkpoint selection is CPU-only after GPU lock release")
    resolved, _, _, _, _ = load_protocol(protocol_path)
    results_path, output = Path(results_path), Path(output)
    training = json.loads((results_path / "training.json").read_text())
    if training["completed_steps"] != 6000:
        raise ValueError("T0 schedule did not complete 6000 updates")
    index = json.loads((results_path / "checkpoints.json").read_text())
    if [entry["step"] for entry in index] != list(range(0, 6001, 500)):
        raise ValueError("T0 did not preserve all declared durable snapshots")
    for entry in index:
        if sha256(results_path / entry["filename"]) != entry["sha256"]:
            raise ValueError("Persisted T0 snapshot checksum differs from training index")
    output.mkdir(parents=True, exist_ok=False)
    evaluations = output / "evaluations"
    evaluations.mkdir()
    started = time.monotonic()
    summaries = []
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as executor:
        jobs = [
            executor.submit(
                _evaluate_checkpoint,
                str(Path(protocol_path).resolve()),
                str((results_path / f"tokenizer-{step:06d}.pt").resolve()),
                step,
                str((evaluations / f"step-{step:06d}.json").resolve()),
            )
            for step in EVALUATION_STEPS
        ]
        for job in as_completed(jobs):
            summaries.append(job.result())
    summaries.sort(key=lambda result: result["step"])
    selected = select_checkpoint(summaries)
    identity = export_candidate(
        results_path / f"tokenizer-{selected['step']:06d}.pt", protocol_path, output, role="tuning_selected"
    )
    report = {
        "arm": "T0",
        "evaluation_steps": list(EVALUATION_STEPS),
        "selection_rule": resolved["checkpoint_selection"],
        "selected_step": selected["step"],
        "checkpoints": summaries,
        "candidate": identity,
        "gate_passed": selected["tuning"]["entity_f1"] >= 90 and selected["tuning"]["target_exact"] >= 80,
        "gate": {"tuning_entity_f1_percent": 90, "tuning_literal_target_exact_percent": 80},
        "cpu_evaluation_wall_seconds": time.monotonic() - started,
        "cpu_workers": 3,
        "threads_per_worker": 1,
        "protected_test_predictions": False,
        "limitation": "Single-seed tuning selection only; no protected-test or comparative generalization claim",
        "next_action": "Report and stop. T1, replication, or generator work requires separate authorization.",
    }
    _json(output / "selection.json", report)
    print("T0_SELECTION " + json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--evaluate-results", type=Path)
    parser.add_argument("--evaluation-output", type=Path)
    args = parser.parse_args()
    if args.preflight:
        resolved, _, _, _, _ = load_protocol(args.protocol)
        print("T0_CONFIG " + json.dumps(resolved, sort_keys=True), flush=True)
    elif args.output_dir is not None:
        train(args.protocol, args.output_dir)
    else:
        if args.evaluation_output is None:
            parser.error("--evaluate-results requires --evaluation-output")
        evaluate(args.protocol, args.evaluate_results, args.evaluation_output)


if __name__ == "__main__":
    main()

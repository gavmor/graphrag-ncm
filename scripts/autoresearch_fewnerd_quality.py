"""Offline Few-NERD quality harness; evaluation only, never trains or downloads."""

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from models.vqvae import VQVAE
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from ncm_graphrag.data.fewnerd_quality_metrics import dictionary_predictions, entities, score_rows
from ncm_graphrag.ncm.config import build_fewnerd_extraction_tokenizer_cfg
from ncm_graphrag.ncm.fewnerd_quality import reconstruct

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "benchmarks/fewnerd"
OUTPUT = ROOT / "artifacts/fewnerd-quality-harness"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    frozen = json.loads((INPUTS / "frozen.json").read_text())
    for filename, expected in frozen["files"].items():
        if sha256(INPUTS / filename) != expected:
            raise ValueError(f"Frozen input checksum mismatch: {filename}")
    pools = {
        name: [json.loads(line) for line in (INPUTS / f"{name}.jsonl").read_text().splitlines()]
        for name in ("generator-train", "tokenizer-train", "tuning")
    }
    groups = {name: {r["group_id"] for r in rows} for name, rows in pools.items()}
    for name, rows in pools.items():
        if len(groups[name]) != len(rows):
            raise ValueError(f"Duplicate input groups in {name}")
    if not groups["tokenizer-train"] <= groups["generator-train"]:
        raise ValueError("Tokenizer data is not a subset of generator training")
    if groups["generator-train"] & groups["tuning"]:
        raise ValueError("Training/tuning input-group leakage")
    if len(pools["tokenizer-train"]) != 6000 or len(pools["tuning"]) != 1000:
        raise ValueError("Protocol requires 6000 tokenizer and 1000 tuning groups")
    return frozen, pools


def load_model(frozen):
    cfg = build_fewnerd_extraction_tokenizer_cfg().vqvae
    if OmegaConf.to_container(cfg, resolve=True) != frozen["resolved_configs"]["vqvae"]:
        raise ValueError("Architecture changed outside the frozen tokenizer experiment protocol")
    model = VQVAE(**cfg).cpu().eval()
    candidate = INPUTS / "candidate.pt"
    provenance = INPUTS / "candidate.provenance.json"
    identity = {"kind": "untrained_seeded_control", "initialization_seed": 0, "completed_steps": 0}
    if candidate.exists() != provenance.exists():
        raise ValueError("Candidate checkpoint and provenance must both exist")
    if candidate.exists():
        identity = json.loads(provenance.read_text())
        expected = {
            "checkpoint_sha256": sha256(candidate),
            "training_manifest_sha256": frozen["files"]["tokenizer-train.jsonl"],
            "vqvae_config": OmegaConf.to_container(cfg, resolve=True),
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(f"Candidate provenance mismatch: {key}")
        if identity.get("kind") != "controlled_tokenizer" or not isinstance(identity.get("completed_steps"), int):
            raise ValueError("Candidate must identify a controlled tokenizer and its completed steps")
        if identity["completed_steps"] <= 0 or not isinstance(identity.get("initialization_seed"), int):
            raise ValueError("Candidate must declare completed training and initialization seed")
        model.load_state_dict(torch.load(candidate, map_location="cpu", weights_only=True)["model"], strict=True)
    return model, identity, OmegaConf.to_container(cfg, resolve=True)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This harness is CPU-only; use bash autoresearch.sh")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    frozen, pools = load_inputs()
    tok = AutoTokenizer.from_pretrained(INPUTS / "tokenizer", local_files_only=True, use_fast=True)
    if tok is None:
        raise RuntimeError("Local GPT-2 tokenizer could not be loaded")
    tok.add_special_tokens({"pad_token": "<|pad|>"})
    train = pools["tokenizer-train"]
    tuning = pools["tuning"]
    for rows in pools.values():
        for row in rows:
            if len(row["prefix_ids"]) > 64 or not 1 <= len(row["target_ids"]) <= 80:
                raise ValueError("Frozen example violates token bounds")
    gold = [r["target_text"] for r in tuning]
    oracle = score_rows(gold, gold)
    if oracle["entity_f1"] != 100 or oracle["entity_set_exact"] != 100 or oracle["strict_validity"] != 100:
        raise ValueError("Scorer oracle/serialization round-trip failed")
    train_names = {name for row in train for name, _ in entities(row["target_text"])}
    exposure = Counter(token for row in train for token in row["target_ids"][:-1])
    dictionary = dictionary_predictions(pools["generator-train"], tuning)
    dictionary_metrics = score_rows(gold, dictionary)
    model, identity, cfg = load_model(frozen)
    print("MODEL_IDENTITY " + json.dumps(identity, sort_keys=True), flush=True)
    print("Evaluating 32 diagnostic training targets and 1000 frozen tuning targets on CPU.", flush=True)
    train_metrics, train_report = reconstruct(model, train[:32], tok, train_names, exposure)
    tuning_metrics, tuning_report = reconstruct(model, tuning, tok, train_names, exposure)
    metrics = {
        "tokenizer_entity_f1": tuning_metrics["entity_f1"],
        "dictionary_entity_f1": dictionary_metrics["entity_f1"],
        "scorer_oracle_entity_f1": oracle["entity_f1"],
    }
    metrics.update({"tokenizer_" + key: value for key, value in tuning_metrics.items() if key != "entity_f1"})
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite quality metric")
    report = {
        "model_identity": identity,
        "vqvae_config": cfg,
        "torch": str(torch.__version__),
        "python": sys.version,
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "scripts/autoresearch_fewnerd_quality.py",
                "src/ncm_graphrag/data/fewnerd_quality_metrics.py",
                "src/ncm_graphrag/ncm/fewnerd_quality.py",
                "src/ncm_graphrag/data/extraction.py",
                "src/ncm_graphrag/ncm/config.py",
                "third_party/ncm/models/vqvae.py",
                "third_party/ncm/models/vae.py",
                "third_party/ncm/models/quant.py",
                "uv.lock",
            )
        },
        "frozen_manifest_sha256": sha256(INPUTS / "frozen.json"),
        "metrics": metrics,
        "oracle": oracle,
        "dictionary": {"metrics": dictionary_metrics, "predictions": dictionary},
        "train_diagnostic_32": {"metrics": train_metrics, **train_report},
        "tuning": {"metrics": tuning_metrics, **tuning_report},
        "limitations": [
            "This entrypoint evaluates only; it does not execute separate training controls.",
            "Initial seed-0 untrained control is not a historical checkpoint result.",
            "32 training targets are diagnostic only, not the 2000-update overfit control.",
            "No reserved-test predictions or metrics; protected protocol is maintained separately.",
            "Sentence-disjoint, not document-disjoint; near duplicates remain possible.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, value in metrics.items():
        print(f"METRIC {name}={value:.8f}")
    print(f"Report: {OUTPUT / 'report.json'}")


if __name__ == "__main__":
    main()

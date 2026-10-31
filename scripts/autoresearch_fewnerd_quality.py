"""Offline Few-NERD quality harness; evaluation only, never trains or downloads."""

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from models.vqvae import VQVAE
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from ncm_graphrag.data.fewnerd_quality_metrics import dictionary_predictions, entities, name_spans, score_rows
from ncm_graphrag.ncm.config import build_fewnerd_extraction_tokenizer_cfg

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


def decode(ids, tok):
    if tok.eos_token_id in ids:
        ids = ids[: ids.index(tok.eos_token_id)]
    return tok.decode([i for i in ids if i != tok.pad_token_id], skip_special_tokens=True)


def alignment(row, tok):
    encoded = tok(row["target_text"], return_offsets_mapping=True, add_special_tokens=False)
    if encoded["input_ids"] + [tok.eos_token_id] != row["target_ids"]:
        raise ValueError("Frozen target IDs no longer match serialized GPT-2 target")
    names = []
    types = set()
    for name, entity_type, start, end in name_spans(row["target_text"]):
        if row["target_text"][start:end] != name:
            raise ValueError("Name alignment disagrees with actual serialization")
        indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
        if not indices or not all(
            any(a <= c < b for a, b in (encoded["offset_mapping"][i] for i in indices)) for c in range(start, end)
        ):
            raise ValueError("GPT-2 name-token offsets do not cover name")
        names.append((name, entity_type, indices))
        type_start = end + len("<|>")
        type_end = type_start + len(entity_type)
        if row["target_text"][type_start:type_end] != entity_type:
            raise ValueError("Type alignment disagrees with actual serialization")
        types.update(i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < type_end and b > type_start)
    return names, types


@torch.inference_mode()
def reconstruct(model, rows, tok, train_names, exposure):
    predictions = []
    counts = Counter()
    names_by_group = {"seen": Counter(), "unseen": Counter(), "multi_subword": Counter()}
    utilization = [set() for _ in model.vq.levels]
    details = []
    vq_loss_sum = 0.0
    nonpad_ce_sum = 0.0
    ce_sum = 0.0
    # Batch geometry is part of the workload and does not depend on environment.
    for start in range(0, len(rows), 8):
        batch_rows = rows[start : start + 8]
        target = torch.tensor(
            [r["target_ids"] + [tok.pad_token_id] * (80 - len(r["target_ids"])) for r in batch_rows], dtype=torch.long
        )
        idx = model.encode(target)
        logits = model.decode(idx)
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite tokenizer reconstruction")
        ce_sum += F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum").item()
        guesses = logits.argmax(-1).tolist()
        latent = model.encode_z(target)
        _, vq_loss, _, _, _ = model.vq(latent)
        if not torch.isfinite(vq_loss):
            raise ValueError("Non-finite evaluation VQ loss")
        vq_loss_sum += float(vq_loss) * len(batch_rows)
        nonpad_ce_sum += F.cross_entropy(
            logits.flatten(0, 1), target.flatten(), ignore_index=tok.pad_token_id, reduction="sum"
        ).item()
        offset = 0
        for level, width in enumerate(model.vq.levels):
            utilization[level].update(idx[:, offset : offset + width].flatten().tolist())
            offset += width
        for row, ids in zip(batch_rows, guesses, strict=True):
            gold = row["target_ids"]
            text = decode(ids, tok)
            predictions.append(text)
            correct = [a == b for a, b in zip(ids, gold)]
            counts["nonpad_correct"] += sum(correct)
            counts["nonpad_tokens"] += len(gold)
            eos = len(gold) - 1
            counts["eos_position_correct"] += ids[eos] == tok.eos_token_id
            counts["eos_first_correct"] += tok.eos_token_id in ids and ids.index(tok.eos_token_id) == eos
            counts["eos_missing"] += tok.eos_token_id not in ids
            aligned, type_positions = alignment(row, tok)
            name_positions = {i for _, _, indices in aligned for i in indices}
            counts["name_token_correct"] += sum(correct[i] for i in name_positions)
            counts["name_tokens"] += len(name_positions)
            counts["type_token_correct"] += sum(correct[i] for i in type_positions)
            counts["type_tokens"] += len(type_positions)
            fixed_positions = set(range(eos)) - name_positions - type_positions
            counts["fixed_token_correct"] += sum(correct[i] for i in fixed_positions)
            counts["fixed_tokens"] += len(fixed_positions)
            predicted_entities = entities(text)
            predicted_names = {name for name, _ in predicted_entities}
            name_details = []
            for name, entity_type, indices in aligned:
                normalized = name.strip().upper()
                exact = normalized in predicted_names
                typed = (normalized, entity_type.strip().upper()) in predicted_entities
                counts["names"] += 1
                counts["exact_names"] += exact
                subgroup = "seen" if normalized in train_names else "unseen"
                for key in [subgroup] + (["multi_subword"] if len(indices) > 1 else []):
                    names_by_group[key]["names"] += 1
                    names_by_group[key]["exact_names"] += exact
                    names_by_group[key]["typed_names_correct"] += typed
                name_details.append(
                    {
                        "name": name,
                        "type": entity_type,
                        "token_positions": indices,
                        "seen_in_tokenizer_training": subgroup == "seen",
                        "exact": exact,
                        "training_target_subword_exposure": [exposure[gold[i]] for i in indices],
                    }
                )
            details.append(
                {"group_id": row["group_id"], "gold": row["target_text"], "reconstruction": text, "names": name_details}
            )
        del logits
    metrics = score_rows([r["target_text"] for r in rows], predictions)
    for label, numerator, denominator in (
        ("nonpad_token_accuracy", "nonpad_correct", "nonpad_tokens"),
        ("name_token_accuracy", "name_token_correct", "name_tokens"),
        ("type_token_accuracy", "type_token_correct", "type_tokens"),
        ("fixed_token_accuracy", "fixed_token_correct", "fixed_tokens"),
        ("exact_name_recall", "exact_names", "names"),
    ):
        metrics[label] = 100 * counts[numerator] / max(counts[denominator], 1)
    metrics["eos_first_position_accuracy"] = 100 * counts["eos_first_correct"] / len(rows)
    metrics["reconstruction_ce_all_positions"] = ce_sum / (80 * len(rows))
    metrics["reconstruction_ce_nonpad"] = nonpad_ce_sum / counts["nonpad_tokens"]
    metrics["vq_loss_eval_no_fine_drop"] = vq_loss_sum / len(rows)
    metrics["examples"] = len(rows)
    subgroups: dict[str, dict[str, float]] = {}
    for number in (1, 2):
        selected = [i for i, row in enumerate(rows) if len(entities(row["target_text"])) == number]
        subgroups[f"{number}_entity"] = {"examples": len(selected)}
        if selected:
            subgroups[f"{number}_entity"].update(
                score_rows([rows[i]["target_text"] for i in selected], [predictions[i] for i in selected])
            )
    return metrics, {
        "counts": dict(counts),
        "name_subgroups": names_by_group,
        "entity_count_subgroups": subgroups,
        "per_level_used_codes": [len(codes) for codes in utilization],
        "predictions": details,
    }


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

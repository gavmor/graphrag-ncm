"""Freeze offline, sentence-disjoint Few-NERD development inputs (never test samples).

Run with PYTHONPATH=third_party/ncm:src. Only tokenizer assets are loaded; no model
checkpoint is read. The output directory must be absent or empty. Source hashes
are provenance, not immutable acceptance constraints for later research code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TOKENIZER_FILES = ("tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.json", "config.json")
SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
PREFIX_BOUND = 64
TARGET_BOUND = 80
GENERATOR_SIZE = 50000
TOKENIZER_SIZE = 6000
TUNING_SIZE = 1000


def canonical_text(text: str) -> str:
    """Identity normalization, deliberately independent of annotations and case."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def group_id(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_json(row) + "\n")


@dataclass
class RawGroup:
    row_ids: dict[str, list[str]] = field(default_factory=dict)
    annotation: str = ""
    conflicting: bool = False


def _annotation_signature(tokens: list[str], tags: list[int], coarse_types: dict[int, str]) -> str:
    """Fingerprint every raw coarse span, even spans the converter would discard.

    Character coordinates refer to canonical text, so equivalent tokenizations
    do not create false conflicts. Repeated mentions at different positions are
    retained, as are short names, unsafe fields, and sentences with >2 entities.
    """
    if len(tokens) != len(tags):
        raise ValueError("Raw tokens/ner_tags length mismatch; cannot audit annotations honestly")
    if any(tag != 0 and tag not in coarse_types for tag in tags):
        raise ValueError("Unknown coarse tag in raw Few-NERD annotations")
    offsets: list[tuple[int, int]] = []
    position = 0
    for token in tokens:
        normalized = canonical_text(token)
        if normalized:
            if position:
                position += 1
            offsets.append((position, position + len(normalized)))
            position += len(normalized)
        else:
            offsets.append((position, position))
    spans: list[tuple[int, int, str]] = []
    start = 0
    while start < len(tags):
        end = start + 1
        while end < len(tags) and tags[end] == tags[start]:
            end += 1
        if tags[start]:
            nonempty = [offsets[i] for i in range(start, end) if offsets[i][1] > offsets[i][0]]
            if nonempty:
                spans.append((nonempty[0][0], nonempty[-1][1], coarse_types[tags[start]]))
        start = end
    return hashlib.sha256(_json(spans).encode("utf-8")).hexdigest()


def _audit_raw(corpus_dir: Path, coarse_types: dict[int, str]) -> tuple[dict[str, RawGroup], dict[str, list[str]]]:
    import pyarrow.parquet as pq

    groups: dict[str, RawGroup] = {}
    split_groups: dict[str, list[str]] = {}
    for split, filename in SPLIT_FILES.items():
        ordered: dict[str, None] = {}
        parquet = pq.ParquetFile(corpus_dir / filename)
        row_number = 0
        for batch in parquet.iter_batches(columns=["tokens", "ner_tags"]):
            data = batch.to_pydict()
            for tokens, tags in zip(data["tokens"], data["ner_tags"], strict=True):
                identity = group_id(" ".join(tokens))
                signature = _annotation_signature(tokens, tags, coarse_types)
                group = groups.setdefault(identity, RawGroup(annotation=signature))
                group.conflicting = group.conflicting or signature != group.annotation
                group.row_ids.setdefault(split, []).append(f"{split}:{row_number}")
                ordered[identity] = None
                row_number += 1
        split_groups[split] = list(ordered)
    return groups, split_groups


def _resolve_corpus_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "information_extraction" / "few_nerd" / "supervised").is_dir():
        return path
    if (
        path.name == "supervised"
        and path.parent.name == "few_nerd"
        and path.parent.parent.name == "information_extraction"
    ):
        return path.parents[2]
    raise ValueError(
        "--corpus-root must contain information_extraction/few_nerd/supervised, or be that supervised directory"
    )


def _eligible_rows(
    split: str, examples: Any, groups: dict[str, RawGroup], excluded: set[str], tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    eligible_conflicts: set[str] = set()
    eligible_exclusions: set[str] = set()
    for example in examples:
        identity = group_id(example.text_unit)
        group = groups[identity]
        if split not in group.row_ids:
            raise ValueError(f"Converter example does not map to raw {split} identity {identity}")
        if identity in seen:
            continue
        seen.add(identity)
        if group.conflicting:
            eligible_conflicts.add(identity)
        if identity in excluded:
            eligible_exclusions.add(identity)
        if group.conflicting or identity in excluded:
            continue
        prefix = f"Entity_types: {example.entity_types}\nText: {example.text_unit}"
        prefix_ids = tokenizer.encode(prefix)
        target_ids = tokenizer.encode(example.target_text) + [tokenizer.eos_token_id]
        if len(prefix_ids) > PREFIX_BOUND or len(target_ids) > TARGET_BOUND:
            raise ValueError("Converter eligibility and frozen tokenization bounds disagree")
        rows.append(
            {
                "group_id": identity,
                "row_ids": group.row_ids[split],
                "text_unit": example.text_unit,
                "entity_types": example.entity_types,
                "target_text": example.target_text,
                "prefix_ids": prefix_ids,
                "target_ids": target_ids,
            }
        )
    return rows, {
        "eligible_rows_before_group_filters": len(examples),
        "eligible_unique_groups_before_filters": len(seen),
        "eligible_conflicting_groups": len(eligible_conflicts),
        "eligible_prior_split_identity_groups": len(eligible_exclusions),
        "clean_eligible_unique_groups": len(rows),
    }


def _token_hashes(rows: list[dict[str, Any]]) -> dict[str, str]:
    hashes = {}
    for key in ("group_id", "prefix_ids", "target_ids"):
        digest = hashlib.sha256()
        for row in rows:
            digest.update((_json(row[key]) + "\n").encode("utf-8"))
        hashes[f"{key}_sha256"] = digest.hexdigest()
    return hashes


def prepare(corpus_root: Path, output: Path, tokenizer_source: Path) -> None:
    from omegaconf import OmegaConf
    from tokenization.hf_tokenizer import HFTokenizer

    from ncm_graphrag.data import fewnerd_extraction as fewnerd
    from ncm_graphrag.data import real_extraction
    from ncm_graphrag.ncm import (
        _ncm_repo,  # noqa: F401 -- installs vendored tokenizer import path
        config,
    )

    corpus_root = _resolve_corpus_root(corpus_root)
    corpus_dir = corpus_root / "information_extraction" / "few_nerd" / "supervised"
    output = output.expanduser().resolve()
    tokenizer_source = tokenizer_source.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite frozen benchmark: {output}")
    if (real_extraction.PREFIX_TOKEN_BUDGET, real_extraction.TARGET_TOKEN_BUDGET, fewnerd.MAX_ENTITIES) != (64, 80, 2):
        raise ValueError("The Phase A protocol requires prefix64, target80, and at most two entities")
    raw_hashes = {
        split: {"filename": filename, "sha256": sha256(corpus_dir / filename)}
        for split, filename in SPLIT_FILES.items()
    }
    tokenizer_hashes = {name: sha256(tokenizer_source / name) for name in TOKENIZER_FILES}
    groups, split_groups = _audit_raw(corpus_dir, fewnerd.COARSE_TYPES)
    conflicts = {identity for identity, group in groups.items() if group.conflicting}
    official_train = set(split_groups["train"])
    official_validation = set(split_groups["validation"])
    split_audit: dict[str, Any] = {}
    for split, identities in split_groups.items():
        excluded = set() if split == "train" else official_train
        if split == "test":
            excluded = official_train | official_validation
        identity_set = set(identities)
        raw_rows = sum(len(groups[identity].row_ids[split]) for identity in identities)
        split_audit[split] = {
            "raw_rows": raw_rows,
            "raw_unique_groups": len(identities),
            "duplicate_rows": raw_rows - len(identities),
            "conflicting_groups": len(identity_set & conflicts),
            "conflicting_rows": sum(len(groups[i].row_ids[split]) for i in identity_set & conflicts),
            "prior_split_identity_excluded_groups": len(identity_set & excluded),
            "prior_split_identity_excluded_rows": sum(len(groups[i].row_ids[split]) for i in identity_set & excluded),
            "remaining_groups_identity_only": len(identity_set - excluded),
            "remaining_groups_after_identity_and_conflict_filters": len(identity_set - excluded - conflicts),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "benchmark"
        tokenizer_dir = staging / "tokenizer"
        tokenizer_dir.mkdir(parents=True)
        for name in TOKENIZER_FILES:
            shutil.copyfile(tokenizer_source / name, tokenizer_dir / name)
        tokenizer = HFTokenizer(model_id=str(tokenizer_dir), pad_token="<|pad|>", local_files_only=True)
        if (tokenizer.eos_token_id, tokenizer.pad_token_id) != (50256, 50257):
            raise ValueError("Unexpected GPT-2 EOS/PAD identity")
        previous_root = os.environ.get("NCM_CORPORA_ROOT")
        previous_tokenizer = real_extraction._TOKENIZER
        os.environ["NCM_CORPORA_ROOT"] = str(corpus_root)
        real_extraction._TOKENIZER = tokenizer
        fewnerd._build.cache_clear()
        try:
            training, train_counts = _eligible_rows("train", fewnerd._build("train"), groups, set(), tokenizer)
            tuning, tuning_counts = _eligible_rows(
                "validation", fewnerd._build("validation"), groups, official_train, tokenizer
            )
        finally:
            fewnerd._build.cache_clear()
            real_extraction._TOKENIZER = previous_tokenizer
            if previous_root is None:
                os.environ.pop("NCM_CORPORA_ROOT", None)
            else:
                os.environ["NCM_CORPORA_ROOT"] = previous_root
        split_audit["train"].update(train_counts)
        split_audit["validation"].update(tuning_counts)
        if len(training) < TOKENIZER_SIZE or len(tuning) < TUNING_SIZE:
            raise ValueError(
                f"Insufficient clean groups: train={len(training)} (need {TOKENIZER_SIZE}), "
                f"tuning={len(tuning)} (need {TUNING_SIZE}); revise protocol"
            )
        random.Random(0).shuffle(training)
        training = training[:GENERATOR_SIZE]
        tokenizer_training = training[:TOKENIZER_SIZE]
        random.Random(42).shuffle(tuning)
        tuning = tuning[:TUNING_SIZE]
        if {row["group_id"] for row in training} & {row["group_id"] for row in tuning}:
            raise ValueError("Frozen train/tuning group overlap")
        manifests = {
            "generator-train.jsonl": training,
            "tokenizer-train.jsonl": tokenizer_training,
            "tuning.jsonl": tuning,
        }
        for filename, rows in manifests.items():
            _write_jsonl(staging / filename, rows)
        _write_jsonl(
            staging / "raw-groups.jsonl",
            (
                {"group_id": identity, "row_ids": group.row_ids, "conflicting_annotations": group.conflicting}
                for identity, group in groups.items()
            ),
        )
        audit = {
            "splits": split_audit,
            "globally_conflicting_groups": len(conflicts),
            "row_id_convention": "official split name:zero-based parquet row ordinal",
            "conflict_policy": (
                "Quarantine globally when any raw coarse span annotation differs, including converter-ineligible rows"
            ),
            "annotation_signature": (
                "SHA256 of canonical-text character spans [start,end,coarse_type], retaining all raw mentions"
            ),
            "identity_only_exclusions_can_overlap_conflict_counts": True,
            "test_eligibility_computed": False,
        }
        _write_json(staging / "audit.json", audit)
        resolved_vq = OmegaConf.to_container(config.build_fewnerd_extraction_tokenizer_cfg().vqvae, resolve=True)
        resolved_configs = {
            "vqvae": resolved_vq,
            "generator": OmegaConf.to_container(config.build_fewnerd_extraction_model_cfg(), resolve=True),
            "generation": OmegaConf.to_container(config.build_fewnerd_extraction_generation_cfg(), resolve=True),
            "evaluation": {
                "control": "untrained_seeded_control",
                "initialization_seed": 0,
                "batch_size": 8,
                "device": "cpu",
                "deterministic": True,
                "training_updates": 0,
            },
            "tiny_training_diagnostic": {
                "manifest": "tokenizer-train.jsonl",
                "selection": "first 32 rows in stored order",
                "rows": 32,
                "overfit_training": False,
            },
        }
        _write_json(staging / "resolved-configs.json", resolved_configs)
        source_names = (
            "scripts/prepare_fewnerd_quality_inputs.py",
            "src/ncm_graphrag/data/fewnerd_quality_metrics.py",
            "src/ncm_graphrag/data/fewnerd_extraction.py",
            "src/ncm_graphrag/data/real_extraction.py",
            "src/ncm_graphrag/data/extraction.py",
            "src/ncm_graphrag/ncm/config.py",
            "third_party/ncm/tokenization/hf_tokenizer.py",
        )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        files = {str(path.relative_to(staging)): sha256(path) for path in sorted(staging.rglob("*")) if path.is_file()}
        metadata = {
            "schema_version": 1,
            "files": files,
            "bounds": {
                "prefix_tokens": PREFIX_BOUND,
                "target_tokens": TARGET_BOUND,
                "max_entities": 2,
                "min_entities": 1,
                "target_includes_eos": True,
                "prefix_padding": False,
            },
            "identity": {
                "normalization": "Unicode NFC, collapsed whitespace, case preserved",
                "hash": "SHA256 UTF-8",
                "includes_gold_types": False,
            },
            "selection": {
                "generator": {
                    "seed": 0,
                    "requested_max": GENERATOR_SIZE,
                    "actual": len(training),
                    "method": (
                        "Python random.Random.shuffle of unique clean eligible official-train groups "
                        "in first converter occurrence order; take first min(50000,pool_size)"
                    ),
                },
                "tokenizer": {
                    "seed": 0,
                    "actual": len(tokenizer_training),
                    "method": "first 6000 rows of selected generator manifest, preserving order; no second shuffle",
                },
                "tuning": {
                    "seed": 42,
                    "actual": len(tuning),
                    "method": (
                        "Python random.Random.shuffle of unique clean eligible validation groups "
                        "excluding all raw official-train identities; first 1000"
                    ),
                },
                "reserved_test": {
                    "proposed_seed": 43,
                    "proposed_size": 1000,
                    "actual": 0,
                    "status": "deferred",
                    "history_clearance": "unavailable",
                    "reason": (
                        "Prior research sample usage has not been cleared; no test sample selected, "
                        "no test eligibility conversion, predictions or metrics"
                    ),
                },
            },
            "tokenizer": {
                "directory": "tokenizer",
                "model_id": "gpt2",
                "revision": GPT2_REVISION,
                "local_files_only": True,
                "pad_token": "<|pad|>",
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "source_files_sha256": tokenizer_hashes,
                "load_contract": (
                    "AutoTokenizer.from_pretrained(local directory, local_files_only=True), "
                    "then add_special_tokens({'pad_token':'<|pad|>'})"
                ),
            },
            "raw_corpus": raw_hashes,
            "manifest_token_hashes": {filename: _token_hashes(rows) for filename, rows in manifests.items()},
            "resolved_configs": resolved_configs,
            "provenance": {
                "code_revision": revision,
                "source_sha256": {name: sha256(PROJECT_ROOT / name) for name in source_names},
                "source_hash_policy": "informational provenance only; not immutable benchmark acceptance constraints",
                "corpus_root": str(corpus_root),
            },
            "limitations": [
                "Sentence-disjoint, not document-disjoint: source rows provide no article identity",
                "Near-duplicate sentences are not excluded",
                "Oracle candidate entity types derived from gold are supplied equally to every control",
                "Shared entity names and target strings alone are not excluded",
            ],
            "reserved_test_predictions_or_metrics": False,
        }
        _write_json(staging / "frozen.json", metadata)
        staging.replace(output)
    print(
        f"Frozen Few-NERD inputs at {output}: generator={len(training)}, "
        f"tokenizer={len(tokenizer_training)}, tuning={len(tuning)}; test selection deferred"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path, help="Corpora root or Few-NERD supervised directory")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "fewnerd")
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub" / "models--gpt2" / "snapshots" / GPT2_REVISION,
        help="Existing local GPT-2 tokenizer assets; never downloaded",
    )
    args = parser.parse_args()
    prepare(args.corpus_root, args.output, args.tokenizer_source)


if __name__ == "__main__":
    main()

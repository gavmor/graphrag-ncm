"""Audit exact input overlap without changing the fixed 300-example result.

CPU-only postprocessing of saved predictions. Reused entity-name targets are
reported separately; they are not excluded, since shared names are normal in NER.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from ncm_graphrag.ncm.task_registry import TASKS


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.evaluation.read_text())
    manifest = json.loads(args.manifest.read_text())
    corpus = Path(os.environ["NCM_CORPORA_ROOT"]) / "information_extraction/few_nerd/supervised"
    for split in ("train", "validation"):
        assert file_sha256(corpus / f"{split}-00000-of-00001.parquet") == manifest["corpus_sha256"][split]
    spec = TASKS[manifest["task"]]
    training = spec.generate_examples(manifest["generator_examples"], manifest["generator_seed"])
    prefixes = {example.prefix_text for example in training}
    tokenizer_examples = spec.generate_examples(manifest["tokenizer_examples"], manifest["tokenizer_seed"])
    tokenizer_targets = {example.target_text for example in tokenizer_examples}
    rows = report["examples"]
    overlap_indices = [i for i, row in enumerate(rows) if row["prefix"] in prefixes]
    unseen = [row for row in rows if row["prefix"] not in prefixes]
    overlapping = [rows[i] for i in overlap_indices]

    def summarize(selected):
        if not selected:
            return {"n": 0}
        return {
            "n": len(selected),
            "entity_f1": 100 * sum(row["entity_f1"] for row in selected) / len(selected),
            "exact_match_rate": 100 * sum(row["exact_match"] for row in selected) / len(selected),
            "valid_rate": sum(row["valid"] for row in selected) / len(selected),
        }

    audit = {
        "evaluation_sha256": file_sha256(args.evaluation),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "generator_train_examples": len(training),
        "registered_validation": summarize(rows),
        "unseen_generator_prefixes": summarize(unseen),
        "seen_generator_prefixes": summarize(overlapping),
        "overlapping_validation_indices": overlap_indices,
        "validation_targets_seen_in_tokenizer_training": sum(row["gold"] in tokenizer_targets for row in rows),
        "selection_rule": "Exact prefix equality only; no prediction- or score-based selection. Main 300 unchanged.",
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

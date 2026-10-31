"""Name-level error analysis for the frozen Few-NERD tokenizer arms.

Reads stored evaluation reports plus the frozen tuning manifest. No new inference,
no protected-test data, and no rescoring of a checkpoint: this only decomposes the
already-recorded greedy reconstructions.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from ncm_graphrag.data.fewnerd_quality_metrics import entities

REPO = Path(__file__).resolve().parents[1]


def _canonical(text):
    return " ".join(text.split()).upper()


def analyse(evaluation, text_by_group, train_names):
    counts = Counter()
    char_bins = Counter()
    for row in evaluation["tuning"]["predictions"]:
        gold = entities(row["gold"])
        predicted = entities(row["reconstruction"])
        text = _canonical(text_by_group[row["group_id"]])
        counts["rows"] += 1
        counts["gold_entities"] += len(gold)
        counts["predicted_entities"] += len(predicted)
        counts["exact_typed"] += len(gold & predicted)
        gold_names = {name for name, _ in gold}
        predicted_names = {name for name, _ in predicted}
        counts["name_only_true_positive"] += len(gold_names & predicted_names)
        counts["type_errors"] += len(gold_names & predicted_names) - len(gold & predicted)
        counts["missing_names"] += len(gold_names - predicted_names)
        counts["extra_names"] += len(predicted_names - gold_names)
        counts["exact_rows"] += gold == predicted
        counts["empty_prediction_rows"] += not predicted
        for name, _ in predicted:
            copied = name in text
            counts["predicted_copied_from_prefix"] += copied
            counts["predicted_absent_from_prefix"] += not copied
        for name, _ in gold:
            bucket = len(name)
            key = "1-5" if bucket <= 5 else "6-10" if bucket <= 10 else "11-20" if bucket <= 20 else "21+"
            char_bins[key + "|seen" if name in train_names else key + "|unseen"] += 1
    total_gold = counts["gold_entities"]
    total_pred = counts["predicted_entities"]
    return {
        "rows": counts["rows"],
        "gold_entities": total_gold,
        "predicted_entities": total_pred,
        "exact_typed_matches": counts["exact_typed"],
        "typed_precision_percent": 100 * counts["exact_typed"] / total_pred if total_pred else None,
        "typed_recall_percent": 100 * counts["exact_typed"] / total_gold if total_gold else None,
        "name_matched_type_wrong": counts["type_errors"],
        "gold_names_missing": counts["missing_names"],
        "predicted_extra_names": counts["extra_names"],
        "predicted_name_copied_from_prefix": counts["predicted_copied_from_prefix"],
        "predicted_name_absent_from_prefix": counts["predicted_absent_from_prefix"],
        "invented_share_of_predicted_names_percent": (
            100 * counts["predicted_absent_from_prefix"] / total_pred if total_pred else None
        ),
        "rows_exactly_correct": counts["exact_rows"],
        "rows_with_empty_prediction": counts["empty_prediction_rows"],
        "gold_name_length_by_seen": dict(sorted(char_bins.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("T0", "T1"))
    parser.add_argument("--step", type=int, default=6000)
    args = parser.parse_args()
    root = REPO / "artifacts" / f"fewnerd-{args.arm.lower()}"
    evaluation = json.loads((root / f"step-{args.step:06d}.json").read_text())
    manifest = REPO / "benchmarks" / "fewnerd-protected"
    tuning = [json.loads(line) for line in (manifest / "tuning.jsonl").read_text().splitlines()]
    training = [json.loads(line) for line in (manifest / "tokenizer-train.jsonl").read_text().splitlines()]
    text_by_group = {row["group_id"]: row["text_unit"] for row in tuning}
    train_names = {name for row in training for name, _ in entities(row["target_text"])}
    result = {
        "arm": args.arm,
        "step": args.step,
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "tuning_entity_f1": evaluation["tuning"]["metrics"]["entity_f1"],
        **analyse(evaluation, text_by_group, train_names),
    }
    destination = root / f"error-analysis-step-{args.step:06d}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

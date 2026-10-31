"""Reusable frozen Few-NERD reconstruction evaluation.

Old numeric metrics are unchanged. Additional diagnostics live in details:
``diagnostic_schema`` freezes bins/exposure thresholds; ``length_subgroups``
counts gold name occurrences, exact names and typed matches by raw character
length and overlapping GPT-2 offset token count. ``seen_unseen`` filters BOTH
gold and predicted typed/name sets using historical strip/upper name identity,
independent of type; includes rows with either gold or predicted subgroup members.
It reports mean per-row and micro F1 with explicit counts/denominators; an absent
subgroup has None scores. ``exposure_subgroups`` counts name occurrences and
subword occurrences with zero/rare (1..5 training occurrences) exposure.
``type_exact`` counts complete gold type fields whose aligned token positions
all match (distinct from token accuracy). ``eos`` reports first-position and
early/missing counts. ``code_utilization`` uses final encoded indices only,
with per-level vocabulary and assignment denominators, never cumulative hits.
Raw predictions and existing name exposure details retain their original shape.
Evaluation requires every module in eval mode and preserves RNG and VQ hit buffers.
"""

from collections import Counter
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from ncm_graphrag.data.fewnerd_quality_metrics import entities, name_spans, score_rows

CHAR_BINS = ((1, 5), (6, 10), (11, 20), (21, None))
TOKEN_BINS = ((1, 1), (2, 2), (3, 4), (5, None))
RARE_EXPOSURE_MAX = 5


def _bin(length, bins):
    return next(
        f"{low}-{high}" if high is not None else f"{low}+"
        for low, high in bins
        if length >= low and (high is None or length <= high)
    )


def subgroup_scores(golds, predictions, train_names, seen):
    """Historical typed/name set scores, including prediction-only rows."""
    counts = Counter(
        rows=0,
        gold_entities=0,
        predicted_entities=0,
        typed_true_positives=0,
        gold_names=0,
        predicted_names=0,
        name_true_positives=0,
    )
    typed_sum = name_sum = 0.0
    for gold, predicted in zip(golds, predictions, strict=True):
        gold = {entry for entry in gold if (entry[0] in train_names) == seen}
        predicted = {entry for entry in predicted if (entry[0] in train_names) == seen}
        if not gold and not predicted:
            continue
        names, guesses = {n for n, _ in gold}, {n for n, _ in predicted}
        overlap, name_overlap = len(gold & predicted), len(names & guesses)
        counts.update(
            rows=1,
            gold_entities=len(gold),
            predicted_entities=len(predicted),
            typed_true_positives=overlap,
            gold_names=len(names),
            predicted_names=len(guesses),
            name_true_positives=name_overlap,
        )
        typed_sum += 2 * overlap / (len(gold) + len(predicted))
        name_sum += 2 * name_overlap / (len(names) + len(guesses))
    typed_den = counts["gold_entities"] + counts["predicted_entities"]
    name_den = counts["gold_names"] + counts["predicted_names"]
    return {
        **counts,
        "typed_micro_f1_denominator": typed_den,
        "name_micro_f1_denominator": name_den,
        "entity_f1": 100 * typed_sum / counts["rows"] if counts["rows"] else None,
        "name_f1": 100 * name_sum / counts["rows"] if counts["rows"] else None,
        "name_reconstruction_recall": 100 * counts["name_true_positives"] / counts["gold_names"]
        if counts["gold_names"]
        else None,
        "name_reconstruction_precision": 100 * counts["name_true_positives"] / counts["predicted_names"]
        if counts["predicted_names"]
        else None,
        "typed_micro_f1": 200 * counts["typed_true_positives"] / typed_den if typed_den else None,
        "name_micro_f1": 200 * counts["name_true_positives"] / name_den if name_den else None,
    }


@contextmanager
def _preserve_evaluation_state(model):
    if any(module.training for module in model.modules()):
        raise ValueError("Reconstruction evaluation requires model.eval()")
    hits = [
        (getattr(model.vq, f"codebook_hit_{i}"), getattr(model.vq, f"codebook_hit_{i}").clone())
        for i in range(len(model.vq.levels))
    ]
    devices = sorted(
        {p.device.index for p in model.parameters() if p.device.type == "cuda" and p.device.index is not None}
    )
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            for buffer, saved in hits:
                buffer.copy_(saved)


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
def _reconstruct(model, rows, tok, train_names, exposure):
    device = next(model.parameters()).device
    predictions = []
    counts = Counter()
    names_by_group = {"seen": Counter(), "unseen": Counter(), "multi_subword": Counter()}
    utilization = [set() for _ in model.vq.levels]
    details = []
    length_groups = {
        kind: {_bin(low, bins): Counter(names=0, exact_names=0, typed_names_correct=0) for low, _ in bins}
        for kind, bins in (("characters", CHAR_BINS), ("tokens", TOKEN_BINS))
    }
    exposure_groups = {
        key: Counter(names=0, exact_names=0, typed_names_correct=0, subword_occurrences=0, subword_correct=0)
        for key in ("zero", "rare_nonzero")
    }
    type_exact_correct = type_exact_count = 0
    vq_loss_sum = 0.0
    nonpad_ce_sum = 0.0
    ce_sum = 0.0
    # Batch geometry is part of the workload and does not depend on environment.
    for start in range(0, len(rows), 8):
        batch_rows = rows[start : start + 8]
        target = torch.tensor(
            [r["target_ids"] + [tok.pad_token_id] * (80 - len(r["target_ids"])) for r in batch_rows],
            dtype=torch.long,
            device=device,
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
            counts["eos_early"] += tok.eos_token_id in ids and ids.index(tok.eos_token_id) < eos
            counts["eos_at_first_position"] += ids[0] == tok.eos_token_id
            aligned, type_positions = alignment(row, tok)
            encoded_offsets = tok(row["target_text"], return_offsets_mapping=True, add_special_tokens=False)[
                "offset_mapping"
            ]
            for _, entity_type, _, end in name_spans(row["target_text"]):
                type_start = end + len("<|>")
                positions = [
                    i
                    for i, (a, b) in enumerate(encoded_offsets)
                    if a < type_start + len(entity_type) and b > type_start
                ]
                type_exact_count += 1
                type_exact_correct += bool(positions) and all(correct[i] for i in positions)
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
                for kind, size, bins in (("characters", len(name), CHAR_BINS), ("tokens", len(indices), TOKEN_BINS)):
                    length_groups[kind][_bin(size, bins)].update(names=1, exact_names=exact, typed_names_correct=typed)
                for key in exposure_groups:
                    positions = [
                        i
                        for i in indices
                        if (exposure[gold[i]] == 0 if key == "zero" else 1 <= exposure[gold[i]] <= RARE_EXPOSURE_MAX)
                    ]
                    if positions:
                        exposure_groups[key].update(
                            names=1,
                            exact_names=exact,
                            typed_names_correct=typed,
                            subword_occurrences=len(positions),
                            subword_correct=sum(correct[i] for i in positions),
                        )
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
    length_details: dict[str, dict[str, dict[str, Any]]] = {
        kind: {key: dict(group) for key, group in bins.items()} for kind, bins in length_groups.items()
    }
    exposure_details: dict[str, dict[str, Any]] = {key: dict(group) for key, group in exposure_groups.items()}
    for group in [group for bins in length_details.values() for group in bins.values()] + list(
        exposure_details.values()
    ):
        group["exact_name_recall"] = 100 * group["exact_names"] / group["names"] if group["names"] else None
        group["typed_name_recall"] = 100 * group["typed_names_correct"] / group["names"] if group["names"] else None
    for group in exposure_details.values():
        group["subword_accuracy"] = (
            100 * group["subword_correct"] / group["subword_occurrences"] if group["subword_occurrences"] else None
        )
    return metrics, {
        "diagnostic_schema": {
            "version": 1,
            "character_bins": CHAR_BINS,
            "token_bins": TOKEN_BINS,
            "zero_exposure": 0,
            "rare_nonzero_exposure_max": RARE_EXPOSURE_MAX,
            "exposure_name_grouping": "At least one qualifying subword; groups can overlap",
            "subgroup_row_inclusion": "gold_or_prediction",
            "name_identity": "strip_upper_independent_of_type",
        },
        "length_subgroups": length_details,
        "exposure_subgroups": exposure_details,
        "seen_unseen": {
            label: subgroup_scores(
                [entities(r["target_text"]) for r in rows], [entities(p) for p in predictions], train_names, seen
            )
            for label, seen in (("seen", True), ("unseen", False))
        },
        "type_exact": {
            "correct": type_exact_correct,
            "fields": type_exact_count,
            "accuracy": 100 * type_exact_correct / type_exact_count if type_exact_count else None,
        },
        "eos": {
            "examples": len(rows),
            "early": counts["eos_early"],
            "missing": counts["eos_missing"],
            "at_first_position": counts["eos_at_first_position"],
            "first_position_correct": counts["eos_first_correct"],
            "gold_position_correct": counts["eos_position_correct"],
        },
        "code_utilization": [
            {
                "level": i,
                "used_codes": len(codes),
                "vocabulary_size": model.vq.vqvae_vocab_sizes[i],
                "assignments": len(rows) * model.vq.levels[i],
                "fraction": len(codes) / model.vq.vqvae_vocab_sizes[i],
            }
            for i, codes in enumerate(utilization)
        ],
        "counts": dict(counts),
        "name_subgroups": names_by_group,
        "entity_count_subgroups": subgroups,
        "per_level_used_codes": [len(codes) for codes in utilization],
        "predictions": details,
    }


def reconstruct(model, rows, tok, train_names, exposure):
    """Evaluate fixed batches of eight without changing caller model/RNG state."""
    if not rows:
        raise ValueError("Reconstruction evaluation requires nonempty rows")
    with torch.inference_mode(), _preserve_evaluation_state(model):
        return _reconstruct(model, rows, tok, train_names, exposure)

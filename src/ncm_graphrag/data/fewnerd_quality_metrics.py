"""Frozen CPU-only scoring and train-only dictionary controls for Few-NERD.

Scoring deliberately retains the historical parser and strip/uppercase typed-set
normalization; NFC/whitespace normalization belongs only to dictionary matching.
The strict grammar check is separate from the historical validity predicate.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from ncm_graphrag.data.extraction import (
    COMPLETION_DELIMITER,
    RECORD_DELIMITER,
    TUPLE_DELIMITER,
    is_well_formed,
    parse_records,
)
from ncm_graphrag.data.fewnerd_extraction import COARSE_TYPES
from ncm_graphrag.data.real_extraction import _record

# Field exclusions are the shared serializer's _is_safe_field exclusions. In
# particular, a field cannot smuggle in a delimiter, extra record, or parenthesis.
_FIELD = r'[^<|>#"()\r\n]+'
_RECORD_PATTERN = (
    r'\("entity"'
    + re.escape(TUPLE_DELIMITER)
    + f"({_FIELD})"
    + re.escape(TUPLE_DELIMITER)
    + "("
    + "|".join(sorted(COARSE_TYPES.values()))
    + ")"
    + re.escape(TUPLE_DELIMITER)
    + f"({_FIELD})"
    + r"\)"
)
_RECORD_RE = re.compile(_RECORD_PATTERN)
_STRICT_TARGET_RE = re.compile(
    _RECORD_PATTERN
    + "(?:"
    + re.escape(RECORD_DELIMITER)
    + _RECORD_PATTERN
    + ")?"
    + re.escape(RECORD_DELIMITER + COMPLETION_DELIMITER)
)


def entities(text: str) -> set[tuple[str, str]]:
    """Return exactly the historical normalized typed entity set."""
    return {
        (entity["name"].strip().upper(), entity["type"].strip().upper()) for entity in parse_records(text)["entities"]
    }


def name_spans(target_text: str) -> list[tuple[str, str, int, int]]:
    """Validate a complete one/two-record target and locate serialized names.

    Only literal ``("entity"<|>name<|>KNOWN_TYPE<|>description)`` records
    separated by ``##`` and ending in ``##<|COMPLETE|>`` are accepted. Names
    must survive the historical parser's three-character minimum; descriptions
    must be nonblank. Names/descriptions cannot contain grammar metacharacters
    or newlines. There is no stripping of the target or trailing-text tolerance.
    Returned half-open character offsets index the original target string, not
    normalized names, so tokenizer offsets can align Unicode and punctuation.
    """
    if _STRICT_TARGET_RE.fullmatch(target_text) is None:
        raise ValueError("target is not a complete one/two-record Few-NERD target")
    spans: list[tuple[str, str, int, int]] = []
    for match in _RECORD_RE.finditer(target_text):
        name, entity_type, description = match.groups()
        if len(name.strip()) < 3 or not description.strip():
            raise ValueError("target has an unscorable name or blank description")
        start, end = match.span(1)
        spans.append((name, entity_type, start, end))
    return spans


def _strict_validity(text: str) -> bool:
    try:
        name_spans(text)
    except ValueError:
        return False
    return True


def score_rows(golds: list[str], predictions: list[str]) -> dict[str, float]:
    """Score aligned rows in percentage points, including historical mean F1.

    Empty predictions receive zero per-example F1, even for an empty gold set.
    Micro metrics use aggregate typed-set counts and zero for zero denominators.
    Target exactness compares literal strings, without stripping or normalization.
    Historical validity remains the registered ``is_well_formed`` predicate;
    strict validity additionally requires the anchored complete-record grammar.
    """
    if not golds or len(golds) != len(predictions):
        raise ValueError("scoring requires nonempty, equally sized gold/prediction lists")
    mean_f1 = 0.0
    true_positives = gold_count = predicted_count = 0
    set_exact = target_exact = valid = strict_valid = 0
    for gold_text, prediction in zip(golds, predictions, strict=True):
        gold, predicted = entities(gold_text), entities(prediction)
        overlap = len(gold & predicted)
        mean_f1 += 2 * overlap / max(len(gold) + len(predicted), 1)
        true_positives += overlap
        gold_count += len(gold)
        predicted_count += len(predicted)
        set_exact += gold == predicted
        target_exact += gold_text == prediction
        valid += is_well_formed(prediction)
        strict_valid += _strict_validity(prediction)
    count = len(golds)
    return {
        "entity_f1": 100 * mean_f1 / count,
        "micro_precision": 100 * true_positives / max(predicted_count, 1),
        "micro_recall": 100 * true_positives / max(gold_count, 1),
        "micro_f1": 200 * true_positives / max(gold_count + predicted_count, 1),
        "entity_set_exact": 100 * set_exact / count,
        "target_exact": 100 * target_exact / count,
        "validity": 100 * valid / count,
        "strict_validity": 100 * strict_valid / count,
    }


def _matching_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _word_character(character: str) -> bool:
    return character.isalnum() or character == "_" or unicodedata.category(character).startswith("M")


def dictionary_predictions(
    training_rows: Sequence[Mapping[str, Any]],
    tuning_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Predict using the following frozen, deterministic train-only rule.

    Entries are unique (name, type) pairs from strict training target records
    only. Match names and input text after NFC and whitespace collapse/strip,
    then uppercase, exactly the scorer's case operation (not casefold or regex
    IGNORECASE). Uppercase expansions must match whole source characters.
    Adjacent source characters outside a match may not be Unicode alphanumeric,
    combining marks, or underscore. Use only each row's supplied Entity_types;
    tuning target_text and all other held-out annotations are never accessed.

    Rank candidates by descending normalized-source character span length, then
    leftmost start, then normalized uppercase name and type lexically. Greedily
    take nonoverlapping spans, at most two; repeated disjoint mentions remain
    candidates. Serialize in sentence order with the shared _record, copying
    normalized-source spelling, and append the shared completion delimiter. No
    match produces only <|COMPLETE|> (historically/strictly invalid, zero F1).
    A character trie avoids rescanning the entire training vocabulary per input.
    """
    # Every node has character edges and, at the empty-string key, a type set.
    # The sentinel cannot collide with a character edge.
    trie: dict[str, Any] = {}
    for row in training_rows:
        for name, entity_type, _, _ in name_spans(row["target_text"]):
            normalized_name = _matching_text(name).upper()
            node = trie
            for character in normalized_name:
                node = node.setdefault(character, {})
            node.setdefault("", set()).add(entity_type)

    predictions: list[str] = []
    for row in tuning_rows:
        text = _matching_text(row["text_unit"])
        candidate_types = {value.strip().upper() for value in row["entity_types"].split(",")}
        upper_parts = [character.upper() for character in text]
        upper_text = "".join(upper_parts)
        # Case conversion can expand one source character, e.g. ß -> SS.
        # Preserve source spans and disallow a match ending inside an expansion.
        source_positions = [index for index, part in enumerate(upper_parts) for _ in part]
        candidates: list[tuple[int, int, str, str]] = []
        upper_start = 0
        for start, part in enumerate(upper_parts):
            if start == 0 or not _word_character(text[start - 1]):
                node = trie
                for position in range(upper_start, len(upper_text)):
                    node = node.get(upper_text[position])
                    if node is None:
                        break
                    upper_end = position + 1
                    if upper_end < len(upper_text) and source_positions[upper_end] == source_positions[position]:
                        continue
                    end = source_positions[position] + 1
                    if end < len(text) and _word_character(text[end]):
                        continue
                    for entity_type in node.get("", ()):
                        if entity_type in candidate_types:
                            candidates.append((start, end, upper_text[upper_start:upper_end], entity_type))
            upper_start += len(part)
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2], item[3]))
        selected: list[tuple[int, int, str, str]] = []
        for candidate in candidates:
            start, end, _, _ = candidate
            if any(start < other[1] and other[0] < end for other in selected):
                continue
            selected.append(candidate)
            if len(selected) == 2:
                break
        selected.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        records = [_record(text[start:end], entity_type) for start, end, _, entity_type in selected]
        predictions.append(RECORD_DELIMITER.join(records + [COMPLETION_DELIMITER]))
    return predictions

"""Bounded GraphRAG entity-and-relationship extraction from REBEL.

REBEL supplies Wikipedia text plus distantly supervised Wikidata triples. Each example
keeps subject/object surface forms verbatim, uses the predicate surface label as the
GraphRAG relationship description, and serializes only complete records that fit the
fixed target budget. Re-DocRED and DocRED adapters use the same serialization for gold
transfer evaluation; they are never part of the REBEL training pool.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import importlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ncm_graphrag.data.corpora import get_corpus_path, load_docred, load_re_docred
from ncm_graphrag.data.extraction import (
    COMPLETION_DELIMITER,
    RECORD_DELIMITER,
    TUPLE_DELIMITER,
    parse_records,
)

PREFIX_TOKEN_BUDGET = 512
TARGET_TOKEN_BUDGET = 128
BENCHMARK_SIZE = 256
UNKNOWN_ENTITY_TYPE = "UNKNOWN"
RELATION_WEIGHT = "1"

_SPLIT_FILES = {
    "train": "en_train.jsonl",
    "validation": "en_val.jsonl",
    "test": "en_test.jsonl",
}


@dataclass(frozen=True)
class GraphExtractionExample:
    prefix_text: str
    target_text: str
    source_id: str
    dataset: str


@dataclass(frozen=True)
class _Triple:
    subject: str
    object: str
    predicate: str
    subject_type: str = UNKNOWN_ENTITY_TYPE
    object_type: str = UNKNOWN_ENTITY_TYPE


_TOK = None


def _pre_tok():
    global _TOK
    if _TOK is None:
        importlib.import_module("ncm_graphrag.ncm._ncm_repo")
        from tokenization.hf_tokenizer import HFTokenizer

        _TOK = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
    return _TOK


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_field(text: str) -> bool:
    return (
        bool(text)
        and text == text.strip()
        and not any(marker in text for marker in (TUPLE_DELIMITER, RECORD_DELIMITER, '"', "(", ")"))
    )


def _clip_prefix(text: str) -> str:
    prefix = f"Text: {_norm(text)}"
    tok = _pre_tok()
    ids = tok.encode(prefix)
    return prefix if len(ids) <= PREFIX_TOKEN_BUDGET else tok.decode(ids[:PREFIX_TOKEN_BUDGET])


def _entity_record(name: str, entity_type: str) -> str:
    return (
        f'("entity"{TUPLE_DELIMITER}{name}{TUPLE_DELIMITER}{entity_type}{TUPLE_DELIMITER}Mentioned in the source text.)'
    )


def _relationship_record(triple: _Triple) -> str:
    return (
        f'("relationship"{TUPLE_DELIMITER}{triple.subject}{TUPLE_DELIMITER}{triple.object}'
        f"{TUPLE_DELIMITER}{triple.predicate}{TUPLE_DELIMITER}{RELATION_WEIGHT})"
    )


def _serialize(triples: list[_Triple]) -> str:
    entities: dict[str, tuple[str, str]] = {}
    for triple in triples:
        entities.setdefault(triple.subject.casefold(), (triple.subject, triple.subject_type))
        entities.setdefault(triple.object.casefold(), (triple.object, triple.object_type))
    records = [_entity_record(name, entity_type) for name, entity_type in entities.values()]
    records.extend(_relationship_record(triple) for triple in triples)
    records.append(COMPLETION_DELIMITER)
    return RECORD_DELIMITER.join(records)


def _bounded_example(
    text: str,
    triples: list[_Triple],
    *,
    source_id: str,
    dataset: str,
    cap_target: bool = True,
) -> GraphExtractionExample | None:
    prefix = _clip_prefix(text)
    selected: list[_Triple] = []
    seen: set[tuple[str, str, str]] = set()
    tok = _pre_tok()

    for triple in triples:
        fields = (triple.subject, triple.object, triple.predicate, triple.subject_type, triple.object_type)
        if not all(_safe_field(field) for field in fields):
            continue
        if triple.subject not in prefix or triple.object not in prefix:
            continue
        key = (triple.subject.casefold(), triple.object.casefold(), triple.predicate.casefold())
        if key in seen:
            continue
        candidate = _serialize([*selected, triple])
        if cap_target and len(tok.encode(candidate)) + 1 > TARGET_TOKEN_BUDGET:
            continue
        selected.append(triple)
        seen.add(key)
    if not selected:
        return None
    target = _serialize(selected)
    return GraphExtractionExample(prefix_text=prefix, target_text=target, source_id=source_id, dataset=dataset)


def _rebel_triples(record: dict[str, Any]) -> list[_Triple]:
    triples: list[_Triple] = []
    for raw in record.get("triples") or []:
        subject = str((raw.get("subject") or {}).get("surfaceform") or "")
        object_ = str((raw.get("object") or {}).get("surfaceform") or "")
        predicate = _norm(str((raw.get("predicate") or {}).get("surfaceform") or ""))
        triples.append(_Triple(subject=subject, object=object_, predicate=predicate))
    return triples


def example_from_rebel(record: dict[str, Any]) -> GraphExtractionExample | None:
    """Convert one raw REBEL record without truncating any tuple record."""
    return _bounded_example(
        str(record.get("text") or ""),
        _rebel_triples(record),
        source_id=str(record.get("title") or record.get("docid") or record.get("uri") or ""),
        dataset="rebel",
    )


def _rebel_path(split: str) -> Path:
    if split not in _SPLIT_FILES:
        raise ValueError(f"Unknown REBEL split {split!r}; expected one of {sorted(_SPLIT_FILES)}")
    return get_corpus_path("rebel") / "extracted" / _SPLIT_FILES[split]


@functools.lru_cache(maxsize=1)
def _gold_holdout_titles() -> frozenset[str]:
    """Exclude gold evaluation articles from REBEL training and selection by Wikipedia title."""
    titles = set()
    for loader in (load_docred, load_re_docred):
        for split in ("dev", "test"):
            for record in loader(split):
                if record.get("title"):
                    titles.add(_norm(str(record["title"])).casefold())
    return frozenset(titles)


def _split_examples(split: str, n: int, seed: int) -> list[GraphExtractionExample]:
    """Read usable examples excluding gold holdout titles, then shuffle deterministically."""
    examples: list[GraphExtractionExample] = []
    with _rebel_path(split).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if _norm(str(record.get("title") or "")).casefold() in _gold_holdout_titles():
                continue
            example = example_from_rebel(record)
            if example is not None:
                examples.append(example)
                if len(examples) >= n:
                    break
    if len(examples) < n:
        raise ValueError(f"requested {n} {split} examples but only {len(examples)} are usable")
    random.Random(seed).shuffle(examples)
    return examples


def validation_partitions(n: int) -> tuple[list[GraphExtractionExample], list[GraphExtractionExample]]:
    """Partition by article so checkpoint selection and reporting cannot share an article."""
    pool = _split_examples("validation", n * 4, 2)
    partitions: tuple[list[GraphExtractionExample], list[GraphExtractionExample]] = ([], [])
    for example in pool:
        bucket = hashlib.sha256(example.source_id.encode("utf-8")).digest()[0] % 2
        if len(partitions[bucket]) < n:
            partitions[bucket].append(example)
    if any(len(partition) < n for partition in partitions):
        raise ValueError("not enough disjoint REBEL validation articles for selection and evaluation")
    return partitions


def generate_examples(n: int, seed: int) -> list[GraphExtractionExample]:
    return _split_examples("train", n, seed)


def benchmark_examples() -> list[GraphExtractionExample]:
    return _split_examples("validation", BENCHMARK_SIZE, 0)


@functools.lru_cache(maxsize=1)
def _relation_names() -> dict[str, str]:
    path = get_corpus_path("docred") / "data" / "rel_info.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data: dict[str, str] = json.load(handle)
    return data


def _mention_for_prefix(cluster: list[dict[str, Any]], prefix: str) -> tuple[str, str] | None:
    for mention in cluster:
        name = _norm(str(mention.get("name") or ""))
        entity_type = _norm(str(mention.get("type") or UNKNOWN_ENTITY_TYPE))
        if name and name in prefix and _safe_field(name) and _safe_field(entity_type):
            return name, entity_type
    return None


def _example_from_docred(record: dict[str, Any], *, dataset: str) -> GraphExtractionExample | None:
    text = " ".join(" ".join(sentence) for sentence in record.get("sents") or [])
    prefix = _clip_prefix(text)
    clusters = record.get("vertexSet") or []
    relation_names = _relation_names()
    triples: list[_Triple] = []
    for label in record.get("labels") or []:
        h, t = int(label["h"]), int(label["t"])
        if h >= len(clusters) or t >= len(clusters):
            continue
        subject = _mention_for_prefix(clusters[h], prefix)
        object_ = _mention_for_prefix(clusters[t], prefix)
        predicate = relation_names.get(str(label.get("r")), str(label.get("r") or ""))
        if subject is None or object_ is None:
            continue
        triples.append(
            _Triple(
                subject=subject[0],
                object=object_[0],
                predicate=predicate,
                subject_type=subject[1],
                object_type=object_[1],
            )
        )
    return _bounded_example(
        text,
        triples,
        source_id=str(record.get("title") or ""),
        dataset=dataset,
        cap_target=False,
    )


def gold_pool(dataset: str, split: str) -> tuple[list[GraphExtractionExample], dict[str, int]]:
    """Build gold examples and report how much of the original split is visible to the model."""
    if dataset == "re_docred":
        rows = load_re_docred(split)
    elif dataset == "docred":
        rows = load_docred(split)
    else:
        raise ValueError("dataset must be 're_docred' or 'docred'")
    examples = [example for row in rows if (example := _example_from_docred(row, dataset=dataset)) is not None]
    stats = {
        "total_documents": len(rows),
        "eligible_documents": len(examples),
        "original_relation_labels": sum(len(row.get("labels") or []) for row in rows),
        "visible_relation_triples": sum(len(graph_sets(example.target_text)[2]) for example in examples),
    }
    return examples, stats


def gold_examples(dataset: str, split: str, n: int, seed: int) -> list[GraphExtractionExample]:
    """Sample eligible gold transfer documents without truncating their visible labels."""
    examples, _ = gold_pool(dataset, split)
    random.Random(seed).shuffle(examples)
    if len(examples) < n:
        raise ValueError(f"requested {n} {dataset}/{split} examples but only {len(examples)} are usable")
    return examples[:n]


def is_well_formed(text: str) -> bool:
    parsed = parse_records(text)
    return bool(parsed["entities"] and parsed["relationships"] and COMPLETION_DELIMITER in text)


def graph_sets(text: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str, str]]]:
    """Return name-only entities, typed entities, and relation triples for scoring."""
    parsed = parse_records(text)
    names = {item["name"].strip().casefold() for item in parsed["entities"]}
    typed = {(item["name"].strip().casefold(), item["type"].strip().casefold()) for item in parsed["entities"]}
    relations = {
        (
            item["source"].strip().casefold(),
            item["target"].strip().casefold(),
            item["description"].strip().casefold(),
        )
        for item in parsed["relationships"]
    }
    return {(name, "") for name in names}, typed, relations

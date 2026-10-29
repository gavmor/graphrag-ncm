"""Real-data entity extraction dataset built from Few-NERD.

Same target grammar and output budget as `ncm_graphrag.data.real_extraction`, but a much
larger and more varied source: Re-DocRED yields ~8.5k single-sentence examples, which a
12k-step run covers ~192 times — measured 42.9% entity F1 in-sample vs 2.9% on held-out
documents, i.e. memorisation rather than copying. Few-NERD supervised provides 131,767
training sentences with real entity spans over 8 coarse types, enough that repeated epochs
no longer dominate.

Source: `DFKI-SLT/few-nerd` (`supervised` split), fetched by `scripts/fetch_corpora.py`.
Coarse tag ids follow the dataset card: 0=O, 1=art, 2=building, 3=event, 4=location,
5=organization, 6=other, 7=person, 8=product. Spans are recovered by grouping consecutive
tokens that carry the same non-zero tag.

Splits stay honest: training reads `train`, the benchmark reads `validation`.
"""

import random
from functools import lru_cache
from pathlib import Path

from ncm_graphrag.data.extraction import (
    COMPLETION_DELIMITER,
    RECORD_DELIMITER,
    ExtractionExample,
)
from ncm_graphrag.data.real_extraction import (
    PREFIX_TOKEN_BUDGET,
    TARGET_TOKEN_BUDGET,
    _is_safe_field,
    _norm,
    _pre_tok,
    _record,
)

CORPUS_NAME = "few_nerd"
_SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

COARSE_TYPES = {
    1: "ART",
    2: "BUILDING",
    3: "EVENT",
    4: "LOCATION",
    5: "ORGANIZATION",
    6: "OTHER",
    7: "PERSON",
    8: "PRODUCT",
}

MAX_ENTITIES = 2


def _corpora_root() -> Path:
    import os

    env = os.environ.get("NCM_CORPORA_ROOT")
    root = (
        Path(env).expanduser()
        if env
        else Path(__file__).resolve().parents[3].parent / ("ncm-graphrag-worktrees") / "training-corpora" / "corpora"
    )
    marker = root / "information_extraction" / CORPUS_NAME
    if not marker.is_dir():
        raise FileNotFoundError(
            f"Few-NERD not found under {root!r}. Fetch it with "
            "`uv run python scripts/fetch_corpora.py --dataset few_nerd` in the "
            "training-corpora worktree, or set NCM_CORPORA_ROOT."
        )
    return root


def _spans(tokens: list[str], tags: list[int]) -> list[tuple[str, str]]:
    """Group consecutive same-tag tokens into entity mentions."""
    out: list[tuple[str, str]] = []
    current: list[str] = []
    current_tag = 0
    for token, tag in zip(tokens, tags, strict=False):
        if tag != current_tag:
            if current_tag in COARSE_TYPES and current:
                out.append((" ".join(current), COARSE_TYPES[current_tag]))
            current, current_tag = [], tag
        if current_tag != 0:
            current.append(token)
    if current_tag in COARSE_TYPES and current:
        out.append((" ".join(current), COARSE_TYPES[current_tag]))
    return out


@lru_cache(maxsize=4)
def _build(split: str) -> tuple[ExtractionExample, ...]:
    import pyarrow.parquet as pq

    tok = _pre_tok()
    path = _corpora_root() / "information_extraction" / CORPUS_NAME / "supervised" / _SPLIT_FILES[split]
    table = pq.read_table(path, columns=["tokens", "ner_tags"])
    out: list[ExtractionExample] = []

    for tokens, tags in zip(table["tokens"].to_pylist(), table["ner_tags"].to_pylist(), strict=False):
        text_unit = _norm(" ".join(tokens))
        if not _is_safe_field(text_unit):
            continue

        seen: set[tuple[str, str]] = set()
        entities: list[tuple[str, str]] = []
        for raw_name, entity_type in _spans(tokens, tags):
            name = _norm(raw_name)
            key = (name.upper(), entity_type)
            # parse_records drops names shorter than 3 characters, so keep gold targets
            # consistent with what the scorer can ever match.
            if key in seen or not _is_safe_field(name) or len(name) < 3:
                continue
            seen.add(key)
            entities.append((name, entity_type))
        if not 1 <= len(entities) <= MAX_ENTITIES:
            continue

        entity_types = ",".join(sorted({t for _, t in entities}))
        target = RECORD_DELIMITER.join([_record(n, t) for n, t in entities] + [COMPLETION_DELIMITER])
        prefix = f"Entity_types: {entity_types}\nText: {text_unit}"
        if len(tok.encode(prefix)) > PREFIX_TOKEN_BUDGET:
            continue
        if len(tok.encode(target)) + 1 > TARGET_TOKEN_BUDGET:
            continue
        out.append(ExtractionExample(text_unit=text_unit, entity_types=entity_types, target_text=target))
    return tuple(out)


def _split_examples(split: str, n: int, seed: int) -> list[ExtractionExample]:
    pool = list(_build(split))
    rng = random.Random(seed)
    rng.shuffle(pool)
    if n > len(pool):
        raise ValueError(f"requested {n} {split} examples but only {len(pool)} are usable")
    return pool[:n]


def generate_examples(n: int, seed: int) -> list[ExtractionExample]:
    """Training split. Deterministic for a given (n, seed)."""
    return _split_examples("train", n, seed)


def eval_examples(split: str, n: int, seed: int) -> list[ExtractionExample]:
    """Held-out split (`validation` or `test`) — sentences never trained on."""
    if split not in ("validation", "test"):
        raise ValueError("eval split must be 'validation' or 'test'")
    return _split_examples(split, n, seed)


if __name__ == "__main__":
    import json

    for name in ("train", "validation"):
        print(f"{name}: {len(_build(name))} usable examples")
    for example in _split_examples("validation", 3, 42):
        print(json.dumps({"text_unit": example.text_unit, "target": example.target_text}, indent=2))

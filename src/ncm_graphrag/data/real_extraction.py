"""Real-data entity extraction dataset built from Re-DocRED.

The synthetic generator in `ncm_graphrag.data.extraction` uses a 40x20 stem/suffix grid.
That is enough to prove the pipeline, but it caps what copying can be shown to do: the
model learned to copy a novel *stem* into name position 1 and then emit a memorised
*suffix*. Real corpora remove that artefact — entity names come from actual Wikipedia
text (arbitrary morphology, punctuation, multi-token) and the entity types are a real
open set (PER / LOC / ORG / MISC / TIME / NUM) rather than a single constant.

Source: Re-DocRED (`tonytan48/Re-DocRED`), the re-annotated DocRED. Fetched by
`scripts/fetch_corpora.py` in the `training-corpora` worktree. Each example is one
sentence plus the gold entities annotated in that sentence, serialised into graphrag's
exact delimiter-tuple grammar.

Splits are kept honest: training reads `train_revised.json`; the benchmark and held-out
checks read `dev_revised.json` / `test_revised.json`, which are different documents.

Examples are filtered to fit the model's existing bounds (prefix <= 64 GPT-2 tokens,
target <= 80), so this dataset can train the current tokenizer/generator architecture
without retuning it — the only variable that changes is the data.
"""

import importlib
import json
import os
import random
import re
from pathlib import Path

from ncm_graphrag.data.extraction import (
    COMPLETION_DELIMITER,
    RECORD_DELIMITER,
    TUPLE_DELIMITER,
    ExtractionExample,
)

_SPLIT_FILES = {
    "train": "train_revised.json",
    "dev": "dev_revised.json",
    "test": "test_revised.json",
}

PREFIX_TOKEN_BUDGET = 64
TARGET_TOKEN_BUDGET = 80

_TOKENIZER = None
_BOUNDS_CHECKED: set[str] = set()


def _corpora_root() -> Path:
    env = os.environ.get("NCM_CORPORA_ROOT")
    if env:
        root = Path(env).expanduser()
    else:
        root = Path(__file__).resolve().parents[3].parent / "ncm-graphrag-worktrees" / "training-corpora" / "corpora"
    marker = root / "information_extraction" / "re_docred"
    if not marker.is_dir():
        raise FileNotFoundError(
            f"Re-DocRED not found under {root!r}. Fetch it with "
            "`uv run python scripts/fetch_corpora.py --dataset re_docred` in the "
            "training-corpora worktree, or set NCM_CORPORA_ROOT to that corpora directory."
        )
    return root


def _pre_tok():
    global _TOKENIZER
    if _TOKENIZER is None:
        # Importing this module is a sys.path side effect that puts the vendored ncm repo
        # on the path; done via importlib so isort cannot reorder it after the import below.
        importlib.import_module("ncm_graphrag.ncm._ncm_repo")
        from tokenization.hf_tokenizer import HFTokenizer

        _TOKENIZER = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
    return _TOKENIZER


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_safe_field(text: str) -> bool:
    """Reject anything that would corrupt the delimiter grammar."""
    return bool(text) and not any(ch in text for ch in '<|>#"()')


def _record(name: str, entity_type: str) -> str:
    """Emit the entity name exactly as it appears in the source text.

    Names used to be upper-cased here, which forced the generator to perform a case
    transform while copying (`Warrick Dunn` -> `WARRICK DUNN`, entirely different BPE
    tokens). Scoring is case-insensitive either way, so the upper-casing bought nothing and
    made an already-hard copy task harder: every gold name now occurs verbatim in the prefix
    (verified 438/438 = 100% on Few-NERD validation).
    """
    desc = "Mentioned in the text."
    return f'("entity"{TUPLE_DELIMITER}{name}{TUPLE_DELIMITER}{entity_type}{TUPLE_DELIMITER}{desc})'


def _iter_docs(split: str):
    path = _corpora_root() / "information_extraction" / "re_docred" / _SPLIT_FILES[split]
    with path.open(encoding="utf-8") as handle:
        yield from json.load(handle)


def _build(split: str, max_entities: int = 2) -> list[ExtractionExample]:
    tok = _pre_tok()
    out: list[ExtractionExample] = []
    for doc in _iter_docs(split):
        sents = doc.get("sents") or []
        per_sentence: dict[int, list[tuple[str, str]]] = {}
        for cluster in doc.get("vertexSet") or []:
            if not cluster:
                continue
            mention = cluster[0]
            per_sentence.setdefault(int(mention["sent_id"]), []).append((_norm(mention["name"]), str(mention["type"])))

        for sent_id, mentions in per_sentence.items():
            if sent_id >= len(sents):
                continue
            seen: set[tuple[str, str]] = set()
            entities: list[tuple[str, str]] = []
            for name, entity_type in mentions:
                key = (name.upper(), entity_type)
                # parse_records drops names shorter than 3 characters, so keep gold targets
                # consistent with what the scorer can ever match.
                if key in seen or not _is_safe_field(name) or len(name) < 3:
                    continue
                seen.add(key)
                entities.append((name, entity_type))
            if not 1 <= len(entities) <= max_entities:
                continue

            text_unit = " ".join(sents[sent_id])
            if not _is_safe_field(text_unit):
                continue

            entity_types = ",".join(sorted({t for _, t in entities}))
            target = RECORD_DELIMITER.join([_record(n, t) for n, t in entities] + [COMPLETION_DELIMITER])
            prefix = f"Entity_types: {entity_types}\nText: {text_unit}"

            if len(tok.encode(prefix)) > PREFIX_TOKEN_BUDGET:
                continue
            if len(tok.encode(target)) + 1 > TARGET_TOKEN_BUDGET:
                continue
            out.append(ExtractionExample(text_unit=text_unit, entity_types=entity_types, target_text=target))
    return out


def _split_examples(split: str, n: int, seed: int) -> list[ExtractionExample]:
    pool = _build(split)
    rng = random.Random(seed)
    rng.shuffle(pool)
    if n > len(pool):
        raise ValueError(f"requested {n} {split} examples but only {len(pool)} are usable")
    return pool[:n]


def generate_examples(n: int, seed: int) -> list[ExtractionExample]:
    """Training split. Deterministic for a given (n, seed)."""
    return _split_examples("train", n, seed)


def eval_examples(split: str, n: int, seed: int) -> list[ExtractionExample]:
    """Held-out split (`dev` or `test`) — different documents from training."""
    if split not in ("dev", "test"):
        raise ValueError("eval split must be 'dev' or 'test'")
    return _split_examples(split, n, seed)


if __name__ == "__main__":
    for name in ("train", "dev", "test"):
        pool = _build(name)
        print(f"{name}: {len(pool)} usable examples")
    for example in _split_examples("dev", 3, 41):
        print(json.dumps({"text_unit": example.text_unit, "target": example.target_text}, indent=2))

"""Closed-vocabulary relation classification on TACRED (Lead 2).

Task. Given a sentence plus two marked entity mentions (head, tail), emit the relation between
them: one of 41 TACRED relations or `NA`. Unlike entity extraction the output is a member of a
FIXED label set, so nothing has to be copied from the input and NCM's parallel coarse-to-fine
decoding faces no open-vocabulary bottleneck (docs/constraints.md §7). This is the "decoupled
relation edge" slice of the brief: GraphRAG already has the entities; it needs the edge type.

Format. `Head: <entity>\\nTail: <entity>\\nText: <sentence>` -> target is the bare relation label
(`per:title`, `org:founded`, `NA`, ...). The label is the whole target, so the model's decoded text
is looked up directly in the label set; anything else counts as invalid.

Data. The copy staged here is OpenNRE-format TACRED (68,124 train / 22,631 val / 15,509 test, 42
classes). It has entity names and token spans but NO entity types. 81% of train is `NA`. The val
file is SORTED, so the first N lines cover only a few classes; benchmarks must use a seeded random
sample, never a prefix.

Metric. Micro-F1 over non-`NA` gold/predicted relations (the standard TACRED convention): `NA`
predictions never count as a hit, so a model that always says `NA` scores 0, not 81%.
"""

import functools
import importlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

CORPUS_REL = Path("information_extraction/tacred/data")
_SPLIT_FILES = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}

NO_RELATION = "NA"
PREFIX_TOKEN_BUDGET = 128
BENCHMARK_SIZE = 3000
BENCHMARK_SEED = 0

_PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}


@dataclass(frozen=True)
class RelationExample:
    prefix_text: str
    target_text: str
    relation: str


def _root() -> Path:
    base = os.environ.get("NCM_CORPORA_ROOT")
    corpora = Path(base) if base else Path(__file__).resolve().parents[3] / "corpora"
    root = corpora / CORPUS_REL
    if not root.exists():
        raise FileNotFoundError(f"TACRED not found at {root}. Fetch the corpora or set NCM_CORPORA_ROOT.")
    return root


_TOK = None


def _pre_tok():
    global _TOK
    if _TOK is None:
        importlib.import_module("ncm_graphrag.ncm._ncm_repo")
        from tokenization.hf_tokenizer import HFTokenizer

        _TOK = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
    return _TOK


@functools.lru_cache(maxsize=1)
def _label_set() -> frozenset[str]:
    with open(_root() / "rel2id.json", encoding="utf-8") as f:
        return frozenset(json.load(f))


def labels() -> list[str]:
    """All relation labels, sorted, including `NA`."""
    return sorted(_label_set())


def _clean(token: str) -> str:
    return _PTB.get(token, token)


def _raw(split: str) -> list[dict]:
    out = []
    with open(_root() / _SPLIT_FILES[split], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _example(rec: dict) -> RelationExample:
    text = " ".join(_clean(t) for t in rec["token"])
    head = " ".join(_clean(t) for t in rec["token"][rec["h"]["pos"][0] : rec["h"]["pos"][1]])
    tail = " ".join(_clean(t) for t in rec["token"][rec["t"]["pos"][0] : rec["t"]["pos"][1]])
    return RelationExample(
        prefix_text=f"Head: {head}\nTail: {tail}\nText: {text}",
        target_text=rec["relation"],
        relation=rec["relation"],
    )


def _fits(ex: RelationExample) -> bool:
    return len(_pre_tok().encode(ex.prefix_text)) <= PREFIX_TOKEN_BUDGET


def generate_examples(n: int, seed: int) -> list[RelationExample]:
    """Deterministic sample of train examples that fit the prefix window (natural class mix)."""
    rng = random.Random(seed)
    rows = _raw("train")
    rng.shuffle(rows)
    out: list[RelationExample] = []
    for rec in rows:
        ex = _example(rec)
        if _fits(ex):
            out.append(ex)
            if len(out) >= n:
                break
    return out


def benchmark_examples() -> list[RelationExample]:
    """Fixed held-out set: a seeded random sample of the *validation* split, natural class mix.

    Random, because val.txt is sorted by relation-ish blocks. Examples over the prefix budget
    (0.3%) are skipped, not truncated, so no benchmark entity is ever cut off.
    """
    rng = random.Random(BENCHMARK_SEED)
    rows = _raw("validation")
    rng.shuffle(rows)
    out: list[RelationExample] = []
    for rec in rows:
        ex = _example(rec)
        if _fits(ex):
            out.append(ex)
            if len(out) >= BENCHMARK_SIZE:
                break
    return out


def is_well_formed(text: str) -> bool:
    return text.strip() in _label_set()


def predicted_label(text: str) -> str | None:
    t = text.strip()
    return t if t in _label_set() else None


def micro_f1(gold: list[str], pred: list[str | None]) -> tuple[float, float, float]:
    """(precision, recall, F1) over non-NA relations. An invalid prediction is a miss."""
    tp = sum(1 for g, p in zip(gold, pred, strict=False) if g != NO_RELATION and p == g)
    n_pred = sum(1 for p in pred if p is not None and p != NO_RELATION)
    n_gold = sum(1 for g in gold if g != NO_RELATION)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

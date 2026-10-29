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
import re
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


def _marked_text(rec: dict) -> str:
    """Sentence with the head wrapped in `@ ... @` and the tail in `# ... #`.

    Inline typed-position markers are the standard relation-extraction input encoding: they tell
    the model WHERE each entity sits and which one is the head (direction matters: per:employee_of
    vs org:top_members/employees), instead of making it match the Head:/Tail: strings back into the
    sentence. Closers are emitted before openers at a shared index so adjacent spans stay balanced.
    """
    toks = [_clean(t) for t in rec["token"]]
    h0, h1 = rec["h"]["pos"]
    t0, t1 = rec["t"]["pos"]
    out: list[str] = []
    for i in range(len(toks) + 1):
        if i == h1:
            out.append("@")
        if i == t1:
            out.append("#")
        if i == h0:
            out.append("@")
        if i == t0:
            out.append("#")
        if i < len(toks):
            out.append(toks[i])
    return " ".join(out)


BETWEEN_WORDS = 12


def _between(rec: dict) -> str:
    """The words between the two entity mentions, at most BETWEEN_WORDS (first half + last half).

    The dominant lexical cue for a relation sits between its arguments ("..., the CEO of ...",
    "... founded by ..."). A softmax regression on these words alone reaches 44.7 micro-F1 on this
    benchmark; surfacing them as their own short field puts that cue next to the entity names
    instead of leaving it to be found inside a ~40-word sentence. A pipeline can compute it as
    cheaply as this does.
    """
    toks = [_clean(t) for t in rec["token"]]
    h0, h1 = rec["h"]["pos"]
    t0, t1 = rec["t"]["pos"]
    left_end, right_start = (h1, t0) if h0 < t0 else (t1, h0)
    mid = toks[left_end:right_start]
    if len(mid) > BETWEEN_WORDS:
        half = BETWEEN_WORDS // 2
        mid = mid[:half] + ["..."] + mid[-half:]
    return " ".join(mid) if mid else "(adjacent)"


def _example(rec: dict) -> RelationExample:
    head = " ".join(_clean(t) for t in rec["token"][rec["h"]["pos"][0] : rec["h"]["pos"][1]])
    tail = " ".join(_clean(t) for t in rec["token"][rec["t"]["pos"][0] : rec["t"]["pos"][1]])
    return RelationExample(
        prefix_text=f"Head: {head}\nTail: {tail}\nBetween: {_between(rec)}\nText: {_marked_text(rec)}",
        target_text=rec["relation"],
        relation=rec["relation"],
    )


_BETWEEN_LINE = re.compile(r"\nBetween: [^\n]*")


def _fits(ex: RelationExample) -> bool:
    """Budget check on the prefix WITHOUT the Between line (the format measured in runs 73-76), so
    adding that field never changes which examples are trained on or benchmarked."""
    reference = _BETWEEN_LINE.sub("", ex.prefix_text)
    return len(_pre_tok().encode(reference)) <= PREFIX_TOKEN_BUDGET


def _na_train_fraction() -> float | None:
    """Target share of `NA` in the training pool, or None for the natural mix (~81%).

    Set via NCM_NA_FRACTION. Only the TRAINING pool is rebalanced; the benchmark always keeps
    the natural validation mix, so precision still pays for every false positive on `NA`.
    """
    raw = os.environ.get("NCM_NA_FRACTION")
    return float(raw) if raw else None


def generate_examples(n: int, seed: int) -> list[RelationExample]:
    """Deterministic sample of train examples that fit the prefix window.

    Natural class mix unless NCM_NA_FRACTION rebalances it: then every non-NA example is kept and
    `NA` is downsampled to that share of the pool (so `n` is an upper bound, not the size).
    """
    rng = random.Random(seed)
    rows = _raw("train")
    rng.shuffle(rows)
    frac = _na_train_fraction()
    if frac is None:
        out: list[RelationExample] = []
        for rec in rows:
            ex = _example(rec)
            if _fits(ex):
                out.append(ex)
                if len(out) >= n:
                    break
        return out
    pos: list[RelationExample] = []
    neg: list[RelationExample] = []
    for rec in rows:
        ex = _example(rec)
        if _fits(ex):
            (neg if ex.relation == NO_RELATION else pos).append(ex)
    pos = pos[:n]
    keep_na = int(len(pos) * frac / (1.0 - frac))
    pool = pos + neg[:keep_na]
    rng.shuffle(pool)
    return pool[:n]


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

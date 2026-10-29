"""Community-report-style drafting on Multi-News (the one remaining untested NCM direction).

A GraphRAG community report summarises several related text units into a short headline-like
statement. Multi-News is the closest staged corpus: each example is a cluster of 2-5 related news
articles with a human-written summary. Target here = the FIRST SENTENCE of that summary (bounded,
XSum-like, <= 48 GPT-2 tokens); source = the lead of each article, concatenated so that every
article is visible inside the fixed prefix window.

Why a trained model at all: the released XSum checkpoint scored 3-8% claim support zero-shot on
GraphRAG text and 27% even on its own domain (docs/constraints.md §11). The open question is whether
an NCM *trained* on multi-document summaries can reach useful faithfulness.

Deterministic metrics (benchmark): ROUGE-1 F against the reference sentence, and *grounding* = the
share of the output's content words that occur in the source. Neither is faithfulness (ROUGE rewards
house style; grounding rewards copying), so final claims use the blind claim-level judge, not these.
"""

import importlib
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_REL = Path("summarization/multinews/data")
_FILES = {"train": ("train.src.cleaned", "train.tgt"), "validation": ("val.src.cleaned", "val.tgt")}

PREFIX_TOKEN_BUDGET = 288
TARGET_TOKEN_BUDGET = 48
WORDS_PER_DOC = 110
BENCHMARK_SIZE = 400
BENCHMARK_SEED = 0

_STOP = frozenset(
    "a an and are as at be been but by for from had has have he her his i if in into is it its of on or "
    "she that the their them there they this to was were what when where which who will with you your we "
    "our us not no do does did so than then about after also more over said says say".split()
)


@dataclass(frozen=True)
class DraftExample:
    prefix_text: str
    target_text: str


def _root() -> Path:
    base = os.environ.get("NCM_CORPORA_ROOT")
    corpora = Path(base) if base else Path(__file__).resolve().parents[3] / "corpora"
    root = corpora / CORPUS_REL
    if not root.exists():
        raise FileNotFoundError(f"Multi-News not found at {root}. Fetch corpora or set NCM_CORPORA_ROOT.")
    return root


_TOK = None


def _pre_tok():
    global _TOK
    if _TOK is None:
        importlib.import_module("ncm_graphrag.ncm._ncm_repo")
        from tokenization.hf_tokenizer import HFTokenizer

        _TOK = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
    return _TOK


def _norm(text: str) -> str:
    text = text.replace("NEWLINE_CHAR", " ")
    text = text.translate({0x2019: "'", 0x2018: "'", 0x201C: '"', 0x201D: '"', 0x2014: "-", 0x2013: "-"})
    return re.sub(r"\s+", " ", text).strip()


def first_sentence(summary: str) -> str:
    s = _norm(summary).lstrip("- ").strip()
    return re.split(r"(?<=[.!?])\s+", s)[0]


def build_source(raw: str) -> str:
    """Lead of each article (<= WORDS_PER_DOC words), joined; clipped to the token budget."""
    docs = [_norm(d) for d in raw.split("|||||")]
    leads = [" ".join(d.split()[:WORDS_PER_DOC]) for d in docs if d]
    text = " || ".join(leads)
    tok = _pre_tok()
    ids = tok.encode(text)
    return text if len(ids) <= PREFIX_TOKEN_BUDGET else tok.decode(ids[:PREFIX_TOKEN_BUDGET])


def _read(split: str) -> list[tuple[str, str]]:
    src_f, tgt_f = _FILES[split]
    with open(_root() / src_f, encoding="utf-8") as a, open(_root() / tgt_f, encoding="utf-8") as b:
        return [(s, t) for s, t in zip(a, b, strict=False)]


def _example(raw_src: str, raw_tgt: str) -> DraftExample | None:
    target = first_sentence(raw_tgt)
    if not 5 <= len(target.split()) or len(_pre_tok().encode(target)) + 1 > TARGET_TOKEN_BUDGET:
        return None
    return DraftExample(prefix_text=build_source(raw_src), target_text=target)


def generate_examples(n: int, seed: int) -> list[DraftExample]:
    rows = _read("train")
    random.Random(seed).shuffle(rows)
    out: list[DraftExample] = []
    for s, t in rows:
        ex = _example(s, t)
        if ex:
            out.append(ex)
            if len(out) >= n:
                break
    return out


def benchmark_examples() -> list[DraftExample]:
    """Fixed held-out set: seeded sample of the Multi-News VALIDATION split."""
    rows = _read("validation")
    random.Random(BENCHMARK_SEED).shuffle(rows)
    out: list[DraftExample] = []
    for s, t in rows:
        ex = _example(s, t)
        if ex:
            out.append(ex)
            if len(out) >= BENCHMARK_SIZE:
                break
    return out


def is_well_formed(text: str) -> bool:
    t = text.strip()
    return len(t.split()) >= 4 and "\ufffd" not in t


def _content(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP]


def rouge1_f(pred: str, ref: str) -> float:
    p, r = _content(pred), _content(ref)
    if not p or not r:
        return 0.0
    rc: dict[str, int] = {}
    for w in r:
        rc[w] = rc.get(w, 0) + 1
    overlap = 0
    for w in p:
        if rc.get(w, 0) > 0:
            rc[w] -= 1
            overlap += 1
    if not overlap:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def grounding(pred: str, source: str) -> float:
    """Share of the output's content words that occur in the source (1.0 = nothing invented)."""
    p = _content(pred)
    if not p:
        return 0.0
    src = set(_content(source))
    return sum(1 for w in p if w in src) / len(p)

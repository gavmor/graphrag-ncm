"""Real map-stage dataset from HotpotQA (human-annotated relevance, no lexical proxy).

Why HotpotQA. The map stage scores one community report against a user query. HotpotQA's
`distractor` setting gives, per question, 10 Wikipedia paragraphs of which exactly the
paragraphs named in `supporting_facts` are *human-annotated* as necessary for the answer; the
rest are retrieved distractors. That is a real relevance label - unlike the QMSum variant in
`map_stage_real.py`, whose grades are a lexical-overlap proxy and which supplies only 3.6k
pairs. ~90k training questions -> ~180k gold + ~720k distractor paragraphs.

Grading (all from human annotation, no model in the loop):

| paragraph | grade | score | description |
|---|---|---|---|
| a supporting-fact paragraph (answer-bearing) | 3 | 95 | "Directly answers the query." |
| a distractor sharing a content-title token with a gold paragraph | 1 | 40 | "Only marginally relates." |
| any other distractor | 0 | 10 | "Unrelated to the query." |

(No grade-2: HotpotQA marks relevance as binary, so inventing a middle tier would label noise.
Grade 1 is kept because HotpotQA's distractors are chosen by retrieval similarity to the
question and some are genuinely adjacent.)

Balanced per question: one gold, one adjacent-distractor, one plain-distractor are emitted so
the label is evenly distributed rather than the natural 2:8 ratio.

Prefix is `Query: ...\\nReport: <paragraph>` with the paragraph clipped to the token budget;
the query comes FIRST because the prefix window truncates the tail (see map_stage_real.py).
"""

import importlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_REL = Path("multi_hop/hotpotqa/distractor")
_SPLIT_FILES = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "validation": ["validation-00000-of-00001.parquet"],
}

PREFIX_TOKEN_BUDGET = 160
GRADE_DESCRIPTIONS = {
    3: "Directly answers the query.",
    1: "Only marginally relates to the query.",
    0: "Unrelated to the query.",
}
GRADE_SCORES = {3: 95, 1: 40, 0: 10}

_STOP = frozenset("a an and are as at be by for from in is it its of on or the to was were with".split())


@dataclass(frozen=True)
class HotpotMapExample:
    prefix_text: str
    target_text: str
    grade: int
    score: int


def _root() -> Path:
    base = os.environ.get("NCM_CORPORA_ROOT")
    corpora = Path(base) if base else Path(__file__).resolve().parents[3] / "corpora"
    root = corpora / CORPUS_REL
    if not root.exists():
        raise FileNotFoundError(f"HotpotQA corpus not found at {root}. Fetch the corpora or set NCM_CORPORA_ROOT.")
    return root


_TOK = None


def _pre_tok():
    global _TOK
    if _TOK is None:
        importlib.import_module("ncm_graphrag.ncm._ncm_repo")
        from tokenization.hf_tokenizer import HFTokenizer

        _TOK = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
    return _TOK


def _title_tokens(title: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", title.lower()) if t not in _STOP and len(t) > 2}


def _head_hit(query: str, paragraph: str) -> int:
    """Query terms found in the paragraph's first 6 content words (capped at 9).

    Measured on held-out HotpotQA: P(gold) rises monotonically 0.30 -> 0.33 -> 0.52 -> 0.62 ->
    0.79 as this goes 0 -> 4+. Wikipedia paragraphs lead with their subject, so this is the
    dominant lexical relevance cue (alone 65.65 AUC; dropping it from a 7-feature classifier
    costs 6.9 AUC). Surfacing it as a number hands the model a cross-span comparison it did
    not learn from raw tokens (surfacing the opening *text* did not help, run #69). A real
    GraphRAG pipeline can compute it as cheaply as this does.
    """
    q = _title_tokens(query)
    first = [t for t in re.findall(r"[a-z0-9]+", paragraph.lower()) if t not in _STOP and len(t) > 2][:6]
    return min(len(q & set(first)), 9)


def _fit_prefix(query: str, paragraph: str) -> str:
    tok = _pre_tok()
    head = f"Query: {query}\nMatch: {_head_hit(query, paragraph)}\nReport: "
    room = PREFIX_TOKEN_BUDGET - len(tok.encode(head))
    ids = tok.encode(paragraph)
    if len(ids) > room:
        paragraph = tok.decode(ids[: max(room, 0)])
    return head + paragraph


def _target(grade: int) -> str:
    return json.dumps(
        {"points": [{"description": GRADE_DESCRIPTIONS[grade], "score": GRADE_SCORES[grade]}]},
        separators=(",", ":"),
    )


def _questions(split: str, limit: int | None = None) -> list[dict]:
    import pyarrow.parquet as pq

    out: list[dict] = []
    for name in _SPLIT_FILES[split]:
        path = _root() / name
        table = pq.read_table(path, columns=["id", "question", "supporting_facts", "context"])
        for qid, q, sf, ctx in zip(
            table["id"].to_pylist(),
            table["question"].to_pylist(),
            table["supporting_facts"].to_pylist(),
            table["context"].to_pylist(),
            strict=False,
        ):
            out.append({"id": qid, "question": q, "gold": set(sf["title"]), "ctx": ctx})
            if limit is not None and len(out) >= limit:
                return out
    return out


def _examples_for(question: dict, rng: random.Random) -> list[HotpotMapExample]:
    titles = question["ctx"]["title"]
    paras = [" ".join(s).strip() for s in question["ctx"]["sentences"]]
    gold_idx = [i for i, t in enumerate(titles) if t in question["gold"]]
    gold_tokens: set[str] = set()
    for i in gold_idx:
        gold_tokens |= _title_tokens(titles[i])
    # Gold paragraphs are systematically shorter than distractors (measured length-only AUC
    # 41.6, i.e. 58.4% the other way), so a model could score "relevance" from size alone.
    # Distractors are therefore drawn only from a +-35% band around the chosen gold length.
    gold_len = sum(len(paras[i].split()) for i in gold_idx) / max(len(gold_idx), 1)
    lo, hi = gold_len * 0.65, gold_len * 1.35
    adjacent: list[int] = []
    plain: list[int] = []
    for i, t in enumerate(titles):
        if i in gold_idx or not paras[i] or not lo <= len(paras[i].split()) <= hi:
            continue
        (adjacent if _title_tokens(t) & gold_tokens else plain).append(i)
    if not gold_idx or not plain:
        return []
    picks: list[tuple[int, int]] = [(rng.choice(gold_idx), 3), (rng.choice(plain), 0)]
    if adjacent:
        picks.append((rng.choice(adjacent), 1))
    out = []
    for idx, grade in picks:
        if not paras[idx]:
            continue
        out.append(
            HotpotMapExample(
                prefix_text=_fit_prefix(question["question"], paras[idx]),
                target_text=_target(grade),
                grade=grade,
                score=GRADE_SCORES[grade],
            )
        )
    return out


def generate_examples(n: int, seed: int) -> list[HotpotMapExample]:
    """Deterministic sample of graded (query, paragraph) pairs from HotpotQA train."""
    rng = random.Random(seed)
    qs = _questions("train", limit=max(n, 100))
    rng.shuffle(qs)
    out: list[HotpotMapExample] = []
    for q in qs:
        out.extend(_examples_for(q, rng))
        if len(out) >= n:
            break
    return out[:n]


def benchmark_examples() -> list[HotpotMapExample]:
    """Fixed held-out set: first 600 HotpotQA *validation* questions, never trained on."""
    rng = random.Random(0)
    out: list[HotpotMapExample] = []
    for q in _questions("validation", limit=600):
        out.extend(_examples_for(q, rng))
    return out


def is_well_formed(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    points = parsed.get("points")
    if not isinstance(points, list) or not points:
        return False
    for p in points:
        score = p.get("score")
        if not isinstance(p.get("description"), str) or not isinstance(score, (int, float)):
            return False
        if not 0 <= score <= 100:
            return False
    return True


def predicted_score(text: str) -> float | None:
    if not is_well_formed(text):
        return None
    return float(json.loads(text)["points"][0]["score"])

"""Real map-stage dataset shaped like GraphRAG's Global Search map-stage call.

Why this task (Lead 1): the map stage maps (one community report, user query) -> strict JSON
`{"points": [{"description": str, "score": 0-100}]}`. Its output is a *bounded ordinal score*
plus a short fixed description, so it exercises NCM's parallel coarse-to-fine decoding without
the open-vocabulary string-copying requirement that caps the extraction slice. See
docs/constraints.md §7 for that measured ceiling.

Source. `corpora/summarization/qmsum_cleaned` (QMSum, query-focused meeting summarization).
Each record is `input` = "<query>\\n<meeting transcript>" plus `output` = the reference
query-focused summary. The Microsoft GraphRAG benchmark corpora (`Kevin Scott Questions.csv`,
`MSFT * Questions.csv`) ship questions but **no reference answers**, so they cannot supervise a
score; QMSum's reference summaries are what make a relevance grade constructible at all.

Labelling rule (fully reproducible, no model in the loop). A transcript is split into
consecutive ~90-word chunks along speaker turns. Each chunk is ranked by how much of the
reference summary's content bigrams it covers. The grade is the chunk's rank within its own
meeting, which keeps the grade balanced per meeting and makes a *graded* score rather than a
binary flag:

| rank within meeting | grade | score | description |
|---|---|---|---|
| 1st | 3 | 95 | "Directly answers the query." |
| 2nd-3rd | 2 | 70 | "Partially answers the query." |
| middle | 1 | 40 | "Only marginally relates to the query." |
| bottom third, or a chunk from another meeting | 0 | 10 | "Unrelated to the query." |

The four descriptions are fixed strings, so the model never has to copy text - the learned
signal is the score itself, which is exactly the part of the map stage NCM is meant to own.

Caveat recorded deliberately: the grade is a lexical-overlap proxy for "this chunk answers the
query", not human relevance judgement. It is a real, deterministic label derived from real
reference summaries, and every example comes from real human-written text, but it is weaker
than a human-annotated relevance set. Benchmarks score a held-out QMSum split.
"""

import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_REL = Path("summarization/qmsum_cleaned")

_SPLIT_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}

CHUNK_WORDS = 60
# Prefix window the generator sees (gpt2 tokens). The query is placed FIRST and the report is
# truncated to fit, because a fixed-length window cuts the *tail*: with "Report ... Query ..."
# 74% of examples lost their query entirely (measured), which made the score unlearnable and
# produced pure memorisation (in-sample AUC 89 vs held-out 58).
PREFIX_TOKEN_BUDGET = 160
GRADE_DESCRIPTIONS = {
    3: "Directly answers the query.",
    2: "Partially answers the query.",
    1: "Only marginally relates to the query.",
    0: "Unrelated to the query.",
}
GRADE_SCORES = {3: 95, 2: 70, 1: 40, 0: 10}

_STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his i if in into is it its
    of on or she that the their them there they this to was were what when where which who will
    with you your we our us not no do does did so than then about""".split()
)


@dataclass(frozen=True)
class MapStageRealExample:
    prefix_text: str
    target_text: str
    grade: int
    score: int


def _corpora_root() -> Path:
    """Locate the fetched corpora tree.

    `NCM_CORPORA_ROOT` wins when set (the corpora are fetched into a separate worktree in
    this checkout); otherwise `corpora/` at the repository root is assumed.
    """
    base = os.environ.get("NCM_CORPORA_ROOT")
    corpora = Path(base) if base else Path(__file__).resolve().parents[3] / "corpora"
    root = corpora / CORPUS_REL
    if not root.exists():
        raise FileNotFoundError(
            f"QMSum corpus not found at {root}. Fetch the corpora (scripts/fetch_corpora.py) "
            "or point NCM_CORPORA_ROOT at the directory containing summarization/."
        )
    return root


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOPWORDS]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False))


def _chunks(transcript: str) -> list[str]:
    """Group speaker turns into consecutive ~CHUNK_WORDS chunks."""
    lines = [ln.strip() for ln in transcript.splitlines() if ln.strip()]
    out: list[str] = []
    buf: list[str] = []
    words = 0
    for line in lines:
        buf.append(line)
        words += len(line.split())
        if words >= CHUNK_WORDS:
            out.append(" ".join(buf))
            buf, words = [], 0
    # The trailing remainder is usually much shorter than CHUNK_WORDS. A short chunk covers
    # less of the reference summary purely because it is short, so ranking it would label
    # length, not relevance (measured: grade-0 chunks averaged 38.8 content words vs ~54 for
    # the rest, and chunk length alone scored 66.4 AUC). Drop it.
    return out


_GENERIC_QUERY = re.compile(
    r"summar(y|ize)\b.*\b(whole|entire|meeting|discussion)\b|\bwhole (meeting|discussion)\b|general discussion",
    re.IGNORECASE,
)


def _is_generic_query(query: str) -> bool:
    """Whole-meeting prompts ("Summarize the whole meeting") are relevant to every chunk, so
    their grade reduces to lexical overlap with the summary - noise for a relevance task.
    Measured at 26% of the QMSum validation set; they are excluded from train and benchmark."""
    return bool(_GENERIC_QUERY.search(query))


def _split_query_transcript(raw_input: str) -> tuple[str, str]:
    head, _, tail = raw_input.partition("\n")
    return head.strip(), tail.strip()


def _records(split: str) -> list[dict]:
    import pyarrow.parquet as pq

    path = _corpora_root() / _SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"QMSum split file missing: {path}")
    table = pq.read_table(path, columns=["id", "input", "output"])
    ids = table["id"].to_pylist()
    inputs = table["input"].to_pylist()
    outputs = table["output"].to_pylist()
    return [{"id": i, "input": a, "output": b} for i, a, b in zip(ids, inputs, outputs, strict=False)]


def _graded_chunks(record: dict) -> list[tuple[str, str, int]]:
    """Return one (query, chunk, grade) per grade for a meeting, ranked by summary coverage.

    Balanced on purpose: consecutive transcript chunks are overwhelmingly "middle" (grade 1),
    so sampling every chunk would train on a near-constant label. Picking the best, second
    best, median and worst chunk of each meeting gives an even 4-way graded signal.
    """
    query, transcript = _split_query_transcript(record["input"])
    summary_grams = _bigrams(_tokens(record["output"]))
    if not query or not transcript or not summary_grams or _is_generic_query(query):
        return []
    chunks = _chunks(transcript)
    if len(chunks) < 4:
        return []
    # Only chunks inside a narrow length band are eligible. Both raw coverage (grows with
    # length) and density (shrinks with length) correlate with chunk length, and a ranking
    # label that tracks length lets a model score "relevance" from size alone. Measured
    # length-only AUCs: 66.4 (raw coverage, short tail kept), 60.1 (tail dropped), 38.6
    # (density). Restricting the candidate pool to a fixed band removes the dependence
    # instead of trading one bias for its opposite.
    lengths = [len(_tokens(c)) for c in chunks]
    median_len = sorted(lengths)[len(lengths) // 2]
    low, high = int(median_len * 0.85), int(median_len * 1.15)
    eligible = [i for i, n in enumerate(lengths) if low <= n <= high]
    if len(eligible) < 4:
        return []
    chunks = [chunks[i] for i in eligible]
    coverage = [len(summary_grams & _bigrams(_tokens(c))) / max(len(summary_grams), 1) for c in chunks]
    order = sorted(range(len(chunks)), key=lambda i: coverage[i], reverse=True)
    picks: list[tuple[int, int]] = [
        (order[0], 3),
        (order[1], 2),
        (order[len(order) // 2], 1),
        (order[-1], 0),
    ]
    seen: set[int] = set()
    graded: list[tuple[str, str, int]] = []
    for idx, grade in picks:
        if idx in seen:
            continue
        seen.add(idx)
        graded.append((query, chunks[idx], grade))
    return graded


_TOK = None


def _pre_tok():
    global _TOK
    if _TOK is None:
        # Importing this module is a sys.path side effect that puts the vendored ncm repo on
        # the path; via importlib so import sorting cannot move it after the import below.
        importlib.import_module("ncm_graphrag.ncm._ncm_repo")
        from tokenization.hf_tokenizer import HFTokenizer

        _TOK = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
    return _TOK


def _fit_prefix(query: str, chunk: str) -> str:
    """ "Query: ...\nReport: ..." with the report clipped so the whole prefix fits the window."""
    tok = _pre_tok()
    head = f"Query: {query}\nReport: "
    room = PREFIX_TOKEN_BUDGET - len(tok.encode(head))
    ids = tok.encode(chunk)
    if len(ids) > room:
        chunk = tok.decode(ids[: max(room, 0)])
    return head + chunk


def _example(query: str, chunk: str, grade: int) -> MapStageRealExample:
    target = json.dumps(
        {"points": [{"description": GRADE_DESCRIPTIONS[grade], "score": GRADE_SCORES[grade]}]},
        separators=(",", ":"),
    )
    return MapStageRealExample(
        prefix_text=_fit_prefix(query, chunk),
        target_text=target,
        grade=grade,
        score=GRADE_SCORES[grade],
    )


def generate_examples(n: int, seed: int) -> list[MapStageRealExample]:
    """Deterministic sample of graded (query, chunk) pairs from the QMSum training split."""
    records = sorted(_records("train"), key=lambda r: r["id"])
    out: list[MapStageRealExample] = []
    seen: set[tuple[str, str]] = set()
    # Round-robin over meetings so a sample never comes from a handful of transcripts.
    graded_per_record = [_graded_chunks(r) for r in records]
    cursor = seed % max(len(graded_per_record), 1)
    while len(out) < n and graded_per_record:
        batch = graded_per_record[cursor % len(graded_per_record)]
        cursor += 1
        for query, chunk, grade in batch:
            key = (query, chunk)
            if key in seen:
                continue
            seen.add(key)
            out.append(_example(query, chunk, grade))
            if len(out) >= n:
                break
        if cursor > seed + len(graded_per_record) * 4:
            break
    return out


def benchmark_examples() -> list[MapStageRealExample]:
    """Fixed held-out set: QMSum *validation* meetings, never used for training."""
    out: list[MapStageRealExample] = []
    for record in sorted(_records("validation"), key=lambda r: r["id"]):
        for query, chunk, grade in _graded_chunks(record):
            out.append(_example(query, chunk, grade))
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
    """Top point's score, or None if the output does not parse."""
    if not is_well_formed(text):
        return None
    return float(json.loads(text)["points"][0]["score"])

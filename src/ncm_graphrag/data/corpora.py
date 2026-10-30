"""Programmatic access and loader helpers for fetched GraphRAG training corpora.

Datasets correspond to those identified in 'GraphRAG Training Dataset Discovery.md':
1. Information Extraction: DocRED, Re-DocRED, Few-NERD, SciRIFF, SciERC, TACRED, Re-TACRED, REBEL, SREDFM
2. Multi-Hop Reasoning: HotpotQA, MuSiQue, KILT Tasks, GraphRAG-Bench, WildGraphBench
3. Query-Focused Summarization: SQuALITY, ZeroSCROLLS/QMSum, GovReport, Multi-News, Microsoft GraphRAG Benchmarks
4. Additional: neo4j/text2cypher
"""

import csv
import gzip
import json
from pathlib import Path
from typing import Any

# Default path to corpora root directory relative to this file
_DATA_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DATA_DIR.parent.parent.parent
CORPORA_ROOT = _REPO_ROOT / "corpora"


def get_corpora_root() -> Path:
    """Return the absolute Path to the local corpora/ directory."""
    return CORPORA_ROOT


def load_manifest() -> dict[str, dict[str, Any]]:
    """Load the dataset manifest mapping dataset names to metadata and disk paths."""
    manifest_path = CORPORA_ROOT / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Corpora manifest not found at {manifest_path}. Run `python scripts/fetch_corpora.py --all` first."
        )
    with open(manifest_path, encoding="utf-8") as f:
        data: dict[str, dict[str, Any]] = json.load(f)
        return data


def list_corpora(category: str | None = None) -> list[dict[str, Any]]:
    """List available corpora metadata, optionally filtered by category.

    Categories: 'information_extraction', 'multi_hop', 'summarization', 'additional'.
    """
    manifest = load_manifest()
    records = list(manifest.values())
    if category is not None:
        records = [r for r in records if r.get("category") == category]
    return records


def get_corpus_path(name: str) -> Path:
    """Resolve the absolute filesystem Path to a dataset directory."""
    manifest = load_manifest()
    if name not in manifest:
        raise KeyError(f"Dataset '{name}' not found in manifest. Available: {sorted(manifest.keys())}")
    rel_path = manifest[name]["path"]
    target = _REPO_ROOT / rel_path
    if not target.exists():
        raise FileNotFoundError(f"Corpus path for '{name}' does not exist on disk: {target}")
    return target


# ==============================================================================
# Specialized dataset loaders
# ==============================================================================


def load_docred(split: str = "dev") -> list[dict[str, Any]]:
    """Load DocRED documents for a split ('train_annotated', 'dev', or 'test')."""
    base = get_corpus_path("docred")
    filename = f"{split}.json.gz" if not split.startswith("data/") else f"{split}.gz"
    full_path = base / "data" / filename if not (base / filename).exists() else base / filename
    if not full_path.exists():
        raise FileNotFoundError(f"DocRED split file not found: {full_path}")
    with gzip.open(full_path, "rt", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
        return data


def load_re_docred(split: str = "dev") -> list[dict[str, Any]]:
    """Load Re-DocRED revised documents ('train', 'dev', or 'test')."""
    base = get_corpus_path("re_docred")
    path = base / f"{split}_revised.json"
    if not path.exists():
        raise FileNotFoundError(f"Re-DocRED split not found: {path}")
    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
        return data


def load_musique(split: str = "dev") -> list[dict[str, Any]]:
    """Load MuSiQue QA examples for 'dev' or 'train'."""
    base = get_corpus_path("musique")
    path = base / f"musique_ans_v1.0_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"MuSiQue split not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_graphrag_bench(domain: str = "medical") -> dict[str, Any]:
    """Load GraphRAG-Bench corpus and questions for a domain ('medical' or 'novel')."""
    base = get_corpus_path("graphrag_bench")
    corpus_file = base / "Datasets" / "Corpus" / f"{domain}.json"
    questions_file = base / "Datasets" / "Questions" / f"{domain}_questions.json"
    if not corpus_file.exists() or not questions_file.exists():
        raise FileNotFoundError(f"GraphRAG-Bench files for domain '{domain}' not found in {base}")
    with open(corpus_file, encoding="utf-8") as f:
        corpus = json.load(f)
    with open(questions_file, encoding="utf-8") as f:
        questions = json.load(f)
    return {"corpus": corpus, "questions": questions}


def load_microsoft_graphrag(subset: str = "podcasts") -> dict[str, Any]:
    """Load Microsoft GraphRAG benchmarking dataset ('podcasts', 'msft_earnings', or 'hotpotqa')."""
    base = get_corpus_path("microsoft_graphrag") / "data"
    if subset == "podcasts":
        csv_path = base / "Kevin Scott Questions.csv"
        text_dir = base / "Kevin Scott Podcast Transcripts Input Text"
        with open(csv_path, encoding="utf-8-sig") as f:
            questions = list(csv.DictReader(f))
        return {"questions": questions, "text_dir": text_dir}
    elif subset in ("msft", "msft_earnings", "earnings"):
        csv_multi = base / "MSFT Multi Transcript Questions.csv"
        csv_single = base / "MSFT Single Transcript Questions.csv"
        text_dir = base / "MSFT Input Text"
        with open(csv_multi, encoding="utf-8-sig") as f:
            multi_q = list(csv.DictReader(f))
        with open(csv_single, encoding="utf-8-sig") as f:
            single_q = list(csv.DictReader(f))
        return {"multi_questions": multi_q, "single_questions": single_q, "text_dir": text_dir}
    elif subset == "hotpotqa":
        csv_path = base / "HotPotQA Filtered Questions.csv"
        text_dir = base / "HotPotQA Filtered Input Text" / "input"
        with open(csv_path, encoding="utf-8-sig") as f:
            questions = list(csv.DictReader(f))
        return {"questions": questions, "text_dir": text_dir}
    else:
        raise ValueError(f"Unknown Microsoft GraphRAG subset: {subset}. Choose 'podcasts', 'earnings', or 'hotpotqa'.")

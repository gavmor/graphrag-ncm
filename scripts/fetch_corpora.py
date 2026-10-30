"""Fetch training datasets mentioned in 'GraphRAG Training Dataset Discovery.md'.

Datasets are organized into 4 categories under `corpora/`:
1. information_extraction (DocRED, Re-DocRED, Few-NERD, SciRIFF, SciERC, TACRED, Re-TACRED, REBEL, SREDFM)
2. multi_hop (HotpotQA, MuSiQue, KILT Tasks, KILT Wikipedia metadata, GraphRAG-Bench, WildGraphBench)
3. summarization (SQuALITY, ZeroSCROLLS/QMSum, GovReport, Multi-News, Microsoft GraphRAG Benchmarking Datasets)
4. additional (neo4j/text2cypher)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = Path(__file__).resolve().parent.parent
CORPORA_DIR = Path(os.environ.get("NCM_CORPORA_ROOT", BASE_DIR / "corpora")).expanduser().resolve()


def _manifest_path(path: Path) -> str:
    """Store paths relative to NCM_CORPORA_ROOT so manifests remain relocatable."""
    return str(path.relative_to(CORPORA_DIR))


def get_dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def format_size(bytes_val: int) -> str:
    val = float(bytes_val)
    for unit in ["B", "KB", "MB", "GB"]:
        if val < 1024.0:
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} TB"


def update_manifest(entry: dict):
    manifest_path = CORPORA_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if manifest_path.exists():
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            manifest = {}
    manifest[entry["name"]] = entry
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


# ==============================================================================
# Category 1: Information Extraction
# ==============================================================================


def fetch_docred():
    name = "docred"
    target = CORPORA_DIR / "information_extraction" / name
    print(f"\n--- Fetching {name} (thunlp/docred) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="thunlp/docred",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "information_extraction",
            "source": "https://huggingface.co/datasets/thunlp/docred",
            "citation": "5, 6",
            "description": "DocRED document-level relation extraction (Wikipedia, 96 Wikidata relations)",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_re_docred():
    name = "re_docred"
    target = CORPORA_DIR / "information_extraction" / name
    print(f"\n--- Fetching {name} (tonytan48/Re-DocRED) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="tonytan48/Re-DocRED",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "information_extraction",
            "source": "https://huggingface.co/datasets/tonytan48/Re-DocRED",
            "citation": "7",
            "description": "Re-DocRED re-annotated document-level relation extraction addressing false-negatives",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_few_nerd():
    name = "few_nerd"
    target = CORPORA_DIR / "information_extraction" / name
    print(f"\n--- Fetching {name} (DFKI-SLT/few-nerd) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="DFKI-SLT/few-nerd",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "information_extraction",
            "source": "https://huggingface.co/datasets/DFKI-SLT/few-nerd",
            "citation": "10, 11",
            "description": "Few-NERD fine-grained entity typing (8 coarse, 66 fine types across 188.2k sentences)",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_sciriff():
    name = "sciriff"
    target = CORPORA_DIR / "information_extraction" / name
    print(f"\n--- Fetching {name} (allenai/SciRIFF) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="allenai/SciRIFF",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "information_extraction",
            "source": "https://huggingface.co/datasets/allenai/SciRIFF",
            "citation": "14",
            "description": "SciRIFF scientific instruction-tuning suite for information extraction and claims",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_scierc():
    name = "scierc"
    target = CORPORA_DIR / "information_extraction" / name
    print(f"\n--- Fetching {name} (UW / AllenAI sciERC processed tarball + HF zj88zj/SCIERC) -> {target} ---")
    t0 = time.time()
    target.mkdir(parents=True, exist_ok=True)

    # 1. Download official UW sciERC tarball
    tarball_url = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"
    tarball_dest = target / "sciERC_processed.tar.gz"
    if not tarball_dest.exists():
        print(f"Downloading official tarball from {tarball_url}...")
        urllib.request.urlretrieve(tarball_url, tarball_dest)
        with tarfile.open(tarball_dest, "r:gz") as tar:
            tar.extractall(path=target / "raw")

    # 2. Also download HF json split representation
    hf_target = target / "hf"
    snapshot_download(
        repo_id="zj88zj/SCIERC",
        repo_type="dataset",
        local_dir=str(hf_target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "information_extraction",
            "source": "http://nlp.cs.washington.edu/sciIE/ & https://huggingface.co/datasets/zj88zj/SCIERC",
            "citation": "12, 13",
            "description": "SciERC scientific abstracts with 6 entity types and 7 directional relations",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_tacred():
    name = "tacred"
    target = CORPORA_DIR / "information_extraction" / name
    print(f"\n--- Fetching {name} (DFKI-SLT/tacred card + xiaobendanyn/tacred data) -> {target} ---")
    t0 = time.time()
    card_target = target / "card"
    snapshot_download(
        repo_id="DFKI-SLT/tacred",
        repo_type="dataset",
        local_dir=str(card_target),
    )
    data_target = target / "data"
    snapshot_download(
        repo_id="xiaobendanyn/tacred",
        repo_type="dataset",
        local_dir=str(data_target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "information_extraction",
            "source": "https://huggingface.co/datasets/DFKI-SLT/tacred & https://huggingface.co/datasets/xiaobendanyn/tacred",
            "citation": "8, 9",
            "description": "TACRED 106k sentence relation extraction instances across 41 relation classes",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_re_tacred():
    name = "re_tacred"
    target = CORPORA_DIR / "information_extraction" / name
    print(f"\n--- Fetching {name} (gstoica27/Re-TACRED GitHub repo) -> {target} ---")
    t0 = time.time()
    if not (target / ".git").exists():
        if target.exists():
            shutil.rmtree(target)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/gstoica27/Re-TACRED.git", str(target)], check=True
        )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "information_extraction",
            "source": "https://github.com/gstoica27/Re-TACRED",
            "citation": "8",
            "description": "Re-TACRED revised labels and patches addressing TACRED label inconsistencies",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_rebel():
    name = "rebel"
    target = CORPORA_DIR / "information_extraction" / name
    print(f"\n--- Fetching {name} (Babelscape/rebel-dataset) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="Babelscape/rebel-dataset",
        repo_type="dataset",
        local_dir=str(target),
    )
    # Also unpack rebel_dataset.zip if present
    zip_path = target / "rebel_dataset.zip"
    if zip_path.exists() and not (target / "extracted").exists():
        print(f"Unpacking {zip_path.name}...")
        shutil.unpack_archive(str(zip_path), str(target / "extracted"))
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "information_extraction",
            "source": "https://huggingface.co/datasets/Babelscape/rebel-dataset",
            "citation": "15, 19, 53",
            "description": "REBEL end-to-end relation extraction seq2seq dataset (triples from Wikipedia)",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_sredfm():
    name = "sredfm"
    target = CORPORA_DIR / "information_extraction" / name
    print(f"\n--- Fetching {name} (Babelscape/sredfm - dev/test/code) -> {target} ---")
    t0 = time.time()
    target.mkdir(parents=True, exist_ok=True)
    # Download English dev and test splits, readme, script
    files_to_download = [
        "README.md",
        "SREDFM.py",
        "data/dev.en.jsonl",
        "data/test.en.jsonl",
    ]
    for rel_path in files_to_download:
        dest = target / rel_path
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {rel_path}...")
            cached = hf_hub_download(
                repo_id="Babelscape/sredfm",
                filename=rel_path,
                repo_type="dataset",
            )
            shutil.copyfile(cached, dest)
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "information_extraction",
            "source": "https://huggingface.co/datasets/Babelscape/sredfm",
            "citation": "11, 15",
            "description": (
                "SREDFM Silver Relation Extraction Dataset with Filtering Models (English evaluation splits)"
            ),
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_information_extraction():
    fetch_docred()
    fetch_re_docred()
    fetch_few_nerd()
    fetch_sciriff()
    fetch_scierc()
    fetch_tacred()
    fetch_re_tacred()
    fetch_rebel()
    fetch_sredfm()


# ==============================================================================
# Category 2: Multi-Hop Reasoning and Subgraph Grounding
# ==============================================================================


def fetch_hotpotqa():
    name = "hotpotqa"
    target = CORPORA_DIR / "multi_hop" / name
    print(f"\n--- Fetching {name} (hotpotqa/hotpot_qa) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="hotpotqa/hotpot_qa",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "multi_hop",
            "source": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
            "citation": "16, 24",
            "description": "HotpotQA multi-hop reasoning benchmark with sentence-level gold supporting facts",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_musique():
    name = "musique"
    target = CORPORA_DIR / "multi_hop" / name
    print(f"\n--- Fetching {name} (dgslibisey/MuSiQue) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="dgslibisey/MuSiQue",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "multi_hop",
            "source": "https://huggingface.co/datasets/dgslibisey/MuSiQue",
            "citation": "19, 25",
            "description": (
                "MuSiQue multi-hop question answering dataset with distractor paragraphs and decomposed hops"
            ),
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_kilt():
    name = "kilt_tasks"
    target = CORPORA_DIR / "multi_hop" / name
    print(f"\n--- Fetching {name} (facebook/kilt_tasks) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="facebook/kilt_tasks",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "multi_hop",
            "source": "https://huggingface.co/datasets/facebook/kilt_tasks",
            "citation": "20, 21",
            "description": "KILT 11 knowledge-intensive tasks benchmark over Wikipedia provenance",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )

    # Also fetch kilt_wikipedia repo card & loader script
    wiki_name = "kilt_wikipedia"
    wiki_target = CORPORA_DIR / "multi_hop" / wiki_name
    print(f"\n--- Fetching {wiki_name} (facebook/kilt_wikipedia metadata & script) -> {wiki_target} ---")
    snapshot_download(
        repo_id="facebook/kilt_wikipedia",
        repo_type="dataset",
        local_dir=str(wiki_target),
    )
    wiki_size = get_dir_size(wiki_target)
    update_manifest(
        {
            "name": wiki_name,
            "category": "multi_hop",
            "source": "https://huggingface.co/datasets/facebook/kilt_wikipedia",
            "citation": "22",
            "description": "KILT Wikipedia knowledge source loader script and schema definition",
            "path": _manifest_path(wiki_target),
            "size_bytes": wiki_size,
            "size_human": format_size(wiki_size),
        }
    )


def fetch_graphrag_bench():
    name = "graphrag_bench"
    target = CORPORA_DIR / "multi_hop" / name
    print(f"\n--- Fetching {name} (GraphRAG-Bench/GraphRAG-Bench) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="GraphRAG-Bench/GraphRAG-Bench",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "multi_hop",
            "source": "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench",
            "citation": "23, 24, 25",
            "description": (
                "GraphRAG-Bench complex multi-constraint graph queries across medical, novel, and algorithmic domains"
            ),
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_wildgraphbench():
    name = "wildgraphbench"
    target = CORPORA_DIR / "multi_hop" / name
    print(f"\n--- Fetching {name} (Bstwpy/WildGraphBench) -> {target} ---")
    t0 = time.time()
    if not (target / ".git").exists():
        if target.exists():
            shutil.rmtree(target)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://huggingface.co/datasets/Bstwpy/WildGraphBench", str(target)],
            check=True,
        )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "multi_hop",
            "source": "https://huggingface.co/datasets/Bstwpy/WildGraphBench",
            "citation": "26",
            "description": (
                "WildGraphBench multi-fact aggregation over complex semi-structured Wikipedia graph topologies"
            ),
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_multi_hop():
    fetch_hotpotqa()
    fetch_musique()
    fetch_kilt()
    fetch_graphrag_bench()
    fetch_wildgraphbench()


# ==============================================================================
# Category 3: Query-Focused Summarization (QFS)
# ==============================================================================


def fetch_squality():
    name = "squality"
    target = CORPORA_DIR / "summarization" / name
    print(f"\n--- Fetching {name} (GEM/squality) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="GEM/squality",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "summarization",
            "source": "https://huggingface.co/datasets/GEM/squality",
            "citation": "29, 37",
            "description": (
                "SQuALITY query-focused summarization over short stories with 4 multi-perspective reference summaries"
            ),
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_zero_scrolls_and_qmsum():
    name = "zero_scrolls"
    target = CORPORA_DIR / "summarization" / name
    print(f"\n--- Fetching {name} (tau/zero_scrolls - QMSum, SQuALITY, GovReport, MuSiQue zips) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="tau/zero_scrolls",
        repo_type="dataset",
        local_dir=str(target),
    )
    # Unpack qmsum.zip, squality.zip, gov_report.zip
    for z in ["qmsum.zip", "squality.zip", "gov_report.zip", "musique.zip"]:
        zp = target / z
        unpacked_dir = target / zp.stem
        if zp.exists() and not unpacked_dir.exists():
            print(f"Unpacking {z}...")
            shutil.unpack_archive(str(zp), str(unpacked_dir))
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "summarization",
            "source": "https://huggingface.co/datasets/tau/zero_scrolls",
            "citation": "19, 27, 38",
            "description": "ZeroSCROLLS benchmark suite including QMSum, SQuALITY, and GovReport",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )

    # Also fetch standalone cleaned QMSum for convenience
    qmsum_name = "qmsum_cleaned"
    qmsum_target = CORPORA_DIR / "summarization" / qmsum_name
    print(f"\n--- Fetching {qmsum_name} (pszemraj/qmsum-cleaned) -> {qmsum_target} ---")
    snapshot_download(
        repo_id="pszemraj/qmsum-cleaned",
        repo_type="dataset",
        local_dir=str(qmsum_target),
    )
    qmsum_size = get_dir_size(qmsum_target)
    update_manifest(
        {
            "name": qmsum_name,
            "category": "summarization",
            "source": "https://huggingface.co/datasets/pszemraj/qmsum-cleaned",
            "citation": "19, 28, 38",
            "description": (
                "QMSum query-based multi-domain meeting summarization (academic, corporate, political meetings)"
            ),
            "path": _manifest_path(qmsum_target),
            "size_bytes": qmsum_size,
            "size_human": format_size(qmsum_size),
        }
    )


def fetch_govreport():
    name = "govreport"
    target = CORPORA_DIR / "summarization" / name
    print(f"\n--- Fetching {name} (ccdv/govreport-summarization) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="ccdv/govreport-summarization",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "summarization",
            "source": "https://huggingface.co/datasets/ccdv/govreport-summarization",
            "citation": "19, 41",
            "description": (
                "GovReport long institutional policy and legislative reports paired with executive summaries"
            ),
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_multinews():
    name = "multinews"
    target = CORPORA_DIR / "summarization" / name
    print(f"\n--- Fetching {name} (alexfabbri/multi_news) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="alexfabbri/multi_news",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "summarization",
            "source": "https://huggingface.co/datasets/alexfabbri/multi_news",
            "citation": "19, 41",
            "description": "Multi-News multi-document news summarization corpus",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_microsoft_graphrag():
    name = "microsoft_graphrag"
    target = CORPORA_DIR / "summarization" / name
    print(f"\n--- Fetching {name} (microsoft/graphrag-benchmarking-datasets GitHub repo) -> {target} ---")
    t0 = time.time()
    if not (target / ".git").exists():
        if target.exists():
            shutil.rmtree(target)
        repo_url = "https://github.com/microsoft/graphrag-benchmarking-datasets.git"
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)
    # Unpack the three benchmarking archives in target/data
    data_dir = target / "data"
    if data_dir.exists():
        for archive_file in data_dir.glob("*.zip"):
            out_subdir = data_dir / archive_file.stem
            if not out_subdir.exists():
                print(f"Unpacking {archive_file.name}...")
                try:
                    shutil.unpack_archive(str(archive_file), str(out_subdir))
                except Exception:
                    # Fallback for tar.gz misnamed as .zip
                    out_subdir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["tar", "-zxf", str(archive_file), "-C", str(out_subdir)], check=True)
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "summarization",
            "source": "https://github.com/microsoft/graphrag-benchmarking-datasets",
            "citation": "4, 18, 39, 40",
            "description": (
                "Official Microsoft GraphRAG benchmarks: Kevin Scott Podcasts, MSFT Earnings Calls, Filtered HotPotQA"
            ),
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_summarization():
    fetch_squality()
    fetch_zero_scrolls_and_qmsum()
    fetch_govreport()
    fetch_multinews()
    fetch_microsoft_graphrag()


# ==============================================================================
# Category 4: Additional Datasets
# ==============================================================================


def fetch_text2cypher():
    name = "text2cypher"
    target = CORPORA_DIR / "additional" / name
    print(f"\n--- Fetching {name} (neo4j/text2cypher) -> {target} ---")
    t0 = time.time()
    snapshot_download(
        repo_id="neo4j/text2cypher",
        repo_type="dataset",
        local_dir=str(target),
    )
    size = get_dir_size(target)
    print(f"Done in {time.time() - t0:.1f}s ({format_size(size)})")
    update_manifest(
        {
            "name": name,
            "category": "additional",
            "source": "https://huggingface.co/datasets/neo4j/text2cypher",
            "citation": "33, 54",
            "description": "neo4j/text2cypher dataset for natural language query to Cypher property graph traversal",
            "path": _manifest_path(target),
            "size_bytes": size,
            "size_human": format_size(size),
        }
    )


def fetch_additional():
    fetch_text2cypher()


# ==============================================================================
# Main CLI Entrypoint
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fetch GraphRAG training corpora")
    parser.add_argument("--all", action="store_true", help="Fetch all corpora categories")
    parser.add_argument(
        "--category",
        choices=["ie", "multi_hop", "summarization", "additional"],
        help="Fetch a specific category",
    )
    parser.add_argument(
        "--dataset",
        help="Fetch a specific dataset by name",
    )
    args = parser.parse_args()

    dispatch = {
        # Category IE
        "docred": fetch_docred,
        "re_docred": fetch_re_docred,
        "few_nerd": fetch_few_nerd,
        "sciriff": fetch_sciriff,
        "scierc": fetch_scierc,
        "tacred": fetch_tacred,
        "re_tacred": fetch_re_tacred,
        "rebel": fetch_rebel,
        "sredfm": fetch_sredfm,
        # Category Multi-hop
        "hotpotqa": fetch_hotpotqa,
        "musique": fetch_musique,
        "kilt": fetch_kilt,
        "graphrag_bench": fetch_graphrag_bench,
        "wildgraphbench": fetch_wildgraphbench,
        # Category Summarization
        "squality": fetch_squality,
        "zero_scrolls": fetch_zero_scrolls_and_qmsum,
        "govreport": fetch_govreport,
        "multinews": fetch_multinews,
        "microsoft_graphrag": fetch_microsoft_graphrag,
        # Additional
        "text2cypher": fetch_text2cypher,
    }

    if args.dataset:
        if args.dataset in dispatch:
            dispatch[args.dataset]()
        else:
            print(f"Unknown dataset '{args.dataset}'. Choose from: {list(dispatch.keys())}")
            sys.exit(1)
    elif args.category == "ie":
        fetch_information_extraction()
    elif args.category == "multi_hop":
        fetch_multi_hop()
    elif args.category == "summarization":
        fetch_summarization()
    elif args.category == "additional":
        fetch_additional()
    elif args.all:
        fetch_information_extraction()
        fetch_multi_hop()
        fetch_summarization()
        fetch_additional()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

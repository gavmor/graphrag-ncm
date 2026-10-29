"""Benchmark script for NCM GraphRAG extraction workload.

Runs a task's fixed evaluation set through NCMTaskGenerator and reports Entity F1, exact
match rate, throughput, and validity rate. The task is chosen by the ``NCM_TASK`` env var
(default ``extraction``); each task supplies its own deterministic benchmark examples via
``TaskSpec.benchmark_examples`` so held-out tasks (``real_extraction``) never score
in-sample data.
"""

import os
import time

import torch

from ncm_graphrag.data.extraction import parse_records
from ncm_graphrag.data.map_stage_real import predicted_score
from ncm_graphrag.ncm.generate import NCMTaskGenerator
from ncm_graphrag.ncm.task_registry import TASKS

TASK = os.environ.get("NCM_TASK", "extraction")


def run_benchmark():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    spec = TASKS[TASK]
    gen = NCMTaskGenerator(task=TASK)
    assert spec.benchmark_examples is not None, f"task {TASK!r} has no benchmark examples"
    examples = spec.benchmark_examples()

    # Warmup
    for e in examples[:5]:
        gen.generate(e.prefix_text)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    total_tokens = 0
    valid_count = 0
    total_f1 = 0.0
    exact_matches = 0

    if spec.metric == "score_auc":
        return _score_benchmark(gen, examples, t0, spec)

    for e in examples:
        out = gen.generate(e.prefix_text)
        total_tokens += gen.prefix_len + gen.net.L
        if gen.is_well_formed(out):
            valid_count += 1

        gt_entities = {
            (item["name"].strip().upper(), item["type"].strip().upper())
            for item in parse_records(e.target_text)["entities"]
        }
        pred_entities = {
            (item["name"].strip().upper(), item["type"].strip().upper()) for item in parse_records(out)["entities"]
        }

        tp = len(gt_entities & pred_entities)
        fp = len(pred_entities - gt_entities)
        fn = len(gt_entities - pred_entities)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        total_f1 += f1
        if gt_entities == pred_entities:
            exact_matches += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    entity_f1 = (total_f1 / len(examples)) * 100.0 if examples else 0.0
    em_rate = (exact_matches / len(examples)) * 100.0 if examples else 0.0
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / len(examples)) * 1000.0 if examples else 0.0
    valid_rate = (valid_count / len(examples)) if examples else 0.0

    print(f"METRIC entity_f1={entity_f1:.2f}")
    print(f"METRIC exact_match_rate={em_rate:.2f}")
    print(f"METRIC throughput_tok_per_sec={throughput:.2f}")
    print(f"METRIC latency_ms={latency_ms:.2f}")
    print(f"METRIC valid_rate={valid_rate:.4f}")


def _auc(pos: list[float], neg: list[float]) -> float:
    """Probability a relevant example scores above an irrelevant one (ties count 1/2)."""
    if not pos or not neg:
        return 0.0
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _score_benchmark(gen, examples, t0, spec) -> None:
    """Report ranking quality for bounded-ordinal-score tasks (the GraphRAG map stage).

    The map stage's job is to rank a report's relevance to a query, so the primary metric is
    the probability that a graded-relevant chunk is scored above an irrelevant one. Graded
    scores are also reported as mean absolute error against the task's canonical scores.
    """
    preds: list[float | None] = []
    for e in examples:
        out = gen.generate(e.prefix_text)
        valid = gen.is_well_formed(out)
        preds.append(predicted_score(out) if valid else None)
        del valid

    valid_count = sum(1 for p in preds if p is not None)
    pairs_pos = [p for p, e in zip(preds, examples, strict=False) if p is not None and e.grade >= 2]
    pairs_neg = [p for p, e in zip(preds, examples, strict=False) if p is not None and e.grade <= 1]
    auc = _auc(pairs_pos, pairs_neg)
    mae = sum(abs(p - e.score) for p, e in zip(preds, examples, strict=False) if p is not None) / max(valid_count, 1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tokens = (gen.prefix_len + gen.net.L) * len(examples)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / len(examples)) * 1000.0 if examples else 0.0

    print(f"METRIC score_auc={auc * 100.0:.2f}")
    print(f"METRIC score_mae={mae:.2f}")
    print(f"METRIC throughput_tok_per_sec={throughput:.2f}")
    print(f"METRIC latency_ms={latency_ms:.2f}")
    print(f"METRIC valid_rate={valid_count / max(len(examples), 1):.4f}")


if __name__ == "__main__":
    run_benchmark()

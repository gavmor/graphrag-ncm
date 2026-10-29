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
from ncm_graphrag.data.relation_tacred import micro_f1, predicted_label
from ncm_graphrag.data.report_multinews import entnum_grounding, grounding, rouge1_f
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
    if spec.metric == "draft_rouge":
        return _draft_benchmark(gen, examples, t0)
    if spec.metric == "relation_f1":
        return _relation_benchmark(gen, examples, t0)

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
    # "Relevant" = the task's directly-answering grade (3). Grades 0-2 are all non-answers, so
    # the ranking question is exactly the map stage's: does a relevant report outscore the rest.
    pairs_pos = [p for p, e in zip(preds, examples, strict=False) if p is not None and e.grade == 3]
    pairs_neg = [p for p, e in zip(preds, examples, strict=False) if p is not None and e.grade < 3]
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


def _draft_benchmark(gen, examples, t0) -> None:
    """ROUGE-1 vs the human reference sentence plus source grounding, over a held-out sample.

    Neither is faithfulness (see report_multinews.py): the blind claim-level judge is the real test.
    The lead-sentence baseline's numbers are printed alongside so the NCM is never read in isolation.
    """
    outs = [gen.generate(e.prefix_text) for e in examples]
    r1 = [rouge1_f(o, e.target_text) for o, e in zip(outs, examples, strict=False)]
    gr = [grounding(o, e.prefix_text) for o, e in zip(outs, examples, strict=False)]
    lead = [e.prefix_text.split(" || ")[0].split(". ")[0] for e in examples]
    lr1 = [rouge1_f(o, e.target_text) for o, e in zip(lead, examples, strict=False)]
    valid = sum(1 for o in outs if gen.is_well_formed(o))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n = max(len(examples), 1)
    print(f"METRIC draft_rouge1={100.0 * sum(r1) / n:.2f}")
    print(f"METRIC draft_grounding={100.0 * sum(gr) / n:.2f}")
    en = [entnum_grounding(o, e.prefix_text) for o, e in zip(outs, examples, strict=False)]
    en_ok, en_tot = sum(a for a, _ in en), sum(b for _, b in en)
    print(f"METRIC draft_entnum_grounding={100.0 * en_ok / max(en_tot, 1):.2f}")
    print(f"METRIC draft_entnum_count={en_tot / n:.2f}")
    print(f"METRIC draft_words={sum(len(o.split()) for o in outs) / n:.2f}")
    print(f"METRIC lead1_rouge1={100.0 * sum(lr1) / n:.2f}")
    print(f"METRIC throughput_tok_per_sec={(gen.prefix_len + gen.net.L) * n / elapsed if elapsed > 0 else 0.0:.2f}")
    print(f"METRIC latency_ms={elapsed / n * 1000.0:.2f}")
    print(f"METRIC valid_rate={valid / n:.4f}")


def _relation_benchmark(gen, examples, t0) -> None:
    """Micro-F1 over non-NA relations (the TACRED convention).

    An invalid output (not one of the task's labels) is a miss, never silently dropped, and `NA`
    predictions never count as hits, so always saying `NA` scores 0.
    """
    gold = [e.relation for e in examples]
    pred = [predicted_label(gen.generate(e.prefix_text)) for e in examples]
    prec, rec, f1 = micro_f1(gold, pred)
    valid = sum(1 for p in pred if p is not None)
    acc = sum(1 for g, p in zip(gold, pred, strict=False) if p == g) / max(len(examples), 1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tokens = (gen.prefix_len + gen.net.L) * len(examples)
    print(f"METRIC relation_f1={f1 * 100.0:.2f}")
    print(f"METRIC relation_precision={prec * 100.0:.2f}")
    print(f"METRIC relation_recall={rec * 100.0:.2f}")
    print(f"METRIC relation_acc={acc * 100.0:.2f}")
    print(f"METRIC throughput_tok_per_sec={total_tokens / elapsed if elapsed > 0 else 0.0:.2f}")
    print(f"METRIC latency_ms={(elapsed / len(examples)) * 1000.0 if examples else 0.0:.2f}")
    print(f"METRIC valid_rate={valid / max(len(examples), 1):.4f}")


if __name__ == "__main__":
    run_benchmark()

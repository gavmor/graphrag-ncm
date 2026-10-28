"""Benchmark script for NCM GraphRAG extraction workload.

Runs a fixed set of synthetic extraction inputs through NCMTaskGenerator,
measuring deterministic wall-clock inference throughput and validity rate on GPU.
"""

import time

import torch

from ncm_graphrag.data.extraction import generate_examples
from ncm_graphrag.ncm.generate import NCMTaskGenerator


def run_benchmark():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    gen = NCMTaskGenerator(task="extraction")
    # 50 fixed examples for steady-state throughput measurement
    examples = generate_examples(50, seed=42)

    # Warmup
    for e in examples[:5]:
        gen.generate(e.prefix_text)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    total_tokens = 0
    valid_count = 0

    for e in examples:
        out = gen.generate(e.prefix_text)
        total_tokens += gen.prefix_len + gen.net.L
        if gen.is_well_formed(out):
            valid_count += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / len(examples)) * 1000.0 if examples else 0.0
    valid_rate = (valid_count / len(examples)) if examples else 0.0

    print(f"METRIC throughput_tok_per_sec={throughput:.2f}")
    print(f"METRIC latency_ms={latency_ms:.2f}")
    print(f"METRIC valid_rate={valid_rate:.4f}")


if __name__ == "__main__":
    run_benchmark()

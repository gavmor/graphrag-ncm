"""Build judge input for the CURRENT report_multinews checkpoint, then run scripts/claim_judge.py.

    NCM_CORPORA_ROOT=... uv run python scripts/judge_drafter.py N OUT_PREFIX [CFG_SCALE]

Systems judged on the same N held-out sources: ncm (this checkpoint), lead1 and mismatch (anchors),
human_ref (what a perfect abstractive reference scores under this lead-only source).
"""

import json
import random
import sys

from ncm_graphrag.data import report_multinews as r
from ncm_graphrag.ncm.generate import NCMTaskGenerator

n, prefix = int(sys.argv[1]), sys.argv[2]
cfg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
bench = r.benchmark_examples()
idx = list(range(len(bench)))
random.Random(2).shuffle(idx)
idx = idx[:n]
gen = NCMTaskGenerator(task="report_multinews")
gen.cfg_scale = cfg
items = []
for k, i in enumerate(idx):
    e, other = bench[i], bench[idx[(k + 1) % len(idx)]]
    items.append(
        {
            "source": e.prefix_text,
            "systems": {
                "ncm": gen.generate(e.prefix_text),
                "lead1": e.prefix_text.split(" || ")[0].split(". ")[0] + ".",
                "mismatch": other.target_text,
                "human_ref": e.target_text,
            },
        }
    )
json.dump(items, open(f"{prefix}_in.json", "w"))
print(f"wrote {len(items)} items to {prefix}_in.json (cfg_scale={cfg})")

"""End-to-end smoke test: GraphRAG's real `GraphExtractor` (entity/relationship
extraction), served by our trained NCM checkpoint over the live litellm proxy.
Compares against a pure-Claude baseline for the same text units.

Per docs/constraints.md's scope note, this runs with `max_gleanings=0` (single-shot,
no gleaning loop) -- the real multi-round gleaning loop
(graph_extractor.py:_process_document) is a variable-length, multi-turn continuation
dialog, a structurally different shape than NCM's bounded single-call output; adapting
NCM to the gleaning loop itself is out of scope for this slice (see
docs/architecture.md).

Run via graphrag's own isolated venv (graphrag/graphrag-llm require Python >=3.11;
this project's main venv is pinned to 3.10 for the NCM/torch stack -- see
docs/constraints.md #5):

    cd third_party/graphrag
    NCM_LITELLM_BASE=http://127.0.0.1:4010/v1 uv run --package graphrag python \\
        ../../src/ncm_graphrag/graphrag_integration/run_extraction_demo.py

Prerequisite: the NCM litellm proxy must be running (see src/ncm_graphrag/serve/,
`uv run litellm --config src/ncm_graphrag/serve/litellm_config.yaml --port 4010`
from this project's own venv, serving both ncm-mapstage and ncm-extraction) and
ANTHROPIC_API_KEY (+ ANTHROPIC_WORKSPACE_ID, if the key requires it) must be set for
the Claude side.
"""

import asyncio
import os
import time

from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig

NCM_BASE = os.environ.get("NCM_LITELLM_BASE", "http://127.0.0.1:4010/v1")
CLAUDE_MODEL = "claude-haiku-5-5"
ANTHROPIC_EXTRA_HEADERS = (
    {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
)

# Same closed-vocabulary shape as src/ncm_graphrag/data/extraction.py's training data.
TEXT_UNITS = [
    "Atlas Corp partnered with Nile Labs this quarter, expanding joint operations in "
    "renewable energy across several regional markets.",
    "Harbor Systems acquired a stake in Crestline LLC this quarter, expanding joint "
    "operations in cybersecurity across several regional markets.",
    "Stratus Energy signed a supply agreement with Orbit Dynamics this quarter, "
    "expanding joint operations in space exploration across several regional markets.",
]
ENTITY_TYPES = ["ORGANIZATION"]


def _ncm_model():
    return create_completion(
        ModelConfig(
            type="litellm",
            model_provider="openai",
            model="ncm-extraction",
            api_base=NCM_BASE,
            api_key="sk-ncm-local",
        )
    )


def _claude_model():
    return create_completion(
        ModelConfig(
            type="litellm",
            model_provider="anthropic",
            model=CLAUDE_MODEL,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            call_args={"extra_headers": ANTHROPIC_EXTRA_HEADERS} if ANTHROPIC_EXTRA_HEADERS else {},
        )
    )


async def _run(model, label: str) -> None:
    extractor = GraphExtractor(model=model, prompt=GRAPH_EXTRACTION_PROMPT, max_gleanings=0)
    print(f"\n=== {label} ===")
    start = time.time()
    for i, text in enumerate(TEXT_UNITS):
        entities_df, relationships_df = await extractor(text, ENTITY_TYPES, source_id=f"unit-{i}")
        print(f"--- text unit {i}: {text[:60]}... ---")
        print("entities:")
        print(entities_df.to_string(index=False) if not entities_df.empty else "  (none)")
        print("relationships:")
        print(relationships_df.to_string(index=False) if not relationships_df.empty else "  (none)")
    print(f"elapsed: {time.time() - start:.1f}s for {len(TEXT_UNITS)} text units")


async def main() -> None:
    await _run(_ncm_model(), "NCM extraction")
    await _run(_claude_model(), "BASELINE: Claude extraction")


if __name__ == "__main__":
    asyncio.run(main())

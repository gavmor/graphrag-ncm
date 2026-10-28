"""End-to-end smoke test: GraphRAG's real `GlobalSearch` orchestration, map stage
served by our trained NCM checkpoint (over the live litellm proxy), reduce/final-
answer stage served by a standard AR LLM (Claude via litellm) -- the hybrid split
the task brief's own table recommends. Compares against a pure-Claude baseline
(both stages AR) for the same query.

Run via graphrag's own isolated venv (graphrag/graphrag-llm require Python >=3.11;
this project's main venv is pinned to 3.10 for the NCM/torch stack -- see
docs/constraints.md #5):

    cd third_party/graphrag
    NCM_LITELLM_BASE=http://127.0.0.1:4010/v1 uv run --package graphrag python \\
        ../../src/ncm_graphrag/graphrag_integration/run_map_stage_demo.py

Prerequisite: the NCM litellm proxy must be running (see src/ncm_graphrag/serve/,
`uv run litellm --config src/ncm_graphrag/serve/litellm_config.yaml --port 4010`
from this project's own venv) and ANTHROPIC_API_KEY (+ ANTHROPIC_WORKSPACE_ID, if
the key requires it) must be set for the Claude side.
"""

import asyncio
import os
import time

from graphrag.data_model.community import Community
from graphrag.data_model.community_report import CommunityReport
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig

from ncm_graphrag.graphrag_integration.hybrid_completion import HybridCompletion

NCM_BASE = os.environ.get("NCM_LITELLM_BASE", "http://127.0.0.1:4010/v1")
CLAUDE_MODEL = "claude-haiku-5-5"
ANTHROPIC_EXTRA_HEADERS = (
    {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
)

# Synthetic community reports -- same domains as src/ncm_graphrag/data/map_stage.py's
# training data, written out as graphrag's real CommunityReport/Community data models.
_REPORTS = [
    (
        "1",
        "Cybersecurity Developments",
        "cybersecurity",
        "Harbor Systems disclosed a coordinated response to a regional cybersecurity "
        "incident, coordinating with national infrastructure operators on patching "
        "timelines and public disclosure.",
    ),
    (
        "2",
        "Renewable Energy Developments",
        "renewable energy",
        "Stratus Energy announced a new offshore wind financing round, with "
        "regional regulators fast-tracking permits amid climate migration pressure "
        "on coastal communities.",
    ),
    (
        "3",
        "Football Transfers Developments",
        "football transfers",
        "Vantage Partners brokered a record transfer deal, with film industry "
        "sponsors attaching documentary rights to the player's upcoming season.",
    ),
    (
        "4",
        "Corporate Mergers Developments",
        "corporate mergers",
        "Fenwick Holdings completed its acquisition of a regional logistics firm, "
        "with monetary policy analysts flagging financing-rate sensitivity.",
    ),
]


def _build_context_builder() -> GlobalCommunityContext:
    reports = [
        CommunityReport(
            id=rid,
            short_id=rid,
            title=title,
            community_id=rid,
            summary=body,
            full_content=f"# {title}\n\n{body}",
            rank=1.0,
        )
        for rid, title, _domain, body in _REPORTS
    ]
    communities = [
        Community(id=rid, short_id=rid, title=title, level="0", parent="", children=[])
        for rid, title, _domain, _body in _REPORTS
    ]
    return GlobalCommunityContext(community_reports=reports, communities=communities)


def _ncm_model():
    return create_completion(
        ModelConfig(
            type="litellm",
            model_provider="openai",
            model="ncm-mapstage",
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


async def _run(model, label: str, query: str) -> None:
    # json_mode=False avoids a duplicate-kwarg TypeError in this graphrag version:
    # GlobalSearch.__init__ (json_mode=True, the default) stashes
    # response_format_json_object=True into self.map_llm_params, which
    # _map_response_single_batch then spreads via **llm_kwargs *in addition to*
    # its own hardcoded response_format_json_object=True kwarg -- a real duplicate-
    # kwarg bug in graphrag/query/structured_search/global_search/search.py:238-242,
    # not something introduced by this integration. The map call still gets
    # response_format_json_object=True from the hardcoded kwarg either way.
    search = GlobalSearch(model=model, context_builder=_build_context_builder(), json_mode=False)
    start = time.time()
    result = await search.search(query)
    elapsed = time.time() - start
    print(f"\n=== {label} ({elapsed:.1f}s, {result.llm_calls} llm calls) ===")
    print("--- map responses (per community report) ---")
    for i, mr in enumerate(result.map_responses):
        print(f"  report {i + 1}: {mr.response}")
    print("--- final answer ---")
    print(result.response)


async def main() -> None:
    query = "How are regional stakeholders responding to cybersecurity incidents?"

    claude = _claude_model()
    ncm = _ncm_model()
    hybrid = HybridCompletion(map_model=ncm, reduce_model=claude)

    await _run(hybrid, "HYBRID: NCM map stage + Claude reduce stage", query)
    await _run(claude, "BASELINE: Claude for both map and reduce", query)


if __name__ == "__main__":
    asyncio.run(main())

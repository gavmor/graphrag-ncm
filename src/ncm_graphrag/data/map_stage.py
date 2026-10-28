"""Synthetic dataset shaped like GraphRAG's Global Search map-stage call.

Real shape (see third_party/graphrag/.../global_search/search.py:_map_response_single_batch
and prompts/query/global_search_map_system_prompt.py): a system prompt carrying one batch
of community-report text (`context_data`) plus a user query; expected output is strict JSON
`{"points": [{"description": str, "score": 0-100}, ...]}`.

This generator produces single-report batches (one community report -> one point) --
the first vertical slice deliberately does not cover GraphRAG's real multi-report-per-batch
packing (see docs/architecture.md "Known scope cuts"). Each example is a closed-form
template fill (domain/entity pairs with a fixed relevance rule), not open-ended text --
appropriate for proving a from-scratch-trained NCM tokenizer+generator pair can produce
well-formed bounded JSON at all, not for evaluating open-domain relevance judgment quality.
"""

import json
import random
from dataclasses import dataclass

DOMAINS = [
    "renewable energy",
    "corporate mergers",
    "football transfers",
    "clinical trials",
    "monetary policy",
    "space exploration",
    "cybersecurity",
    "agriculture",
    "film industry",
    "urban housing",
    "archaeology",
    "climate migration",
]

# A few related-domain links so scoring isn't purely binary same/different.
RELATED = {
    "renewable energy": "climate migration",
    "climate migration": "agriculture",
    "corporate mergers": "monetary policy",
    "monetary policy": "corporate mergers",
    "cybersecurity": "space exploration",
    "space exploration": "cybersecurity",
    "football transfers": "film industry",
    "film industry": "football transfers",
    "clinical trials": "agriculture",
    "agriculture": "climate migration",
    "urban housing": "corporate mergers",
    "archaeology": "urban housing",
}

ENTITIES = [
    "Atlas Corp",
    "Nile Labs",
    "Meridian Group",
    "Solace Inc",
    "Vantage Partners",
    "Orbit Dynamics",
    "Harbor Systems",
    "Crestline LLC",
    "Union Robotics",
    "Alder Biotech",
    "Fenwick Holdings",
    "Stratus Energy",
]


@dataclass(frozen=True)
class MapStageExample:
    prefix_text: str
    target_text: str
    report_domain: str
    query_domain: str
    score: int


def _report_text(domain: str, entity: str) -> str:
    return (
        f"Community Report: {domain.title()} Developments. {entity} announced updates "
        f"in {domain} affecting regional stakeholders and ongoing projects."
    )


def _query_text(query_domain: str) -> str:
    return f"How does this relate to {query_domain}?"


def _target_json(description: str, score: int) -> str:
    # Matches GraphRAG's expected shape exactly: {"points": [{"description": ..., "score": ...}]}
    return json.dumps({"points": [{"description": description, "score": score}]}, separators=(",", ":"))


def _make_example(rng: random.Random) -> MapStageExample:
    report_domain = rng.choice(DOMAINS)
    entity = rng.choice(ENTITIES)
    roll = rng.random()
    if roll < 0.4:
        query_domain = report_domain
        score = rng.randint(75, 95)
        desc = f"Report on {report_domain} directly answers the query via {entity}."
    elif roll < 0.7 and report_domain in RELATED:
        query_domain = RELATED[report_domain]
        score = rng.randint(35, 60)
        desc = f"Report on {report_domain} partially relates via {entity}."
    else:
        others = [d for d in DOMAINS if d != report_domain]
        query_domain = rng.choice(others)
        score = rng.randint(0, 20)
        desc = f"Report on {report_domain} is unrelated to the query."

    return MapStageExample(
        prefix_text=_report_text(report_domain, entity) + " \n" + _query_text(query_domain),
        target_text=_target_json(desc, score),
        report_domain=report_domain,
        query_domain=query_domain,
        score=score,
    )


def generate_examples(n: int, seed: int) -> list[MapStageExample]:
    rng = random.Random(seed)
    return [_make_example(rng) for _ in range(n)]

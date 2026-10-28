"""Synthetic dataset shaped like GraphRAG's entity extraction call.

Real shape (see third_party/graphrag/.../index/operations/extract_graph/graph_extractor.py
and prompts/index/extract_graph.py): a single user message built from
GRAPH_EXTRACTION_PROMPT (few-shot instructions + `entity_types` + `input_text`);
expected output is graphrag's exact delimiter-tuple grammar, parsed by
`GraphExtractor._process_result` with zero tolerance for reformatting:

    ("entity"<|><NAME><|><TYPE><|><description>)##("entity"<|>...)##<|COMPLETE|>

(`TUPLE_DELIMITER="<|>"`, `RECORD_DELIMITER="##"`, `COMPLETION_DELIMITER="<|COMPLETE|>"`.)

**Entity-only, not entity+relationship.** An earlier version of this generator also
emitted a `("relationship"...)` record (3 records, ~140 gpt2 tokens, 6 VQ levels / 284
total codes). At that length the trained generator's free-running (non-teacher-forced)
output degraded badly past roughly the first entity -- 0/30 well-formed even with
best-of-32 reranking, a real measured instance of the coarse-to-fine exposure-bias
failure mode both `models/quant.py`'s own code comments and the Gemini background doc
(docs/NCM Training for GraphRAG.md ##5 "A primary failure mode... is exposure bias")
name: a 98.5% per-level-code teacher-forced accuracy still means ~0.985^284 =~ 1% of
sequences fully correct, and free-running generation is measurably worse than that
teacher-forced figure once a level conditions on its own (possibly wrong) prior output.
This generator instead follows the doc's own suggested mitigation directly: "decouple
extraction into two dedicated, parallel models: an entity model ... and a relation/
covariate model" -- this is the entity model (see docs/constraints.md ##6 for the
measured before/after and docs/architecture.md for the full account); a relationship
model following the same recipe at a shorter bound is the natural next slice.

This generator produces a bounded, closed-vocabulary version of the grammar -- 2
entities per text unit -- per docs/constraints.md's "closed-vocabulary synthetic data"
scope note (same pattern as map_stage.py). The real graph_extractor.py few-shot
examples sometimes omit the final "##" before "<|COMPLETE|>", which makes the last
record's trailing field fail to parse cleanly in graphrag's own parser -- this
generator always includes the trailing "##" so every synthetic example round-trips
through the *unmodified* real parser without hitting that edge case.
"""

import json
import random
import re
from dataclasses import dataclass

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

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

RELATION_VERBS = [
    "partnered with",
    "acquired a stake in",
    "signed a supply agreement with",
    "filed a joint patent with",
    "formed a research alliance with",
]


@dataclass(frozen=True)
class ExtractionExample:
    text_unit: str
    entity_types: str
    target_text: str

    @property
    def prefix_text(self) -> str:
        """Matches the "-Real Data-" section of GRAPH_EXTRACTION_PROMPT
        (`Entity_types: {entity_types}\\nText: {input_text}`) -- what
        serve/custom_handler.py's extraction handler extracts from the real prompt
        before feeding it to the model, so training and inference distributions match."""
        return f"Entity_types: {self.entity_types}\nText: {self.text_unit}"


def _record_entity(name: str, domain: str) -> str:
    desc = f"Operates in {domain}."
    return f'("entity"{TUPLE_DELIMITER}{name.upper()}{TUPLE_DELIMITER}ORGANIZATION{TUPLE_DELIMITER}{desc})'


def _make_example(rng: random.Random) -> ExtractionExample:
    domain = rng.choice(DOMAINS)
    src, tgt = rng.sample(ENTITIES, 2)
    verb = rng.choice(RELATION_VERBS)

    text_unit = (
        f"{src} {verb} {tgt} this quarter, expanding joint operations in {domain} across several regional markets."
    )

    target_text = RECORD_DELIMITER.join(
        [
            _record_entity(src, domain),
            _record_entity(tgt, domain),
            COMPLETION_DELIMITER,
        ]
    )

    return ExtractionExample(text_unit=text_unit, entity_types="ORGANIZATION", target_text=target_text)


def generate_examples(n: int, seed: int) -> list[ExtractionExample]:
    rng = random.Random(seed)
    return [_make_example(rng) for _ in range(n)]


def parse_records(result: str) -> dict[str, list]:
    """Minimal reimplementation of GraphExtractor._process_result's parsing rules
    (graph_extractor.py:124-178), reused locally so generate.py's best-of-N validity
    check doesn't need graphrag installed (it runs in this project's own Python 3.10
    venv; graphrag only installs into third_party/graphrag's own 3.11+ venv -- see
    docs/constraints.md #5). Kept in exact lockstep with the real parser's rules."""

    entities: list[dict] = []
    relationships: list[dict] = []
    records = [r.strip() for r in result.split(RECORD_DELIMITER)]
    for raw in records:
        record = re.sub(r"^\(|\)$", "", raw.strip())
        if not record or record == COMPLETION_DELIMITER:
            continue
        attrs = record.split(TUPLE_DELIMITER)
        if attrs[0] == '"entity"' and len(attrs) >= 4:
            entities.append({"name": attrs[1], "type": attrs[2], "description": attrs[3]})
        elif attrs[0] == '"relationship"' and len(attrs) >= 5:
            try:
                weight = float(attrs[-1])
            except ValueError:
                weight = 1.0
            relationships.append(
                {
                    "source": attrs[1],
                    "target": attrs[2],
                    "description": attrs[3],
                    "weight": weight,
                }
            )
    return {"entities": entities, "relationships": relationships}


def is_well_formed(result: str) -> bool:
    """At least one entity, the completion delimiter present, no dangling tuple
    fragments (every non-empty, non-COMPLETE record parses as entity or relationship)."""
    if COMPLETION_DELIMITER not in result:
        return False

    for raw in result.split(RECORD_DELIMITER):
        record = re.sub(r"^\(|\)$", "", raw.strip())
        if not record or record == COMPLETION_DELIMITER:
            continue
        attrs = record.split(TUPLE_DELIMITER)
        if attrs[0] == '"entity"' and len(attrs) >= 4:
            continue
        if attrs[0] == '"relationship"' and len(attrs) >= 5:
            continue
        return False
    parsed = parse_records(result)
    return len(parsed["entities"]) > 0


if __name__ == "__main__":
    for ex in generate_examples(3, seed=0):
        print(json.dumps({"text_unit": ex.text_unit, "target": ex.target_text}, indent=2))

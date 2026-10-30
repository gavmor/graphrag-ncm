"""Registry mapping a task name to its dataset generator, config builders, training
hyperparameters, and output-validity checker -- lets train_tokenizer.py/train_generator.py/
generate.py stay single implementations shared across tasks instead of duplicated
per-task copies (see docs/architecture.md for why a second task, "extraction", exists
alongside the original "mapstage" slice).

Every example object needs `.prefix_text` and `.target_text` string attributes
(`MapStageExample`/`ExtractionExample` both satisfy this by duck typing).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from ncm_graphrag.data import (
    extraction,
    fewnerd_extraction,
    map_stage,
    map_stage_hotpot,
    map_stage_real,
    real_extraction,
    rebel_extraction,
    relation_tacred,
    report_multinews,
)
from ncm_graphrag.ncm import config


@dataclass(frozen=True)
class TaskSpec:
    generate_examples: Callable[[int, int], list[Any]]
    build_tokenizer_cfg: Callable[[], DictConfig]
    build_model_cfg: Callable[[], DictConfig]
    build_generation_cfg: Callable[[], DictConfig]
    is_well_formed: Callable[[str], bool]
    tokenizer_steps: int
    tokenizer_n_examples: int
    generator_steps: int
    generator_n_examples: int
    # Examples the benchmark scores. Defaults to the task's own generator; held-out tasks
    # override this so the benchmark never scores in-sample data.
    benchmark_examples: Callable[[], list[Any]] | None = None
    # Which metric the benchmark reports for this task: "entity_f1" for extraction-style
    # tasks, "score_auc" for bounded-ordinal-score tasks (the map stage), "relation_f1" for
    # closed-vocabulary relation classification.
    metric: str = "entity_f1"


def _mapstage_is_well_formed(text: str) -> bool:
    import json

    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    points = parsed.get("points")
    return isinstance(points, list) and all("description" in p and "score" in p for p in points)


TASKS: dict[str, TaskSpec] = {
    "mapstage": TaskSpec(
        generate_examples=map_stage.generate_examples,
        build_tokenizer_cfg=config.build_tokenizer_cfg,
        build_model_cfg=config.build_model_cfg,
        build_generation_cfg=config.build_generation_cfg,
        is_well_formed=_mapstage_is_well_formed,
        tokenizer_steps=3000,
        tokenizer_n_examples=6000,
        generator_steps=4000,
        generator_n_examples=8000,
        benchmark_examples=lambda: map_stage.generate_examples(50, 42),
    ),
    "map_stage_real": TaskSpec(
        generate_examples=map_stage_real.generate_examples,
        build_tokenizer_cfg=config.build_map_stage_real_tokenizer_cfg,
        build_model_cfg=config.build_map_stage_real_model_cfg,
        build_generation_cfg=config.build_map_stage_real_generation_cfg,
        is_well_formed=map_stage_real.is_well_formed,
        tokenizer_steps=3000,
        tokenizer_n_examples=6000,
        generator_steps=4000,
        generator_n_examples=8000,
        benchmark_examples=map_stage_real.benchmark_examples,
        metric="score_auc",
    ),
    "map_stage_hotpot": TaskSpec(
        generate_examples=map_stage_hotpot.generate_examples,
        build_tokenizer_cfg=config.build_map_stage_hotpot_tokenizer_cfg,
        build_model_cfg=config.build_map_stage_hotpot_model_cfg,
        build_generation_cfg=config.build_map_stage_hotpot_generation_cfg,
        is_well_formed=map_stage_hotpot.is_well_formed,
        tokenizer_steps=3000,
        tokenizer_n_examples=6000,
        generator_steps=8000,
        generator_n_examples=60000,
        benchmark_examples=map_stage_hotpot.benchmark_examples,
        metric="score_auc",
    ),
    "relation_tacred": TaskSpec(
        generate_examples=relation_tacred.generate_examples,
        build_tokenizer_cfg=config.build_relation_tacred_tokenizer_cfg,
        build_model_cfg=config.build_relation_tacred_model_cfg,
        build_generation_cfg=config.build_relation_tacred_generation_cfg,
        is_well_formed=relation_tacred.is_well_formed,
        tokenizer_steps=3000,
        tokenizer_n_examples=6000,
        generator_steps=8000,
        generator_n_examples=67000,
        benchmark_examples=relation_tacred.benchmark_examples,
        metric="relation_f1",
    ),
    "report_multinews": TaskSpec(
        generate_examples=report_multinews.generate_examples,
        build_tokenizer_cfg=config.build_report_multinews_tokenizer_cfg,
        build_model_cfg=config.build_report_multinews_model_cfg,
        build_generation_cfg=config.build_report_multinews_generation_cfg,
        is_well_formed=report_multinews.is_well_formed,
        tokenizer_steps=6000,
        tokenizer_n_examples=8000,
        generator_steps=12000,
        generator_n_examples=40000,
        benchmark_examples=report_multinews.benchmark_examples,
        metric="draft_rouge",
    ),
    "rebel_extraction": TaskSpec(
        generate_examples=rebel_extraction.generate_examples,
        build_tokenizer_cfg=config.build_rebel_tokenizer_cfg,
        build_model_cfg=config.build_rebel_model_cfg,
        build_generation_cfg=config.build_rebel_generation_cfg,
        is_well_formed=rebel_extraction.is_well_formed,
        tokenizer_steps=6000,
        tokenizer_n_examples=50000,
        generator_steps=12000,
        generator_n_examples=200000,
        benchmark_examples=rebel_extraction.benchmark_examples,
        metric="graph_f1",
    ),
    "extraction": TaskSpec(
        generate_examples=extraction.generate_examples,
        build_tokenizer_cfg=config.build_extraction_tokenizer_cfg,
        build_model_cfg=config.build_extraction_model_cfg,
        build_generation_cfg=config.build_extraction_generation_cfg,
        is_well_formed=extraction.is_well_formed,
        tokenizer_steps=3500,
        tokenizer_n_examples=6000,
        generator_steps=12000,
        generator_n_examples=200000,
        benchmark_examples=lambda: extraction.generate_examples(50, 42),
    ),
    "real_extraction": TaskSpec(
        generate_examples=real_extraction.generate_examples,
        build_tokenizer_cfg=config.build_real_extraction_tokenizer_cfg,
        build_model_cfg=config.build_real_extraction_model_cfg,
        build_generation_cfg=config.build_real_extraction_generation_cfg,
        is_well_formed=extraction.is_well_formed,
        tokenizer_steps=6000,
        tokenizer_n_examples=6000,
        generator_steps=12000,
        generator_n_examples=8000,
        # Re-DocRED dev documents are never trained on.
        benchmark_examples=lambda: real_extraction.eval_examples("dev", 300, 42),
    ),
    "fewnerd_extraction": TaskSpec(
        generate_examples=fewnerd_extraction.generate_examples,
        build_tokenizer_cfg=config.build_fewnerd_extraction_tokenizer_cfg,
        build_model_cfg=config.build_fewnerd_extraction_model_cfg,
        build_generation_cfg=config.build_fewnerd_extraction_generation_cfg,
        is_well_formed=extraction.is_well_formed,
        tokenizer_steps=6000,
        tokenizer_n_examples=6000,
        generator_steps=12000,
        generator_n_examples=50000,
        # Few-NERD `validation` sentences are never trained on.
        benchmark_examples=lambda: fewnerd_extraction.eval_examples("validation", 300, 42),
    ),
}

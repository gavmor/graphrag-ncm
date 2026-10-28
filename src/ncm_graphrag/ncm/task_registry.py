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

from ncm_graphrag.data import extraction, map_stage
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
    ),
    "extraction": TaskSpec(
        generate_examples=extraction.generate_examples,
        build_tokenizer_cfg=config.build_extraction_tokenizer_cfg,
        build_model_cfg=config.build_extraction_model_cfg,
        build_generation_cfg=config.build_extraction_generation_cfg,
        is_well_formed=extraction.is_well_formed,
        tokenizer_steps=3500,
        tokenizer_n_examples=6000,
        generator_steps=5000,
        generator_n_examples=8000,
    ),
}

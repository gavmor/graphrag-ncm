"""Module-level singleton loading the trained map-stage NCM pair once per process.

Mirrors the pattern already working at /home/user/Desktop/ncm/ncm_runtime.py (module-
level load on import, `generate(text) -> text` entrypoint) so `custom_handler.py` can
plug into litellm's `CustomLLM` adapter the same way.
"""

from ncm_graphrag.ncm.generate import MapStageGenerator

_GENERATOR: MapStageGenerator | None = None


def _get_generator() -> MapStageGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = MapStageGenerator()
    return _GENERATOR


def generate(text: str) -> str:
    """Run one conditional NCM generation: "report + query" text in, JSON text out.

    Uses best-of-N JSON-validity reranking (see MapStageGenerator.generate_best_of_n)
    since single-sample greedy decode at this param count is rough (docs/constraints.md)
    and a map-stage caller already generates many short outputs per query -- the
    reranking budget is effectively free (one batched forward pass).
    """
    return _get_generator().generate_best_of_n(text)

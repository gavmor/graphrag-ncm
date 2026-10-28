"""Module-level singletons loading each task's trained NCM pair once per process.

Mirrors the pattern already working at /home/user/Desktop/ncm/ncm_runtime.py (module-
level load on import, `generate(text) -> text` entrypoint) so `custom_handler.py` can
plug into litellm's `CustomLLM` adapter the same way, generalized to serve more than
one task (mapstage, extraction) from the same proxy process.
"""

from ncm_graphrag.ncm.generate import NCMTaskGenerator

_GENERATORS: dict[str, NCMTaskGenerator] = {}


def _get_generator(task: str) -> NCMTaskGenerator:
    if task not in _GENERATORS:
        _GENERATORS[task] = NCMTaskGenerator(task=task)
    return _GENERATORS[task]


def generate(task: str, text: str) -> str:
    """Run one conditional NCM generation for `task`: bounded prefix text in, bounded
    task-shaped text out.

    Uses best-of-N well-formedness reranking (see NCMTaskGenerator.generate_best_of_n)
    since single-sample greedy decode at this param count is rough (docs/constraints.md)
    and both tasks already generate many short outputs per query/document -- the
    reranking budget is effectively free (one batched forward pass).
    """
    return _get_generator(task).generate_best_of_n(text)

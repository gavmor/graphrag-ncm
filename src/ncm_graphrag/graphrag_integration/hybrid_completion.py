"""Routes GraphRAG Global Search calls to two different backends, per the brief's
hybrid architecture (point 3): NCM for the bounded-output map stage, a standard AR LLM
for the reduce / final-answer-synthesis stage.

`GlobalSearch` (graphrag/query/structured_search/global_search/search.py) takes a
single `model: LLMCompletion` for both the map and reduce calls -- there is no native
two-model split in the class. This wraps two already-constructed `LLMCompletion`
instances (built via graphrag's own `create_completion` factory, see
run_map_stage_demo.py) and routes on the one reliable signal that distinguishes the
two call shapes: the map stage always passes `response_format_json_object=True`
(search.py:97 sets it unconditionally when `json_mode=True`, and `_map_response_single_batch`
at search.py:240 always forwards it); the reduce stage never does (search.py:396-404
calls with `stream=True` instead).
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from graphrag_llm.completion import LLMCompletion


class HybridCompletion(LLMCompletion):
    """Delegates to `map_model` for map-stage (bounded JSON) calls, `reduce_model` for
    everything else. Not a general-purpose router -- a routing heuristic specific to
    GlobalSearch's two known call shapes (see module docstring)."""

    def __init__(self, map_model: LLMCompletion, reduce_model: LLMCompletion) -> None:
        self._map_model = map_model
        self._reduce_model = reduce_model

    def _route(self, kwargs: dict[str, Any]) -> LLMCompletion:
        return self._map_model if kwargs.get("response_format_json_object") else self._reduce_model

    def completion(self, /, **kwargs: Any) -> Any:
        return self._route(kwargs).completion(**kwargs)

    async def completion_async(self, /, **kwargs: Any) -> Any:
        return await self._route(kwargs).completion_async(**kwargs)

    def streaming(self, /, **kwargs: Any) -> Iterator[Any]:
        return self._route(kwargs).completion(**kwargs)  # type: ignore[return-value]

    async def astreaming(self, /, **kwargs: Any) -> AsyncIterator[Any]:
        return await self._route(kwargs).completion_async(**kwargs)  # type: ignore[return-value]

    @property
    def metrics_store(self) -> Any:
        return self._reduce_model.metrics_store

    @property
    def tokenizer(self) -> Any:
        return self._reduce_model.tokenizer

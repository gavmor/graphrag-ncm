"""litellm CustomLLM adapter for the trained map-stage NCM generator.

litellm handles all OpenAI wire-protocol plumbing (chat/completions, response_format
JSON-mode, SSE framing, /v1/models). This file concatenates the community-report
text and the query into one prefix string, matching how this model was trained
(see src/ncm_graphrag/data/map_stage.py: "<report text> \\n<query text>").

GraphRAG's real map-stage call (graphrag/query/structured_search/global_search/search.py:
_map_response_single_batch) sends the *entire* MAP_SYSTEM_PROMPT as the system message --
~400 tokens of fixed instructions wrapping the actual community-report `context_data` in
the middle (prompts/query/global_search_map_system_prompt.py, between the literal
"---Data tables---" and "---Goal---" markers, repeated twice in the template). Feeding
that whole block to a model trained on short "report + query" pairs is badly off-
distribution -- this extracts just the data-table section so what reaches the model at
least matches our training format's intent (report text, not prompt boilerplate).

This does NOT fully close the training/production distribution gap: GraphRAG's real
community-report formatting (produced by `build_community_context`) still differs
textually from this project's closed-template synthetic data (see docs/constraints.md
#4) -- a production deployment needs a tokenizer+generator trained on real graphrag-
produced community reports, not just boilerplate-stripped input to this synthetic model.

Adapted from the working pattern at /home/user/Desktop/ncm/custom_handler.py.
"""

import time
import uuid
from collections.abc import AsyncIterator, Iterator

from litellm import CustomLLM
from litellm.types.utils import Choices, GenericStreamingChunk, Message, ModelResponse, Usage

from ncm_graphrag.serve.ncm_runtime import generate

_DATA_START = "---Data tables---"
_DATA_END = "---Goal---"


def _extract_context_data(system_text: str) -> str:
    """Pull the community-report block out of GraphRAG's MAP_SYSTEM_PROMPT, if present."""
    start = system_text.find(_DATA_START)
    if start == -1:
        return system_text
    start += len(_DATA_START)
    end = system_text.find(_DATA_END, start)
    data = system_text[start:end] if end != -1 else system_text[start:]
    return data.strip()


def _build_prefix(messages) -> str:
    system_text, user_text = "", ""
    for m in messages or []:
        content = m.get("content")
        text = (
            content
            if isinstance(content, str)
            else "".join(b.get("text", "") for b in (content or []) if isinstance(b, dict) and b.get("type") == "text")
        )
        if m.get("role") == "system":
            system_text = text
        elif m.get("role") == "user":
            user_text = text
    if not system_text:
        return user_text
    return f"{_extract_context_data(system_text)} \n{user_text}"


def _build_response(requested_model: str, out_text: str, prompt_tokens: int) -> ModelResponse:
    completion_tokens = len(out_text.split())
    return ModelResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=requested_model,
        object="chat.completion",
        choices=[Choices(index=0, finish_reason="stop", message=Message(role="assistant", content=out_text))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class NCMMapStageLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> ModelResponse:
        prefix = _build_prefix(kwargs.get("messages"))
        out_text = generate(prefix) if prefix.strip() else ""
        return _build_response(kwargs.get("model", "ncm-mapstage"), out_text, len(prefix.split()))

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        return self.completion(*args, **kwargs)

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        prefix = _build_prefix(kwargs.get("messages"))
        out_text = generate(prefix) if prefix.strip() else ""
        yield GenericStreamingChunk(
            finish_reason="stop",
            index=0,
            is_finished=True,
            text=out_text,
            tool_use=None,
            usage={"completion_tokens": len(out_text.split()), "prompt_tokens": 0, "total_tokens": 0},
        )

    async def astreaming(  # ty: ignore[invalid-method-override]  # litellm's base is a coroutine fn by signature, async-generator by convention
        self, *args, **kwargs
    ) -> AsyncIterator[GenericStreamingChunk]:
        for chunk in self.streaming(*args, **kwargs):
            yield chunk


ncm_mapstage_llm = NCMMapStageLLM()

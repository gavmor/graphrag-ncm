"""litellm CustomLLM adapters for the trained map-stage and extraction NCM generators.

litellm handles all OpenAI wire-protocol plumbing (chat/completions, response_format
JSON-mode, SSE framing, /v1/models). Each handler's job is just to turn graphrag's real
prompt (full of fixed instruction boilerplate neither model was trained on) into the
short bounded prefix text each model *was* trained on, matching
src/ncm_graphrag/data/{map_stage,extraction}.py's `.prefix_text` shape:

- map-stage: `_map_response_single_batch` (search.py) sends system=MAP_SYSTEM_PROMPT
  (~400 tokens of instructions wrapping the community-report `context_data` between
  literal "---Data tables---"/"---Goal---" markers) + user=query. We extract just the
  data-table section and recombine as "<report text> \\n<query text>".
- extraction: `_process_document` (graph_extractor.py) sends a single user message built
  from GRAPH_EXTRACTION_PROMPT (~900 tokens: goal, format rules, 3 worked few-shot
  examples, then a "-Real Data-" section with `Entity_types: ...` / `Text: ...`). We
  extract just that "-Real Data-" section, matching ExtractionExample.prefix_text.

Neither extraction fully closes the training/production distribution gap: GraphRAG's
real community-report and text-unit formatting still differs textually from this
project's closed-template synthetic data (see docs/constraints.md #4) -- a production
deployment needs tokenizer+generator pairs trained on real graphrag-produced data, not
just boilerplate-stripped input to these synthetic models.

Adapted from the working pattern at /home/user/Desktop/ncm/custom_handler.py.
"""

import time
import uuid
from collections.abc import AsyncIterator, Iterator

from litellm import CustomLLM
from litellm.types.utils import Choices, GenericStreamingChunk, Message, ModelResponse, Usage

from ncm_graphrag.serve.ncm_runtime import generate

_MAP_DATA_START = "---Data tables---"
_MAP_DATA_END = "---Goal---"

_EXTRACTION_REAL_DATA = "-Real Data-"
_EXTRACTION_TEXT_MARKER = "Text:"
_EXTRACTION_SECTION_END = "######################"
_EXTRACTION_OUTPUT_MARKER = "Output:"


def _messages_text(messages, role: str) -> str:
    for m in reversed(messages or []):
        if m.get("role") != role:
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        return "".join(b.get("text", "") for b in (content or []) if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _build_prefix_mapstage(messages) -> str:
    system_text = _messages_text(messages, "system")
    user_text = _messages_text(messages, "user")
    if not system_text:
        return user_text
    start = system_text.find(_MAP_DATA_START)
    if start == -1:
        data = system_text
    else:
        start += len(_MAP_DATA_START)
        end = system_text.find(_MAP_DATA_END, start)
        data = system_text[start:end] if end != -1 else system_text[start:]
    return f"{data.strip()} \n{user_text}"


def _build_prefix_extraction(messages) -> str:
    """Extract the "-Real Data-" section's `Entity_types: ...` / `Text: ...` body from
    GRAPH_EXTRACTION_PROMPT, matching `ExtractionExample.prefix_text` exactly."""
    user_text = _messages_text(messages, "user")
    idx = user_text.find(_EXTRACTION_REAL_DATA)
    if idx == -1:
        return user_text
    section = user_text[idx:]
    t_idx = section.find(_EXTRACTION_TEXT_MARKER)
    if t_idx == -1:
        return section.strip()
    entity_types_line = section[:t_idx].splitlines()
    entity_types = next((ln for ln in entity_types_line if ln.strip().startswith("Entity_types:")), "")
    body = section[t_idx:]
    end = body.find(_EXTRACTION_SECTION_END)
    if end == -1:
        end = body.find(_EXTRACTION_OUTPUT_MARKER)
    if end != -1:
        body = body[:end]
    return f"{entity_types.strip()}\n{body.strip()}"


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


class NCMTaskLLM(CustomLLM):
    """Generic litellm CustomLLM wrapping one registered NCM task (see
    ncm_graphrag.ncm.task_registry.TASKS)."""

    def __init__(self, task: str, build_prefix) -> None:
        super().__init__()
        self._task = task
        self._build_prefix = build_prefix

    def completion(self, *args, **kwargs) -> ModelResponse:
        prefix = self._build_prefix(kwargs.get("messages"))
        out_text = generate(self._task, prefix) if prefix.strip() else ""
        return _build_response(kwargs.get("model", f"ncm-{self._task}"), out_text, len(prefix.split()))

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        return self.completion(*args, **kwargs)

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        prefix = self._build_prefix(kwargs.get("messages"))
        out_text = generate(self._task, prefix) if prefix.strip() else ""
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


ncm_mapstage_llm = NCMTaskLLM("mapstage", _build_prefix_mapstage)
ncm_extraction_llm = NCMTaskLLM("extraction", _build_prefix_extraction)

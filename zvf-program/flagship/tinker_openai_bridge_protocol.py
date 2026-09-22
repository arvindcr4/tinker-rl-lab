"""Pure protocol helpers for the secured Tinker OpenAI-compatible bridge."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping


USD_PER_M_PREFILL = 0.54
USD_PER_M_SAMPLE = 1.335

_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(?P<payload>.*?)\s*</tool_call>",
    flags=re.DOTALL,
)
_XML_FUNCTION_BLOCK = re.compile(
    r"^<function=(?P<name>[^>]+)>\s*(?P<body>.*?)\s*</function>$",
    flags=re.DOTALL,
)
_XML_PARAMETER_BLOCK = re.compile(
    r"<parameter=(?P<name>[^>]+)>\s*(?P<value>.*?)\s*</parameter>",
    flags=re.DOTALL,
)


def estimate_tinker_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Return the pinned Tinker inference price for one request."""

    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return round(
        prompt_tokens / 1_000_000 * USD_PER_M_PREFILL
        + completion_tokens / 1_000_000 * USD_PER_M_SAMPLE,
        9,
    )


def bearer_token(authorization: str | None) -> str | None:
    """Extract a strict HTTP Bearer credential without logging it."""

    if not authorization:
        return None
    scheme, separator, credential = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not credential.strip():
        return None
    return credential.strip()


def _normalise_tool_call(payload: Mapping[str, Any], raw: str) -> dict[str, Any] | None:
    function = payload.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(arguments, str):
        argument_text = arguments
        try:
            json.loads(argument_text)
        except json.JSONDecodeError:
            return None
    else:
        argument_text = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    call_id = payload.get("id")
    if not isinstance(call_id, str) or not call_id:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        call_id = f"call_{digest}"
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name.strip(), "arguments": argument_text},
    }


def _parse_xml_tool_call(raw: str) -> Mapping[str, Any] | None:
    """Parse the native Qwen3 ``function/parameter`` tool-call notation."""

    function_match = _XML_FUNCTION_BLOCK.fullmatch(raw.strip())
    if function_match is None:
        return None
    arguments: dict[str, Any] = {}
    body = function_match.group("body")
    matches = list(_XML_PARAMETER_BLOCK.finditer(body))
    # Qwen emits an empty function body for valid zero-argument tools such as
    # ``toolbelt_list_tools``. Reject only unparsed body content; an empty body
    # is the canonical representation of an empty argument object.
    if _XML_PARAMETER_BLOCK.sub("", body).strip():
        return None
    for match in matches:
        name = match.group("name").strip()
        if not name or name in arguments:
            return None
        value = match.group("value").strip()
        try:
            arguments[name] = json.loads(value)
        except json.JSONDecodeError:
            arguments[name] = value
    return {"name": function_match.group("name").strip(), "arguments": arguments}


def parse_qwen_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert Qwen ``<tool_call>`` blocks into OpenAI tool-call objects.

    Invalid blocks remain visible in assistant content instead of being silently
    converted into executable calls.
    """

    calls: list[dict[str, Any]] = []
    valid_spans: list[tuple[int, int]] = []
    for match in _TOOL_CALL_BLOCK.finditer(text):
        raw = match.group("payload")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = _parse_xml_tool_call(raw)
        if not isinstance(payload, Mapping):
            continue
        call = _normalise_tool_call(payload, raw)
        if call is None:
            continue
        calls.append(call)
        valid_spans.append(match.span())

    content = text
    for start, end in reversed(valid_spans):
        content = content[:start] + content[end:]
    content = content.strip()
    return (content or None), calls


def normalise_openai_messages_for_qwen(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI wire-format tool arguments for Qwen's chat template.

    OpenAI carries ``function.arguments`` as a JSON string. The pinned Qwen
    template iterates the arguments as a mapping, so multi-turn tool sessions
    need the string decoded before rendering.
    """

    normalised = deepcopy(messages)
    for message in normalised:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise ValueError("tool_calls entries must be objects")
            function = tool_call.get("function")
            if not isinstance(function, dict):
                raise ValueError("tool call function must be an object")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError("tool call arguments must contain JSON") from exc
            if not isinstance(arguments, Mapping):
                raise ValueError("tool call arguments must decode to an object")
            function["arguments"] = dict(arguments)
    return normalised


def openai_chat_stream_events(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str | None,
    tool_calls: list[dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
) -> list[str]:
    """Serialize a completed sample as OpenAI chat-completion SSE events.

    Tinker returns a completed sample rather than token deltas. Emitting that
    sample as one content/tool-call delta still satisfies clients that require
    the streaming Chat Completions transport.
    """

    def event(choices: list[dict[str, Any]], usage: dict[str, int] | None = None) -> str:
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": choices,
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    events = [
        event(
            [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ]
        )
    ]
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = [
            {"index": index, **tool_call} for index, tool_call in enumerate(tool_calls)
        ]
    if delta:
        events.append(event([{"index": 0, "delta": delta, "finish_reason": None}]))
    events.append(
        event(
            [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ]
        )
    )
    events.append(
        event(
            [],
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )
    )
    events.append("data: [DONE]\n\n")
    return events


def _responses_text(part_output: Any) -> str:
    """Stringify a function_call_output output (string or content parts)."""
    if isinstance(part_output, str):
        return part_output
    if isinstance(part_output, list):
        texts = []
        for part in part_output:
            if isinstance(part, Mapping):
                texts.append(str(part.get("text", part.get("output_text", ""))))
            else:
                texts.append(str(part))
        return "\n".join(t for t in texts if t)
    return "" if part_output is None else str(part_output)


def responses_input_to_chat_messages(input: Any) -> list[dict[str, Any]]:
    """Convert a Responses API input (string or item list) to chat messages.

    History items the server regenerates (function_call, reasoning) are
    skipped; function_call_output items become tool messages. Raises
    ValueError on an unusable shape.
    """
    if isinstance(input, str):
        return [{"role": "user", "content": input}]
    if not isinstance(input, list):
        raise ValueError("responses input must be a string or a list of items")
    messages: list[dict[str, Any]] = []
    for item in input:
        if not isinstance(item, Mapping):
            raise ValueError("responses input items must be objects")
        kind = item.get("type", "message")
        if kind == "message":
            parts = []
            for chunk in item.get("content", []):
                if not isinstance(chunk, Mapping):
                    continue
                ctype = chunk.get("type")
                if ctype in ("input_text", "output_text"):
                    parts.append(str(chunk.get("text", "")))
                elif ctype == "refusal":
                    parts.append("refusal: " + str(chunk.get("refusal", "")))
            text = "\n".join(p for p in parts if p)
            messages.append(
                {"role": item.get("role", "user"), "content": text})
        elif kind == "function_call_output":
            call_id = item.get("call_id", "")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("function_call_output item lacks call_id")
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": _responses_text(item.get("output"))})
    if not messages:
        raise ValueError("responses input carried no usable messages")
    return messages


def responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]] | None:
    """Convert Responses function tools to chat-completions tools."""
    chat: list[dict[str, Any]] = []
    for tool in tools or []:
        if isinstance(tool, Mapping) and tool.get("type") == "function":
            chat.append({"type": "function", "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }})
    return chat or None


def build_responses_object(*, response_id: str, model: str,
                           content: str | None,
                           tool_calls: list[dict[str, Any]],
                           prompt_tokens: int, completion_tokens: int,
                           created_at: int) -> dict[str, Any]:
    """Build a Responses API response object (non-streaming shape)."""
    output: list[dict[str, Any]] = []
    if content:
        output.append({"id": f"msg_{response_id}", "type": "message",
                       "status": "completed", "role": "assistant",
                       "content": [{"type": "output_text", "text": content,
                                    "annotations": []}]})
    for call in tool_calls:
        output.append({"id": f"fc_{call['id']}", "type": "function_call",
                       "status": "completed", "call_id": call["id"],
                       "name": call["function"]["name"],
                       "arguments": call["function"]["arguments"]})
    return {"id": response_id, "object": "response", "created_at": created_at,
            "model": model, "status": "completed", "output": output,
            "usage": {"input_tokens": prompt_tokens,
                      "output_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}}


def _responses_sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def iter_responses_sse_events(response: dict[str, Any],
                              text_chunk_size: int = 500) -> Any:
    """Yield SSE event strings replaying a completed Responses object."""
    rid = response["id"]
    header = {"id": rid, "object": "response", "model": response["model"]}
    yield _responses_sse("response.created",
                         {"type": "response.created", "response": header})
    index = 0
    for item in response["output"]:
        if item["type"] == "message":
            text = item["content"][0]["text"] if item["content"] else ""
            yield _responses_sse(
                "response.output_item.added",
                {"type": "response.output_item.added",
                 "output_index": index, "item": item})
            yield _responses_sse(
                "response.content_part.added",
                {"type": "response.content_part.added",
                 "item_id": item["id"], "output_index": index,
                 "part": {"type": "output_text", "text": "",
                          "annotations": []}})
            for start in range(0, len(text), text_chunk_size):
                yield _responses_sse(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta",
                     "item_id": item["id"], "output_index": index,
                     "content_index": 0,
                     "delta": text[start:start + text_chunk_size]})
            yield _responses_sse(
                "response.output_text.done",
                {"type": "response.output_text.done",
                 "item_id": item["id"], "output_index": index,
                 "content_index": 0, "text": text})
            yield _responses_sse(
                "response.content_part.done",
                {"type": "response.content_part.done",
                 "item_id": item["id"], "output_index": index,
                 "content_index": 0,
                 "part": {"type": "output_text", "text": text,
                          "annotations": []}})
            yield _responses_sse(
                "response.output_item.done",
                {"type": "response.output_item.done",
                 "output_index": index, "item": item})
            index += 1
        elif item["type"] == "function_call":
            yield _responses_sse(
                "response.output_item.added",
                {"type": "response.output_item.added",
                 "output_index": index, "item": item})
            yield _responses_sse(
                "response.output_item.done",
                {"type": "response.output_item.done",
                 "output_index": index, "item": item})
            index += 1
    yield _responses_sse(
        "response.completed",
        {"type": "response.completed", "response": response})

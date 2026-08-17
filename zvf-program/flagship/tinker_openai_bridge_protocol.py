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
    if not matches or _XML_PARAMETER_BLOCK.sub("", body).strip():
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
        events.append(
            event([{"index": 0, "delta": delta, "finish_reason": None}])
        )
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

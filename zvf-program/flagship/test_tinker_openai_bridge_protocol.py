from __future__ import annotations

import json
import unittest

from flagship.tinker_openai_bridge_protocol import (
    bearer_token,
    build_responses_object,
    estimate_tinker_usd,
    iter_responses_sse_events,
    normalise_openai_messages_for_qwen,
    openai_chat_stream_events,
    parse_qwen_tool_calls,
    responses_input_to_chat_messages,
    responses_tools_to_chat_tools,
)
import pytest


class TinkerOpenAIBridgeProtocolTests(unittest.TestCase):
    def test_extracts_qwen_tool_call_and_preserves_content(self) -> None:
        content, calls = parse_qwen_tool_calls(
            'I will inspect it.\n<tool_call>{"name":"shell","arguments":{"cmd":"pwd"}}</tool_call>'
        )
        assert content == "I will inspect it."
        assert calls[0]["function"]["name"] == "shell"
        assert json.loads(calls[0]["function"]["arguments"]) == {"cmd": "pwd"}
        assert calls[0]["id"].startswith("call_")

    def test_accepts_openai_shaped_tool_call(self) -> None:
        content, calls = parse_qwen_tool_calls(
            '<tool_call>{"id":"call_fixed","function":{"name":"read_file","arguments":"{\\"path\\":\\"a.txt\\"}"}}</tool_call>'
        )
        assert content is None
        assert calls[0]["id"] == "call_fixed"
        assert calls[0]["function"]["name"] == "read_file"

    def test_accepts_native_qwen_xml_tool_call(self) -> None:
        content, calls = parse_qwen_tool_calls(
            "Plan first.\n<tool_call>\n"
            "<function=todowrite>\n"
            "<parameter=todos>\n"
            '[{"content":"Inspect files","status":"in_progress"}]\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        assert content == "Plan first."
        assert calls[0]["function"]["name"] == "todowrite"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "todos": [{"content": "Inspect files", "status": "in_progress"}]
        }

    def test_native_qwen_xml_plain_string_parameter(self) -> None:
        _, calls = parse_qwen_tool_calls(
            "<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>"
        )
        assert json.loads(calls[0]["function"]["arguments"]) == {"command": "pwd"}

    def test_accepts_native_qwen_zero_argument_tool_call(self) -> None:
        content, calls = parse_qwen_tool_calls(
            "<tool_call><function=toolbelt_list_tools></function></tool_call>"
        )
        assert content is None
        assert calls[0]["function"]["name"] == "toolbelt_list_tools"
        assert json.loads(calls[0]["function"]["arguments"]) == {}

    def test_invalid_tool_call_remains_non_executable_content(self) -> None:
        text = '<tool_call>{"name":"shell","arguments":"not-json"}</tool_call>'
        content, calls = parse_qwen_tool_calls(text)
        assert content == text
        assert calls == []

    def test_bearer_token_is_strict(self) -> None:
        assert bearer_token("Bearer secret") == "secret"
        assert bearer_token("bearer secret") == "secret"
        assert bearer_token("Basic secret") is None
        assert bearer_token("Bearer") is None

    def test_cost_projection_uses_pinned_prices(self) -> None:
        assert estimate_tinker_usd(1000000, 1000000) == 1.875
        with pytest.raises(ValueError):
            estimate_tinker_usd(-1, 0)

    def test_stream_events_follow_openai_sse_shape(self) -> None:
        events = openai_chat_stream_events(
            completion_id="chatcmpl-fixed",
            created=123,
            model="pavlov-model",
            content="done",
            tool_calls=[],
            prompt_tokens=10,
            completion_tokens=2,
        )
        payloads = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
        assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
        assert payloads[1]["choices"][0]["delta"]["content"] == "done"
        assert payloads[2]["choices"][0]["finish_reason"] == "stop"
        assert payloads[3]["usage"]["total_tokens"] == 12
        assert events[-1] == "data: [DONE]\n\n"

    def test_stream_events_include_indexed_tool_calls(self) -> None:
        _, calls = parse_qwen_tool_calls(
            '<tool_call>{"name":"shell","arguments":{"cmd":"pwd"}}</tool_call>'
        )
        events = openai_chat_stream_events(
            completion_id="chatcmpl-fixed",
            created=123,
            model="pavlov-model",
            content=None,
            tool_calls=calls,
            prompt_tokens=10,
            completion_tokens=2,
        )
        delta = json.loads(events[1].removeprefix("data: "))["choices"][0]["delta"]
        assert delta["tool_calls"][0]["index"] == 0
        assert delta["tool_calls"][0]["function"]["name"] == "shell"
        finish = json.loads(events[2].removeprefix("data: "))["choices"][0]
        assert finish["finish_reason"] == "tool_calls"

    def test_normalises_openai_tool_arguments_for_qwen_template(self) -> None:
        source = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"pwd"}',
                        },
                    }
                ],
            }
        ]
        normalised = normalise_openai_messages_for_qwen(source)
        assert normalised[0]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
        assert isinstance(source[0]["tool_calls"][0]["function"]["arguments"], str)

    def test_rejects_non_object_tool_arguments(self) -> None:
        with pytest.raises(ValueError):
            normalise_openai_messages_for_qwen(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "bash", "arguments": "[]"}}],
                    }
                ]
            )


class ResponsesConversionTests(unittest.TestCase):
    def test_string_input_becomes_user_message(self) -> None:
        assert responses_input_to_chat_messages("hello") == [{"role": "user", "content": "hello"}]

    def test_item_list_maps_messages_and_tool_outputs(self) -> None:
        messages = responses_input_to_chat_messages(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "run it"}],
                },
                {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "done"},
            ]
        )
        assert messages[0] == {"role": "user", "content": "run it"}
        assert messages[1] == {"role": "tool", "tool_call_id": "call_1", "content": "done"}
        assert len(messages) == 2  # history call skipped

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError):
            responses_input_to_chat_messages([])
        with pytest.raises(ValueError):
            responses_input_to_chat_messages(
                [{"type": "function_call", "call_id": "c", "name": "x", "arguments": "{}"}]
            )

    def test_tools_mapping(self) -> None:
        assert responses_tools_to_chat_tools(None) is None
        assert responses_tools_to_chat_tools([]) is None
        chat = responses_tools_to_chat_tools(
            [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "run",
                    "parameters": {"type": "object"},
                },
                {"type": "web_search"},
            ]
        )
        assert len(chat) == 1
        assert chat[0]["function"]["name"] == "shell"

    def test_object_shape_with_tool_call(self) -> None:
        response = build_responses_object(
            response_id="resp_1",
            model="openai/m",
            content="hi",
            tool_calls=[
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
                }
            ],
            prompt_tokens=10,
            completion_tokens=3,
            created_at=7,
        )
        assert response["object"] == "response"
        assert response["status"] == "completed"
        kinds = [item["type"] for item in response["output"]]
        assert kinds == ["message", "function_call"]
        call = response["output"][1]
        assert call["call_id"] == "call_9"
        assert call["arguments"] == '{"cmd":"pwd"}'
        assert response["usage"]["total_tokens"] == 13

    def test_sse_replays_text_and_calls_in_order(self) -> None:
        response = build_responses_object(
            response_id="resp_2",
            model="openai/m",
            content="ab",
            tool_calls=[
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
            prompt_tokens=1,
            completion_tokens=1,
            created_at=7,
        )
        events = list(iter_responses_sse_events(response, text_chunk_size=1))
        types = [json.loads(e.split("data: ", 1)[1])["type"] for e in events]
        assert types[0] == "response.created"
        assert types[-1] == "response.completed"
        deltas = [
            json.loads(e.split("data: ", 1)[1])["delta"]
            for e in events
            if json.loads(e.split("data: ", 1)[1])["type"] == "response.output_text.delta"
        ]
        assert "".join(deltas) == "ab"
        assert "response.output_item.done" in types


if __name__ == "__main__":
    unittest.main()

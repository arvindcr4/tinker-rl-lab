from __future__ import annotations

import json
import unittest

from flagship.tinker_openai_bridge_protocol import (
    bearer_token,
    estimate_tinker_usd,
    normalise_openai_messages_for_qwen,
    openai_chat_stream_events,
    parse_qwen_tool_calls,
)


class TinkerOpenAIBridgeProtocolTests(unittest.TestCase):
    def test_extracts_qwen_tool_call_and_preserves_content(self) -> None:
        content, calls = parse_qwen_tool_calls(
            'I will inspect it.\n<tool_call>{"name":"shell","arguments":{"cmd":"pwd"}}</tool_call>'
        )
        self.assertEqual(content, "I will inspect it.")
        self.assertEqual(calls[0]["function"]["name"], "shell")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"cmd": "pwd"})
        self.assertTrue(calls[0]["id"].startswith("call_"))

    def test_accepts_openai_shaped_tool_call(self) -> None:
        content, calls = parse_qwen_tool_calls(
            '<tool_call>{"id":"call_fixed","function":{"name":"read_file","arguments":"{\\"path\\":\\"a.txt\\"}"}}</tool_call>'
        )
        self.assertIsNone(content)
        self.assertEqual(calls[0]["id"], "call_fixed")
        self.assertEqual(calls[0]["function"]["name"], "read_file")

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
        self.assertEqual(content, "Plan first.")
        self.assertEqual(calls[0]["function"]["name"], "todowrite")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"todos": [{"content": "Inspect files", "status": "in_progress"}]},
        )

    def test_native_qwen_xml_plain_string_parameter(self) -> None:
        _, calls = parse_qwen_tool_calls(
            "<tool_call><function=bash><parameter=command>pwd</parameter>"
            "</function></tool_call>"
        )
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"command": "pwd"}
        )

    def test_invalid_tool_call_remains_non_executable_content(self) -> None:
        text = '<tool_call>{"name":"shell","arguments":"not-json"}</tool_call>'
        content, calls = parse_qwen_tool_calls(text)
        self.assertEqual(content, text)
        self.assertEqual(calls, [])

    def test_bearer_token_is_strict(self) -> None:
        self.assertEqual(bearer_token("Bearer secret"), "secret")
        self.assertEqual(bearer_token("bearer secret"), "secret")
        self.assertIsNone(bearer_token("Basic secret"))
        self.assertIsNone(bearer_token("Bearer"))

    def test_cost_projection_uses_pinned_prices(self) -> None:
        self.assertEqual(estimate_tinker_usd(1_000_000, 1_000_000), 1.875)
        with self.assertRaises(ValueError):
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
        self.assertEqual(payloads[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(payloads[1]["choices"][0]["delta"]["content"], "done")
        self.assertEqual(payloads[2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payloads[3]["usage"]["total_tokens"], 12)
        self.assertEqual(events[-1], "data: [DONE]\n\n")

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
        self.assertEqual(delta["tool_calls"][0]["index"], 0)
        self.assertEqual(delta["tool_calls"][0]["function"]["name"], "shell")
        finish = json.loads(events[2].removeprefix("data: "))["choices"][0]
        self.assertEqual(finish["finish_reason"], "tool_calls")

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
        self.assertEqual(
            normalised[0]["tool_calls"][0]["function"]["arguments"],
            {"command": "pwd"},
        )
        self.assertIsInstance(
            source[0]["tool_calls"][0]["function"]["arguments"], str
        )

    def test_rejects_non_object_tool_arguments(self) -> None:
        with self.assertRaises(ValueError):
            normalise_openai_messages_for_qwen(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": "[]"}}
                        ],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()

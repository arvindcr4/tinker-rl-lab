"""Structured correctness helpers, protocol v3 (whole-response, case- and JSON-type-aware)."""

import json
from dataclasses import dataclass


def json_value_equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            json_value_equal(left[k], right[k]) for k in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(json_value_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def _json_decoder():
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON value")

    return json.JSONDecoder(object_pairs_hook=pairs, parse_constant=invalid)


def parse_json_object(text):
    """Require exactly one JSON object, allowing only surrounding whitespace."""
    if not isinstance(text, str):
        return None
    try:
        value = _json_decoder().decode(text.strip())
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def extract_json_object(text):
    """Extract an embedded call only for protocols explicitly allowing prose.

    Single-call responses must use ``parse_json_object`` instead. This helper
    permits a trailing final answer in the alternative tool-and-answer task.
    """
    if not isinstance(text, str):
        return None
    decoder = _json_decoder()
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = decoder.raw_decode(text[start:])
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class ToolCorrectness:
    valid_json: bool
    schema_valid: bool
    tool_correct: bool
    arguments_correct: bool
    execution_correct: bool | None = None
    shaping_reward: float = 0.0

    @property
    def correct(self):
        return self.schema_valid and self.tool_correct and self.arguments_correct


def tool_correctness(predicted, expected_name, expected_args):
    if not isinstance(predicted, dict):
        return ToolCorrectness(False, False, False, False)
    names = [predicted[k] for k in ("tool", "name") if k in predicted]
    args = [predicted[k] for k in ("arguments", "parameters", "args") if k in predicted]
    valid = (
        bool(names)
        and all(isinstance(v, str) and v for v in names)
        and all(json_value_equal(names[0], v) for v in names)
        and bool(args)
        and all(isinstance(v, dict) for v in args)
        and all(json_value_equal(args[0], v) for v in args)
        and not (set(predicted) - {"tool", "name", "arguments", "parameters", "args"})
    )
    if not valid:
        return ToolCorrectness(True, False, False, False)
    return ToolCorrectness(
        True, True, names[0] == expected_name, json_value_equal(args[0], expected_args)
    )


def authoritative_math_answer(text):
    """The last complete box is authoritative under the boxed-output contract."""
    import re

    answer = ""
    for match in re.finditer(r"\\boxed\{", text):
        depth = 1
        for index in range(match.end(), len(text)):
            depth += (text[index] == "{") - (text[index] == "}")
            if depth == 0:
                answer = text[match.start() : index + 1]
                break
    return answer


def verify_answer(gold, prediction, verify_fn=None):
    """Math-Verify is asymmetric: the reference must be the first argument."""
    if verify_fn is None:
        from math_verify import verify as verify_fn
    return bool(verify_fn(gold, prediction))

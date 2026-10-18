"""Offline, versioned contracts for experiments. No service/model side effects.

Protocol v2 deliberately rejects legacy unverified evidence. Missing measurements
are not zeroes, simulated observations, or bootstrap samples.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

PROTOCOL_VERSION = "hybrid-research-v2"
GRADER_VERSION = "hybrid-graders-v2"
SIMULATION_METADATA = dict(
    execution_kind="simulation",
    measured=False,
    claim_eligible=False,
    artifact_type="illustrative_metrics",
    protocol_version=PROTOCOL_VERSION,
)


class EvidenceUnavailable(ValueError):
    """A required measured observation or provenance is absent."""


class IsolatedEvaluatorUnavailable(RuntimeError):
    """No trusted generated-code isolation backend is configured."""


def require_empirical(record):
    if (
        record.get("execution_kind") != "empirical"
        or record.get("measured") is not True
        or record.get("claim_eligible") is not True
        or record.get("status") != "completed"
        or not record.get("run_id")
        or not record.get("receipt_id")
    ):
        raise EvidenceUnavailable("Measured completed run with run_id and receipt_id required")
    return record


def inherit_evidence(record, parent, *, inherit_missing=True):
    """Preserve explicit provenance across container loaders, including vetoes.

    A containing simulation/ineligible/failed artifact cannot be promoted by a
    nested row. Missing metadata is never filled with an empirical default.
    Identity fields remain row-local unless explicitly supplied by the parent.
    """
    result = dict(record)
    expected = {"execution_kind": "empirical", "measured": True,
                "claim_eligible": True, "status": "completed"}
    for field in (*expected, "run_id", "receipt_id"):
        if inherit_missing and field in parent:
            result.setdefault(field, parent[field])
    for field, value in expected.items():
        if field in parent and (parent[field] is not value if isinstance(value, bool)
                                else parent[field] != value):
            result[field] = parent[field]
    return result


def measured_observations(records, endpoint):
    values, seen = [], set()
    for record in records:
        require_empirical(record)
        key = record["run_id"]
        if key in seen:
            raise EvidenceUnavailable(f"Duplicate experimental unit: {key}")
        seen.add(key)
        value = record.get(endpoint)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise EvidenceUnavailable(f"Missing or invalid {endpoint}: {key}")
        values.append(float(value))
    if not values:
        raise EvidenceUnavailable("No measured independent runs")
    return values


def simulation_result(name, metrics, *, simulation=False, output_dir=None):
    if not simulation:
        raise EvidenceUnavailable(
            f"{name}: empirical implementation unavailable; explicit simulation=True required"
        )
    result = {**SIMULATION_METADATA, "name": name, "illustrative_metrics": metrics}
    if output_dir is not None:
        # Never let a demo choose a paper/results/checkpoint destination.
        out = Path(output_dir) / "simulations"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def completion_alignment(prompt_ids, response_ids):
    if not prompt_ids or not response_ids:
        raise ValueError("Nonempty prompt and response tokens required")
    full = list(prompt_ids) + list(response_ids)
    return full[:-1], full[1:], ([0.0] * len(prompt_ids) + [1.0] * len(response_ids))[1:]


def final_box(text):
    """Last box wins, including malformed final boxes; balanced nested braces."""
    starts = list(re.finditer(r"\\boxed\s*\{", str(text)))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def numeric_value(text):
    text = str(text).strip().replace(",", "").replace("$", "")
    frac = re.fullmatch(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", text)
    if frac:
        return numeric_value(frac[1]) / numeric_value(frac[2])
    value = Fraction(text)
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite answer")
    return value


def final_numeric(text):
    text = str(text)
    if "\\boxed" in text:
        value = final_box(text)
        if value is None:
            return None
    elif "####" in text:
        tail = text.rsplit("####", 1)[1].strip()
        match = re.match(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:/\d+)?", tail)
        value = match[0] if match else None
    else:
        value = text.strip()
    try:
        return float(numeric_value(value)) if value is not None else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def math_reward(response, answer, *, require_box=False):
    pred = final_box(response) if require_box else response
    if pred is None:
        return 0.0
    target = final_box(str(answer)) if "\\boxed" in str(answer) else str(answer)
    try:
        return float(numeric_value(pred) == numeric_value(target))
    except (ValueError, ZeroDivisionError, OverflowError):
        return float(pred.strip() == target.strip())


def gsm8k_reward(response, answer):
    pred, target = final_numeric(response), final_numeric(answer)
    return float(
        pred is not None
        and target is not None
        and math.isclose(pred, target, rel_tol=1e-9, abs_tol=1e-9)
    )


def json_object(text):
    """Decode a complete JSON object, preserving nested structure."""
    text = str(text).strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
        )
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def json_objects(text):
    """Extract complete nested objects embedded in a schema/prompt (no regex)."""
    decoder, result, i = json.JSONDecoder(), [], 0
    while i < len(text):
        if text[i] == "{":
            try:
                value, size = decoder.raw_decode(text[i:])
            except ValueError:
                i += 1
                continue
            if isinstance(value, dict):
                result.append(value)
            i += size
        else:
            i += 1
    return result


def exact_json_equal(left, right):
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            exact_json_equal(left[k], right[k]) for k in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(exact_json_equal(a, b) for a, b in zip(left, right))
    return (
        type(left) is type(right)
        and left == right
        or (type(left) in (int, float) and type(right) in (int, float) and left == right)
    )


def tool_score(text, name, arguments):
    obj = json_object(text)
    if not obj or not isinstance(name, str) or not name:
        return 0.0
    name_keys, arg_keys = set(obj) & {"name", "tool"}, set(obj) & {"arguments", "parameters"}
    if len(name_keys) != 1 or len(arg_keys) != 1 or set(obj) != name_keys | arg_keys:
        return 0.0
    actual_name = obj.get("name", obj.get("tool"))
    actual_args = obj.get("arguments", obj.get("parameters"))
    return float(
        actual_name == name
        and isinstance(actual_args, dict)
        and exact_json_equal(actual_args, arguments)
    )


def humaneval_program(prompt, completion, tests, entry_point):
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError("Invalid HumanEval entry_point")
    return prompt + completion + "\n" + tests + f"\ncheck({entry_point})\n"


def humaneval_reward(completion, tests, entry_point, prompt="", **kwargs):
    # A restricted globals dict, subprocess, or timeout is NOT a security sandbox.
    # Do not execute or silently turn missing infrastructure into a model failure.
    raise IsolatedEvaluatorUnavailable(
        "HumanEval disabled: a trusted isolated evaluator with a deadline is required"
    )


def group_metrics(rewards):
    values = [float(r) for r in rewards]
    if not values or any(not math.isfinite(r) for r in values):
        raise ValueError("Finite nonempty reward group required")
    mean = sum(values) / len(values)
    variance = sum((r - mean) ** 2 for r in values) / len(values)
    return dict(
        reward_mean=mean,
        reward_variance=variance,
        reward_zvf=float(variance == 0),
        homogeneous_success=all(r == 1 for r in values),
        homogeneous_failure=all(r == 0 for r in values),
    )


def safe_arithmetic(expression):
    """Small bounded numeric AST interpreter; never eval/exec generated text."""
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("Invalid expression")
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 64:
        raise ValueError("Expression too complex")
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            value = ops[type(node.op)](walk(node.left), walk(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            value = math.sqrt(walk(node.args[0]))
        else:
            raise ValueError("Unsupported expression")
        if not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("Expression out of bounds")
        return value

    return walk(tree.body)


def execute_fixture_tool(name, arguments):
    if not isinstance(arguments, dict):
        return {"error": "arguments must be an object"}
    try:
        if name == "calculate_density" and set(arguments) == {"population", "area"}:
            if any(
                type(arguments[k]) not in (int, float) or not math.isfinite(arguments[k])
                for k in arguments
            ):
                raise ValueError("finite numbers required")
            return {"result": arguments["population"] / arguments["area"]}
        if name == "calculate" and set(arguments) == {"expression"}:
            return {"result": safe_arithmetic(arguments["expression"])}
    except (ValueError, ZeroDivisionError, SyntaxError, OverflowError):
        return {"error": "invalid arguments"}
    return {"error": "unknown tool or unsupported signature"}


@dataclass(frozen=True)
class TurnRecord:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    generated_ids: tuple[int, ...]
    behavior_policy_version: str
    sampling_settings: tuple[tuple[str, object], ...]
    tool_observation: str | None = None


def response_token_mask(tokens, eos_token_id):
    """Include first genuine EOS; all generated positions after it are padding."""
    mask, ended = [], False
    for token in tokens:
        mask.append(0.0 if ended else 1.0)
        if token == eos_token_id:
            ended = True
    return mask


def topk_overlap(predictor, terminal, k):
    if (
        not predictor
        or len(predictor) != len(terminal)
        or k < 1
        or any(not math.isfinite(x) for x in list(predictor) + list(terminal))
        or sum(abs(x) for x in predictor) == 0
        or sum(abs(x) for x in terminal) == 0
    ):
        return None
    left = set(sorted(range(len(predictor)), key=lambda i: predictor[i], reverse=True)[:k])
    right = set(sorted(range(len(terminal)), key=lambda i: terminal[i], reverse=True)[:k])
    return len(left & right) / k


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    attention_mask,
    response_mask,
    temperature=2.0,
    alpha=0.5,
):
    import torch
    import torch.nn.functional as F

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher and student vocabulary/logit shapes must match")
    if labels is None or response_mask is None or attention_mask is None:
        raise ValueError("Causal labels, attention and explicit response masks required")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= alpha <= 1:
        raise ValueError("Invalid temperature/alpha")
    logits, teacher = student_logits[:, :-1], teacher_logits[:, :-1].detach()
    targets = labels[:, 1:]
    mask = attention_mask[:, 1:].bool() & response_mask[:, 1:].bool() & (targets != -100)
    if not mask.any():
        raise ValueError("No valid response targets")
    task = F.cross_entropy(logits[mask], targets[mask])
    kl_tokens = F.kl_div(
        F.log_softmax(logits / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="none",
    ).sum(-1)
    kl = kl_tokens[mask].mean() * temperature**2
    return alpha * task + (1 - alpha) * kl, task, kl


def lagged_pairs(step_data, lag):
    if lag <= 0:
        raise ValueError("Positive lag required")
    by_step = {r["step"]: r for r in step_data}
    if len(by_step) != len(step_data):
        raise ValueError("Duplicate actual step IDs")
    return [(r, by_step[r["step"] + lag]) for r in step_data if r["step"] + lag in by_step]


def content_hash(value):
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_manifest(manifest):
    required = (
        "code_revision",
        "package_versions",
        "model_revision",
        "objective",
        "seed",
        "rng_state",
        "data_hash",
        "prompt_hash",
        "grader_version",
        "budgets",
        "checkpoint_paths",
    )
    if manifest.get("protocol_version") != PROTOCOL_VERSION or any(
        k not in manifest for k in required
    ):
        raise EvidenceUnavailable("Complete protocol-v2 execution manifest required")
    if manifest["grader_version"] != GRADER_VERSION:
        raise EvidenceUnavailable("Grader contract mismatch")
    return manifest


def validate_resume(expected, saved):
    validate_manifest(expected)
    validate_manifest(saved)
    for key in (
        "code_revision",
        "package_versions",
        "model_revision",
        "objective",
        "seed",
        "data_hash",
        "prompt_hash",
        "grader_version",
        "budgets",
    ):
        if expected[key] != saved[key]:
            raise EvidenceUnavailable(f"Resume contract mismatch: {key}")
    return saved["rng_state"]

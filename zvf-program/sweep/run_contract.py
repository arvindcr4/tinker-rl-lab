"""Sweep protocol v2: one effective specification and one admission rule.

The budget here is an upper bound, not measured matched compute. A runner must
retain effective_spec and resource_usage; existing unversioned results cannot
be silently relabelled as v2 evidence.
"""

import hashlib
import json
import math
import string

SETTINGS = (
    "model_id",
    "task",
    "loss",
    "seed",
    "group_size",
    "lr",
    "steps",
    "rank",
    "difficulty",
    "clip_low",
    "clip_high",
    "importance_level",
    "dynamic_sampling",
    "prompt_filter",
    "batch_prompts",
    "max_tokens",
)


def bind_spec(cell):
    spec = {key: cell[key] for key in SETTINGS}
    for key in ("group_size", "steps", "rank", "batch_prompts", "max_tokens"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "clip_low", "clip_high"):
        if type(spec[key]) not in (int, float) or not math.isfinite(spec[key]) or spec[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    spec["protocol_version"] = "sweep-effective-v2"
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return spec, digest


def render_command(template, substitutions):
    used = {key for _, key, _, _ in string.Formatter().parse(template) if key}
    missing = set(SETTINGS) - used
    if missing:
        raise ValueError(f"runner template omits scientific settings: {sorted(missing)}")
    return " ".join(template.split()).format(**substitutions)


def result_complete(data, planned_steps, expected_spec=None):
    if (
        not isinstance(expected_spec, dict)
        or not isinstance(data, dict)
        or data.get("status") != "completed"
    ):
        return False
    rows = data.get("step_log")
    if (
        type(planned_steps) is not int
        or planned_steps <= 0
        or not isinstance(rows, list)
        or len(rows) != planned_steps
    ):
        return False
    ids = [row.get("step") if isinstance(row, dict) else None for row in rows]
    if any(type(i) is not int for i in ids) or ids not in (
        list(range(planned_steps)),
        list(range(1, planned_steps + 1)),
    ):
        return False
    for row in rows:
        for key in ("reward", "zvf", "gu"):
            v = row.get(key)
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
                return False
        if not math.isclose(row["gu"], 1 - row["zvf"], abs_tol=1e-9):
            return False
    if expected_spec is not None:
        if data.get("effective_spec") != expected_spec:
            return False
        usage = data.get("resource_usage", {})
        for key in (
            "attempted_groups",
            "accepted_groups",
            "generated_tokens",
            "training_tokens",
            "optimizer_steps",
        ):
            if type(usage.get(key)) is not int or usage[key] < 0:
                return False
        if (
            usage["optimizer_steps"] != planned_steps
            or usage["accepted_groups"] > usage["attempted_groups"]
            or usage["attempted_groups"] <= 0
            or usage["generated_tokens"] <= 0
            or usage["training_tokens"] > usage["generated_tokens"]
        ):
            return False
        if (
            type(usage.get("elapsed_gpu_seconds")) not in (int, float)
            or not math.isfinite(usage["elapsed_gpu_seconds"])
            or usage["elapsed_gpu_seconds"] < 0
        ):
            return False
        if (
            usage["generated_tokens"]
            > expected_spec["steps"]
            * expected_spec["group_size"]
            * expected_spec["batch_prompts"]
            * expected_spec["max_tokens"]
        ):
            return False
    return True


def claim_attempt(result_path, identity):
    import sys
    from pathlib import Path

    support = str(Path(__file__).resolve().parents[1] / "e1_wave10")
    if support not in sys.path:
        sys.path.insert(0, support)
    from attempt_store import claim, transition

    result_path = Path(result_path)
    if result_path.exists():
        raise FileExistsError("existing evidence must be reconciled, never overwritten")
    directory = result_path.parent / ".attempts" / result_path.stem
    nonce = claim(directory, identity)
    transition(directory, nonce, "DISPATCHED")
    return nonce, directory


def result_file_complete(path, planned_steps, expected_spec):
    """Resume/aggregate need durable attempt completion and current evidence bytes."""
    from pathlib import Path

    path = Path(path)
    try:
        data = json.loads(path.read_text())
        directory = path.parent / ".attempts" / path.stem
        prepared = json.loads((directory / "attempt.json").read_text())
        completed = json.loads((directory / "completed.json").read_text())
        return (
            result_complete(data, planned_steps, expected_spec)
            and prepared["identity"]["effective_spec"] == expected_spec
            and bool(data.get("attempt_nonce"))
            and data["attempt_nonce"] == prepared["nonce"] == completed["nonce"]
            and completed["state"] == "COMPLETED"
            and completed["result_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False

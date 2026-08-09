#!/usr/bin/env python3
"""Deterministic, fail-closed config and validator for the first paid xLAM smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from decimal import Decimal
from typing import Any, Mapping

SCHEMA_VERSION = "pavlov-xlam-smoke-config-v1"
SMOKE_ID = "pavlov-xlam-first-paid-smoke-v1"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
XLAM_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"

SEED = 809
STEPS = 10
GROUP = 4
BATCH = 2
RANK = 32
LEARNING_RATE = 2e-05
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_PROMPT_TOKENS = 1200
MAX_RESPONSE_TOKENS = 128
SAVE_EVERY_STEPS = 5

EXPECTED_CHECKPOINT_STEPS = (0, 5, 10)
EXPECTED_CHECKPOINT_VISIBILITY = ("public", "private")
REQUIRED_CHECKPOINT_STAGES = ("initial", "periodic", "final")

HARD_CAP_USD = Decimal("18.00")
OPERATIONAL_CAP_USD = Decimal("16.50")
SAFETY_RESERVE_USD = Decimal("1.50")
RESERVATION_USD = Decimal("0.50")

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_FLOAT_TOLERANCE = Decimal("0.000000001")

ROOT_KEYS = (
    "schema_version",
    "smoke_id",
    "component",
    "model",
    "model_revision",
    "xlam_revision",
    "seed",
    "steps",
    "group",
    "batch",
    "rank",
    "learning_rate",
    "temperature",
    "top_p",
    "max_prompt_tokens",
    "max_response_tokens",
    "save_every_steps",
    "run_order",
    "wandb",
    "sampler_checkpoints",
    "runtime_constraints",
    "budget",
    "component_only",
    "primary_eval",
    "heldout",
    "portfolio_claim",
    "config_signature",
)

WANDB_KEYS = ("mode", "required_before_tinker")
CHECKPOINT_KEYS = (
    "required_stages",
    "periodic_every_steps",
    "required_steps",
    "allowed_visibility",
    "safe_public_artifact",
)
RUNTIME_KEYS = ("allow_network", "allow_credentials", "allow_paid_run")
BUDGET_KEYS = ("maximum_usd", "operational_cap_usd", "safety_reserve_usd", "reservation_usd")


class SmokeConfigError(ValueError):
    """Raised for malformed smoke configs that fail strict validation."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _to_decimal(name: str, value: Any) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise SmokeConfigError(f"{name} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (ValueError, TypeError) as exc:
        raise SmokeConfigError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise SmokeConfigError(f"{name} must be a finite decimal")
    return parsed


def generate_smoke_config() -> dict[str, Any]:
    """Return a deterministic smoke config for the first paid xLAM component run."""

    config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "smoke_id": SMOKE_ID,
        "component": "xlam",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "xlam_revision": XLAM_REVISION,
        "seed": SEED,
        "steps": STEPS,
        "group": GROUP,
        "batch": BATCH,
        "rank": RANK,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "save_every_steps": SAVE_EVERY_STEPS,
        "run_order": ["wandb_init", "tinker_client"],
        "wandb": {"mode": "online", "required_before_tinker": True},
        "sampler_checkpoints": {
            "required_stages": list(REQUIRED_CHECKPOINT_STAGES),
            "periodic_every_steps": [SAVE_EVERY_STEPS],
            "required_steps": list(EXPECTED_CHECKPOINT_STEPS),
            "allowed_visibility": list(EXPECTED_CHECKPOINT_VISIBILITY),
            "safe_public_artifact": True,
        },
        "runtime_constraints": {
            "allow_network": False,
            "allow_credentials": False,
            "allow_paid_run": False,
        },
        "budget": {
            "maximum_usd": float(HARD_CAP_USD),
            "operational_cap_usd": float(OPERATIONAL_CAP_USD),
            "safety_reserve_usd": float(SAFETY_RESERVE_USD),
            "reservation_usd": float(RESERVATION_USD),
        },
        "component_only": True,
        "primary_eval": False,
        "heldout": False,
        "portfolio_claim": False,
    }
    config["config_signature"] = _sha256(
        {
            key: config[key]
            for key in config
            if key != "config_signature"
        }
    )
    return config


def _validate_revision(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        raise SmokeConfigError(f"{name} must be a 40-character lowercase hex revision")


def _validate_float(name: str, value: Any, expected: float) -> None:
    actual = _to_decimal(name, value)
    target = _to_decimal(name, expected)
    if abs(actual - target) > _FLOAT_TOLERANCE:
        raise SmokeConfigError(f"{name} must be exactly {expected}")


def _validate_int(name: str, value: Any, expected: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise SmokeConfigError(f"{name} must be exactly {expected}")


def validate_smoke_config(config: Mapping[str, Any]) -> list[str]:
    """Return ordered fail-closed validation errors for the config payload."""

    errors: list[str] = []
    if not isinstance(config, Mapping):
        return ["smoke config must be an object"]

    # Ensure no unknown top-level fields are accepted.
    extras = sorted(set(config) - set(ROOT_KEYS))
    if extras:
        errors.append(f"unexpected fields: {', '.join(extras)}")

    for key in ROOT_KEYS:
        if key not in config:
            errors.append(f"missing required field: {key}")

    if errors:
        return errors

    if config["schema_version"] != SCHEMA_VERSION:
        errors.append("schema_version is incorrect")
    if config["smoke_id"] != SMOKE_ID:
        errors.append("smoke_id is incorrect")
    if config["component"] != "xlam":
        errors.append("component must remain xlam")
    if config["model"] != MODEL_ID:
        errors.append("model must be Qwen/Qwen3.6-35B-A3B")
    try:
        _validate_revision("model_revision", config["model_revision"])
    except SmokeConfigError as exc:
        errors.append(str(exc))
    if config["model_revision"] != MODEL_REVISION:
        errors.append("model_revision must match the pinned primary model revision")
    try:
        _validate_revision("xlam_revision", config["xlam_revision"])
    except SmokeConfigError as exc:
        errors.append(str(exc))
    if config["xlam_revision"] != XLAM_REVISION:
        errors.append("xlam_revision must match the pinned xLAM revision")

    for name, value, expected in (
        ("seed", config["seed"], SEED),
        ("steps", config["steps"], STEPS),
        ("group", config["group"], GROUP),
        ("batch", config["batch"], BATCH),
        ("rank", config["rank"], RANK),
        ("max_prompt_tokens", config["max_prompt_tokens"], MAX_PROMPT_TOKENS),
        ("max_response_tokens", config["max_response_tokens"], MAX_RESPONSE_TOKENS),
        ("save_every_steps", config["save_every_steps"], SAVE_EVERY_STEPS),
    ):
        try:
            _validate_int(name, value, expected)
        except SmokeConfigError as exc:
            errors.append(str(exc))

    for name, value, expected in (
        ("learning_rate", config["learning_rate"], LEARNING_RATE),
        ("temperature", config["temperature"], TEMPERATURE),
        ("top_p", config["top_p"], TOP_P),
    ):
        try:
            _validate_float(name, value, expected)
        except SmokeConfigError as exc:
            errors.append(str(exc))

    if config["save_every_steps"] not in config["sampler_checkpoints"].get("periodic_every_steps", []):
        errors.append("sampler_checkpoints.periodic_every_steps must include save_every_steps")

    # Stable run gate: W&B must be initialized online before the Tinker client.
    wandb = config["wandb"]
    if not isinstance(wandb, Mapping) or set(wandb) != set(WANDB_KEYS):
        errors.append("wandb must contain only mode and required_before_tinker")
    else:
        if wandb["mode"] != "online":
            errors.append("wandb mode must be online")
        if wandb["required_before_tinker"] is not True:
            errors.append("wandb.required_before_tinker must be true")
    if config["run_order"] != ["wandb_init", "tinker_client"]:
        errors.append("run_order must be [\"wandb_init\", \"tinker_client\"]")

    # xLAM sampler checkpoints must be explicit and complete.
    checkpoints = config["sampler_checkpoints"]
    if not isinstance(checkpoints, Mapping):
        errors.append("sampler_checkpoints must be an object")
    else:
        extras_checkpoints = sorted(set(checkpoints) - set(CHECKPOINT_KEYS))
        if extras_checkpoints:
            errors.append(
                f"sampler_checkpoints has unexpected fields: {', '.join(extras_checkpoints)}"
            )
        for key in CHECKPOINT_KEYS:
            if key not in checkpoints:
                errors.append(f"sampler_checkpoints missing required field: {key}")
        if isinstance(checkpoints.get("required_stages"), list):
            if tuple(checkpoints["required_stages"]) != REQUIRED_CHECKPOINT_STAGES:
                errors.append("required_stages must be [initial, periodic, final]")
        else:
            errors.append("required_stages must be a list")
        if tuple(checkpoints.get("required_steps", ())) != EXPECTED_CHECKPOINT_STEPS:
            errors.append(
                "required_steps must be [0, save_every_steps, steps]"
            )
        if not isinstance(checkpoints.get("periodic_every_steps"), list) or checkpoints["periodic_every_steps"] != [SAVE_EVERY_STEPS]:
            errors.append("periodic_every_steps must be [5]")
        visibility = checkpoints.get("allowed_visibility")
        if tuple(visibility or ()) != EXPECTED_CHECKPOINT_VISIBILITY:
            errors.append("allowed_visibility must be exactly ['public', 'private']")
        if checkpoints.get("safe_public_artifact") is not True:
            errors.append("safe_public_artifact must be true")

    runtime = config["runtime_constraints"]
    if not isinstance(runtime, Mapping):
        errors.append("runtime_constraints must be an object")
    else:
        if tuple(sorted(runtime)) != ("allow_credentials", "allow_network", "allow_paid_run"):
            errors.append("runtime_constraints must contain only allowed controls")
        if runtime.get("allow_network") is not False:
            errors.append("runtime_constraints.allow_network must be false")
        if runtime.get("allow_credentials") is not False:
            errors.append("runtime_constraints.allow_credentials must be false")
        if runtime.get("allow_paid_run") is not False:
            errors.append("runtime_constraints.allow_paid_run must be false")

    budget = config["budget"]
    if not isinstance(budget, Mapping):
        errors.append("budget must be an object")
    else:
        budget_extras = sorted(set(budget) - set(BUDGET_KEYS))
        if budget_extras:
            errors.append(f"budget has unexpected fields: {', '.join(budget_extras)}")
        for key in BUDGET_KEYS:
            if key not in budget:
                errors.append(f"budget missing required field: {key}")
        for key, expected, message in (
            ("maximum_usd", HARD_CAP_USD, "budget.maximum_usd must preserve $18.00 hard cap"),
            ("operational_cap_usd", OPERATIONAL_CAP_USD, "budget.operational_cap_usd must preserve $16.50 cap"),
            ("safety_reserve_usd", SAFETY_RESERVE_USD, "budget.safety_reserve_usd must preserve $1.50 reserve"),
            ("reservation_usd", RESERVATION_USD, "budget.reservation_usd must be exactly $0.50"),
        ):
            try:
                if _to_decimal(key, budget.get(key)) != expected:
                    errors.append(message)
            except SmokeConfigError as exc:
                errors.append(str(exc))

    expected_signature = _sha256({key: config[key] for key in ROOT_KEYS if key != "config_signature"})
    if config.get("config_signature") != expected_signature:
        errors.append("config_signature is invalid or missing")

    if config["component_only"] is not True:
        errors.append("component_only must be true")
    if config["primary_eval"] is not False:
        errors.append("primary_eval must be false")
    if config["heldout"] is not False:
        errors.append("heldout must be false")
    if config["portfolio_claim"] is not False:
        errors.append("portfolio_claim must be false")

    return errors


def assert_smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and raise when fail-closed checks are violated."""

    errors = validate_smoke_config(config)
    if errors:
        raise SmokeConfigError("smoke config is invalid: " + "; ".join(errors))
    return dict(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate config JSON from stdin")
    args = parser.parse_args(argv)

    if not args.validate:
        print(_canonical_json(generate_smoke_config()))
        return 0

    try:
        config = json.load(sys.stdin)
    except Exception as exc:  # pragma: no cover - exercised only via CLI use
        raise SystemExit(f"invalid stdin JSON: {exc}")

    errors = validate_smoke_config(config)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

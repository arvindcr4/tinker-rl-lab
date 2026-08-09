"""Zero-cost, fail-closed configuration boundary for the T3 BrowserGym smoke.

This module validates a declarative pilot configuration only.  It does not
install BrowserGym, import a browser/runtime, read credentials, contact W&B or
HF, call Tinker, or authorize a paid launch.  The default configuration is a
schema-valid but receipt-gated smoke plan: its conservative cost envelope is
at most $0.60, while the paid gate remains false until an observed receipt is
attached and independently verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

try:  # Package import used by the flagship test suite.
    from . import pavlov_browsergym_adapter as adapter
    from . import pavlov_browsergym_receipt as receipt
except ImportError:  # Direct execution from this directory remains offline-safe.
    import pavlov_browsergym_adapter as adapter
    import pavlov_browsergym_receipt as receipt


SCHEMA_VERSION = "pavlov-browsergym-t3-launch-config-v1"
PILOT_ID = "t3_browsergym_miniwob_smoke"
SUITE_ID = adapter.SUITE_ID
SUITE_ROLE = adapter.SUITE_ROLE
BENCHMARK_NAME = adapter.BENCHMARK_NAME
DATASET_ID = adapter.DATASET_ID
DATASET_REVISION = adapter.PINNED_DATASET_REVISION
ENVIRONMENT_REVISION = adapter.PINNED_ENVIRONMENT_REVISION
E6_SUITE_ID = adapter.E6_SUITE_ID
AUTHORIZED_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

SMOKE_UPDATES = 10
SMOKE_BATCH_SIZE = 2
SMOKE_GROUP_SIZE = 2
SMOKE_HORIZON = 8
SMOKE_MAX_PROMPT_TOKENS = 1024
SMOKE_MAX_RESPONSE_TOKENS = 128
SMOKE_LR = 1e-5
SMOKE_SEED = 42
SMOKE_CAP_USD = Decimal("0.60")
PREFILL_USD_PER_MILLION = Decimal("0.54")
SAMPLE_USD_PER_MILLION = Decimal("1.335")
TRAIN_USD_PER_MILLION = Decimal("1.177")

PILOT_ENV_IDS = (
    "browsergym/miniwob.click-button",
    "browsergym/miniwob.choose-list",
    "browsergym/miniwob.enter-text",
)

# These are the exact safe task identities used by the offline manifest.  The
# adapter derives each task_id/task_id_hash from all identity-bearing fields.
_TASK_BLUEPRINTS = (
    {
        "env_id": PILOT_ENV_IDS[0],
        "seed": 42,
        "goal": "Click the button labeled Continue.",
        "initial_observation": {
            "url": "about:blank",
            "open_pages": 1,
            "button_label": "Continue",
            "clicked": False,
        },
    },
    {
        "env_id": PILOT_ENV_IDS[1],
        "seed": 43,
        "goal": "Choose the option labeled Red.",
        "initial_observation": {
            "url": "about:blank",
            "open_pages": 1,
            "options": ["Red", "Blue"],
            "selected": None,
        },
    },
    {
        "env_id": PILOT_ENV_IDS[2],
        "seed": 44,
        "goal": "Enter the text BrowserGym.",
        "initial_observation": {
            "url": "about:blank",
            "open_pages": 1,
            "field_label": "Text",
            "text_length": 0,
        },
    },
)

WANDB_CONFIG_FIELDS = (
    "campaign",
    "suite_id",
    "suite_role",
    "model_id",
    "model_revision",
    "adapter_revision",
    "dataset_revision",
    "split_manifest_hash",
    "container_digest",
    "seed",
    "horizon",
    "group_size",
    "batch_size",
    "lr",
    "max_prompt_tokens",
    "max_response_tokens",
    "reward_type",
    "verifier_type",
    "stateful",
    "artifact_or_side_effect",
    "git_sha",
    "budget_cap_usd",
)
WANDB_METRIC_KEYS = (
    "train/loss",
    "train/reward",
    "train/step",
    "train/browser_success_rate",
    "train/browser_reward_mean",
    "train/browser_action_count_mean",
    "eval/browser_success_rate",
    "eval/browser_reward_mean",
    "eval/browser_action_count_mean",
)
WANDB_SAMPLE_FIELDS = (
    "env_id",
    "actions",
    "terminal_state",
    "error",
    "state_hash",
    "action_hash",
    "artifact_digest",
)
TINKER_REQUIRED_FLAGS = (
    "run_id_required",
    "initial_sampler_path_required",
    "periodic_sampler_path_required",
    "final_sampler_path_required",
    "local_checkpoint_json_required",
)
HF_REQUIRED_FLAGS = (
    "initial_sampler_export_required",
    "periodic_sampler_export_required",
    "final_sampler_export_required",
    "c0_receipt_required",
    "wandb_manifest_link_required",
    "model_revision_required",
    "adapter_revision_required",
)
REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "pilot_id",
    "suite_id",
    "suite_role",
    "benchmark",
    "dataset_id",
    "dataset_revision",
    "environment_revision",
    "task_manifest",
    "split_manifest_hash",
    "stateful",
    "artifact_or_side_effect",
    "telemetry",
    "tracking_config",
    "wandb",
    "tinker",
    "hf",
    "cost",
    "receipt_gate",
    "e6_suite_id",
    "e6_substitute",
    "portfolio_evidence",
    "paid_launch_allowed",
)


class LaunchConfigError(ValueError):
    """Raised by strict config helpers for malformed declarative metadata."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LaunchConfigError(f"value is not canonical JSON: {exc}") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not _is_mapping(value):
        raise LaunchConfigError(f"{label} must be an object")
    copied = dict(value)
    try:
        adapter.assert_secret_free(copied)
    except adapter.SecretMaterialError as exc:
        raise LaunchConfigError(str(exc)) from exc
    return copied


def _bool(value: Any, label: str, errors: list[str]) -> bool:
    if not isinstance(value, bool):
        errors.append(f"{label} must be boolean")
        return False
    return value


def _nonempty(value: Any, label: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string")
        return ""
    return value


def _decimal(value: Any, label: str, errors: list[str]) -> Decimal | None:
    if isinstance(value, bool):
        errors.append(f"{label} must be a finite non-negative number")
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        errors.append(f"{label} must be a finite non-negative number")
        return None
    if not number.is_finite() or number < 0:
        errors.append(f"{label} must be a finite non-negative number")
        return None
    return number


def _expected_task_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for blueprint in _TASK_BLUEPRINTS:
        task = adapter.TaskSpec(
            env_id=blueprint["env_id"],
            seed=blueprint["seed"],
            goal=blueprint["goal"],
            initial_observation=blueprint["initial_observation"],
        )
        manifest.append(task.to_dict())
    return manifest


def task_manifest_hash(task_manifest: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered, exact train-task manifest."""

    return sha256_json([dict(task) for task in task_manifest])


def estimate_smoke_cost(
    *,
    updates: int = SMOKE_UPDATES,
    batch_size: int = SMOKE_BATCH_SIZE,
    group_size: int = SMOKE_GROUP_SIZE,
    horizon: int = SMOKE_HORIZON,
    max_prompt_tokens: int = SMOKE_MAX_PROMPT_TOKENS,
    max_response_tokens: int = SMOKE_MAX_RESPONSE_TOKENS,
) -> dict[str, Any]:
    """Return the conservative full-token smoke envelope without charging."""

    values = {
        "updates": updates,
        "batch_size": batch_size,
        "group_size": group_size,
        "horizon": horizon,
        "max_prompt_tokens": max_prompt_tokens,
        "max_response_tokens": max_response_tokens,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in values.values()
    ):
        raise LaunchConfigError("cost dimensions must be positive integers")
    sequence_count = updates * batch_size * group_size * horizon
    per_sequence = (
        Decimal(max_prompt_tokens) * PREFILL_USD_PER_MILLION
        + Decimal(max_response_tokens) * SAMPLE_USD_PER_MILLION
        + Decimal(max_response_tokens) * TRAIN_USD_PER_MILLION
    ) / Decimal(1_000_000)
    nominal = Decimal(sequence_count) * per_sequence
    envelope = nominal * Decimal(2)
    return {
        **values,
        "sequence_count": sequence_count,
        "prefill_usd_per_million": float(PREFILL_USD_PER_MILLION),
        "sample_usd_per_million": float(SAMPLE_USD_PER_MILLION),
        "train_usd_per_million": float(TRAIN_USD_PER_MILLION),
        "nominal_usd": float(nominal),
        "conservative_envelope_usd": float(envelope),
        "cap_usd": float(SMOKE_CAP_USD),
        "currency": "USD",
        "charged_usd": 0.0,
        "paid_launch": False,
    }


def _offline_revision(label: str) -> str:
    return f"offline://{label}-pending"


def build_offline_smoke_config() -> dict[str, Any]:
    """Build the deterministic config fixture; it never authorizes spending."""

    manifest = _expected_task_manifest()
    manifest_hash = task_manifest_hash(manifest)
    model_revision = _offline_revision("model-revision")
    adapter_revision = _offline_revision("adapter-revision")
    container_digest = _offline_revision("container-digest")
    git_sha = _offline_revision("git-sha")
    tracking_config = {
        "campaign": PILOT_ID,
        "suite_id": SUITE_ID,
        "suite_role": SUITE_ROLE,
        "model_id": AUTHORIZED_MODEL_ID,
        "model_revision": model_revision,
        "adapter_revision": adapter_revision,
        "dataset_revision": DATASET_REVISION,
        "split_manifest_hash": manifest_hash,
        "container_digest": container_digest,
        "seed": SMOKE_SEED,
        "horizon": SMOKE_HORIZON,
        "group_size": SMOKE_GROUP_SIZE,
        "batch_size": SMOKE_BATCH_SIZE,
        "lr": SMOKE_LR,
        "max_prompt_tokens": SMOKE_MAX_PROMPT_TOKENS,
        "max_response_tokens": SMOKE_MAX_RESPONSE_TOKENS,
        "reward_type": "browsergym_native_success",
        "verifier_type": receipt.NATIVE_VERIFIER_NAME,
        "stateful": True,
        "artifact_or_side_effect": True,
        "git_sha": git_sha,
        "budget_cap_usd": float(SMOKE_CAP_USD),
    }
    config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pilot_id": PILOT_ID,
        "suite_id": SUITE_ID,
        "suite_role": SUITE_ROLE,
        "benchmark": BENCHMARK_NAME,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "environment_revision": ENVIRONMENT_REVISION,
        "model_id": AUTHORIZED_MODEL_ID,
        "model_revision": model_revision,
        "adapter_revision": adapter_revision,
        "container_digest": container_digest,
        "git_sha": git_sha,
        "task_manifest": manifest,
        "split_manifest_hash": manifest_hash,
        "stateful": True,
        "artifact_or_side_effect": True,
        "telemetry": {
            "per_step_observation_required": True,
            "per_step_action_required": True,
            "state_hash_required": True,
            "action_hash_required": True,
            "artifact_digest_required": True,
            "terminal_task_success_required": True,
            "native_verifier": receipt.NATIVE_VERIFIER_NAME,
            "artifact_names": list(adapter.REQUIRED_ARTIFACT_NAMES),
        },
        "tracking_config": tracking_config,
        "wandb": {
            "online_required": True,
            "initialize_before_tinker": True,
            "project": "tinker-rl-lab-pavlov",
            "required_config_fields": list(WANDB_CONFIG_FIELDS),
            "required_metric_keys": list(WANDB_METRIC_KEYS),
            "required_sample_fields": list(WANDB_SAMPLE_FIELDS),
        },
        "tinker": {
            "model_id": AUTHORIZED_MODEL_ID,
            "run_id_field": "tinker_run_id",
            "initial_sampler_path_field": "initial_sampler_path",
            "periodic_sampler_path_field": "periodic_sampler_path",
            "final_sampler_path_field": "final_sampler_path",
            "local_checkpoint_json_field": "checkpoint_receipt_path",
            **{name: True for name in TINKER_REQUIRED_FLAGS},
        },
        "hf": {
            "repository_ref_field": "checkpoint_repo_id",
            "revision_field": "checkpoint_revision",
            "checkpoint_manifest_field": "checkpoint_manifest",
            **{name: True for name in HF_REQUIRED_FLAGS},
        },
        "cost": estimate_smoke_cost(),
        "receipt_gate": {
            "receipt_schema_version": receipt.SCHEMA_VERSION,
            "observed_result_required": True,
            "native_success_required": True,
            "wandb_evidence_required": True,
            "tinker_evidence_required": True,
            "hf_evidence_required": True,
            "cost_receipt_required": True,
            "receipt_attached": False,
            "receipts_verified": False,
            "paid_launch_allowed": False,
            "portfolio_evidence": False,
        },
        "e6_suite_id": E6_SUITE_ID,
        "e6_substitute": False,
        "portfolio_evidence": False,
        "paid_launch_allowed": False,
    }
    return config


@dataclass(frozen=True)
class LaunchConfigValidationResult:
    """Config outcome; schema validity never implies launch authorization."""

    ok: bool
    errors: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    paid_launch_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "metrics": dict(self.metrics),
            "paid_launch_authorized": False,
            "receipt_gate_closed": True,
        }


def _validate_task_manifest(value: Any, errors: list[str]) -> bool:
    if not isinstance(value, list):
        errors.append("task_manifest must be a list")
        return False
    expected = _expected_task_manifest()
    if value != expected:
        errors.append("task_manifest does not match the exact pinned MiniWoB train tasks")
    for index, item in enumerate(value):
        try:
            task = adapter.TaskSpec.from_dict(item)
            if task.task_id != item.get("task_id") or task.task_id_hash != item.get("task_id_hash"):
                errors.append(f"task_manifest[{index}] task hash is not deterministic")
        except (KeyError, TypeError, ValueError, adapter.AdapterSchemaError) as exc:
            errors.append(f"task_manifest[{index}] invalid: {exc}")
    return value == expected and not any("task_manifest" in error for error in errors)


def _validate_cost(value: Any, errors: list[str]) -> bool:
    if not _is_mapping(value):
        errors.append("cost must be an object")
        return False
    try:
        cost = _mapping(value, "cost")
    except LaunchConfigError as exc:
        errors.append(str(exc))
        return False
    expected = estimate_smoke_cost(
        updates=SMOKE_UPDATES,
        batch_size=SMOKE_BATCH_SIZE,
        group_size=SMOKE_GROUP_SIZE,
        horizon=SMOKE_HORIZON,
        max_prompt_tokens=SMOKE_MAX_PROMPT_TOKENS,
        max_response_tokens=SMOKE_MAX_RESPONSE_TOKENS,
    )
    for key in (
        "updates",
        "batch_size",
        "group_size",
        "horizon",
        "max_prompt_tokens",
        "max_response_tokens",
        "sequence_count",
    ):
        if cost.get(key) != expected[key]:
            errors.append(f"cost.{key} does not match the smoke envelope")
    for key in ("nominal_usd", "conservative_envelope_usd", "cap_usd", "charged_usd"):
        actual = _decimal(cost.get(key), f"cost.{key}", errors)
        wanted = _decimal(expected[key], f"expected.{key}", errors)
        if actual is not None and wanted is not None and actual != wanted:
            errors.append(f"cost.{key} does not match the conservative estimate")
    if cost.get("currency") != "USD":
        errors.append("cost.currency must be USD")
    if cost.get("paid_launch") is not False:
        errors.append("cost.paid_launch must remain false")
    envelope = _decimal(cost.get("conservative_envelope_usd"), "cost.conservative_envelope_usd", errors)
    cap = _decimal(cost.get("cap_usd"), "cost.cap_usd", errors)
    if envelope is not None and cap is not None and envelope > cap:
        errors.append("conservative smoke envelope exceeds cap")
    if cap is not None and cap > SMOKE_CAP_USD:
        errors.append("smoke cap exceeds $0.60")
    return not any(error.startswith("cost") or "smoke envelope" in error for error in errors)


def validate_t3_pilot_config(config: Mapping[str, Any]) -> LaunchConfigValidationResult:
    """Validate the declarative T3 smoke config and keep paid launch closed."""

    errors: list[str] = []
    if not _is_mapping(config):
        return LaunchConfigValidationResult(False, ("config must be an object",), {})
    try:
        adapter.assert_secret_free(config)
    except adapter.SecretMaterialError as exc:
        errors.append(str(exc))
    data = dict(config)
    for key in REQUIRED_TOP_LEVEL_FIELDS:
        if key not in data:
            errors.append(f"missing required config field: {key}")

    exact_values = {
        "schema_version": SCHEMA_VERSION,
        "pilot_id": PILOT_ID,
        "suite_id": SUITE_ID,
        "suite_role": SUITE_ROLE,
        "benchmark": BENCHMARK_NAME,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "environment_revision": ENVIRONMENT_REVISION,
        "e6_suite_id": E6_SUITE_ID,
    }
    for key, expected in exact_values.items():
        if data.get(key) != expected:
            errors.append(f"{key} must be {expected!r}")
    if _bool(data.get("stateful"), "stateful", errors) is not True:
        errors.append("T3 pilot must set stateful=true")
    if _bool(data.get("artifact_or_side_effect"), "artifact_or_side_effect", errors) is not True:
        errors.append("T3 pilot must set artifact_or_side_effect=true")
    if _bool(data.get("e6_substitute"), "e6_substitute", errors) is not False:
        errors.append("BrowserGym cannot substitute for E6/WebBench")
    if _bool(data.get("portfolio_evidence"), "portfolio_evidence", errors) is not False:
        errors.append("pilot config cannot claim portfolio evidence")
    if _bool(data.get("paid_launch_allowed"), "paid_launch_allowed", errors) is not False:
        errors.append("paid launch must remain false in this config validator")

    task_valid = _validate_task_manifest(data.get("task_manifest"), errors)
    split_hash = data.get("split_manifest_hash")
    if not isinstance(split_hash, str) or len(split_hash) != 64:
        errors.append("split_manifest_hash must be a SHA-256 digest")
    elif isinstance(data.get("task_manifest"), list) and split_hash != task_manifest_hash(data["task_manifest"]):
        errors.append("split_manifest_hash does not match task_manifest")

    telemetry = data.get("telemetry")
    telemetry_valid = True
    if not _is_mapping(telemetry):
        errors.append("telemetry must be an object")
        telemetry_valid = False
    else:
        telemetry_data = dict(telemetry)
        for field_name in (
            "per_step_observation_required",
            "per_step_action_required",
            "state_hash_required",
            "action_hash_required",
            "artifact_digest_required",
            "terminal_task_success_required",
        ):
            if _bool(telemetry_data.get(field_name), f"telemetry.{field_name}", errors) is not True:
                errors.append(f"telemetry.{field_name} must be true")
        if telemetry_data.get("native_verifier") != receipt.NATIVE_VERIFIER_NAME:
            errors.append("telemetry.native_verifier must use the native BrowserGym verifier")
        if tuple(telemetry_data.get("artifact_names", ())) != adapter.REQUIRED_ARTIFACT_NAMES:
            errors.append("telemetry.artifact_names must match the adapter artifact contract")

    tracking = data.get("tracking_config")
    tracking_valid = True
    if not _is_mapping(tracking):
        errors.append("tracking_config must be an object")
        tracking_valid = False
    else:
        tracking_data = dict(tracking)
        for field_name in WANDB_CONFIG_FIELDS:
            if field_name not in tracking_data:
                errors.append(f"tracking_config missing {field_name}")
        expected_tracking = {
            "campaign": PILOT_ID,
            "suite_id": SUITE_ID,
            "suite_role": SUITE_ROLE,
            "model_id": AUTHORIZED_MODEL_ID,
            "dataset_revision": DATASET_REVISION,
            "split_manifest_hash": split_hash,
            "seed": SMOKE_SEED,
            "horizon": SMOKE_HORIZON,
            "group_size": SMOKE_GROUP_SIZE,
            "batch_size": SMOKE_BATCH_SIZE,
            "lr": SMOKE_LR,
            "max_prompt_tokens": SMOKE_MAX_PROMPT_TOKENS,
            "max_response_tokens": SMOKE_MAX_RESPONSE_TOKENS,
            "reward_type": "browsergym_native_success",
            "verifier_type": receipt.NATIVE_VERIFIER_NAME,
            "stateful": True,
            "artifact_or_side_effect": True,
            "budget_cap_usd": float(SMOKE_CAP_USD),
        }
        for field_name, expected in expected_tracking.items():
            if tracking_data.get(field_name) != expected:
                errors.append(f"tracking_config.{field_name} is not pinned")
        if tracking_data.get("split_manifest_hash") != split_hash:
            errors.append("tracking_config.split_manifest_hash does not match top-level hash")

    wandb = data.get("wandb")
    wandb_valid = True
    if not _is_mapping(wandb):
        errors.append("wandb must be an object")
        wandb_valid = False
    else:
        wandb_data = dict(wandb)
        for field_name in ("online_required", "initialize_before_tinker"):
            if _bool(wandb_data.get(field_name), f"wandb.{field_name}", errors) is not True:
                errors.append(f"wandb.{field_name} must be true")
        if wandb_data.get("project") != "tinker-rl-lab-pavlov":
            errors.append("wandb.project must be the Pavlov project")
        for key, expected in (
            ("required_config_fields", WANDB_CONFIG_FIELDS),
            ("required_metric_keys", WANDB_METRIC_KEYS),
            ("required_sample_fields", WANDB_SAMPLE_FIELDS),
        ):
            values = wandb_data.get(key)
            if not isinstance(values, list) or not set(expected).issubset(values):
                errors.append(f"wandb.{key} is missing required telemetry fields")

    tinker = data.get("tinker")
    tinker_valid = True
    if not _is_mapping(tinker):
        errors.append("tinker must be an object")
        tinker_valid = False
    else:
        tinker_data = dict(tinker)
        if tinker_data.get("model_id") != AUTHORIZED_MODEL_ID:
            errors.append("tinker.model_id is not the authorized model")
        for field_name in TINKER_REQUIRED_FLAGS:
            if _bool(tinker_data.get(field_name), f"tinker.{field_name}", errors) is not True:
                errors.append(f"tinker.{field_name} must be true")

    hf = data.get("hf")
    hf_valid = True
    if not _is_mapping(hf):
        errors.append("hf must be an object")
        hf_valid = False
    else:
        hf_data = dict(hf)
        for field_name in HF_REQUIRED_FLAGS:
            if _bool(hf_data.get(field_name), f"hf.{field_name}", errors) is not True:
                errors.append(f"hf.{field_name} must be true")
        for field_name in ("repository_ref_field", "revision_field", "checkpoint_manifest_field"):
            if not isinstance(hf_data.get(field_name), str) or not hf_data[field_name]:
                errors.append(f"hf.{field_name} must identify a receipt field")

    gate = data.get("receipt_gate")
    gate_valid = True
    if not _is_mapping(gate):
        errors.append("receipt_gate must be an object")
        gate_valid = False
    else:
        gate_data = dict(gate)
        if gate_data.get("receipt_schema_version") != receipt.SCHEMA_VERSION:
            errors.append("receipt_gate must use the T3 receipt schema")
        for field_name in (
            "observed_result_required",
            "native_success_required",
            "wandb_evidence_required",
            "tinker_evidence_required",
            "hf_evidence_required",
            "cost_receipt_required",
        ):
            if _bool(gate_data.get(field_name), f"receipt_gate.{field_name}", errors) is not True:
                errors.append(f"receipt_gate.{field_name} must be true")
        for field_name in ("receipt_attached", "receipts_verified", "paid_launch_allowed", "portfolio_evidence"):
            if _bool(gate_data.get(field_name), f"receipt_gate.{field_name}", errors) is not False:
                errors.append(f"receipt_gate.{field_name} must remain false before receipts")

    cost_valid = _validate_cost(data.get("cost"), errors)
    metrics = {
        "schema_valid": not errors,
        "task_manifest_valid": task_valid,
        "telemetry_valid": telemetry_valid,
        "tracking_config_valid": tracking_valid,
        "wandb_requirements_valid": wandb_valid,
        "tinker_requirements_valid": tinker_valid,
        "hf_requirements_valid": hf_valid,
        "receipt_gate_valid": gate_valid,
        "cost_valid": cost_valid,
        "cost_within_cap": cost_valid,
        "receipt_attached": False,
        "paid_launch_authorized": False,
        "e6_substitute": False,
        "portfolio_evidence": False,
    }
    return LaunchConfigValidationResult(not errors, tuple(errors), metrics, False)


def _cli_payload() -> dict[str, Any]:
    config = build_offline_smoke_config()
    return {"config": config, "validation": validate_t3_pilot_config(config).to_dict()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the offline config JSON")
    args = parser.parse_args(argv)
    payload = _cli_payload()
    if args.json:
        print(_canonical_json(payload))
    else:
        print(
            "T3 BrowserGym launch config: "
            f"{payload['validation']['ok']} "
            "(receipt-gated; paid launch unauthorized)"
        )
    return 0 if payload["validation"]["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Offline, fail-closed validation for a completed Pavlov xLAM smoke receipt.

The GRPO runner writes a local completion JSON after the final sampler export.
This module intentionally does not import Tinker, W&B, Hugging Face, or a
dataset library.  It only parses that JSON and checks the evidence contract for
the ten-step xLAM component smoke.  A receipt that is incomplete, internally
inconsistent, or broader than the xLAM component boundary is inadmissible.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
DATASET_ID = "Salesforce/xlam-function-calling-60k"
DATASET_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
SMOKE_SEED = 809
SMOKE_STEPS = 10
SMOKE_SAVE_STEPS = (0, 5, 10, "final")
XLAM_COMPONENT_SCOPE = "xlam_component_only"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SOURCE_PATH_PREFIX = "tinker://"

_REQUIRED_COST_FIELDS = (
    "estimated_cost_usd",
    "maximum_authorized_cost_usd",
    "operational_cap_usd",
    "safety_reserve_usd",
    "prompt_tokens",
    "sample_tokens",
    "train_tokens",
)
_COST_USD_FIELDS = (
    "estimated_cost_usd",
    "maximum_authorized_cost_usd",
    "operational_cap_usd",
    "safety_reserve_usd",
    "actual_cost_usd",
)
_TOKEN_FIELDS = ("prompt_tokens", "sample_tokens", "train_tokens")


class ReceiptValidationError(ValueError):
    """Raised when a smoke receipt cannot be admitted as evidence."""


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKeyError("duplicate JSON object key")
    return dict(pairs)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except (OSError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as exc:
        raise ReceiptValidationError(f"cannot parse receipt: {exc}") from exc
    if not isinstance(value, dict):
        raise ReceiptValidationError("receipt root must be a JSON object")
    return value


def _json_copy(value: Any) -> Any:
    """Return JSON-compatible data while rejecting NaN/Infinity and odd values."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReceiptValidationError(f"value is not strict JSON: {exc}") from exc


def _same_json(left: Any, right: Any, path: str = "$") -> None:
    """Compare JSON values with exact container, key, and scalar types."""
    if type(left) is not type(right):
        raise ReceiptValidationError(
            f"{path} changed type: {type(left).__name__} != {type(right).__name__}"
        )
    if isinstance(left, dict):
        if set(left) != set(right):
            missing = sorted(set(right) - set(left))
            extra = sorted(set(left) - set(right))
            raise ReceiptValidationError(
                f"{path} keys differ: missing={missing!r} extra={extra!r}"
            )
        for key in sorted(left):
            _same_json(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list):
        if len(left) != len(right):
            raise ReceiptValidationError(
                f"{path} length differs: {len(left)} != {len(right)}"
            )
        for index, (actual, expected) in enumerate(zip(left, right)):
            _same_json(actual, expected, f"{path}[{index}]")
        return
    if left != right:
        # Do not echo arbitrary receipt values.  A malformed payload could
        # contain a credential-like string, and validation errors are allowed
        # to reach a terminal or CI log.
        raise ReceiptValidationError(f"{path} scalar differs")


def _canonical_config(*, checkpoint_dir: str = "checkpoints/grpo") -> dict[str, Any]:
    """Return the exact ten-step Pavlov xLAM fingerprint expected by the runner."""
    training_suite_ids = [
        "agentdojo_train",
        "api_bank_rlvr_train",
        "bfcl_train",
        "browsergym_train",
        "crafter_train",
        "openr1_math_train",
        "openreward_train",
        "rtlcoder_train",
        "scienceworld_train",
        "swe_gym_train",
        "unix_ctf_train",
        "visual_app_train",
    ]
    primary_evaluation_suite_ids = [
        "agentharm_eval",
        "apex_agents_eval",
        "appbench_eval",
        "banker_toolbench_eval",
        "binaryaudit_eval",
        "frontier_swe_eval",
        "frontiermath_eval",
        "lifescibench_eval",
        "mle_bench_eval",
        "openreward_games_eval",
        "sdab_eval",
        "swe_bench_pro_eval",
        "verilog_eval",
        "webbench_eval",
    ]
    heldout_suite_ids = [
        "agentharm_eval",
        "apex_agents_eval",
        "appbench_eval",
        "banker_toolbench_eval",
        "frontiermath_eval",
        "openreward_games_eval",
    ]
    declared_domains = [
        "alignment",
        "browser",
        "chip_design",
        "code",
        "computer_use",
        "design",
        "enterprise",
        "finance",
        "games",
        "long_horizon",
        "math",
        "ml",
        "multi_domain",
        "science",
        "security",
        "tool_use",
    ]
    domain_tags = [
        "code",
        "browser",
        "computer_use",
        "finance",
        "enterprise",
        "science",
        "security",
        "chip_design",
        "design",
        "ml",
        "games",
        "alignment",
        "tools",
        "long_horizon",
    ]
    return {
        "name": "pavlov_xlam_qwen36",
        "model": MODEL_ID,
        "lora_rank": 32,
        "steps": SMOKE_STEPS,
        "group_size": 4,
        "batch_size": 2,
        "lr": 2e-5,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_prompt_tokens": 1200,
        "max_response_tokens": 128,
        "save_every": 5,
        "seed": SMOKE_SEED,
        "num_seeds": 1,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "evaluate_heldout": True,
        "base_url": None,
        "checkpoint_dir": checkpoint_dir,
        "wandb_project": "tinker-rl-lab-pavlov",
        "wandb_entity": "arvindcr4-pes-university",
        "wandb_group": "pavlov-tinker-18usd-20260809",
        "wandb_tags": ["grpo", "tinker"],
        "wandb_mode": "online",
        "wandb_enabled": True,
        "require_wandb": True,
        "hf_owner": None,
        "hf_public": True,
        "hf_repo_prefix": "pavlov-xlam-qwen36",
        "checkpoint_name_prefix": "pavlov-xlam-qwen36",
        "hf_enabled": True,
        "require_hf": True,
        "campaign_status": "authorized",
        "budget_status": "AUTHORIZED_TINKER_ONLY",
        "paid_jobs_may_launch": True,
        "authorized_budget_usd": 18.0,
        "maximum_usd": 18.0,
        "training_suite_ids": training_suite_ids,
        "heldout_suite_ids": heldout_suite_ids,
        "held_out_suite_ids": heldout_suite_ids,
        "primary_evaluation_suite_ids": primary_evaluation_suite_ids,
        "domain_tags": domain_tags,
        "declared_domains": declared_domains,
        "training_domain_union": declared_domains,
        "primary_evaluation_domain_union": declared_domains,
        "dataset_revision": DATASET_REVISION,
        "model_revision": MODEL_REVISION,
    }


PAVLOV_XLAM_SMOKE_CONFIG = _canonical_config()
EXPECTED_IMMUTABLE_CONFIG = PAVLOV_XLAM_SMOKE_CONFIG


def canonical_pavlov_xlam_smoke_config(
    *, checkpoint_dir: str = "checkpoints/grpo"
) -> dict[str, Any]:
    """Return a detached copy of the frozen smoke config for test/CLI callers."""
    return copy.deepcopy(_canonical_config(checkpoint_dir=checkpoint_dir))


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptValidationError(f"{path} must be an object")
    return value


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ReceiptValidationError(f"{path} must be a non-empty string")
    return value


def _finite_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptValidationError(f"{path} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ReceiptValidationError(f"{path} must be finite") from exc
    if not math.isfinite(number):
        raise ReceiptValidationError(f"{path} must be finite")
    if minimum is not None and number < minimum:
        raise ReceiptValidationError(f"{path} must be >= {minimum}")
    return number


def _require_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReceiptValidationError(f"{path} must be a non-negative integer")
    return value


def _extract_run_id(payload: Mapping[str, Any], *, kind: str) -> str:
    if kind == "wandb":
        candidates = [payload.get("wandb_run_id"), payload.get("wandb_id")]
        nested = payload.get("wandb")
        if isinstance(nested, Mapping):
            candidates.extend((nested.get("run_id"), nested.get("id")))
        path = "wandb_run_id"
    else:
        candidates = [
            payload.get("tinker_run_id"),
            payload.get("tinker_id"),
            payload.get("run_id"),
        ]
        nested = payload.get("tinker")
        if isinstance(nested, Mapping):
            candidates.extend((nested.get("run_id"), nested.get("id")))
        result = payload.get("result")
        if isinstance(result, Mapping):
            candidates.append(result.get("run_id"))
        path = "tinker_run_id/run_id"

    present = [candidate for candidate in candidates if candidate is not None]
    if not present:
        raise ReceiptValidationError(f"{path} is required")
    normalized = [_require_nonempty_string(candidate, path) for candidate in present]
    if any(candidate != normalized[0] for candidate in normalized[1:]):
        raise ReceiptValidationError(f"{path} aliases disagree")
    return normalized[0]


def _validate_scope(payload: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    for broad_key in (
        "campaign_claim",
        "company_claim",
        "company_ids",
        "all_company_readiness",
    ):
        if payload.get(broad_key):
            raise ReceiptValidationError(f"xLAM receipt cannot contain {broad_key}")
    scope = payload.get("scope", payload.get("claim_scope"))
    if isinstance(scope, str):
        if scope != XLAM_COMPONENT_SCOPE:
            raise ReceiptValidationError("receipt scope is not xLAM component-only")
        return scope
    scope_obj = _require_mapping(scope, "scope")
    kind = scope_obj.get("kind", scope_obj.get("scope"))
    if kind != XLAM_COMPONENT_SCOPE:
        raise ReceiptValidationError("receipt scope is not xLAM component-only")
    dataset = scope_obj.get("dataset")
    if dataset is not None and dataset != DATASET_ID:
        raise ReceiptValidationError("xLAM scope names a different dataset")
    revision = scope_obj.get("dataset_revision")
    if revision is not None and revision != config.get("dataset_revision"):
        raise ReceiptValidationError("xLAM scope dataset revision disagrees with config")
    component = scope_obj.get("component")
    if component is not None and component not in {"strict_tool_call", "function_calling"}:
        raise ReceiptValidationError("xLAM scope contains a non-xLAM component")
    claim_boundary = scope_obj.get("claim_boundary")
    if claim_boundary is not None and claim_boundary != "component_only":
        raise ReceiptValidationError("xLAM scope broadens the claim boundary")
    for broad_key in ("domains", "companies", "company_ids", "campaign_claim"):
        if scope_obj.get(broad_key):
            raise ReceiptValidationError(f"xLAM scope cannot contain {broad_key}")
    return XLAM_COMPONENT_SCOPE


def _expected_campaign_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: list(config[key])
        for key in (
            "training_suite_ids",
            "heldout_suite_ids",
            "primary_evaluation_suite_ids",
            "domain_tags",
            "declared_domains",
            "training_domain_union",
            "primary_evaluation_domain_union",
        )
    }


def _validate_receipts(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("checkpoint_receipts")
    if not isinstance(raw, list):
        raise ReceiptValidationError("checkpoint_receipts must be a list")
    if len(raw) != len(SMOKE_SAVE_STEPS):
        raise ReceiptValidationError("exactly four checkpoint receipts are required")
    receipts: list[dict[str, Any]] = []
    seen_repos: set[str] = set()
    seen_revisions: set[str] = set()
    seen_commits: set[str] = set()
    for index, (raw_receipt, expected_step) in enumerate(zip(raw, SMOKE_SAVE_STEPS)):
        receipt = dict(_require_mapping(raw_receipt, f"checkpoint_receipts[{index}]"))
        step = receipt.get("step")
        if step != expected_step:
            raise ReceiptValidationError(
                f"checkpoint_receipts[{index}].step must be {expected_step!r}"
            )
        repo_id = _require_nonempty_string(receipt.get("repo_id"), f"checkpoint_receipts[{index}].repo_id")
        revision = _require_nonempty_string(
            receipt.get("revision"), f"checkpoint_receipts[{index}].revision"
        )
        commit_sha = _require_nonempty_string(
            receipt.get("commit_sha"), f"checkpoint_receipts[{index}].commit_sha"
        )
        if not _REPO_ID_RE.fullmatch(repo_id):
            raise ReceiptValidationError(f"checkpoint_receipts[{index}].repo_id is not sanitized")
        if not _REVISION_RE.fullmatch(revision):
            raise ReceiptValidationError(f"checkpoint_receipts[{index}].revision is not sanitized")
        if not _COMMIT_RE.fullmatch(commit_sha):
            raise ReceiptValidationError(f"checkpoint_receipts[{index}].commit_sha is not immutable")
        if repo_id in seen_repos or revision in seen_revisions or commit_sha in seen_commits:
            raise ReceiptValidationError("checkpoint repositories, revisions, and commits must be unique")
        seen_repos.add(repo_id)
        seen_revisions.add(revision)
        seen_commits.add(commit_sha)

        repo_url = f"https://huggingface.co/{repo_id}"
        if receipt.get("repo_url") != repo_url:
            raise ReceiptValidationError(f"checkpoint_receipts[{index}].repo_url is not canonical")
        if receipt.get("revision_url") != f"{repo_url}/tree/{revision}":
            raise ReceiptValidationError(
                f"checkpoint_receipts[{index}].revision_url is not canonical"
            )
        if receipt.get("commit_url") != f"{repo_url}/commit/{commit_sha}":
            raise ReceiptValidationError(
                f"checkpoint_receipts[{index}].commit_url is not canonical"
            )
        source_path = _require_nonempty_string(
            receipt.get("source_path"), f"checkpoint_receipts[{index}].source_path"
        )
        if not source_path.startswith(_SOURCE_PATH_PREFIX):
            raise ReceiptValidationError(f"checkpoint_receipts[{index}].source_path is not Tinker")

        aliases = {
            "hf_revision": revision,
            "hf_commit_sha": commit_sha,
            "hf_repo_url": repo_url,
            "hf_revision_url": f"{repo_url}/tree/{revision}",
        }
        for key, expected in aliases.items():
            if key in receipt and receipt[key] != expected:
                raise ReceiptValidationError(f"checkpoint_receipts[{index}].{key} disagrees")
        receipts.append(receipt)
    return receipts


def _validate_reward_fields(payload: Mapping[str, Any], result: Mapping[str, Any]) -> list[float]:
    trace_value = result.get("reward_trace")
    if not isinstance(trace_value, list) or len(trace_value) != SMOKE_STEPS:
        raise ReceiptValidationError(f"result.reward_trace must contain exactly {SMOKE_STEPS} values")
    trace: list[float] = []
    for index, value in enumerate(trace_value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReceiptValidationError(f"result.reward_trace[{index}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ReceiptValidationError(f"result.reward_trace[{index}] must be finite in [0, 1]")
        trace.append(number)
    if "reward_trace" in payload:
        _same_json(_json_copy(payload["reward_trace"]), _json_copy(trace_value), "$.reward_trace")

    required = ("avg_first5", "avg_last10", "peak_reward", "zero_loss_steps", "zero_reward_steps")
    for key in required:
        if key not in result:
            raise ReceiptValidationError(f"result.{key} is required")
    expected_first5 = sum(trace[:5]) / 5
    expected_last10 = sum(trace) / SMOKE_STEPS
    expected_peak = max(trace)
    for key, expected in (
        ("avg_first5", expected_first5),
        ("avg_last10", expected_last10),
        ("peak_reward", expected_peak),
    ):
        actual = _finite_number(result[key], f"result.{key}", minimum=0.0)
        if actual > 1.0 or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ReceiptValidationError(f"result.{key} disagrees with reward_trace")
    _require_nonnegative_int(result["zero_loss_steps"], "result.zero_loss_steps")
    if result["zero_loss_steps"] > SMOKE_STEPS:
        raise ReceiptValidationError("result.zero_loss_steps exceeds smoke steps")
    zero_reward_steps = _require_nonnegative_int(
        result["zero_reward_steps"], "result.zero_reward_steps"
    )
    if zero_reward_steps > SMOKE_STEPS:
        raise ReceiptValidationError("result.zero_reward_steps exceeds smoke steps")
    expected_zero_rewards = sum(value == 0.0 for value in trace)
    if zero_reward_steps != expected_zero_rewards:
        raise ReceiptValidationError("result.zero_reward_steps disagrees with reward_trace")
    heldout = result.get("heldout_reward")
    if heldout is None:
        raise ReceiptValidationError("result.heldout_reward is required for a completed smoke")
    _finite_number(heldout, "result.heldout_reward", minimum=0.0)
    if float(heldout) > 1.0:
        raise ReceiptValidationError("result.heldout_reward must be in [0, 1]")
    return trace


def _validate_cost(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    cost = _require_mapping(payload.get("cost"), "cost")
    for key in _REQUIRED_COST_FIELDS:
        if key not in cost:
            raise ReceiptValidationError(f"cost.{key} is required")
    values: dict[str, Any] = dict(cost)
    estimated = _finite_number(values["estimated_cost_usd"], "cost.estimated_cost_usd", minimum=0.0)
    maximum = _finite_number(
        values["maximum_authorized_cost_usd"], "cost.maximum_authorized_cost_usd", minimum=0.0
    )
    operational = _finite_number(values["operational_cap_usd"], "cost.operational_cap_usd", minimum=0.0)
    reserve = _finite_number(values["safety_reserve_usd"], "cost.safety_reserve_usd", minimum=0.0)
    if estimated > operational + 1e-12:
        raise ReceiptValidationError("cost.estimated_cost_usd exceeds operational cap")
    configured_maximum = config.get("maximum_usd")
    if configured_maximum is None:
        raise ReceiptValidationError("config.maximum_usd is required for cost reconciliation")
    if not math.isclose(maximum, float(configured_maximum), rel_tol=0.0, abs_tol=1e-12):
        raise ReceiptValidationError("cost maximum does not match immutable config")
    if not math.isclose(maximum, operational + reserve, rel_tol=0.0, abs_tol=1e-12):
        raise ReceiptValidationError("cost cap and safety reserve do not reconcile")
    for key in _TOKEN_FIELDS:
        _require_nonnegative_int(values[key], f"cost.{key}")
    if sum(values[key] for key in _TOKEN_FIELDS) <= 0:
        raise ReceiptValidationError("cost token counts must contain work")
    if "actual_cost_usd" in values:
        actual = _finite_number(values["actual_cost_usd"], "cost.actual_cost_usd", minimum=0.0)
        if actual > operational + 1e-12:
            raise ReceiptValidationError("cost.actual_cost_usd exceeds operational cap")
    return values


def validate_completed_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_wandb_run_id: str | None = None,
    expected_tinker_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate one already-loaded completed checkpoint payload offline."""
    if not isinstance(payload, Mapping):
        raise ReceiptValidationError("receipt payload must be an object")
    actual = _json_copy(dict(payload))
    if actual.get("status") != "completed" or actual.get("final_status") != "success":
        raise ReceiptValidationError("receipt must be a successful completed run")
    if actual.get("step") != SMOKE_STEPS:
        raise ReceiptValidationError("completed receipt must end at smoke step 10")

    config = _require_mapping(actual.get("config"), "config")
    expected = (
        _canonical_config()
        if expected_config is None
        else _json_copy(dict(expected_config))
    )
    _same_json(dict(config), expected, "$.config")
    for config_alias in ("immutable_config", "wandb_config"):
        if config_alias in actual:
            _same_json(actual[config_alias], expected, f"$.{config_alias}")
    if config.get("dataset_revision") != DATASET_REVISION:
        raise ReceiptValidationError("config dataset revision is not the pinned xLAM revision")
    if config.get("model_revision") != MODEL_REVISION:
        raise ReceiptValidationError("config model revision is not the pinned model revision")

    expected_campaign_metadata = _expected_campaign_metadata(config)
    campaign_metadata = actual.get("campaign_metadata")
    _same_json(campaign_metadata, expected_campaign_metadata, "$.campaign_metadata")

    wandb_run_id = _extract_run_id(actual, kind="wandb")
    tinker_run_id = _extract_run_id(actual, kind="tinker")
    if expected_wandb_run_id is not None and wandb_run_id != expected_wandb_run_id:
        raise ReceiptValidationError("W&B run ID differs from the expected run")
    if expected_tinker_run_id is not None and tinker_run_id != expected_tinker_run_id:
        raise ReceiptValidationError("Tinker run ID differs from the expected run")

    result = _require_mapping(actual.get("result"), "result")
    if result.get("seed") != config.get("seed"):
        raise ReceiptValidationError("result.seed disagrees with immutable config")
    if result.get("run_id") != tinker_run_id:
        raise ReceiptValidationError("result.run_id disagrees with the Tinker run ID")
    _same_json(
        result.get("campaign_metadata"),
        expected_campaign_metadata,
        "$.result.campaign_metadata",
    )
    receipts = _validate_receipts(actual)
    result_receipts = result.get("checkpoint_receipts")
    if not isinstance(result_receipts, list):
        raise ReceiptValidationError("result.checkpoint_receipts must be a list")
    _same_json(result_receipts, receipts, "$.result.checkpoint_receipts")
    expected_urls = [receipt["revision_url"] for receipt in receipts]
    expected_commits = [receipt["commit_sha"] for receipt in receipts]
    for path, value, expected_value in (
        ("checkpoint_urls", actual.get("checkpoint_urls"), expected_urls),
        ("checkpoint_commit_shas", actual.get("checkpoint_commit_shas"), expected_commits),
        ("result.checkpoint_urls", result.get("checkpoint_urls"), expected_urls),
        ("result.checkpoint_commit_shas", result.get("checkpoint_commit_shas"), expected_commits),
    ):
        _same_json(value, expected_value, f"$.{path}")

    trace = _validate_reward_fields(actual, result)
    cost = _validate_cost(actual, config)
    scope = _validate_scope(actual, config)

    normalized = copy.deepcopy(actual)
    normalized["wandb_run_id"] = wandb_run_id
    normalized["tinker_run_id"] = tinker_run_id
    normalized["reward_trace"] = trace
    normalized["cost"] = cost
    normalized["scope"] = scope
    return normalized


def parse_completed_checkpoint(
    path: str | Path,
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_wandb_run_id: str | None = None,
    expected_tinker_run_id: str | None = None,
) -> dict[str, Any]:
    """Parse and validate a completed checkpoint JSON without external calls."""
    receipt_path = Path(path)
    payload = _load_json(receipt_path)
    return validate_completed_checkpoint(
        payload,
        expected_config=expected_config,
        expected_wandb_run_id=expected_wandb_run_id,
        expected_tinker_run_id=expected_tinker_run_id,
    )


# Short aliases make the validator convenient from small local gate scripts.
parse_receipt = parse_completed_checkpoint
validate_receipt = validate_completed_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--expected-wandb-run-id")
    parser.add_argument("--expected-tinker-run-id")
    args = parser.parse_args(argv)
    try:
        validated = parse_completed_checkpoint(
            args.receipt,
            expected_wandb_run_id=args.expected_wandb_run_id,
            expected_tinker_run_id=args.expected_tinker_run_id,
        )
    except ReceiptValidationError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "accepted",
                "wandb_run_id": validated["wandb_run_id"],
                "tinker_run_id": validated["tinker_run_id"],
                "checkpoint_steps": [
                    receipt["step"] for receipt in validated["checkpoint_receipts"]
                ],
                "scope": validated["scope"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

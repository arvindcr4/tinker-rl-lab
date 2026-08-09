#!/usr/bin/env python3
"""Evaluate a Tinker base model or adapter on an observed seed-809 xLAM evaluation slice.

The xLAM result is one component of the Pavlov primary-evaluation portfolio.  The
evaluation is deliberately fail-closed: an online W&B run must be initialized
and remain able to receive its immutable config, metrics, and receipt artifact
before a Tinker ``ServiceClient`` is created.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PREFILL_RATE = 0.54
SAMPLE_RATE = 1.335

# These defaults match the Pavlov campaign's durable W&B namespace.  There is
# intentionally no offline/disabled flag: paid Tinker work requires online
# provenance.
PAVLOV_WANDB_ENTITY = "arvindcr4-pes-university"
PAVLOV_WANDB_PROJECT = "tinker-rl-lab-pavlov"
PAVLOV_WANDB_GROUP = "pavlov-xlam-evaluation"
PAVLOV_WANDB_NAME = "pavlov-xlam-eval"

PAVLOV_PRIMARY_EVAL_PORTFOLIO_ID = "pavlov-primary-eval-14-suite-v1"
PAVLOV_PRIMARY_EVAL_SUITE_COUNT = 14
DEFAULT_SUITE_ID = "xlam_component"
DEFAULT_DOMAINS = ("tool_use",)
XLAM_DATASET_ID = "Salesforce/xlam-function-calling-60k"
XLAM_SPLIT_ID = "observed test_examples slice"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$", re.IGNORECASE)
_REVISION_RE = re.compile(
    r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|sha256:[0-9a-fA-F]{64})$",
    re.IGNORECASE,
)
_PLACEHOLDER_MARKERS = (
    "placeholder",
    "to_be_pinned",
    "to-be-pinned",
    "tbd",
    "todo",
    "changeme",
    "replace_me",
    "replace-me",
)

# Primary-evaluation suites from pavlovs_domain_contract.json.  Keeping this
# manifest in every component receipt makes the portfolio boundary auditable;
# ``DEFAULT_SUITE_ID`` is intentionally not one of these suites.
PAVLOV_PRIMARY_EVAL_SUITE_DOMAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("swe_bench_pro_eval", ("code", "long_horizon")),
    ("frontier_swe_eval", ("code", "ml", "long_horizon")),
    ("sdab_eval", ("code", "ml", "long_horizon", "enterprise")),
    ("banker_toolbench_eval", ("finance", "enterprise", "tool_use", "long_horizon")),
    ("apex_agents_eval", ("multi_domain", "finance", "enterprise", "long_horizon", "tool_use")),
    ("webbench_eval", ("browser", "computer_use", "enterprise")),
    ("binaryaudit_eval", ("security", "code", "long_horizon")),
    ("lifescibench_eval", ("science", "long_horizon", "tool_use")),
    ("mle_bench_eval", ("ml", "code", "long_horizon")),
    ("agentharm_eval", ("alignment", "security", "tool_use")),
    ("verilog_eval", ("chip_design", "code")),
    ("appbench_eval", ("design", "computer_use", "code")),
    ("openreward_games_eval", ("games", "long_horizon", "tool_use")),
    ("frontiermath_eval", ("math",)),
)
PAVLOV_PRIMARY_EVAL_SUITE_IDS = tuple(
    suite_id for suite_id, _ in PAVLOV_PRIMARY_EVAL_SUITE_DOMAINS
)


def maximum_eval_cost(
    examples: int, max_prompt_tokens: int, max_response_tokens: int
) -> float:
    return (
        examples * max_prompt_tokens * PREFILL_RATE
        + examples * max_response_tokens * SAMPLE_RATE
    ) / 1_000_000


def _require_positive_finite(name: str, value: int | float) -> int | float:
    try:
        numeric_value = float(value)
    except (TypeError, OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and greater than zero") from exc
    if isinstance(value, bool) or not math.isfinite(numeric_value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_provenance_text(name: str, value: str | None) -> str:
    if value is None:
        raise ValueError(f"{name} is required")
    normalized = value.strip()
    lowered = normalized.lower()
    if not normalized or lowered in {"none", "null", "na", "n/a", "unknown", "unset"}:
        raise ValueError(f"{name} is missing")
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        raise ValueError(f"{name} is a placeholder")
    zero_candidate = lowered.removeprefix("sha256:")
    if len(zero_candidate) in {40, 64} and set(zero_candidate) == {"0"}:
        raise ValueError(f"{name} is a placeholder")
    return normalized


def _validate_sha256(name: str, value: str | None) -> str:
    normalized = _require_provenance_text(name, value)
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be exactly 64 hexadecimal characters")
    return normalized.lower()


def _validate_digest(name: str, value: str | None) -> str:
    normalized = _require_provenance_text(name, value)
    if not _DIGEST_RE.fullmatch(normalized):
        raise ValueError(f"{name} must match sha256:<64 hexadecimal characters>")
    return normalized.lower()


def _validate_revision(name: str, value: str | None) -> str:
    normalized = _require_provenance_text(name, value)
    if not _REVISION_RE.fullmatch(normalized):
        raise ValueError(
            f"{name} must be an immutable 40/64-character revision or sha256 digest"
        )
    return normalized.lower()


def validate_provenance(
    *,
    dataset_revision: str | None,
    split_manifest_sha256: str | None,
    task_id_sha256: str | None,
    license_id: str | None,
    license_receipt: str | None,
    decontamination_sha256: str | None,
    decontamination_receipt: str | None,
    container_digest: str | None,
    runtime_digest: str | None,
    verifier_sha256: str | None,
    base_model_revision: str | None,
    tokenizer_revision: str | None,
    adapter_revision: str | None,
    sampler_path: str | None,
) -> dict[str, str | None]:
    """Validate Phase-0 provenance before any W&B or Tinker side effect."""

    provenance: dict[str, str | None] = {
        "dataset_revision": _validate_revision("dataset_revision", dataset_revision),
        "split_manifest_sha256": _validate_sha256(
            "split_manifest_sha256", split_manifest_sha256
        ),
        "task_id_sha256": _validate_sha256("task_id_sha256", task_id_sha256),
        "license_id": _require_provenance_text("license_id", license_id),
        "license_receipt": _require_provenance_text(
            "license_receipt", license_receipt
        ),
        "decontamination_sha256": _validate_sha256(
            "decontamination_sha256", decontamination_sha256
        ),
        "decontamination_receipt": _require_provenance_text(
            "decontamination_receipt", decontamination_receipt
        ),
        "container_digest": _validate_digest("container_digest", container_digest),
        "runtime_digest": _validate_digest("runtime_digest", runtime_digest),
        "verifier_sha256": _validate_sha256("verifier_sha256", verifier_sha256),
        "base_model_revision": _validate_revision(
            "base_model_revision", base_model_revision
        ),
        "tokenizer_revision": _validate_revision(
            "tokenizer_revision", tokenizer_revision
        ),
        "adapter_revision": None,
    }
    if sampler_path:
        provenance["adapter_revision"] = _validate_revision(
            "adapter_revision", adapter_revision
        )
    elif adapter_revision is not None:
        # An adapter revision without an adapter evaluation is ambiguous and
        # must not silently enter the run's immutable provenance record.
        raise ValueError("adapter_revision is only valid with --sampler-path")
    return provenance


def _normalise_domains(raw_domains: list[str] | None) -> tuple[str, ...]:
    """Return deterministic, non-empty domain tags from repeated/comma flags."""

    if not raw_domains:
        return DEFAULT_DOMAINS
    domains: list[str] = []
    for raw_value in raw_domains:
        for value in raw_value.split(","):
            domain = value.strip()
            if domain and domain not in domains:
                domains.append(domain)
    if not domains:
        raise ValueError("at least one non-empty --domain/--domains value is required")
    return tuple(domains)


def _validate_xlam_component_identity(
    suite_id: str, domains: tuple[str, ...]
) -> None:
    if suite_id != DEFAULT_SUITE_ID:
        raise ValueError(
            f"xLAM evaluation suite is fixed to component id {DEFAULT_SUITE_ID!r}"
        )
    if domains != DEFAULT_DOMAINS:
        raise ValueError(
            "xLAM evaluation domains are fixed to the component-only tool_use domain"
        )


class _RevisionBoundDataset:
    def __init__(self, train: list[Any], test: list[Any]) -> None:
        self._train = train
        self._test = test

    def train_examples(self) -> list[Any]:
        return self._train

    def test_examples(self) -> list[Any]:
        return self._test


def _convert_xlam_rows(rows: Any, *, seed: int) -> _RevisionBoundDataset:
    """Mirror the runner's xLAM conversion while keeping the revision-bound load."""

    system_prompt = (
        "You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n"
        '{"tool": "<name>", "arguments": {<key>: <value>}}\n'
        "No prose. Only JSON."
    )
    examples: list[Any] = []
    for row in rows:
        try:
            tools = (
                json.loads(row.get("tools", "[]"))
                if isinstance(row.get("tools"), str)
                else row.get("tools", [])
            )
            answers = (
                json.loads(row.get("answers", "[]"))
                if isinstance(row.get("answers"), str)
                else row.get("answers", [])
            )
            if not isinstance(answers, list) or not answers:
                continue
            answer = answers[0]
            tool = answer.get("name", answer.get("tool", ""))
            arguments = answer.get("arguments", answer.get("parameters", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not tool:
                continue
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\nAvailable tools:\n{json.dumps(tools[:8])}\n\n"
                f"User: {row.get('query', row.get('instruction', ''))}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            examples.append(
                SimpleNamespace(
                    prompt=prompt,
                    target={"tool": tool, "arguments": arguments},
                )
            )
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    random.Random(seed).shuffle(examples)
    return _RevisionBoundDataset(examples[:3000], examples[3000:3500])


def _load_xlam_dataset(
    *, seed: int, revision: str
) -> tuple[Any, dict[str, Any]]:
    """Load xLAM through a revision-aware runner API or a pinned local seam."""

    from platform_tinker.tinkerrl.grpo import make_xlam_dataset

    revision_parameter = _revision_parameter(
        make_xlam_dataset,
        candidates=("revision", "dataset_revision"),
        allow_var_keyword=False,
    )
    if revision_parameter is not None:
        loader_kwargs = {"seed": seed, revision_parameter: revision}
        return (
            make_xlam_dataset(**loader_kwargs),
            {
                "mode": "runner_revision_argument",
                "parameter": revision_parameter,
                "revision": revision,
            },
        )

    from datasets import load_dataset

    rows = load_dataset(XLAM_DATASET_ID, split="train", revision=revision)
    return (
        _convert_xlam_rows(rows, seed=seed),
        {
            "mode": "local_revision_bound_loader",
            "loader": "datasets.load_dataset",
            "dataset_id": XLAM_DATASET_ID,
            "split": "train",
            "revision": revision,
        },
    )


def _primary_eval_manifest() -> list[dict[str, Any]]:
    return [
        {"suite_id": suite_id, "domains": list(domains), "role": "primary_eval"}
        for suite_id, domains in PAVLOV_PRIMARY_EVAL_SUITE_DOMAINS
    ]


def _wandb_config(
    *,
    args: argparse.Namespace,
    source_kind: str,
    evaluated_path: str,
    domains: tuple[str, ...],
    projected_cost: float,
    provenance: Mapping[str, str | None],
    provenance_sha256: str,
) -> dict[str, Any]:
    """Build the only config sent to W&B; it contains no prompts/responses."""

    return {
        "config_schema_version": "pavlov-xlam-wandb-config-v1",
        "config_immutable": True,
        "artifact_ack_required": True,
        "provenance_schema_version": "pavlov-phase0-provenance-v1",
        "provenance_sha256": provenance_sha256,
        "provenance": dict(provenance),
        "phase0_provenance_sha256": provenance_sha256,
        "phase0_provenance": dict(provenance),
        "campaign": "pavlov-18usd",
        "stage": "primary-evaluation",
        "source_kind": source_kind,
        "evaluated_path": evaluated_path,
        "tokenizer_model": args.tokenizer_model,
        "dataset_id": XLAM_DATASET_ID,
        "dataset_split": XLAM_SPLIT_ID,
        "reward": "StrictToolCallReward",
        "sampling_temperature": 0.1,
        "sampling_top_p": 0.95,
        "seed": args.seed,
        "limit": args.limit,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_response_tokens": args.max_response_tokens,
        "projected_max_cost_usd": projected_cost,
        "maximum_authorized_cost_usd": args.max_cost_usd,
        "suite_id": args.suite_id,
        "suite_role": "component",
        "domains": list(domains),
        "domain_tags": list(domains),
        "portfolio_id": PAVLOV_PRIMARY_EVAL_PORTFOLIO_ID,
        "portfolio_role": "primary_eval",
        "portfolio_suite_count": PAVLOV_PRIMARY_EVAL_SUITE_COUNT,
        "portfolio_suite_ids": list(PAVLOV_PRIMARY_EVAL_SUITE_IDS),
        "portfolio_suites": _primary_eval_manifest(),
        "portfolio_component_only": True,
        "portfolio_coverage_claim": "xLAM component only; no full-portfolio claim",
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "wandb_name": args.wandb_name,
    }


def _freeze_wandb_config(run: Any, config: Mapping[str, Any]) -> None:
    """Reassert the initial config without permitting value changes."""

    config_store = getattr(run, "config", None)
    update = getattr(config_store, "update", None)
    if not callable(update):
        raise RuntimeError("W&B run does not expose config storage")
    try:
        update(dict(config), allow_val_change=False)
    except TypeError:
        # A plain dict is useful in mocked tests but does not accept W&B's
        # keyword.  Real W&B Config accepts it; other failures stay fatal.
        if type(config_store) is dict:
            update(dict(config))
        else:
            raise
    sentinel = object()
    getter = getattr(config_store, "get", None)
    if not callable(getter):
        raise RuntimeError("W&B config storage is not mapping-like")
    for key, expected in config.items():
        if getter(key, sentinel) != expected:
            raise RuntimeError(f"W&B config storage did not retain {key}")


def _assert_online_run(run: Any) -> None:
    """Reject a fake/ambient offline run even when ``wandb.init`` returned."""

    mode = getattr(run, "mode", None)
    if mode is None:
        settings = getattr(run, "settings", None) or getattr(run, "_settings", None)
        mode = getattr(settings, "mode", None)
    mode = getattr(mode, "value", mode)
    if mode != "online":
        raise RuntimeError(
            "W&B run is not verifiably online; refusing to start Tinker evaluation"
        )


def _assert_wandb_identity(run: Any) -> None:
    run_id = getattr(run, "id", None)
    run_url = getattr(run, "url", None)
    if not isinstance(run_id, str) or not run_id.strip():
        raise RuntimeError("W&B run did not expose a non-empty run id")
    if not isinstance(run_url, str) or not run_url.startswith(("https://", "http://")):
        raise RuntimeError("W&B run did not expose a valid run URL")


def _revision_parameter(
    callable_object: Any,
    *,
    candidates: tuple[str, ...],
    allow_var_keyword: bool = True,
) -> str | None:
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return None
    for candidate in candidates:
        if candidate in parameters:
            return candidate
    if allow_var_keyword and any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        if "revision" in candidates:
            return "revision"
        return candidates[0] if candidates else None
    return None


def _load_tokenizer(
    tokenizer_loader: Any, *, model_name: str, revision: str
) -> tuple[Any, dict[str, Any]]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    revision_parameter = _revision_parameter(
        tokenizer_loader, candidates=("revision",)
    )
    if revision_parameter is not None:
        kwargs[revision_parameter] = revision
    return tokenizer_loader(model_name, **kwargs), {
        "requested_revision": revision,
        "parameter": revision_parameter,
        "passed": revision_parameter is not None,
    }


def _create_service_client(
    service_client: Any,
    *,
    user_metadata: Mapping[str, str],
    base_model_revision: str,
) -> Any:
    kwargs: dict[str, Any] = {"user_metadata": dict(user_metadata)}
    revision_parameter = _revision_parameter(
        service_client, candidates=("base_model_revision", "model_revision", "revision")
    )
    if revision_parameter is not None:
        kwargs[revision_parameter] = base_model_revision
    return service_client(**kwargs)


def _create_sampling_client(
    service: Any,
    *,
    source_kind: str,
    evaluated_path: str,
    revision: str,
) -> tuple[Any, dict[str, Any]]:
    if source_kind == "base_model":
        kwargs: dict[str, Any] = {"base_model": evaluated_path}
        candidates = ("base_model_revision", "revision", "model_revision")
    else:
        kwargs = {"model_path": evaluated_path}
        candidates = ("adapter_revision", "model_path_revision", "revision")
    revision_parameter = _revision_parameter(
        service.create_sampling_client, candidates=candidates
    )
    if revision_parameter is not None:
        kwargs[revision_parameter] = revision
    return service.create_sampling_client(**kwargs), {
        "requested_revision": revision,
        "parameter": revision_parameter,
        "passed": revision_parameter is not None,
    }


def _observed_revision(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in (
            "revision",
            "model_revision",
            "base_model_revision",
            "adapter_revision",
            "_commit_hash",
            "commit_hash",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return None


def _object_revision(obj: Any) -> str | None:
    for attribute in (
        "revision",
        "model_revision",
        "base_model_revision",
        "adapter_revision",
        "_commit_hash",
        "commit_hash",
        "init_kwargs",
    ):
        observed_revision = _observed_revision(getattr(obj, attribute, None))
        if observed_revision is not None:
            return observed_revision
    return None


def _verify_revision(
    label: str, obj: Any, *, expected_revision: str
) -> dict[str, Any]:
    observed_revision = _object_revision(obj)
    if (
        observed_revision is not None
        and observed_revision.lower() != expected_revision.lower()
    ):
        raise RuntimeError(
            f"{label} revision mismatch: expected {expected_revision}, "
            f"got {observed_revision}"
        )
    return {
        "expected_revision": expected_revision,
        "observed_revision": observed_revision,
        "status": "verified" if observed_revision is not None else "not_exposed_by_api",
    }


def _verify_identity(
    label: str, obj: Any, *, expected_revision: str, expected_name: str
) -> dict[str, Any]:
    identity = _verify_revision(label, obj, expected_revision=expected_revision)
    observed_name = getattr(obj, "name_or_path", None)
    if observed_name is None:
        observed_name = getattr(obj, "model_name", None)
    if observed_name is not None and observed_name != expected_name:
        raise RuntimeError(
            f"{label} identity mismatch: expected {expected_name!r}, got {observed_name!r}"
        )
    return {
        "expected_name": expected_name,
        "expected_revision": expected_revision,
        "observed_name": observed_name,
        "observed_revision": identity["observed_revision"],
        "status": identity["status"],
    }


def _start_wandb_run(wandb: Any, config: Mapping[str, Any], args: argparse.Namespace) -> Any:
    """Start and validate an online W&B run before importing Tinker."""

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        job_type="evaluation",
        mode="online",
        tags=[f"suite:{config['suite_id']}", *[f"domain:{domain}" for domain in config["domains"]]],
        config=dict(config),
    )
    if run is None:
        raise RuntimeError("W&B online initialization returned no run")
    try:
        _assert_online_run(run)
        _assert_wandb_identity(run)
        _freeze_wandb_config(run, config)
    except Exception:
        try:
            _finish_wandb_run(run, exit_code=1)
        except Exception as finish_error:
            raise RuntimeError(
                "W&B initialization failed and could not be marked failed"
            ) from finish_error
        raise
    return run


def _finish_wandb_run(run: Any, *, exit_code: int) -> None:
    finish = getattr(run, "finish", None)
    if not callable(finish):
        raise RuntimeError("W&B run has no finish method")
    finish(exit_code=exit_code)


def _log_receipt(
    *,
    wandb: Any,
    run: Any,
    receipt: Mapping[str, Any],
    metrics: Mapping[str, Any],
    receipt_path: Path,
    artifact_name: str,
) -> None:
    """Log only aggregate metrics and the hash-only JSON receipt artifact."""

    if not receipt_path.is_file():
        raise RuntimeError("local receipt is missing before artifact upload")
    run.log(dict(metrics))
    artifact = wandb.Artifact(
        name=artifact_name,
        type="evaluation-receipt",
        metadata={
            "schema_version": receipt["schema_version"],
            "config_sha256": receipt["config_sha256"],
            "provenance_sha256": receipt["provenance_sha256"],
            "suite_id": receipt["suite_id"],
            "portfolio_id": receipt["portfolio"]["portfolio_id"],
            "portfolio_suite_count": receipt["portfolio"]["suite_count"],
        },
    )
    artifact.add_file(str(receipt_path), name="receipt.json")
    acknowledgement = run.log_artifact(artifact)
    if acknowledgement is False:
        raise RuntimeError("W&B did not acknowledge the receipt artifact")
    explicit_ack = getattr(run, "artifact_acknowledged", None)
    if explicit_ack is False:
        raise RuntimeError("W&B receipt artifact acknowledgement was negative")
    if acknowledgement is None and explicit_ack is not True:
        raise RuntimeError("W&B did not expose receipt artifact acknowledgement")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-model")
    source.add_argument("--sampler-path")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    parser.add_argument("--max-response-tokens", type=int, default=128)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--split-manifest-sha256", "--split-manifest-hash", dest="split_manifest_sha256")
    parser.add_argument("--task-id-sha256", "--task-id-hash", dest="task_id_sha256")
    parser.add_argument("--license-id", "--license-identifier", dest="license_id")
    parser.add_argument("--license-receipt")
    parser.add_argument(
        "--decontamination-sha256",
        "--decontamination-hash",
        dest="decontamination_sha256",
    )
    parser.add_argument("--decontamination-receipt")
    parser.add_argument("--container-digest")
    parser.add_argument("--runtime-digest")
    parser.add_argument("--verifier-sha256", "--verifier-hash", dest="verifier_sha256")
    parser.add_argument("--base-model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--suite-id", "--suite", dest="suite_id", default=DEFAULT_SUITE_ID)
    parser.add_argument(
        "--domain",
        "--domains",
        dest="domains",
        action="append",
        help="Domain tag(s), repeatable or comma-separated; defaults to tool_use",
    )
    parser.add_argument("--wandb-entity", default=PAVLOV_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=PAVLOV_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=PAVLOV_WANDB_GROUP)
    parser.add_argument("--wandb-name", "--wandb-run-name", dest="wandb_name", default=PAVLOV_WANDB_NAME)
    args = parser.parse_args()

    for name, value in (
        ("limit", args.limit),
        ("max_prompt_tokens", args.max_prompt_tokens),
        ("max_response_tokens", args.max_response_tokens),
        ("max_cost_usd", args.max_cost_usd),
    ):
        try:
            _require_positive_finite(name, value)
        except ValueError as exc:
            parser.error(str(exc))

    if not args.suite_id.strip():
        parser.error("--suite-id must not be empty")
    try:
        domains = _normalise_domains(args.domains)
        _validate_xlam_component_identity(args.suite_id, domains)
    except ValueError as exc:
        parser.error(str(exc))

    source_kind = "base_model" if args.base_model else "sampler_path"
    evaluated_path = args.base_model or args.sampler_path
    try:
        provenance = validate_provenance(
            dataset_revision=args.dataset_revision,
            split_manifest_sha256=args.split_manifest_sha256,
            task_id_sha256=args.task_id_sha256,
            license_id=args.license_id,
            license_receipt=args.license_receipt,
            decontamination_sha256=args.decontamination_sha256,
            decontamination_receipt=args.decontamination_receipt,
            container_digest=args.container_digest,
            runtime_digest=args.runtime_digest,
            verifier_sha256=args.verifier_sha256,
            base_model_revision=args.base_model_revision,
            tokenizer_revision=args.tokenizer_revision,
            adapter_revision=args.adapter_revision,
            sampler_path=args.sampler_path,
        )
    except ValueError as exc:
        raise SystemExit(f"Phase-0 provenance gate failed: {exc}") from exc
    provenance_sha256 = _sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    )

    projected = maximum_eval_cost(
        args.limit, args.max_prompt_tokens, args.max_response_tokens
    )
    if projected > args.max_cost_usd:
        raise SystemExit(
            f"projected maximum ${projected:.4f} exceeds cap ${args.max_cost_usd:.4f}"
        )

    config = _wandb_config(
        args=args,
        source_kind=source_kind,
        evaluated_path=evaluated_path,
        domains=domains,
        projected_cost=projected,
        provenance=provenance,
        provenance_sha256=provenance_sha256,
    )
    config_sha256 = _sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"))
    )

    # Import and initialize W&B first.  In particular, do not import/create a
    # Tinker client until this online run exists and its config is immutable.
    wandb_run: Any | None = None
    try:
        import wandb

        wandb_run = _start_wandb_run(wandb, config, args)

        import tinker
        import tinker.types as T
        from transformers import AutoTokenizer

        from platform_tinker.tinkerrl.grpo import StrictToolCallReward

        dataset, dataset_binding = _load_xlam_dataset(
            seed=args.seed, revision=provenance["dataset_revision"]
        )
        examples = list(dataset.test_examples())[: args.limit]
        if not examples:
            raise RuntimeError("xLAM dataset produced no evaluation examples")
        tokenizer, tokenizer_binding = _load_tokenizer(
            AutoTokenizer.from_pretrained,
            model_name=args.tokenizer_model,
            revision=provenance["tokenizer_revision"],
        )
        tokenizer_identity = _verify_identity(
            "tokenizer",
            tokenizer,
            expected_revision=provenance["tokenizer_revision"],
            expected_name=args.tokenizer_model,
        )
        service = _create_service_client(
            tinker.ServiceClient,
            base_model_revision=provenance["base_model_revision"],
            user_metadata={
                "campaign": "pavlov-18usd",
                "stage": "primary-evaluation",
                "seed": str(args.seed),
                "suite_id": args.suite_id,
                "domains": ",".join(domains),
                "dataset_revision": provenance["dataset_revision"],
                "base_model_revision": provenance["base_model_revision"],
                "tokenizer_revision": provenance["tokenizer_revision"],
                "decontamination_sha256": provenance["decontamination_sha256"],
            },
        )
        service_identity = _verify_revision(
            "service client",
            service,
            expected_revision=provenance["base_model_revision"],
        )
        sampling_revision = provenance["adapter_revision"] or provenance[
            "base_model_revision"
        ]
        sampler, sampler_binding = _create_sampling_client(
            service,
            source_kind=source_kind,
            evaluated_path=evaluated_path,
            revision=sampling_revision,
        )
        sampler_identity = _verify_identity(
            "sampling client",
            sampler,
            expected_revision=sampling_revision,
            expected_name=evaluated_path,
        )

        reward = StrictToolCallReward()
        scores: list[float] = []
        prompt_tokens = 0
        sample_tokens = 0
        rows: list[dict[str, Any]] = []
        for index, example in enumerate(examples):
            prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False)[
                : args.max_prompt_tokens
            ]
            result = sampler.sample(
                T.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=T.SamplingParams(
                    max_tokens=args.max_response_tokens,
                    temperature=0.1,
                    top_p=0.95,
                ),
            ).result()
            sequence = result.sequences[0]
            response_tokens = list(sequence.tokens)
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            score = reward.score(response, example)
            scores.append(score)
            prompt_tokens += len(prompt_ids)
            sample_tokens += len(response_tokens)
            rows.append(
                {
                    "index": index,
                    "prompt_sha256": _sha256(example.prompt),
                    "target_sha256": _sha256(
                        json.dumps(example.target, sort_keys=True, separators=(",", ":"))
                    ),
                    "response_sha256": _sha256(response),
                    "score": score,
                    "prompt_tokens": len(prompt_ids),
                    "sample_tokens": len(response_tokens),
                }
            )
            if (index + 1) % 25 == 0:
                print(
                    f"evaluated={index + 1}/{len(examples)} "
                    f"mean_reward={sum(scores) / len(scores):.4f}",
                    flush=True,
                )

        mean_reward = sum(scores) / len(scores) if scores else None
        perfect_call_rate = (
            sum(score == 1.0 for score in scores) / len(scores) if scores else None
        )
        estimated_cost = (
            prompt_tokens * PREFILL_RATE + sample_tokens * SAMPLE_RATE
        ) / 1_000_000
        portfolio = {
            "portfolio_id": PAVLOV_PRIMARY_EVAL_PORTFOLIO_ID,
            "portfolio_role": "primary_eval",
            "suite_count": PAVLOV_PRIMARY_EVAL_SUITE_COUNT,
            "suite_ids": list(PAVLOV_PRIMARY_EVAL_SUITE_IDS),
            "suites": _primary_eval_manifest(),
            "component_suite_id": args.suite_id,
            "component_domains": list(domains),
            "component_domain_tags": list(domains),
            "component_only": True,
            "coverage_claim": "xLAM component only; does not cover all Pavlov domains",
        }
        run_identity = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "group": args.wandb_group,
            "name": args.wandb_name,
        }
        for key in ("id", "url"):
            value = getattr(wandb_run, key, None)
            if value is not None:
                run_identity[key] = value
        receipt = {
            "schema_version": "pavlov-xlam-eval-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha256,
            "provenance_schema_version": "pavlov-phase0-provenance-v1",
            "provenance_sha256": provenance_sha256,
            "provenance": dict(provenance),
            "phase0_provenance_sha256": provenance_sha256,
            "phase0_provenance": dict(provenance),
            "dataset_loader_binding": dataset_binding,
            "service_client_identity": service_identity,
            "tokenizer_binding": tokenizer_binding,
            "tokenizer_identity": tokenizer_identity,
            "sampling_client_binding": sampler_binding,
            "sampling_client_identity": sampler_identity,
            "artifact_ack_required": True,
            "source_kind": source_kind,
            "evaluated_path": evaluated_path,
            "tokenizer_model": args.tokenizer_model,
            "dataset_id": XLAM_DATASET_ID,
            "dataset_split": XLAM_SPLIT_ID,
            "seed": args.seed,
            "suite_id": args.suite_id,
            "suite_role": "component",
            "domains": list(domains),
            "domain_tags": list(domains),
            "portfolio_suite_count": PAVLOV_PRIMARY_EVAL_SUITE_COUNT,
            "portfolio_suite_ids": list(PAVLOV_PRIMARY_EVAL_SUITE_IDS),
            "portfolio_suites": _primary_eval_manifest(),
            "portfolio": portfolio,
            "wandb": run_identity,
            "examples": len(examples),
            "mean_strict_reward": mean_reward,
            "perfect_call_rate": perfect_call_rate,
            "prompt_tokens": prompt_tokens,
            "sample_tokens": sample_tokens,
            "estimated_cost_usd": estimated_cost,
            "maximum_authorized_cost_usd": args.max_cost_usd,
            "pricing": {
                "prefill_usd_per_million": PREFILL_RATE,
                "sample_usd_per_million": SAMPLE_RATE,
                "source": "https://tinker-docs.thinkingmachines.ai/tinker/models.json",
                "accessed_at": "2026-08-09",
            },
            "rows": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        metrics = {
            "examples": len(examples),
            "mean_strict_reward": mean_reward,
            "perfect_call_rate": perfect_call_rate,
            "prompt_tokens": prompt_tokens,
            "sample_tokens": sample_tokens,
            "estimated_cost_usd": estimated_cost,
            "maximum_authorized_cost_usd": args.max_cost_usd,
        }
        _log_receipt(
            wandb=wandb,
            run=wandb_run,
            receipt=receipt,
            metrics=metrics,
            receipt_path=args.out,
            artifact_name=f"{args.wandb_name}-receipt",
        )
        print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    except Exception:
        if wandb_run is not None:
            try:
                _finish_wandb_run(wandb_run, exit_code=1)
            except Exception as finish_error:
                raise RuntimeError("W&B failed and could not be marked failed") from finish_error
        raise

    try:
        _finish_wandb_run(wandb_run, exit_code=0)
    except Exception as finish_error:
        # A successful evaluation without a successful W&B finish is not an
        # admissible result, even though the local receipt remains available.
        try:
            _finish_wandb_run(wandb_run, exit_code=1)
        except Exception:
            pass
        raise RuntimeError("W&B failed to record successful completion") from finish_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline contract adapter for the Unix-CTF procedural training suite.

The adapter validates a prospective training boundary and its receipts; it does
not generate Unix tasks, execute a shell, contact Vmax, W&B, Tinker, or
Hugging Face, and never manufactures a training result.  The training task
IDs must be disjoint from both the E7 ``binaryaudit_eval`` task IDs and the
complete primary-evaluation task registry supplied in the boundary.

The native contract is intentionally explicit: a pinned container, an
isolated network-disabled shell sandbox, an artifact manifest, and an
immutable verifier revision are required.  A valid receipt is training
provenance only.  It cannot be promoted to a primary evaluation or portfolio
claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any


AUTHORITATIVE_SOURCE_ID = "VmaxAI/unix-ctf"
AUTHORITATIVE_SOURCE_URL = (
    "https://vmax.ai/team/unix-ctf-procedural-environments-for-unix-competence-reinforcement-learning"
)
SUITE_ID = "unix_ctf_train"
ROLE = "train"
BOUNDARY_SCHEMA_VERSION = "unix-ctf-train-boundary-v1"
RESULT_SCHEMA_VERSION = "unix-ctf-train-result-v1"
BINARYAUDIT_SUITE_ID = "binaryaudit_eval"
PRIMARY_EVAL_REGISTRY_ID = "primary_eval_registry"
TRAINING_ONLY_CLAIM = "unix_ctf_training_only"
VERIFIER_NAME = "unix-ctf-native-verifier"
DEFAULT_PROVIDER = "Tinker"
DEFAULT_MAXIMUM_USD = 18.0
DEFAULT_OPERATIONAL_CAP_USD = 16.5
DEFAULT_RESERVE_USD = 1.5

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_SUCCESS_STATUSES = {"complete", "completed", "success", "succeeded"}


class UnixCtfTrainBoundaryError(ValueError):
    """Raised when the Unix-CTF training contract is not admissible."""

    def __init__(self, diagnostics: Iterable[str]):
        unique = tuple(dict.fromkeys(str(item) for item in diagnostics if str(item)))
        self.diagnostics = unique or ("Unix-CTF training boundary validation failed",)
        super().__init__("; ".join(self.diagnostics))


BoundaryValidationError = UnixCtfTrainBoundaryError


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON for reproducible procedural/config receipts."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnixCtfTrainBoundaryError((f"{name} must be a non-empty string",))
    return value.strip()


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise UnixCtfTrainBoundaryError((f"{name} must be a finite real number",))
    result = float(value)
    if not math.isfinite(result):
        raise UnixCtfTrainBoundaryError((f"{name} must be finite",))
    return result


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    result = _finite(name, value)
    if not result.is_integer() or int(result) < minimum:
        raise UnixCtfTrainBoundaryError((f"{name} must be an integer >= {minimum}",))
    return int(result)


def _hash64(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise UnixCtfTrainBoundaryError((f"{name} must be a 64-character hexadecimal hash",))
    return value.lower()


def _immutable_revision(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnixCtfTrainBoundaryError((f"{name} must be immutable",))
    revision = value.strip()
    if not (_HEX40.fullmatch(revision) or _DIGEST.fullmatch(revision)):
        raise UnixCtfTrainBoundaryError(
            (f"{name} must be a 40-character commit or sha256 digest, not a mutable tag/branch",)
        )
    return revision.lower()


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UnixCtfTrainBoundaryError((f"{name} must be a mapping",))
    return value


def _ids(name: str, values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Iterable):
        raise UnixCtfTrainBoundaryError((f"{name} must be an iterable of task IDs",))
    materialized = tuple(values)
    if not materialized:
        raise UnixCtfTrainBoundaryError((f"{name} must be non-empty",))
    normalized = tuple(_string(f"{name}[{index}]", value) for index, value in enumerate(materialized))
    if len(set(normalized)) != len(normalized):
        raise UnixCtfTrainBoundaryError((f"{name} must not contain duplicate IDs",))
    if normalized != tuple(sorted(normalized)):
        raise UnixCtfTrainBoundaryError((f"{name} must be lexically sorted for deterministic hashing",))
    return normalized


def task_id_manifest_sha256(task_ids: Iterable[str]) -> str:
    return canonical_sha256(list(_ids("task_ids", task_ids)))


def _read_mapping(source: Mapping[str, Any] | Path | str, name: str) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if not isinstance(source, (Path, str)):
        raise UnixCtfTrainBoundaryError((f"{name} must be a mapping or JSON path",))
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UnixCtfTrainBoundaryError((f"could not read {name}: {exc}",)) from exc
    return _mapping(name, value)


def _successful_status(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _SUCCESS_STATUSES


def _validate_source(boundary: Mapping[str, Any], diagnostics: list[str]) -> dict[str, Any]:
    source = boundary.get("source_identity", boundary.get("source"))
    if not isinstance(source, Mapping):
        diagnostics.append("source_identity is required")
        return {}
    source_id = source.get("id", source.get("source_id"))
    source_url = source.get("url", source.get("source_url"))
    if source_id != AUTHORITATIVE_SOURCE_ID:
        diagnostics.append(f"source identity must be {AUTHORITATIVE_SOURCE_ID!r}")
    if source_url != AUTHORITATIVE_SOURCE_URL:
        diagnostics.append(f"source URL must be {AUTHORITATIVE_SOURCE_URL!r}")
    license_record = source.get("license") if isinstance(source.get("license"), Mapping) else {}
    try:
        revision = _immutable_revision(
            "source_identity.revision",
            source.get("revision", source.get("immutable_revision")),
        )
        license_spdx = _string(
            "source_identity.license_spdx",
            source.get("license_spdx", license_record.get("spdx")),
        )
        license_hash = _hash64(
            "source_identity.license_text_sha256",
            source.get("license_text_sha256", license_record.get("text_sha256")),
        )
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        revision = license_spdx = license_hash = ""
    return {
        "id": AUTHORITATIVE_SOURCE_ID,
        "url": AUTHORITATIVE_SOURCE_URL,
        "revision": revision,
        "license_spdx": license_spdx,
        "license_text_sha256": license_hash,
    }


def _validate_procedural_generation(
    boundary: Mapping[str, Any], diagnostics: list[str]
) -> dict[str, Any]:
    procedural = boundary.get("procedural_generation")
    if not isinstance(procedural, Mapping):
        diagnostics.append("procedural_generation is required")
        return {}
    generator_id = procedural.get("generator_id")
    if not isinstance(generator_id, str) or not generator_id.strip():
        diagnostics.append("procedural_generation.generator_id must be non-empty")
        generator_id = ""
    try:
        generator_revision = _immutable_revision(
            "procedural_generation.generator_revision", procedural.get("generator_revision")
        )
        seed = _integer("procedural_generation.seed", procedural.get("seed"), minimum=0)
        seed_hash = _hash64("procedural_generation.seed_sha256", procedural.get("seed_sha256"))
        if seed_hash != canonical_sha256(seed):
            diagnostics.append("procedural seed hash does not match seed")
        task_ids = _ids("procedural_generation.task_ids", procedural.get("task_ids"))
        task_hash = _hash64(
            "procedural_generation.task_id_manifest_sha256",
            procedural.get("task_id_manifest_sha256"),
        )
        if task_hash != task_id_manifest_sha256(task_ids):
            diagnostics.append("procedural task hash does not match task_ids")
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        generator_revision = ""
        seed = 0
        seed_hash = ""
        task_ids = ()
        task_hash = ""
    parameters = procedural.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        diagnostics.append("procedural_generation.parameters must be a non-empty mapping")
        parameters = {}
    try:
        parameter_hash = _hash64(
            "procedural_generation.parameters_sha256", procedural.get("parameters_sha256")
        )
        try:
            expected_parameter_hash = canonical_sha256(parameters)
        except (TypeError, ValueError) as exc:
            raise UnixCtfTrainBoundaryError(
                (f"procedural_generation.parameters must be JSON-canonicalizable: {exc}",)
            ) from exc
        if parameter_hash != expected_parameter_hash:
            diagnostics.append("procedural parameters hash does not match parameters")
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        parameter_hash = ""
    return {
        "generator_id": str(generator_id),
        "generator_revision": generator_revision,
        "seed": seed,
        "seed_sha256": seed_hash,
        "task_ids": list(task_ids),
        "task_id_manifest_sha256": task_hash,
        "parameters": dict(parameters),
        "parameters_sha256": parameter_hash,
    }


def _normalise_disjointness(
    boundary: Mapping[str, Any], diagnostics: list[str]
) -> dict[str, Any]:
    disjointness = boundary.get("disjointness")
    if not isinstance(disjointness, Mapping):
        diagnostics.append("disjointness is required")
        return {}
    binary = disjointness.get("binaryaudit_eval", disjointness.get(BINARYAUDIT_SUITE_ID))
    primary = disjointness.get("primary_evals", disjointness.get(PRIMARY_EVAL_REGISTRY_ID))
    if not isinstance(binary, Mapping):
        diagnostics.append("disjointness.binaryaudit_eval is required")
        binary = {}
    if not isinstance(primary, Mapping):
        diagnostics.append("disjointness.primary_evals is required")
        primary = {}
    try:
        binary_ids = _ids("disjointness.binaryaudit_eval.task_ids", binary.get("task_ids"))
        binary_hash = _hash64(
            "disjointness.binaryaudit_eval.task_id_manifest_sha256",
            binary.get("task_id_manifest_sha256"),
        )
        if binary_hash != task_id_manifest_sha256(binary_ids):
            diagnostics.append("BinaryAudit disjointness hash does not match task_ids")
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        binary_ids, binary_hash = (), ""
    if binary.get("suite_id") != BINARYAUDIT_SUITE_ID:
        diagnostics.append("disjointness.binaryaudit_eval.suite_id must be binaryaudit_eval")
    try:
        primary_ids = _ids("disjointness.primary_evals.task_ids", primary.get("task_ids"))
        primary_hash = _hash64(
            "disjointness.primary_evals.task_id_manifest_sha256",
            primary.get("task_id_manifest_sha256"),
        )
        if primary_hash != task_id_manifest_sha256(primary_ids):
            diagnostics.append("primary-evaluation disjointness hash does not match task_ids")
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        primary_ids, primary_hash = (), ""
    if primary.get("registry_id") != PRIMARY_EVAL_REGISTRY_ID:
        diagnostics.append("disjointness.primary_evals.registry_id must be primary_eval_registry")
    return {
        "binaryaudit_eval": {
            "suite_id": BINARYAUDIT_SUITE_ID,
            "task_ids": list(binary_ids),
            "task_id_manifest_sha256": binary_hash,
        },
        "primary_evals": {
            "registry_id": PRIMARY_EVAL_REGISTRY_ID,
            "task_ids": list(primary_ids),
            "task_id_manifest_sha256": primary_hash,
        },
    }


def _validate_native_environment(
    boundary: Mapping[str, Any], diagnostics: list[str]
) -> dict[str, Any]:
    environment = boundary.get("native_environment", boundary.get("environment"))
    if not isinstance(environment, Mapping):
        diagnostics.append("native_environment is required")
        return {}
    if environment.get("mode") != "native":
        diagnostics.append("native_environment.mode must be native")
    try:
        container = environment.get("container_digest")
        if not isinstance(container, str) or not _DIGEST.fullmatch(container):
            raise UnixCtfTrainBoundaryError(("native_environment.container_digest must be sha256:<64 hex>",))
        environment_hash = _hash64(
            "native_environment.environment_manifest_sha256",
            environment.get("environment_manifest_sha256", environment.get("manifest_sha256")),
        )
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        container, environment_hash = "", ""
    sandbox = environment.get("shell_sandbox")
    if not isinstance(sandbox, Mapping):
        diagnostics.append("native_environment.shell_sandbox is required")
        sandbox = {}
    if sandbox.get("enabled") is not True:
        diagnostics.append("shell_sandbox.enabled must be true")
    if sandbox.get("network") != "disabled":
        diagnostics.append("shell_sandbox.network must be disabled")
    if sandbox.get("filesystem") != "isolated":
        diagnostics.append("shell_sandbox.filesystem must be isolated")
    try:
        policy_hash = _hash64("shell_sandbox.policy_sha256", sandbox.get("policy_sha256"))
        limits_hash = _hash64("shell_sandbox.limits_sha256", sandbox.get("limits_sha256"))
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        policy_hash = limits_hash = ""
    allowlist = sandbox.get("command_allowlist")
    if isinstance(allowlist, (str, bytes, bytearray)) or not allowlist:
        diagnostics.append("shell_sandbox.command_allowlist must be non-empty")
        allowlist = ()
    else:
        try:
            allowlist = tuple(_string("shell_sandbox.command_allowlist[]", item) for item in allowlist)
        except (UnixCtfTrainBoundaryError, TypeError) as exc:
            diagnostics.extend(
                exc.diagnostics if isinstance(exc, UnixCtfTrainBoundaryError) else ("shell_sandbox.command_allowlist must be iterable",)
            )
            allowlist = ()
    artifact = environment.get("artifact_contract")
    if not isinstance(artifact, Mapping):
        diagnostics.append("native_environment.artifact_contract is required")
        artifact = {}
    if artifact.get("required") is not True:
        diagnostics.append("artifact_contract.required must be true")
    try:
        artifact_hash = _hash64(
            "artifact_contract.manifest_sha256",
            artifact.get("manifest_sha256", artifact.get("artifact_manifest_sha256")),
        )
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        artifact_hash = ""
    artifact_types = artifact.get("types")
    if isinstance(artifact_types, (str, bytes, bytearray)) or not artifact_types:
        diagnostics.append("artifact_contract.types must be non-empty")
        artifact_types = ()
    else:
        try:
            artifact_types = tuple(_string("artifact_contract.types[]", item) for item in artifact_types)
        except (UnixCtfTrainBoundaryError, TypeError) as exc:
            diagnostics.extend(
                exc.diagnostics if isinstance(exc, UnixCtfTrainBoundaryError) else ("artifact_contract.types must be iterable",)
            )
            artifact_types = ()
    verifier = environment.get("verifier_contract", environment.get("verifier"))
    if not isinstance(verifier, Mapping):
        diagnostics.append("native_environment.verifier_contract is required")
        verifier = {}
    if verifier.get("name") != VERIFIER_NAME:
        diagnostics.append(f"verifier_contract.name must be {VERIFIER_NAME!r}")
    try:
        verifier_revision = _immutable_revision(
            "verifier_contract.revision", verifier.get("revision")
        )
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        verifier_revision = ""
    receipt_schema = verifier.get("receipt_schema")
    if receipt_schema != RESULT_SCHEMA_VERSION:
        diagnostics.append(f"verifier_contract.receipt_schema must be {RESULT_SCHEMA_VERSION!r}")
    checks = verifier.get("checks")
    if isinstance(checks, (str, bytes, bytearray)) or not checks:
        diagnostics.append("verifier_contract.checks must be non-empty")
        checks = ()
    else:
        checks = tuple(_string("verifier_contract.checks[]", item) for item in checks)
    return {
        "mode": "native",
        "container_digest": str(container).lower(),
        "environment_manifest_sha256": environment_hash,
        "shell_sandbox": {
            "enabled": True,
            "network": "disabled",
            "filesystem": "isolated",
            "policy_sha256": policy_hash,
            "limits_sha256": limits_hash,
            "command_allowlist": list(allowlist),
        },
        "artifact_contract": {
            "required": True,
            "manifest_sha256": artifact_hash,
            "types": list(artifact_types),
        },
        "verifier_contract": {
            "name": VERIFIER_NAME,
            "revision": verifier_revision,
            "receipt_schema": RESULT_SCHEMA_VERSION,
            "checks": list(checks),
        },
    }


def _validate_budget_gate(boundary: Mapping[str, Any], diagnostics: list[str]) -> dict[str, Any]:
    budget = boundary.get("budget_gate")
    if not isinstance(budget, Mapping):
        diagnostics.append("budget_gate is required")
        return {}
    if budget.get("provider") != DEFAULT_PROVIDER:
        diagnostics.append("budget_gate.provider must be Tinker")
    if budget.get("authorized") is not True:
        diagnostics.append("budget_gate.authorized must be true")
    try:
        maximum = _finite("budget_gate.maximum_usd", budget.get("maximum_usd"))
        cap = _finite("budget_gate.operational_cap_usd", budget.get("operational_cap_usd"))
        reserve = _finite("budget_gate.reserve_usd", budget.get("reserve_usd"))
        if maximum != DEFAULT_MAXIMUM_USD or cap != DEFAULT_OPERATIONAL_CAP_USD or reserve != DEFAULT_RESERVE_USD:
            diagnostics.append("budget_gate must match the authorized $18/$16.50/$1.50 ledger")
        if cap + reserve > maximum:
            diagnostics.append("budget_gate cap plus reserve exceeds maximum")
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        maximum, cap, reserve = 0.0, 0.0, 0.0
    return {
        "provider": DEFAULT_PROVIDER,
        "authorized": True,
        "maximum_usd": maximum,
        "operational_cap_usd": cap,
        "reserve_usd": reserve,
    }


def validate_unix_ctf_training_boundary(
    source: Mapping[str, Any] | Path | str,
) -> dict[str, Any]:
    """Validate a complete, prospective Unix-CTF training boundary."""

    boundary = _read_mapping(source, "boundary")
    diagnostics: list[str] = []
    if boundary.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
        diagnostics.append(f"schema_version must be {BOUNDARY_SCHEMA_VERSION!r}")
    if boundary.get("suite_id") != SUITE_ID or boundary.get("role") != ROLE:
        diagnostics.append("boundary must identify unix_ctf_train with role train")
    source_identity = _validate_source(boundary, diagnostics)
    procedural = _validate_procedural_generation(boundary, diagnostics)
    disjointness = _normalise_disjointness(boundary, diagnostics)
    environment = _validate_native_environment(boundary, diagnostics)
    budget = _validate_budget_gate(boundary, diagnostics)
    train_ids = tuple(procedural.get("task_ids", ()))
    binary_ids = tuple(disjointness.get("binaryaudit_eval", {}).get("task_ids", ()))
    primary_ids = tuple(disjointness.get("primary_evals", {}).get("task_ids", ()))
    if set(train_ids).intersection(binary_ids):
        diagnostics.append("Unix-CTF training IDs overlap BinaryAudit evaluation IDs")
    if set(train_ids).intersection(primary_ids):
        diagnostics.append("Unix-CTF training IDs overlap primary-evaluation IDs")
    if diagnostics:
        raise UnixCtfTrainBoundaryError(diagnostics)
    return {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "role": ROLE,
        "source_identity": source_identity,
        "procedural_generation": procedural,
        "disjointness": disjointness,
        "native_environment": environment,
        "budget_gate": budget,
        "training_only": True,
        "primary_eval_claim_permitted": False,
        "binaryaudit_substitution_permitted": False,
    }


def _validate_tracking(
    receipt: Mapping[str, Any], expected_artifact_hash: str, diagnostics: list[str]
) -> dict[str, Any]:
    tracking: dict[str, Any] = {}
    wandb = receipt.get("wandb")
    if not isinstance(wandb, Mapping):
        diagnostics.append("missing W&B receipt")
    else:
        for field in ("run_id", "url"):
            if not isinstance(wandb.get(field), str) or not wandb[field].strip():
                diagnostics.append(f"W&B receipt missing {field}")
        if wandb.get("mode") != "online":
            diagnostics.append("W&B receipt mode must be online")
        try:
            config_hash = _hash64("W&B config_sha256", wandb.get("config_sha256"))
            metrics_hash = _hash64("W&B metrics_receipt_sha256", wandb.get("metrics_receipt_sha256"))
        except UnixCtfTrainBoundaryError as exc:
            diagnostics.extend(exc.diagnostics)
            config_hash = metrics_hash = ""
        if wandb.get("metrics_logged") is not True:
            diagnostics.append("W&B receipt metrics_logged must be true")
        tracking["wandb"] = {"run_id": str(wandb.get("run_id", "")), "url": str(wandb.get("url", "")), "mode": "online", "config_sha256": config_hash, "metrics_receipt_sha256": metrics_hash}
    tinker = receipt.get("tinker")
    if not isinstance(tinker, Mapping):
        diagnostics.append("missing Tinker receipt")
    else:
        if not isinstance(tinker.get("run_id"), str) or not tinker["run_id"].strip():
            diagnostics.append("Tinker receipt missing run_id")
        if not _successful_status(tinker.get("status")):
            diagnostics.append("Tinker receipt status must be completed/success")
        try:
            config_hash = _hash64("Tinker config_sha256", tinker.get("config_sha256"))
            cost = _finite("Tinker cost_usd", tinker.get("cost_usd"))
            if cost < 0:
                raise UnixCtfTrainBoundaryError(("Tinker cost_usd must be non-negative",))
        except UnixCtfTrainBoundaryError as exc:
            diagnostics.extend(exc.diagnostics)
            config_hash, cost = "", 0.0
        tracking["tinker"] = {"run_id": str(tinker.get("run_id", "")), "status": str(tinker.get("status", "")), "config_sha256": config_hash, "cost_usd": cost}
    hf = receipt.get("hf")
    if not isinstance(hf, Mapping):
        diagnostics.append("missing Hugging Face receipt")
    else:
        if not isinstance(hf.get("repo"), str) or not hf["repo"].strip():
            diagnostics.append("Hugging Face receipt missing repo")
        try:
            revision = _immutable_revision("Hugging Face revision", hf.get("revision", hf.get("commit")))
            artifact_hash = _hash64("Hugging Face artifact_manifest_sha256", hf.get("artifact_manifest_sha256"))
        except UnixCtfTrainBoundaryError as exc:
            diagnostics.extend(exc.diagnostics)
            revision, artifact_hash = "", ""
        if hf.get("visibility") != "private":
            diagnostics.append("Hugging Face checkpoint visibility must be private")
        if artifact_hash != expected_artifact_hash:
            diagnostics.append("Hugging Face artifact manifest differs from artifact contract")
        tracking["hf"] = {"repo": str(hf.get("repo", "")), "revision": revision, "visibility": "private", "artifact_manifest_sha256": artifact_hash}
    return tracking


def _validate_budget_receipt(receipt: Mapping[str, Any], boundary_budget: Mapping[str, Any], diagnostics: list[str]) -> dict[str, Any]:
    budget = receipt.get("budget_receipt")
    if not isinstance(budget, Mapping):
        diagnostics.append("budget_receipt is required")
        return {}
    if budget.get("provider") != boundary_budget.get("provider"):
        diagnostics.append("budget receipt provider differs from boundary")
    if budget.get("authorized") is not True:
        diagnostics.append("budget receipt authorized must be true")
    try:
        spent = _finite("budget_receipt.spent_usd", budget.get("spent_usd"))
        projected = _finite("budget_receipt.projected_next_cost_usd", budget.get("projected_next_cost_usd"))
        if spent < 0 or projected < 0:
            raise UnixCtfTrainBoundaryError(("budget receipt costs must be non-negative",))
    except UnixCtfTrainBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        spent, projected = 0.0, 0.0
    cap = boundary_budget.get("operational_cap_usd", 0.0)
    maximum = boundary_budget.get("maximum_usd", 0.0)
    if spent + projected > cap:
        diagnostics.append("projected training spend exceeds operational cap")
    if spent + projected > maximum:
        diagnostics.append("projected training spend exceeds hard maximum")
    if budget.get("within_operational_cap") is not True:
        diagnostics.append("budget receipt within_operational_cap must be true")
    return {"provider": str(budget.get("provider", "")), "authorized": True, "spent_usd": spent, "projected_next_cost_usd": projected, "within_operational_cap": True}


def validate_unix_ctf_training_receipt(
    boundary_source: Mapping[str, Any] | Path | str,
    receipt_source: Mapping[str, Any] | Path | str,
) -> dict[str, Any]:
    """Validate a completed Unix-CTF training receipt without claiming eval."""

    boundary = validate_unix_ctf_training_boundary(boundary_source)
    receipt = _read_mapping(receipt_source, "result receipt")
    diagnostics: list[str] = []
    if receipt.get("schema_version") != RESULT_SCHEMA_VERSION:
        diagnostics.append(f"schema_version must be {RESULT_SCHEMA_VERSION!r}")
    if receipt.get("suite_id") != SUITE_ID or receipt.get("role") != ROLE:
        diagnostics.append("receipt must identify unix_ctf_train training")
    if receipt.get("evaluation_role") in {"primary_eval", "receipt_proven_heldout"} or receipt.get("primary_eval") is True:
        diagnostics.append("Unix-CTF training receipt cannot be promoted to primary evaluation")
    if receipt.get("claim_scope", TRAINING_ONLY_CLAIM) != TRAINING_ONLY_CLAIM:
        diagnostics.append("claim_scope must remain unix_ctf_training_only")
    source = receipt.get("source_identity", receipt.get("source"))
    if not isinstance(source, Mapping):
        diagnostics.append("receipt source_identity is required")
    else:
        license_record = source.get("license") if isinstance(source.get("license"), Mapping) else {}
        fields = {
            "id": source.get("id", source.get("source_id")),
            "url": source.get("url", source.get("source_url")),
            "revision": source.get("revision", source.get("immutable_revision")),
            "license_spdx": source.get("license_spdx", license_record.get("spdx")),
            "license_text_sha256": source.get("license_text_sha256", license_record.get("text_sha256")),
        }
        for field, expected in boundary["source_identity"].items():
            if fields.get(field) != expected:
                diagnostics.append(f"receipt source_identity.{field} differs from boundary")
    procedural = receipt.get("procedural_generation")
    expected_procedural = boundary["procedural_generation"]
    if not isinstance(procedural, Mapping):
        diagnostics.append("receipt procedural_generation is required")
    else:
        for field in ("generator_id", "generator_revision", "seed", "seed_sha256", "task_id_manifest_sha256", "parameters_sha256"):
            if procedural.get(field) != expected_procedural.get(field):
                diagnostics.append(f"receipt procedural_generation.{field} differs from boundary")
        try:
            receipt_ids = _ids("receipt procedural_generation.task_ids", procedural.get("task_ids"))
            if tuple(receipt_ids) != tuple(expected_procedural["task_ids"]):
                diagnostics.append("receipt procedural task_ids differ from boundary")
        except UnixCtfTrainBoundaryError as exc:
            diagnostics.extend(exc.diagnostics)
    disjoint_receipt = receipt.get("disjointness_receipt")
    expected_disjoint = boundary["disjointness"]
    if not isinstance(disjoint_receipt, Mapping):
        diagnostics.append("disjointness_receipt is required")
    else:
        if disjoint_receipt.get("binaryaudit_eval_task_id_manifest_sha256") != expected_disjoint["binaryaudit_eval"]["task_id_manifest_sha256"]:
            diagnostics.append("receipt BinaryAudit task hash differs from boundary")
        if disjoint_receipt.get("primary_eval_task_id_manifest_sha256") != expected_disjoint["primary_evals"]["task_id_manifest_sha256"]:
            diagnostics.append("receipt primary-evaluation task hash differs from boundary")
        if disjoint_receipt.get("verified") is not True or disjoint_receipt.get("overlap_count") != 0:
            diagnostics.append("receipt disjointness must be verified with zero overlap")
    environment = receipt.get("native_environment_receipt")
    expected_environment = boundary["native_environment"]
    if not isinstance(environment, Mapping):
        diagnostics.append("native_environment_receipt is required")
    else:
        for field in ("container_digest", "environment_manifest_sha256"):
            if environment.get(field) != expected_environment.get(field):
                diagnostics.append(f"receipt environment {field} differs from boundary")
        sandbox = environment.get("shell_sandbox")
        expected_sandbox = expected_environment.get("shell_sandbox", {})
        if not isinstance(sandbox, Mapping):
            diagnostics.append("receipt shell_sandbox is required")
        else:
            for field in ("enabled", "network", "filesystem", "policy_sha256", "limits_sha256"):
                if sandbox.get(field) != expected_sandbox.get(field):
                    diagnostics.append(f"receipt shell_sandbox.{field} differs from boundary")
    artifact = receipt.get("artifact_receipt")
    expected_artifact = expected_environment.get("artifact_contract", {})
    if not isinstance(artifact, Mapping):
        diagnostics.append("artifact_receipt is required")
    else:
        if artifact.get("manifest_sha256", artifact.get("artifact_manifest_sha256")) != expected_artifact.get("manifest_sha256"):
            diagnostics.append("artifact receipt manifest differs from artifact contract")
        paths = artifact.get("paths", artifact.get("artifacts"))
        if isinstance(paths, (str, bytes, bytearray)) or not paths:
            diagnostics.append("artifact_receipt.paths must be non-empty")
    verifier = receipt.get("verifier_receipt")
    expected_verifier = expected_environment.get("verifier_contract", {})
    if not isinstance(verifier, Mapping):
        diagnostics.append("verifier_receipt is required")
    else:
        for field in ("name", "revision", "receipt_schema"):
            if verifier.get(field) != expected_verifier.get(field):
                diagnostics.append(f"verifier receipt {field} differs from boundary")
    tracking = _validate_tracking(receipt, str(expected_artifact.get("manifest_sha256", "")), diagnostics)
    budget = _validate_budget_receipt(receipt, boundary["budget_gate"], diagnostics)
    tinker_cost = tracking.get("tinker", {}).get("cost_usd", 0.0)
    if tinker_cost > budget.get("projected_next_cost_usd", 0.0):
        diagnostics.append("Tinker cost exceeds budget receipt projected_next_cost_usd")
    if tinker_cost + budget.get("spent_usd", 0.0) > boundary["budget_gate"].get(
        "operational_cap_usd", 0.0
    ):
        diagnostics.append("Tinker cost plus spent budget exceeds operational cap")
    if not _successful_status(receipt.get("status")):
        diagnostics.append("training receipt status must be completed/success")
    metrics = receipt.get("metrics", {})
    if metrics:
        if not isinstance(metrics, Mapping):
            diagnostics.append("metrics must be a mapping when present")
        else:
            for name, value in metrics.items():
                if not isinstance(name, str) or not name.strip():
                    diagnostics.append("metric names must be non-empty strings")
                else:
                    try:
                        _finite(f"metrics.{name}", value)
                    except UnixCtfTrainBoundaryError as exc:
                        diagnostics.extend(exc.diagnostics)
    if diagnostics:
        raise UnixCtfTrainBoundaryError(diagnostics)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "admissible_training_receipt",
        "suite_id": SUITE_ID,
        "role": ROLE,
        "training_only": True,
        "primary_eval": False,
        "primary_eval_claim_permitted": False,
        "binaryaudit_substitution_permitted": False,
        "claim_scope": TRAINING_ONLY_CLAIM,
        "source_identity": boundary["source_identity"],
        "procedural_generation": boundary["procedural_generation"],
        "disjointness": boundary["disjointness"],
        "native_environment": boundary["native_environment"],
        "tracking_receipts": tracking,
        "budget_receipt": budget,
        "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
    }


validate_unix_ctf_receipt = validate_unix_ctf_training_receipt


__all__ = [
    "AUTHORITATIVE_SOURCE_ID",
    "AUTHORITATIVE_SOURCE_URL",
    "BINARYAUDIT_SUITE_ID",
    "BOUNDARY_SCHEMA_VERSION",
    "BoundaryValidationError",
    "DEFAULT_MAXIMUM_USD",
    "DEFAULT_OPERATIONAL_CAP_USD",
    "DEFAULT_PROVIDER",
    "DEFAULT_RESERVE_USD",
    "PRIMARY_EVAL_REGISTRY_ID",
    "RESULT_SCHEMA_VERSION",
    "ROLE",
    "SUITE_ID",
    "TRAINING_ONLY_CLAIM",
    "UnixCtfTrainBoundaryError",
    "canonical_sha256",
    "task_id_manifest_sha256",
    "validate_unix_ctf_receipt",
    "validate_unix_ctf_training_boundary",
    "validate_unix_ctf_training_receipt",
]

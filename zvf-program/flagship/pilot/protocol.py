from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
FLAGSHIP_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = FLAGSHIP_ROOT / "pilot_preregistration.json"
CONTROL_PLANE_SOURCES = (
    "zvf-program/flagship/pilot/protocol.py",
    "zvf-program/flagship/pilot/plan_screening.py",
    "zvf-program/flagship/pilot/replay.py",
    "zvf-program/flagship/pilot/objective.py",
    "zvf-program/flagship/pilot/test_stack_differential.py",
    "zvf-program/flagship/pilot/artifacts.py",
    "zvf-program/flagship/pilot/training.py",
    "zvf-program/flagship/pilot/remote_core.py",
    "zvf-program/flagship/pilot/flops.py",
    "zvf-program/flagship/pilot/remote_unit.py",
    "zvf-program/flagship/pilot/checkpointing.py",
    "zvf-program/flagship/pilot/evaluation.py",
    "zvf-program/flagship/pilot/remote_training.py",
    "zvf-program/flagship/pilot/bootstrap.py",
    "zvf-program/flagship/pilot/launcher.py",
    "zvf-program/flagship/pilot/verifier.py",
    "zvf-program/flagship/pilot/supervisor.py",
    "zvf-program/flagship/pilot/analysis.py",
)

CONDITION_ORDER = (
    "intended_full",
    "native_trl",
    "epsilon_only",
    "reduction_only",
)
REGIME_ORDER = ("balanced_equal_length", "filtered_variable_length")

CONDITION_FACTORS: Mapping[str, Mapping[str, str]] = {
    "intended_full": {
        "advantage": "selected_rows_sample_std_no_epsilon",
        "selection": "remove_unselected_rows_before_advantage_and_loss",
        "reduction": "mean_of_per_completion_masked_token_means",
        "importance_ratio": "canonical_grpo_token_level",
    },
    "native_trl": {
        "advantage": "trl_1_2_0_all_rows_sample_std_plus_1e-4",
        "selection": "retain_unselected_rows_with_zero_completion_mask",
        "reduction": "trl_1_2_0_dapo_global_active_token_mean",
        "importance_ratio": "trl_1_2_0_dapo_token_level_asymmetric_clip",
    },
    "epsilon_only": {
        "advantage": "selected_rows_sample_std_plus_1e-4",
        "selection": "remove_unselected_rows_before_advantage_and_loss",
        "reduction": "mean_of_per_completion_masked_token_means",
        "importance_ratio": "canonical_grpo_token_level",
    },
    "reduction_only": {
        "advantage": "selected_rows_sample_std_no_epsilon",
        "selection": "retain_unselected_rows_with_zero_completion_mask",
        "reduction": "trl_1_2_0_dapo_global_active_token_mean",
        "importance_ratio": "canonical_grpo_token_level",
    },
}

FORBIDDEN_E1_ID_FRAGMENTS = (
    "tinker-rl-lab-e1",
    "e1__",
    "e1-con",
    "e1-confirmatory",
)


class ProtocolError(RuntimeError):
    """The frozen pilot protocol is internally inconsistent or stale."""


class AuthorizationError(RuntimeError):
    """The protocol does not authorize external compute allocation."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repo_path(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    if REPO_ROOT.resolve() not in path.parents and path != REPO_ROOT.resolve():
        raise ProtocolError(f"source binding escapes repository: {relative}")
    return path


def _assert_hash(record: Mapping[str, Any], *, path_key: str = "path") -> None:
    path = _repo_path(str(record[path_key]))
    if not path.is_file():
        raise ProtocolError(f"bound source is missing: {path.relative_to(REPO_ROOT)}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise ProtocolError(
            f"bound source hash changed for {path.relative_to(REPO_ROOT)}: "
            f"expected {record['sha256']}, got {actual}"
        )


@dataclass(frozen=True, slots=True)
class PilotUnit:
    condition: str
    regime: str
    seed: int

    @property
    def unit_id(self) -> str:
        return f"fpilot__{self.condition}__{self.regime}__s{self.seed}"


@dataclass(frozen=True, slots=True)
class PilotProtocol:
    path: Path
    payload: Mapping[str, Any]
    sha256: str

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def gpu_authorized(self) -> bool:
        return bool(self.payload["authorization"]["gpu"])

    def screening_units(self) -> Iterator[PilotUnit]:
        seeds = tuple(int(seed) for seed in self.payload["runtime"]["screening_seeds"])
        for regime in REGIME_ORDER:
            for seed in seeds:
                for condition in CONDITION_ORDER:
                    yield PilotUnit(condition=condition, regime=regime, seed=seed)

    def require_gpu_authorization(self) -> None:
        if self.status != "ready_to_run" or not self.gpu_authorized:
            raise AuthorizationError(
                "pilot GPU allocation is forbidden: protocol must be ready_to_run and "
                "authorization.gpu must be true"
            )


def _validate_protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ProtocolError("unsupported pilot protocol schema")
    if payload.get("status") not in {"locked_not_authorized", "ready_to_run"}:
        raise ProtocolError("pilot status must be locked_not_authorized or ready_to_run")

    for key in ("parent_protocol", "s1_freeze"):
        _assert_hash(payload[key])
    theory = payload["theory_gate"]
    _assert_hash({"path": theory["audit_path"], "sha256": theory["audit_sha256"]})

    freeze = json.loads(_repo_path(payload["s1_freeze"]["path"]).read_text())
    if freeze.get("status") != payload["s1_freeze"]["required_status"]:
        raise ProtocolError("S1 implementation freeze no longer passes")

    if tuple(payload["conditions"]) != CONDITION_ORDER:
        raise ProtocolError("condition order or membership changed")
    if tuple(payload["regimes"]) != REGIME_ORDER:
        raise ProtocolError("regime order or membership changed")
    if set(CONDITION_FACTORS) != set(CONDITION_ORDER):
        raise ProtocolError("condition-factor map is incomplete")

    seeds = tuple(int(seed) for seed in payload["runtime"]["screening_seeds"])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ProtocolError("screening requires exactly three distinct seeds")
    confirmatory = {int(seed) for seed in payload["runtime"]["confirmatory_seeds"]}
    if set(seeds) & confirmatory:
        raise ProtocolError("screening and confirmatory seeds overlap")

    expected = len(CONDITION_ORDER) * len(REGIME_ORDER) * len(seeds)
    count = payload["unit_count"]
    if expected != 24 or count.get("total") != expected:
        raise ProtocolError("screening matrix is not exactly 24 units")

    checkpoints = tuple(int(step) for step in payload["runtime"]["checkpoint_steps"])
    if checkpoints != (20, 40, 60, 80, 100):
        raise ProtocolError("checkpoint schedule is not the frozen 20-step cadence")
    if int(payload["runtime"]["steps"]) != checkpoints[-1]:
        raise ProtocolError("final checkpoint does not match the training horizon")
    if payload["runtime"]["accelerator"] != "A100 only":
        raise ProtocolError("accelerator substitution is forbidden")

    authorized = bool(payload["authorization"]["gpu"])
    if payload["status"] == "locked_not_authorized" and authorized:
        raise ProtocolError("locked_not_authorized protocol cannot authorize a GPU")
    if payload["status"] == "ready_to_run" and not authorized:
        raise ProtocolError("ready_to_run protocol must explicitly authorize a GPU")


def load_protocol(path: Path = PROTOCOL_PATH) -> PilotProtocol:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_protocol(payload)
    return PilotProtocol(path=path, payload=payload, sha256=sha256_file(path))


def execution_blockers(protocol: PilotProtocol) -> tuple[str, ...]:
    """Return missing information that makes accelerator allocation invalid."""
    runtime = protocol.payload["runtime"]
    execution = runtime.get("execution_contract", {})
    required_execution_fields = (
        "charged_generated_token_ceiling",
        "matched_budget_horizon_rule",
        "train_split",
        "train_order_hash",
        "heldout_split",
        "heldout_n",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "generation_batch_size",
        "max_prompt_length",
        "max_completion_length",
        "learning_rate",
        "optimizer",
        "lr_scheduler_type",
        "warmup_steps",
        "lora",
        "decoding",
        "replay_gradient_steps",
        "selection_mask_algorithm",
        "flop_counter",
        "checkpoint_resume_contract",
    )
    blockers = [
        f"missing runtime.execution_contract.{field}"
        for field in required_execution_fields
        if field not in execution
    ]
    for regime in REGIME_ORDER:
        record = protocol.payload["regimes"][regime]
        if "dataset_revision" not in record:
            blockers.append(f"missing regimes.{regime}.dataset_revision")
        if "dataset_split" not in record:
            blockers.append(f"missing regimes.{regime}.dataset_split")
    if protocol.status != "ready_to_run":
        blockers.append("protocol status is not ready_to_run")
    if not protocol.gpu_authorized:
        blockers.append("authorization.gpu is false")
    return tuple(blockers)


def _assert_isolated_identifiers(values: Mapping[str, str]) -> None:
    for field, value in values.items():
        lowered = value.lower()
        for forbidden in FORBIDDEN_E1_ID_FRAGMENTS:
            if forbidden in lowered:
                raise ProtocolError(f"pilot {field} overlaps frozen E1 namespace: {value}")


def build_screening_plan(protocol: PilotProtocol, unit: PilotUnit) -> dict[str, Any]:
    if unit.condition not in CONDITION_ORDER:
        raise ProtocolError(f"unknown pilot condition: {unit.condition}")
    if unit.regime not in REGIME_ORDER:
        raise ProtocolError(f"unknown pilot regime: {unit.regime}")
    if unit.seed not in protocol.payload["runtime"]["screening_seeds"]:
        raise ProtocolError(f"seed is outside the screening matrix: {unit.seed}")

    source_bindings = {
        str(protocol.path.relative_to(REPO_ROOT)): protocol.sha256,
        protocol.payload["parent_protocol"]["path"]: protocol.payload["parent_protocol"]["sha256"],
        protocol.payload["s1_freeze"]["path"]: protocol.payload["s1_freeze"]["sha256"],
        protocol.payload["theory_gate"]["audit_path"]: protocol.payload["theory_gate"]["audit_sha256"],
        **{path: sha256_file(_repo_path(path)) for path in CONTROL_PLANE_SOURCES},
    }
    stem = f"{unit.condition}-{unit.regime}-s{unit.seed}"
    protocol_suffix = protocol.sha256[:8]
    source_suffix = canonical_fingerprint(source_bindings)[:8]
    identity_suffix = f"{protocol_suffix}-{source_suffix}"
    corpus_stem = f"{unit.regime}-s{unit.seed}-{identity_suffix}"
    identity = {
        "local_record": (
            f"zvf-program/flagship/pilot/runs/{unit.unit_id}-{identity_suffix}/full_record.json"
        ),
        "corpus_hf_repo": f"arvindcr4/tinker-rl-lab-flagship-pilot-corpus-{corpus_stem}",
        "corpus_wandb_run": f"flagship-pilot-corpus-{corpus_stem}",
        "hf_repo": f"arvindcr4/tinker-rl-lab-flagship-pilot-{stem}-{identity_suffix}",
        "wandb_group": "flagship-s1-conformance-screening",
        "wandb_run": f"flagship-pilot-{stem}-{identity_suffix}",
        "colab_session": (
            f"fpilot-{unit.condition[:4]}-{unit.regime[:4]}-s{unit.seed}-"
            f"{protocol_suffix[:4]}{source_suffix[:4]}"
        )[:40],
    }
    _assert_isolated_identifiers(identity)
    blockers = list(execution_blockers(protocol))
    plan: dict[str, Any] = {
        "schema_version": "flagship-pilot-screening-plan-v1",
        "status": "dry_run_only",
        "stage": "screening",
        "unit": {
            "id": unit.unit_id,
            "condition": unit.condition,
            "regime": unit.regime,
            "seed": unit.seed,
        },
        "protocol": {
            "path": str(protocol.path.relative_to(REPO_ROOT)),
            "sha256": protocol.sha256,
            "status": protocol.status,
            "gpu_authorized": protocol.gpu_authorized,
            "source_bundle_sha256": canonical_fingerprint(source_bindings),
        },
        "source_bindings": source_bindings,
        "runtime": {
            "stack": protocol.payload["runtime"]["primary_stack"],
            "model": protocol.payload["runtime"]["model"],
            "accelerator": protocol.payload["runtime"]["accelerator"],
            "group_size": protocol.payload["runtime"]["group_size"],
            "steps": protocol.payload["runtime"]["steps"],
            "checkpoint_steps": protocol.payload["runtime"]["checkpoint_steps"],
            "tracking": protocol.payload["runtime"]["execution_contract"]["tracking"],
        },
        "condition_factors": dict(CONDITION_FACTORS[unit.condition]),
        "regime_contract": protocol.payload["regimes"][unit.regime],
        "compute_contract": protocol.payload["matched_compute"],
        "identity": identity,
        "allocation": {
            "allowed": False,
            "command": None,
            "reason": "dry-run plans never contain an accelerator-allocation command",
        },
        "readiness": {
            "ready": not blockers,
            "authorization_blockers": blockers,
        },
    }
    plan["fingerprint"] = canonical_fingerprint(plan)
    return plan

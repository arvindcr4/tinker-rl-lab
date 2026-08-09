"""Fail-closed, metadata-first boundary for the Pavlov T3 BrowserGym suite.

This module intentionally does *not* import BrowserGym, Playwright, Tinker,
W&B, or a model client.  It provides the receipt shape and an offline fixture
that can be validated before those runtime concerns exist.  A valid fixture is
adapter/preflight evidence only; it is never a training result, an E6 result,
or portfolio evidence.

The public launch function is deliberately separate from preflight and always
returns ``authorized=False``.  Paid work belongs to the campaign owner after
the external C0, tracking, checkpoint, license, and budget gates have been
verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pavlov-browsergym-t3-v1"
SUITE_ID = "browsergym_train"
SUITE_ROLE = "train"
BENCHMARK_NAME = "browsergym"
DATASET_ID = "browsergym/miniwob"

# The local setup script pins this MiniWoB++ commit.  The package pin is the
# BrowserGym MiniWoB adapter declared in platform_tinker/atropos/pyproject.toml.
PINNED_DATASET_REVISION = (
    "miniwob-plusplus@7fd85d71a4b60325c6585396ec4f48377d049838"
)
PINNED_ENVIRONMENT_REVISION = "browsergym-miniwob==0.14.3"

# This is intentionally named as the primary held-out suite ID.  BrowserGym,
# WebArena, and MiniWoB are not WebBench and cannot fill E6.
E6_SUITE_ID = "webbench_eval"
VERIFIER_NAME = "pavlov-browsergym-t3-offline-verifier"
VERIFIER_REVISION = "offline-t3-verifier-v1"
EVIDENCE_STATUS = "OFFLINE_FIXTURE_ONLY"
CLAIM_BOUNDARY = "T3_ADAPTER_VALIDATION_ONLY"
REQUIRED_ARTIFACT_NAMES = ("browser_state", "task_success")
VERIFIER_REQUIRED_FIELDS = (
    "schema_version",
    "suite_id",
    "suite_role",
    "benchmark",
    "dataset_id",
    "dataset_revision",
    "environment_revision",
    "stateful",
    "artifact_or_side_effect",
    "task",
    "observations",
    "actions",
    "artifacts",
    "terminal",
    "status",
    "evidence_status",
    "claim_boundary",
    "e6_substitute",
    "portfolio_evidence",
    "verifier",
    "trace_hash",
)
VERIFIER_CONTRACT = {
    "name": VERIFIER_NAME,
    "revision": VERIFIER_REVISION,
    "required_fields": list(VERIFIER_REQUIRED_FIELDS),
    "required_artifacts": list(REQUIRED_ARTIFACT_NAMES),
    "stateful": True,
    "artifact_or_side_effect": True,
    "e6_substitute": False,
    "portfolio_evidence": False,
}
ALLOWED_TRACE_STATUSES = frozenset(("offline_fixture", "preflight_only"))
REQUIRED_RUNTIME_COMPONENTS = (
    "browsergym",
    "tinker",
    "wandb_online_gate",
    "model_server",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|"
    r"secret|cookie|session(?:[_-]?id)?|credential|private[_-]?key|"
    r"hf[_-]?token|wandb[_-]?(?:key|token))",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?:bearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"(?:sk|hf|wandb)[_-][A-Za-z0-9_-]{8,}|"
    r"(?:api[_-]?key|access[_-]?token|token|password|secret)\s*[:=]\s*\S+|"
    r"-----BEGIN [A-Z0-9 ]+ KEY-----)",
    re.IGNORECASE,
)


class AdapterSchemaError(ValueError):
    """Raised when a task, trajectory, or artifact violates this boundary."""


class SecretMaterialError(AdapterSchemaError):
    """Raised when raw credential-like material appears in a receipt."""


def _canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation for identity and receipts."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AdapterSchemaError(f"value is not canonicalizable: {exc}") from exc


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value without using runtime or environment state."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def assert_secret_free(value: Any, *, _path: str = "root") -> None:
    """Reject likely raw credentials instead of attempting to redact them.

    This function only inspects the supplied value.  It never reads process
    environment variables, credential stores, files, or network responses.
    Failing closed is important because a redacted-looking receipt must not be
    mistaken for proof that a secret was absent from the original trace.
    """

    if _is_mapping(value):
        for key, child in value.items():
            key_text = str(key)
            if _SECRET_KEY_RE.search(key_text):
                raise SecretMaterialError(f"secret-like field at {_path}.{key_text}")
            assert_secret_free(child, _path=f"{_path}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_secret_free(child, _path=f"{_path}[{index}]")
        return
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        raise SecretMaterialError(f"secret-like value at {_path}")


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AdapterSchemaError(f"{label} must be a lowercase SHA-256 hex digest")


def _mapping_copy(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not _is_mapping(value):
        raise AdapterSchemaError(f"{label} must be an object")
    copied = dict(value)
    assert_secret_free(copied)
    return copied


def _identity_payload(
    *,
    suite_id: str,
    split: str,
    env_id: str,
    seed: int,
    goal: str,
    initial_observation: Mapping[str, Any],
    dataset_revision: str,
    environment_revision: str,
    artifact_contract: Sequence[str],
) -> dict[str, Any]:
    return {
        "suite_id": suite_id,
        "split": split,
        "env_id": env_id,
        "seed": seed,
        "goal": goal,
        "initial_observation": dict(initial_observation),
        "dataset_revision": dataset_revision,
        "environment_revision": environment_revision,
        "artifact_contract": list(artifact_contract),
    }


def deterministic_task_id(
    *,
    env_id: str,
    seed: int,
    split: str = SUITE_ROLE,
    goal: str = "",
    initial_observation: Mapping[str, Any] | None = None,
    dataset_revision: str = PINNED_DATASET_REVISION,
    environment_revision: str = PINNED_ENVIRONMENT_REVISION,
    artifact_contract: Sequence[str] = REQUIRED_ARTIFACT_NAMES,
) -> str:
    """Build the stable task ID from all identity-bearing task metadata."""

    if initial_observation is None:
        initial_observation = {}
    payload = _identity_payload(
        suite_id=SUITE_ID,
        split=split,
        env_id=env_id,
        seed=seed,
        goal=goal,
        initial_observation=initial_observation,
        dataset_revision=dataset_revision,
        environment_revision=environment_revision,
        artifact_contract=artifact_contract,
    )
    assert_secret_free(payload)
    return f"t3-browsergym-{sha256_json(payload)[:24]}"


@dataclass(frozen=True)
class TaskSpec:
    """Pinned identity and safe initial state for one T3 training task."""

    env_id: str
    seed: int
    goal: str
    initial_observation: Mapping[str, Any]
    split: str = SUITE_ROLE
    suite_id: str = SUITE_ID
    dataset_revision: str = PINNED_DATASET_REVISION
    environment_revision: str = PINNED_ENVIRONMENT_REVISION
    artifact_contract: tuple[str, ...] = REQUIRED_ARTIFACT_NAMES
    task_id: str = ""
    task_id_hash: str = ""

    def __post_init__(self) -> None:
        if self.suite_id != SUITE_ID:
            raise AdapterSchemaError(f"task suite must be {SUITE_ID!r}")
        if self.split != SUITE_ROLE:
            raise AdapterSchemaError("T3 adapter accepts training tasks only")
        if not isinstance(self.env_id, str) or not self.env_id.strip():
            raise AdapterSchemaError("env_id must be non-empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise AdapterSchemaError("seed must be an integer")
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise AdapterSchemaError("goal must be non-empty")
        assert_secret_free(self.env_id)
        assert_secret_free(self.goal)
        if self.dataset_revision != PINNED_DATASET_REVISION:
            raise AdapterSchemaError("dataset revision is not the pinned T3 revision")
        if self.environment_revision != PINNED_ENVIRONMENT_REVISION:
            raise AdapterSchemaError(
                "environment revision is not the pinned BrowserGym revision"
            )
        if not self.artifact_contract:
            raise AdapterSchemaError("artifact_contract cannot be empty")
        if tuple(self.artifact_contract) != REQUIRED_ARTIFACT_NAMES:
            raise AdapterSchemaError(
                "T3 artifact contract must include browser_state and task_success"
            )
        initial = _mapping_copy(self.initial_observation, "initial_observation")
        object.__setattr__(self, "initial_observation", initial)
        payload = _identity_payload(
            suite_id=self.suite_id,
            split=self.split,
            env_id=self.env_id,
            seed=self.seed,
            goal=self.goal,
            initial_observation=initial,
            dataset_revision=self.dataset_revision,
            environment_revision=self.environment_revision,
            artifact_contract=self.artifact_contract,
        )
        expected_hash = sha256_json(payload)
        expected_id = f"t3-browsergym-{expected_hash[:24]}"
        if self.task_id and self.task_id != expected_id:
            raise AdapterSchemaError("task_id does not match canonical task metadata")
        if self.task_id_hash and self.task_id_hash != expected_hash:
            raise AdapterSchemaError("task_id_hash does not match canonical task metadata")
        object.__setattr__(self, "task_id", expected_id)
        object.__setattr__(self, "task_id_hash", expected_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "split": self.split,
            "env_id": self.env_id,
            "seed": self.seed,
            "goal": self.goal,
            "initial_observation": dict(self.initial_observation),
            "dataset_revision": self.dataset_revision,
            "environment_revision": self.environment_revision,
            "artifact_contract": list(self.artifact_contract),
            "task_id": self.task_id,
            "task_id_hash": self.task_id_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        data = _mapping_copy(value, "task")
        return cls(
            env_id=data["env_id"],
            seed=data["seed"],
            goal=data["goal"],
            initial_observation=data["initial_observation"],
            split=data.get("split", SUITE_ROLE),
            suite_id=data.get("suite_id", SUITE_ID),
            dataset_revision=data.get("dataset_revision", PINNED_DATASET_REVISION),
            environment_revision=data.get(
                "environment_revision", PINNED_ENVIRONMENT_REVISION
            ),
            artifact_contract=tuple(
                data.get("artifact_contract", REQUIRED_ARTIFACT_NAMES)
            ),
            task_id=data.get("task_id", ""),
            task_id_hash=data.get("task_id_hash", ""),
        )


@dataclass(frozen=True)
class ObservationRecord:
    """One stateful observation with a hashable, secret-free state summary."""

    step: int
    axtree: str
    state: Mapping[str, Any]
    last_action_error: str | None = None
    state_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise AdapterSchemaError("observation step must be a non-negative integer")
        if not isinstance(self.axtree, str) or not self.axtree.strip():
            raise AdapterSchemaError("observation axtree must be non-empty")
        state = _mapping_copy(self.state, "observation.state")
        object.__setattr__(self, "state", state)
        expected_hash = sha256_json(state)
        if self.state_hash and self.state_hash != expected_hash:
            raise AdapterSchemaError("state_hash does not match observation.state")
        object.__setattr__(self, "state_hash", expected_hash)
        assert_secret_free(self.axtree)
        if self.last_action_error is not None:
            assert_secret_free(self.last_action_error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "axtree": self.axtree,
            "state": dict(self.state),
            "state_hash": self.state_hash,
            "last_action_error": self.last_action_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationRecord":
        data = _mapping_copy(value, "observation")
        return cls(
            step=data["step"],
            axtree=data["axtree"],
            state=data["state"],
            last_action_error=data.get("last_action_error"),
            state_hash=data.get("state_hash", ""),
        )


@dataclass(frozen=True)
class ActionRecord:
    """An action linked to the state hash it produced."""

    step: int
    action: str
    next_state_hash: str
    valid: bool = True
    action_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise AdapterSchemaError("action step must be a non-negative integer")
        if not isinstance(self.action, str) or not self.action.strip():
            raise AdapterSchemaError("action must be non-empty")
        _require_sha256(self.next_state_hash, "next_state_hash")
        if not isinstance(self.valid, bool):
            raise AdapterSchemaError("action valid flag must be boolean")
        assert_secret_free(self.action)
        expected_hash = sha256_json({"step": self.step, "action": self.action})
        if self.action_hash and self.action_hash != expected_hash:
            raise AdapterSchemaError("action_hash does not match step and action")
        object.__setattr__(self, "action_hash", expected_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "valid": self.valid,
            "next_state_hash": self.next_state_hash,
            "action_hash": self.action_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionRecord":
        data = _mapping_copy(value, "action")
        return cls(
            step=data["step"],
            action=data["action"],
            next_state_hash=data["next_state_hash"],
            valid=data.get("valid", True),
            action_hash=data.get("action_hash", ""),
        )


def deterministic_artifact_digest(
    *, name: str, kind: str, metadata: Mapping[str, Any]
) -> str:
    """Hash artifact metadata, never artifact bytes or secret-bearing content."""

    return sha256_json({"name": name, "kind": kind, "metadata": dict(metadata)})


@dataclass(frozen=True)
class ArtifactRecord:
    """Metadata-only artifact receipt; raw files are intentionally not embedded."""

    name: str
    kind: str
    metadata: Mapping[str, Any]
    exists: bool = True
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AdapterSchemaError("artifact name must be non-empty")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise AdapterSchemaError("artifact kind must be non-empty")
        assert_secret_free(self.name)
        assert_secret_free(self.kind)
        if not isinstance(self.exists, bool):
            raise AdapterSchemaError("artifact exists flag must be boolean")
        metadata = _mapping_copy(self.metadata, "artifact.metadata")
        object.__setattr__(self, "metadata", metadata)
        expected_digest = deterministic_artifact_digest(
            name=self.name, kind=self.kind, metadata=metadata
        )
        if self.digest and self.digest != expected_digest:
            raise AdapterSchemaError("artifact digest does not match metadata")
        object.__setattr__(self, "digest", expected_digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "exists": self.exists,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRecord":
        data = _mapping_copy(value, "artifact")
        return cls(
            name=data["name"],
            kind=data["kind"],
            metadata=data.get("metadata", {}),
            exists=data.get("exists", True),
            digest=data.get("digest", ""),
        )


@dataclass(frozen=True)
class EpisodeTrace:
    """A deterministic observation/action/artifact trace for one task."""

    task: TaskSpec
    observations: tuple[ObservationRecord, ...]
    actions: tuple[ActionRecord, ...]
    artifacts: tuple[ArtifactRecord, ...]
    terminal: Mapping[str, Any]
    status: str = "offline_fixture"
    verifier_name: str = VERIFIER_NAME
    verifier_revision: str = VERIFIER_REVISION

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": SUITE_ID,
            "suite_role": SUITE_ROLE,
            "benchmark": BENCHMARK_NAME,
            "dataset_id": DATASET_ID,
            "dataset_revision": PINNED_DATASET_REVISION,
            "environment_revision": PINNED_ENVIRONMENT_REVISION,
            "stateful": True,
            "artifact_or_side_effect": True,
            "task": self.task.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "actions": [item.to_dict() for item in self.actions],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "terminal": dict(self.terminal),
            "status": self.status,
            "evidence_status": EVIDENCE_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
            "e6_substitute": False,
            "portfolio_evidence": False,
            "verifier": {
                "name": self.verifier_name,
                "revision": self.verifier_revision,
            },
        }
        assert_secret_free(payload)
        payload["trace_hash"] = sha256_json(payload)
        return payload


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    trace_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "metrics": dict(self.metrics),
            "trace_hash": self.trace_hash,
            "evidence_status": EVIDENCE_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
            "e6_substitute": False,
            "portfolio_evidence": False,
        }


def offline_dry_run_fixture() -> dict[str, Any]:
    """Return a deterministic, secret-free stateful fixture without launching a browser."""

    initial_state = {
        "url": "about:blank",
        "open_pages": 1,
        "button_label": "Continue",
        "clicked": False,
    }
    task = TaskSpec(
        env_id="browsergym/miniwob.click-button",
        seed=42,
        goal="Click the button labeled Continue.",
        initial_observation=initial_state,
    )
    before = ObservationRecord(
        step=0,
        axtree="button [ref=continue] Continue",
        state=initial_state,
    )
    after_state = dict(initial_state)
    after_state["clicked"] = True
    after = ObservationRecord(
        step=1,
        axtree="button [ref=continue] Continue (pressed)",
        state=after_state,
    )
    action = ActionRecord(
        step=0,
        action="click('continue')",
        next_state_hash=after.state_hash,
        valid=True,
    )
    browser_state_artifact = ArtifactRecord(
        name="browser_state",
        kind="state_summary",
        metadata={
            "task_id": task.task_id,
            "final_state_hash": after.state_hash,
            "side_effect": "button_clicked",
        },
    )
    task_success_artifact = ArtifactRecord(
        name="task_success",
        kind="verifier_observation",
        metadata={
            "task_id": task.task_id,
            "success": True,
            "final_state_hash": after.state_hash,
        },
    )
    trace = EpisodeTrace(
        task=task,
        observations=(before, after),
        actions=(action,),
        artifacts=(browser_state_artifact, task_success_artifact),
        terminal={
            "task_success": True,
            "terminated": True,
            "truncated": False,
            "final_state_hash": after.state_hash,
            "artifact_hashes": [
                browser_state_artifact.digest,
                task_success_artifact.digest,
            ],
        },
    )
    return trace.to_dict()


def verify_episode(trace: Mapping[str, Any]) -> VerificationResult:
    """Verify the receipt contract while preserving its non-evidence boundary."""

    errors: list[str] = []
    trace_hash = ""
    try:
        assert_secret_free(trace)
    except SecretMaterialError as exc:
        errors.append(str(exc))

    if not _is_mapping(trace):
        return VerificationResult(False, ("trace must be an object",), {})

    for key in VERIFIER_REQUIRED_FIELDS:
        if key not in trace:
            errors.append(f"missing required field: {key}")

    if trace.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version is not the T3 BrowserGym schema")
    if trace.get("suite_id") != SUITE_ID:
        errors.append("suite_id must be browsergym_train; E6/WebBench is not a substitute")
    if trace.get("suite_role") != SUITE_ROLE:
        errors.append("suite_role must be train")
    if trace.get("benchmark") != BENCHMARK_NAME:
        errors.append("benchmark must be browsergym")
    if trace.get("dataset_id") != DATASET_ID:
        errors.append("dataset_id must be browsergym/miniwob")
    if trace.get("dataset_revision") != PINNED_DATASET_REVISION:
        errors.append("dataset_revision is not pinned")
    if trace.get("environment_revision") != PINNED_ENVIRONMENT_REVISION:
        errors.append("environment_revision is not pinned")
    if trace.get("stateful") is not True:
        errors.append("T3 requires stateful=true")
    if trace.get("artifact_or_side_effect") is not True:
        errors.append("T3 requires artifact_or_side_effect=true")
    if trace.get("status") not in ALLOWED_TRACE_STATUSES:
        errors.append("status must remain offline_fixture or preflight_only")
    if trace.get("evidence_status") != EVIDENCE_STATUS:
        errors.append("receipt must remain offline-fixture-only evidence")
    if trace.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("claim_boundary must remain adapter-validation-only")
    if trace.get("e6_substitute") is not False:
        errors.append("BrowserGym cannot be labelled as E6/WebBench")
    if trace.get("portfolio_evidence") is not False:
        errors.append("offline T3 fixture cannot claim portfolio evidence")

    task: TaskSpec | None = None
    observations: list[ObservationRecord] = []
    actions: list[ActionRecord] = []
    artifacts: list[ArtifactRecord] = []
    try:
        task = TaskSpec.from_dict(trace["task"])
    except (KeyError, TypeError, ValueError, AdapterSchemaError) as exc:
        errors.append(f"invalid task: {exc}")
    try:
        observations = [ObservationRecord.from_dict(item) for item in trace["observations"]]
        if not observations:
            errors.append("observations cannot be empty")
    except (KeyError, TypeError, ValueError, AdapterSchemaError) as exc:
        errors.append(f"invalid observations: {exc}")
    try:
        actions = [ActionRecord.from_dict(item) for item in trace["actions"]]
    except (KeyError, TypeError, ValueError, AdapterSchemaError) as exc:
        errors.append(f"invalid actions: {exc}")
    try:
        artifacts = [ArtifactRecord.from_dict(item) for item in trace["artifacts"]]
    except (KeyError, TypeError, ValueError, AdapterSchemaError) as exc:
        errors.append(f"invalid artifacts: {exc}")

    if task is not None:
        if task.dataset_revision != trace.get("dataset_revision"):
            errors.append("task and trace dataset revisions differ")
        if task.environment_revision != trace.get("environment_revision"):
            errors.append("task and trace environment revisions differ")
        if task.suite_id != trace.get("suite_id"):
            errors.append("task and trace suite IDs differ")
        if observations and task.initial_observation != observations[0].state:
            errors.append("task initial_observation does not match observation step zero")

    if observations:
        if [item.step for item in observations] != list(range(len(observations))):
            errors.append("observation steps must be contiguous from zero")
    if len(actions) != max(0, len(observations) - 1):
        errors.append("there must be exactly one action per state transition")
    for index, action in enumerate(actions):
        if action.step != index:
            errors.append(f"action step {action.step} is not transition {index}")
        if index + 1 < len(observations) and action.next_state_hash != observations[index + 1].state_hash:
            errors.append(f"action {index} does not link to the next state hash")

    terminal = trace.get("terminal")
    if not _is_mapping(terminal):
        errors.append("terminal must be an object")
    elif observations:
        if not isinstance(terminal.get("task_success"), bool):
            errors.append("terminal.task_success must be boolean")
        if terminal.get("final_state_hash") != observations[-1].state_hash:
            errors.append("terminal final_state_hash does not match final observation")
        terminal_artifacts = terminal.get("artifact_hashes")
        if not isinstance(terminal_artifacts, list):
            errors.append("terminal.artifact_hashes must be a list")
        else:
            for digest in terminal_artifacts:
                try:
                    _require_sha256(digest, "terminal artifact hash")
                except AdapterSchemaError as exc:
                    errors.append(str(exc))
            actual_artifacts = [item.digest for item in artifacts if item.exists]
            if terminal_artifacts != actual_artifacts:
                errors.append("terminal artifact hashes do not match artifact receipts")
            actual_names = {item.name for item in artifacts if item.exists}
            missing_names = [
                name for name in REQUIRED_ARTIFACT_NAMES if name not in actual_names
            ]
            if missing_names:
                errors.append(
                    "missing required artifact receipts: " + ", ".join(missing_names)
                )
            if task is not None:
                for artifact in artifacts:
                    if artifact.exists and artifact.metadata.get("task_id") != task.task_id:
                        errors.append(
                            f"artifact {artifact.name} is not bound to the task ID"
                        )
            if terminal.get("final_state_hash"):
                browser_states = [
                    item
                    for item in artifacts
                    if item.exists and item.name == "browser_state"
                ]
                if browser_states and any(
                    item.metadata.get("final_state_hash")
                    != terminal.get("final_state_hash")
                    for item in browser_states
                ):
                    errors.append("browser_state artifact is not bound to final state")
                success_receipts = [
                    item
                    for item in artifacts
                    if item.exists and item.name == "task_success"
                ]
                if success_receipts and any(
                    item.metadata.get("success") != terminal.get("task_success")
                    for item in success_receipts
                ):
                    errors.append("task_success artifact disagrees with terminal result")

    verifier = trace.get("verifier")
    if not _is_mapping(verifier):
        errors.append("verifier must be an object")
    else:
        if verifier.get("name") != VERIFIER_NAME:
            errors.append("unexpected verifier name")
        if verifier.get("revision") != VERIFIER_REVISION:
            errors.append("unexpected verifier revision")

    supplied_hash = trace.get("trace_hash")
    if isinstance(supplied_hash, str):
        trace_hash = supplied_hash
        payload = {key: value for key, value in trace.items() if key != "trace_hash"}
        expected_trace_hash = sha256_json(payload)
        if supplied_hash != expected_trace_hash:
            errors.append("trace_hash does not match canonical receipt payload")
    else:
        errors.append("trace_hash must be a SHA-256 string")

    metrics = {
        "trajectory_valid": not errors,
        "task_success": bool(terminal.get("task_success"))
        if _is_mapping(terminal)
        else False,
        "state_integrity": bool(observations) and not any(
            "state" in error or "observation" in error for error in errors
        ),
        "artifact_integrity": bool(artifacts)
        and not any("artifact" in error for error in errors),
        "e6_substitute": False,
        "portfolio_evidence": False,
    }
    return VerificationResult(not errors, tuple(errors), metrics, trace_hash)


@dataclass(frozen=True)
class PreflightReadiness:
    """Offline fixture readiness, intentionally independent of paid launch."""

    status: str
    fixture_valid: bool
    runtime: Mapping[str, bool] = field(default_factory=dict)
    missing_runtime: tuple[str, ...] = ()
    browser_launch_attempted: bool = False
    network_allowed: bool = False
    paid_launch_authorized: bool = False
    evidence_status: str = EVIDENCE_STATUS
    claim_boundary: str = CLAIM_BOUNDARY
    e6_substitute: bool = False
    portfolio_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "suite_id": SUITE_ID,
            "schema_version": SCHEMA_VERSION,
            "fixture_valid": self.fixture_valid,
            "runtime": dict(self.runtime),
            "missing_runtime": list(self.missing_runtime),
            "browser_launch_attempted": self.browser_launch_attempted,
            "network_allowed": self.network_allowed,
            "paid_launch_authorized": self.paid_launch_authorized,
            "evidence_status": self.evidence_status,
            "claim_boundary": self.claim_boundary,
            "e6_substitute": self.e6_substitute,
            "portfolio_evidence": self.portfolio_evidence,
        }


def preflight_readiness(
    *, runtime: Mapping[str, bool] | None = None
) -> PreflightReadiness:
    """Validate only the offline fixture and explicitly report missing runtime.

    ``runtime`` is caller-provided metadata.  The adapter does not probe imports,
    launch a browser, inspect credentials, or contact a service.
    """

    supplied_runtime = {} if runtime is None else _mapping_copy(runtime, "runtime")
    normalized_runtime: dict[str, bool] = {}
    for component in REQUIRED_RUNTIME_COMPONENTS:
        value = supplied_runtime.get(component, False)
        if not isinstance(value, bool):
            raise AdapterSchemaError(f"runtime.{component} must be boolean")
        normalized_runtime[component] = value
    fixture_result = verify_episode(offline_dry_run_fixture())
    missing = tuple(
        component
        for component in REQUIRED_RUNTIME_COMPONENTS
        if not normalized_runtime[component]
    )
    return PreflightReadiness(
        status="READY_OFFLINE_FIXTURE" if fixture_result.ok else "BLOCKED_FIXTURE",
        fixture_valid=fixture_result.ok,
        runtime=normalized_runtime,
        missing_runtime=missing,
    )


@dataclass(frozen=True)
class PaidLaunchAuthorization:
    """Separate, fail-closed answer to the question of spending money."""

    authorized: bool
    status: str
    reasons: tuple[str, ...]
    operational_cap_usd: float | None = None
    network_allowed: bool = False
    online_wandb_confirmed: bool = False
    hf_checkpoint_export_confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized": self.authorized,
            "status": self.status,
            "reasons": list(self.reasons),
            "operational_cap_usd": self.operational_cap_usd,
            "network_allowed": self.network_allowed,
            "online_wandb_confirmed": self.online_wandb_confirmed,
            "hf_checkpoint_export_confirmed": self.hf_checkpoint_export_confirmed,
            "suite_id": SUITE_ID,
            "evidence_status": EVIDENCE_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
            "e6_substitute": False,
            "portfolio_evidence": False,
        }


def paid_launch_authorization(
    readiness: PreflightReadiness,
    *,
    operational_cap_usd: float | None = None,
    network_allowed: bool = False,
    online_wandb_confirmed: bool = False,
    hf_checkpoint_export_confirmed: bool = False,
) -> PaidLaunchAuthorization:
    """Return a non-authorizing decision; this adapter cannot spend money.

    Even complete caller-supplied metadata is insufficient here.  The primary
    campaign owner must independently authorize a paid run after the live C0,
    budget, license, tracking, and checkpoint gates pass.  No launch function
    exists in this module.
    """

    if not isinstance(readiness, PreflightReadiness):
        raise TypeError("readiness must be a PreflightReadiness instance")
    reasons = [
        "metadata_first_adapter_never_authorizes_paid_launch",
        "primary_campaign_owner_authorization_required",
        "no_network_browser_credentials_or_paid_calls_in_this_module",
    ]
    if not readiness.fixture_valid:
        reasons.append("offline_fixture_failed")
    if not readiness.runtime or readiness.missing_runtime:
        reasons.append("required_runtime_components_not_verified")
    if not network_allowed:
        reasons.append("network_not_allowed")
    if not online_wandb_confirmed:
        reasons.append("online_wandb_gate_not_confirmed")
    if not hf_checkpoint_export_confirmed:
        reasons.append("hf_checkpoint_export_gate_not_confirmed")
    return PaidLaunchAuthorization(
        authorized=False,
        status="NOT_AUTHORIZED",
        reasons=tuple(reasons),
        operational_cap_usd=operational_cap_usd,
        network_allowed=network_allowed,
        online_wandb_confirmed=online_wandb_confirmed,
        hf_checkpoint_export_confirmed=hf_checkpoint_export_confirmed,
    )


def _cli_payload() -> dict[str, Any]:
    readiness = preflight_readiness()
    authorization = paid_launch_authorization(readiness)
    fixture = offline_dry_run_fixture()
    verification = verify_episode(fixture)
    return {
        "fixture": fixture,
        "verification": verification.to_dict(),
        "preflight": readiness.to_dict(),
        "paid_launch": authorization.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the offline fixture/readiness report; never launch work."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the deterministic fixture and gate reports as JSON",
    )
    args = parser.parse_args(argv)
    payload = _cli_payload()
    if args.json:
        print(_canonical_json(payload))
    else:
        print(
            "T3 BrowserGym offline fixture: "
            f"{payload['verification']['ok']} "
            "(preflight only; paid launch unauthorized)"
        )
    return 0 if payload["verification"]["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised by a smoke command
    raise SystemExit(main())

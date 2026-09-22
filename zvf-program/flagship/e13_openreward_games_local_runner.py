#!/usr/bin/env python3
"""Local-side machinery for the E13 ``openreward_games_eval`` suite.

Scope and honesty boundary
--------------------------
This module does **not** produce a benchmark score and cannot be made to.
It builds the four pieces that must exist locally before any OpenReward game
result could be trusted:

1. a strict schema for ORS game task / seed-split manifests,
2. seed-separation (procedural disjointness) proof logic,
3. a game-state verifier interface with a fail-closed default implementation,
4. fail-closed local receipt emission whose ``score`` is always ``None``.

Running a game environment against a gold action proves the plumbing works.
That is ``harness_validation`` with ``is_model_score: false``; it is never a
benchmark score and this module refuses to promote it.

Upstream shape this mirrors
---------------------------
The public OpenReward game environments (``github.com/EnvCommons/<game>``)
declare their splits in source, e.g. ``EnvCommons/wordle@92bea32``::

    seed = seed_idx if split == "train" else seed_idx + 10000

so train seeds occupy ``[0, N)`` and test seeds ``[10000, 10000+N)``.
``prove_seed_separation`` checks that disjointness rather than assuming it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import base64
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

SCHEMA_VERSION = "e13-openreward-games-receipt-v1"
PROVIDER_GRANT_SCHEMA = "e13-openreward-provider-grant-v1"
E13_PROVIDER_REQUEST_SCHEMA = "e13-provider-execution-request-v1"
E13_PROVIDER_RESULT_SCHEMA = "e13-provider-signed-result-v1"
E13_TRUST_ROOT_SCHEMA = "provider-lane-trust-root-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


#: Marker stamped into anything produced from a synthetic fixture. Its presence
#: in a receipt is a hard bar to that receipt ever carrying a score.
SYNTHETIC_FIXTURE_MARKER = "SYNTHETIC-FIXTURE-NOT-A-BENCHMARK-ARTIFACT"

_ENV_NAME = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

VALID_STATUSES = ("RUNNING", "PARTIAL", "BLOCKED")


class GameManifestError(ValueError):
    """Raised when a game task / split manifest is malformed."""


class ReceiptIntegrityError(RuntimeError):
    """Raised when a receipt would claim more than the evidence supports."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GameManifestError(message)


# --------------------------------------------------------------------------
# 1. Task / seed manifest schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GameTaskSpec:
    """One ORS game task: a seeded instance of a named environment variant."""

    id: str
    env_id: str
    seed: int
    variant: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "env_id": self.env_id, "seed": self.seed, "variant": self.variant}


@dataclass(frozen=True)
class SplitManifest:
    """The full task list for one split of one environment, at a pinned revision."""

    environment: str
    split: str
    source_revision: str
    tasks: tuple[GameTaskSpec, ...]
    synthetic: bool = False

    @property
    def seeds(self) -> frozenset[int]:
        return frozenset(task.seed for task in self.tasks)

    @property
    def instance_keys(self) -> frozenset[tuple[str, int]]:
        """``(variant, seed)`` pairs — the actual unit of a procedural instance.

        Upstream reuses the same seed index across variants (``Wordle-v0_seed0``
        and ``Wordle-v0-hardcore_seed0`` both exist), so a bare seed is not an
        instance identity and must not be treated as one.
        """
        return frozenset((task.variant, task.seed) for task in self.tasks)

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task.id for task in self.tasks)

    @property
    def variants(self) -> frozenset[str]:
        return frozenset(task.variant for task in self.tasks)

    def digest(self) -> str:
        """Order-independent content hash over the task tuples."""
        rows = sorted(f"{t.id}\x1f{t.env_id}\x1f{t.seed}\x1f{t.variant}" for t in self.tasks)
        payload = "\x1e".join([self.environment, self.split, self.source_revision, *rows])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_task_spec(payload: Mapping[str, Any], *, index: int = 0) -> GameTaskSpec:
    """Parse one task mapping. Strict: unknown keys and loose types are rejected."""

    _require(isinstance(payload, Mapping), f"task[{index}] must be a mapping")
    allowed = {"id", "env_id", "seed", "variant"}
    extra = set(payload) - allowed
    _require(not extra, f"task[{index}] has unexpected keys: {sorted(extra)}")

    task_id = payload.get("id")
    env_id = payload.get("env_id")
    seed = payload.get("seed")
    variant = payload.get("variant", env_id)

    _require(
        isinstance(task_id, str) and task_id.strip() != "",
        f"task[{index}].id must be a non-empty string",
    )
    _require(
        isinstance(env_id, str) and env_id.strip() != "",
        f"task[{index}].env_id must be a non-empty string",
    )
    # bool is a subclass of int; a boolean seed is a bug, not a seed.
    _require(
        isinstance(seed, int) and not isinstance(seed, bool), f"task[{index}].seed must be an int"
    )
    _require(seed >= 0, f"task[{index}].seed must be non-negative")
    _require(
        isinstance(variant, str) and variant.strip() != "",
        f"task[{index}].variant must be a non-empty string",
    )

    return GameTaskSpec(id=task_id, env_id=env_id, seed=seed, variant=variant)


def parse_split_manifest(payload: Mapping[str, Any]) -> SplitManifest:
    """Parse a split manifest. Duplicate task ids or duplicate seeds are rejected."""

    _require(isinstance(payload, Mapping), "manifest must be a mapping")

    environment = payload.get("environment")
    split = payload.get("split")
    revision = payload.get("source_revision")
    raw_tasks = payload.get("tasks")
    synthetic = bool(payload.get("synthetic", False))

    _require(
        isinstance(environment, str) and _ENV_NAME.fullmatch(environment) is not None,
        "manifest.environment must look like 'Owner/Name'",
    )
    _require(
        isinstance(split, str) and split.strip() != "", "manifest.split must be a non-empty string"
    )
    _require(
        isinstance(revision, str) and _HEX40.fullmatch(revision) is not None,
        "manifest.source_revision must be a 40-hex git commit",
    )
    _require(
        isinstance(raw_tasks, Sequence) and not isinstance(raw_tasks, (str, bytes)),
        "manifest.tasks must be a list",
    )
    _require(len(raw_tasks) > 0, "manifest.tasks cannot be empty")

    tasks = tuple(parse_task_spec(row, index=i) for i, row in enumerate(raw_tasks))

    ids = [t.id for t in tasks]
    _require(len(ids) == len(set(ids)), "manifest.tasks contains duplicate task ids")
    # A seed is only unique within a variant: upstream emits the same seed index
    # for every variant of a game. The instance identity is (variant, seed).
    keys = [(t.variant, t.seed) for t in tasks]
    _require(
        len(keys) == len(set(keys)), "manifest.tasks contains duplicate (variant, seed) instances"
    )

    return SplitManifest(
        environment=environment,
        split=split,
        source_revision=revision,
        tasks=tasks,
        synthetic=synthetic,
    )


# --------------------------------------------------------------------------
# 2. Seed separation (procedural disjointness) proof
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedSeparationProof:
    """Evidence that an eval split shares no procedural instance with train."""

    environment: str
    train_split: str
    eval_split: str
    train_digest: str
    eval_digest: str
    train_instance_count: int
    eval_instance_count: int
    shared_instances: tuple[tuple[str, int], ...]
    shared_seeds: tuple[int, ...]
    shared_task_ids: tuple[str, ...]
    variant_coverage_matches: bool
    violations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def holds(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "train_split": self.train_split,
            "eval_split": self.eval_split,
            "train_manifest_sha256": self.train_digest,
            "eval_manifest_sha256": self.eval_digest,
            "instance_key": "(variant, seed)",
            "train_instance_count": self.train_instance_count,
            "eval_instance_count": self.eval_instance_count,
            "shared_instances": [list(k) for k in self.shared_instances],
            "shared_seeds": list(self.shared_seeds),
            "shared_task_ids": list(self.shared_task_ids),
            "variant_coverage_matches": self.variant_coverage_matches,
            "violations": list(self.violations),
            "holds": self.holds,
        }


def prove_seed_separation(train: SplitManifest, evaluation: SplitManifest) -> SeedSeparationProof:
    """Check that ``evaluation`` shares no seed or task id with ``train``.

    Disjointness is *checked*, never assumed. A proof with a non-empty
    ``violations`` tuple has ``holds is False`` and blocks receipt scoring.
    """

    violations: list[str] = []

    if train.environment != evaluation.environment:
        violations.append(
            f"environment mismatch: train={train.environment!r} eval={evaluation.environment!r}"
        )
    if train.source_revision != evaluation.source_revision:
        violations.append(
            "source_revision mismatch: both splits must be derived from one pinned revision "
            f"(train={train.source_revision}, eval={evaluation.source_revision})"
        )
    if train.split == evaluation.split:
        violations.append(f"train and eval refer to the same split name {train.split!r}")

    shared_instances = tuple(sorted(train.instance_keys & evaluation.instance_keys))
    if shared_instances:
        violations.append(
            f"{len(shared_instances)} (variant, seed) instance(s) appear in both splits: "
            f"{[list(k) for k in shared_instances[:10]]}"
        )

    # Reported for visibility. A shared bare seed across *different* variants is
    # not by itself a leak, so it is not a violation on its own.
    shared_seeds = tuple(sorted(train.seeds & evaluation.seeds))

    shared_ids = tuple(sorted(train.task_ids & evaluation.task_ids))
    if shared_ids:
        violations.append(
            f"{len(shared_ids)} task id(s) appear in both splits: {list(shared_ids[:10])}"
        )

    variant_match = train.variants == evaluation.variants
    if not variant_match:
        violations.append(
            "variant coverage differs between splits: "
            f"train-only={sorted(train.variants - evaluation.variants)} "
            f"eval-only={sorted(evaluation.variants - train.variants)}"
        )

    return SeedSeparationProof(
        environment=evaluation.environment,
        train_split=train.split,
        eval_split=evaluation.split,
        train_digest=train.digest(),
        eval_digest=evaluation.digest(),
        train_instance_count=len(train.instance_keys),
        eval_instance_count=len(evaluation.instance_keys),
        shared_instances=shared_instances,
        shared_seeds=shared_seeds,
        shared_task_ids=shared_ids,
        variant_coverage_matches=variant_match,
        violations=tuple(violations),
    )


# --------------------------------------------------------------------------
# 3. Game-state verifier interface
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeRecord:
    """One rollout against one seeded task."""

    task: GameTaskSpec
    steps: int
    finished: bool
    terminal_reward: float | None
    #: Hash of the observation stream, so replays can be compared byte-for-byte.
    transcript_sha256: str | None = None


@dataclass(frozen=True)
class VerifierOutcome:
    accepted: bool
    reward: float | None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reward": self.reward, "reasons": list(self.reasons)}


class GameStateVerifier(Protocol):
    """Adjudicates a single episode. Implementations must be side-effect free."""

    name: str

    def verify(self, episode: EpisodeRecord) -> VerifierOutcome: ...


@dataclass(frozen=True)
class ProgrammaticRewardVerifier:
    """Fail-closed verifier for programmatically graded game environments.

    Rejects — rather than coerces — anything it cannot adjudicate: an episode
    that never terminated, a missing reward, a reward outside ``[lo, hi]``, or
    a non-finite reward. The OpenReward game environments grade programmatically
    (no LLM grader), so a reward that falls outside the declared band means the
    harness is wrong, not that the model scored oddly.
    """

    name: str = "programmatic-reward"
    reward_low: float = 0.0
    reward_high: float = 1.0
    require_finished: bool = True

    def verify(self, episode: EpisodeRecord) -> VerifierOutcome:
        reasons: list[str] = []

        if self.require_finished and not episode.finished:
            reasons.append("episode did not reach a terminal state")
        if episode.steps <= 0:
            reasons.append("episode recorded no steps")

        reward = episode.terminal_reward
        if reward is None:
            reasons.append("no terminal reward was recorded")
        else:
            if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                reasons.append(f"terminal reward has non-numeric type {type(reward).__name__}")
                reward = None
            elif reward != reward or reward in (float("inf"), float("-inf")):
                reasons.append("terminal reward is not finite")
                reward = None
            elif not (self.reward_low <= reward <= self.reward_high):
                reasons.append(
                    f"terminal reward {reward} outside declared band "
                    f"[{self.reward_low}, {self.reward_high}]"
                )
                reward = None

        if reasons:
            return VerifierOutcome(accepted=False, reward=None, reasons=tuple(reasons))
        return VerifierOutcome(accepted=True, reward=float(reward), reasons=())


def verify_episodes(
    verifier: GameStateVerifier,
    episodes: Iterable[EpisodeRecord],
) -> tuple[VerifierOutcome, ...]:
    return tuple(verifier.verify(episode) for episode in episodes)


# --------------------------------------------------------------------------
# 4. Fail-closed receipt emission
# --------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_receipt(
    *,
    lane: str,
    suite: str,
    status: str,
    separation: SeedSeparationProof | None,
    outcomes: Sequence[VerifierOutcome] = (),
    run_kind: str = "harness_validation",
    is_model_score: bool = False,
    synthetic: bool = False,
    evidence: Mapping[str, Any] | None = None,
    blockers: Sequence[str] = (),
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a local E13 receipt that can never carry a benchmark score.

    Local outcomes can demonstrate seed separation or verifier plumbing, but
    caller-supplied rewards are not native OpenReward suite evidence. Exact
    completion therefore requires a separately implemented provider-signed
    result collector bound to the provider grant, suite, trust root, immutable
    task manifest, runtime, native grader, model, metric, and completion time.
    That collector is a distinct intake path below; this local receipt builder
    never delegates to it, so every local/public/harness receipt is permanently
    scoreless.
    """

    if status not in VALID_STATUSES:
        raise ReceiptIntegrityError(f"status must be one of {VALID_STATUSES}, got {status!r}")
    if is_model_score and run_kind != "model_rollout":
        raise ReceiptIntegrityError(
            f"is_model_score=True is incompatible with run_kind={run_kind!r}; "
            "harness validation is never a model score"
        )
    if is_model_score and synthetic:
        raise ReceiptIntegrityError("a synthetic fixture can never carry a model score")

    gate_failures: list[str] = []
    if run_kind != "model_rollout" or not is_model_score:
        gate_failures.append(f"run_kind={run_kind!r} is not a scored model rollout")
    if synthetic:
        gate_failures.append("run is synthetic")
    if separation is None:
        gate_failures.append("no seed-separation proof supplied")
    elif not separation.holds:
        gate_failures.append("seed-separation proof does not hold")
    if not outcomes:
        gate_failures.append("no verified episodes")
    elif any(not o.accepted for o in outcomes):
        rejected = sum(1 for o in outcomes if not o.accepted)
        gate_failures.append(f"{rejected}/{len(outcomes)} episodes rejected by the verifier")
    if status == "BLOCKED":
        gate_failures.append("status is BLOCKED")

    gate_failures.append(
        "local verifier outcomes are not a provider-signed native OpenReward result"
    )
    score: float | None = None

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "lane": lane,
        "suite": suite,
        "generated_at_utc": generated_at or _utc_now(),
        "status": status,
        "run_kind": run_kind,
        "is_model_score": bool(is_model_score),
        "score": score,
        "score_withheld_because": gate_failures,
        "episodes_verified": len(outcomes),
        "episodes_accepted": sum(1 for o in outcomes if o.accepted),
        "seed_separation": separation.as_dict() if separation is not None else None,
        "evidence": dict(evidence or {}),
        "blockers": list(blockers),
    }
    if synthetic:
        receipt["synthetic_fixture"] = SYNTHETIC_FIXTURE_MARKER
    return receipt


def emit_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Write a receipt to disk after re-checking its own integrity invariant."""

    if receipt.get("score") is not None:
        raise ReceiptIntegrityError(
            "local E13 receipts cannot carry a benchmark score; a provider-signed "
            "native result collector is required"
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_manifest(path: str) -> SplitManifest:
    return parse_split_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def _load_e13_trust_root(trust_root: Mapping[str, Any] | str | Path | None) -> dict[str, str]:
    try:
        raw = (
            json.loads(Path(trust_root).read_text())
            if isinstance(trust_root, (str, Path))
            else trust_root
        )
        required = {"schema_version", "lane", "suite_id", "provider", "key_id", "public_key_hex"}
        if (
            not isinstance(raw, Mapping)
            or set(raw) != required
            or not all(isinstance(raw[key], str) for key in required)
        ):
            raise ValueError("schema")
        root = {key: str(value) for key, value in raw.items()}
        if (
            (root["schema_version"], root["lane"], root["suite_id"], root["provider"])
            != (E13_TRUST_ROOT_SCHEMA, "E13", "openreward_games_eval", "OpenReward")
            or not root["key_id"]
            or not re.fullmatch(r"[0-9a-f]{64}", root["public_key_hex"])
        ):
            raise ValueError("identity")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(root["public_key_hex"]))
        root["document_sha256"] = hashlib.sha256(canonical_json(raw).encode()).hexdigest()
        root["key_fingerprint"] = hashlib.sha256(bytes.fromhex(root["public_key_hex"])).hexdigest()
        return root
    except Exception as exc:
        raise ValueError("explicit valid E13 provider trust root is required") from exc


def _exact_mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} has missing or unknown fields")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a 64-hex SHA-256 digest")
    return value


def _require_revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 40-hex revision")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _parse_e13_utc(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _verify_e13_signature(
    payload: Mapping[str, Any], trust_root: Mapping[str, Any] | str | Path | None
) -> dict[str, str]:
    try:
        root = _load_e13_trust_root(trust_root)
        if payload.get("signature_key_id") != root["key_id"]:
            raise ValueError("wrong lane key")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(root["public_key_hex"])).verify(
            base64.b64decode(str(payload["signature"]), validate=True),
            canonical_json({k: v for k, v in payload.items() if k != "signature"}).encode(),
        )
        issued = _parse_e13_utc(payload.get("issued_at"), "issued_at")
        expires = _parse_e13_utc(payload.get("expires_at"), "expires_at")
        now = datetime.now(timezone.utc)
        if issued > now or expires <= now or expires <= issued:
            raise ValueError("invalid validity window")
        return root
    except Exception as exc:
        raise ValueError("provider signature or validity window is invalid") from exc


def validate_provider_suite_grant(
    grant: Mapping[str, Any],
    train: SplitManifest,
    evaluation: SplitManifest,
    *,
    trust_root: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Reject locally selected Wordle manifests until OpenReward binds a suite."""
    required = {
        "schema_version",
        "suite_id",
        "provider",
        "issued_at",
        "expires_at",
        "grant_id",
        "license",
        "deployment",
        "runtime",
        "grader",
        "train_manifest_sha256",
        "heldout_manifest_sha256",
        "train_task_count",
        "heldout_task_count",
        "signature_key_id",
        "signature",
    }
    if not isinstance(grant, Mapping) or set(grant) != required:
        raise ValueError("provider grant has missing or unknown fields")
    data = dict(grant)
    if (
        data["schema_version"] != PROVIDER_GRANT_SCHEMA
        or data["suite_id"] != "openreward_games_eval"
    ):
        raise ValueError("provider grant schema, suite, or trust root is invalid")
    root = _verify_e13_signature(data, trust_root)
    if train.synthetic or evaluation.synthetic or data.get("provider") != "OpenReward":
        raise ValueError("local fixtures and non-provider grants are rejected")
    for key in ("license", "deployment", "runtime", "grader"):
        block = data[key]
        if (
            not isinstance(block, Mapping)
            or block.get("approved") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", str(block.get("sha256", "")))
            or (
                key in {"deployment", "runtime"}
                and _HEX40.fullmatch(str(block.get("revision", ""))) is None
            )
        ):
            raise ValueError(f"provider grant {key} is not approved and pinned")
    if (
        data["train_manifest_sha256"] != train.digest()
        or data["heldout_manifest_sha256"] != evaluation.digest()
        or not data["deployment"].get("revision") == evaluation.source_revision
        or not data["runtime"].get("revision") == evaluation.source_revision
        or _require_positive_int(data["train_task_count"], "grant.train_task_count")
        != len(train.tasks)
        or _require_positive_int(data["heldout_task_count"], "grant.heldout_task_count")
        != len(evaluation.tasks)
    ):
        raise ValueError("provider deployment/heldout binding does not match manifests")
    data["grant_fingerprint"] = hashlib.sha256(canonical_json(data).encode()).hexdigest()
    data["trust_root_sha256"] = root["document_sha256"]
    data["trust_root_key_fingerprint"] = root["key_fingerprint"]
    return data


def build_e13_provider_execution_request(
    grant: Mapping[str, Any],
    train: SplitManifest,
    evaluation: SplitManifest,
    *,
    model_revision: str,
    trust_root: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Build a canonical, no-spend E13 provider request from exact suite assets.

    This function only creates a deterministic instruction for the provider; it
    neither starts Tinker nor contacts OpenReward. Its score is permanently
    ``None`` and its zero spend is explicit, so it cannot be mistaken for a
    local evaluation or a launch authorization.
    """
    verified = validate_provider_suite_grant(grant, train, evaluation, trust_root=trust_root)
    separation = prove_seed_separation(train, evaluation)
    if not separation.holds:
        raise ValueError("provider execution request requires separated train/eval manifests")
    return {
        "schema_version": E13_PROVIDER_REQUEST_SCHEMA,
        "status": "PROVIDER_EXECUTION_REQUIRED",
        "suite_id": "openreward_games_eval",
        "provider": "OpenReward",
        "paid_launch_allowed": False,
        "spent_usd": 0.0,
        "score": None,
        "grant_fingerprint": verified["grant_fingerprint"],
        "trust_root_sha256": verified["trust_root_sha256"],
        "trust_root_key_fingerprint": verified["trust_root_key_fingerprint"],
        "train_manifest_sha256": train.digest(),
        "heldout_manifest_sha256": evaluation.digest(),
        "train_task_count": len(train.tasks),
        "heldout_task_count": len(evaluation.tasks),
        "deployment": dict(verified["deployment"]),
        "runtime": dict(verified["runtime"]),
        "grader": dict(verified["grader"]),
        "model_revision": _require_revision(model_revision, "model_revision"),
        "wandb_before_tinker": True,
        "immutable_hf_checkpoint_required": True,
    }


def _validate_result_receipts(value: Any) -> dict[str, dict[str, str]]:
    receipts = _exact_mapping(value, {"wandb", "tinker", "hugging_face"}, "result.receipts")
    wandb = _exact_mapping(
        receipts["wandb"], {"project", "entity", "run_id", "run_url"}, "result.receipts.wandb"
    )
    tinker = _exact_mapping(
        receipts["tinker"], {"job_id", "receipt_sha256"}, "result.receipts.tinker"
    )
    hugging_face = _exact_mapping(
        receipts["hugging_face"],
        {"repo_id", "commit", "receipt_sha256"},
        "result.receipts.hugging_face",
    )
    if (
        not all(isinstance(wandb[field], str) and wandb[field] for field in wandb)
        or not str(wandb["run_url"]).startswith("https://wandb.ai/")
        or re.fullmatch(r"[0-9A-Za-z]{8}", str(wandb["run_id"])) is None
        or not isinstance(tinker["job_id"], str)
        or re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", tinker["job_id"]) is None
    ):
        raise ValueError("result W&B/Tinker receipts are malformed")
    return {
        "wandb": {field: str(wandb[field]) for field in wandb},
        "tinker": {
            "job_id": str(tinker["job_id"]),
            "receipt_sha256": _require_sha256(
                tinker["receipt_sha256"], "result.tinker.receipt_sha256"
            ),
        },
        "hugging_face": {
            "repo_id": str(hugging_face["repo_id"]),
            "commit": _require_revision(hugging_face["commit"], "result.hugging_face.commit"),
            "receipt_sha256": _require_sha256(
                hugging_face["receipt_sha256"], "result.hugging_face.receipt_sha256"
            ),
        },
    }


def collect_e13_provider_signed_result(
    grant: Mapping[str, Any],
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    train: SplitManifest,
    evaluation: SplitManifest,
    *,
    trust_root: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Validate the sole E13 native-score path: an exact provider-signed result."""
    verified = validate_provider_suite_grant(grant, train, evaluation, trust_root=trust_root)
    request_required = {
        "schema_version",
        "status",
        "suite_id",
        "provider",
        "paid_launch_allowed",
        "spent_usd",
        "score",
        "grant_fingerprint",
        "trust_root_sha256",
        "trust_root_key_fingerprint",
        "train_manifest_sha256",
        "heldout_manifest_sha256",
        "train_task_count",
        "heldout_task_count",
        "deployment",
        "runtime",
        "grader",
        "model_revision",
        "wandb_before_tinker",
        "immutable_hf_checkpoint_required",
    }
    if not isinstance(request, Mapping) or set(request) != request_required:
        raise ValueError("provider execution request has missing or unknown fields")
    canonical_request = build_e13_provider_execution_request(
        grant,
        train,
        evaluation,
        model_revision=request.get("model_revision"),
        trust_root=trust_root,
    )
    if dict(request) != canonical_request:
        raise ValueError("provider execution request is not the exact validated no-spend request")

    result_required = {
        "schema_version",
        "suite_id",
        "provider",
        "issued_at",
        "expires_at",
        "signature_key_id",
        "signature",
        "grant_fingerprint",
        "request_fingerprint",
        "train_manifest_sha256",
        "heldout_manifest_sha256",
        "train_task_count",
        "heldout_task_count",
        "license_sha256",
        "deployment_sha256",
        "deployment_revision",
        "runtime_sha256",
        "runtime_revision",
        "grader_sha256",
        "model_revision",
        "checkpoint_sha256",
        "artifact_sha256",
        "wandb_before_tinker",
        "receipts",
        "metric",
        "score",
        "completed_at",
    }
    if not isinstance(result, Mapping) or set(result) != result_required:
        raise ValueError("provider signed result has missing or unknown fields")
    payload = dict(result)
    root = _verify_e13_signature(payload, trust_root)
    receipts = _validate_result_receipts(payload["receipts"])
    score = payload["score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise ValueError("result.score must be a finite number")
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError("result.score must be bounded to [0, 1]")
    request_fingerprint = hashlib.sha256(canonical_json(canonical_request).encode()).hexdigest()
    if (
        payload["schema_version"] != E13_PROVIDER_RESULT_SCHEMA
        or payload["suite_id"] != "openreward_games_eval"
        or payload["provider"] != "OpenReward"
        or payload["issued_at"] != verified["issued_at"]
        or payload["expires_at"] != verified["expires_at"]
        or root["document_sha256"] != verified["trust_root_sha256"]
        or root["key_fingerprint"] != verified["trust_root_key_fingerprint"]
        or payload["grant_fingerprint"] != verified["grant_fingerprint"]
        or payload["request_fingerprint"] != request_fingerprint
        or payload["train_manifest_sha256"] != train.digest()
        or payload["heldout_manifest_sha256"] != evaluation.digest()
        or _require_positive_int(payload["train_task_count"], "result.train_task_count")
        != len(train.tasks)
        or _require_positive_int(payload["heldout_task_count"], "result.heldout_task_count")
        != len(evaluation.tasks)
        or _require_sha256(payload["license_sha256"], "result.license_sha256")
        != verified["license"]["sha256"]
        or _require_sha256(payload["deployment_sha256"], "result.deployment_sha256")
        != verified["deployment"]["sha256"]
        or _require_revision(payload["deployment_revision"], "result.deployment_revision")
        != verified["deployment"]["revision"]
        or _require_sha256(payload["runtime_sha256"], "result.runtime_sha256")
        != verified["runtime"]["sha256"]
        or _require_revision(payload["runtime_revision"], "result.runtime_revision")
        != verified["runtime"]["revision"]
        or _require_sha256(payload["grader_sha256"], "result.grader_sha256")
        != verified["grader"]["sha256"]
        or _require_revision(payload["model_revision"], "result.model_revision")
        != canonical_request["model_revision"]
        or _require_sha256(payload["checkpoint_sha256"], "result.checkpoint_sha256")
        != payload["checkpoint_sha256"]
        or _require_sha256(payload["artifact_sha256"], "result.artifact_sha256")
        != payload["artifact_sha256"]
        or receipts["hugging_face"]["commit"] != canonical_request["model_revision"]
        or payload["wandb_before_tinker"] is not True
        or payload["metric"] != "openreward_games_score"
    ):
        raise ValueError("provider result bindings, receipts, or native metric are invalid")
    completed = _parse_e13_utc(payload["completed_at"], "result.completed_at")
    issued = _parse_e13_utc(verified["issued_at"], "grant.issued_at")
    expires = _parse_e13_utc(verified["expires_at"], "grant.expires_at")
    if completed < issued or completed > expires or completed > datetime.now(timezone.utc):
        raise ValueError("result.completed_at is outside the signed validity window")
    return {
        "status": "COMPLETE",
        "suite_id": "openreward_games_eval",
        "score": float(score),
        "metric": payload["metric"],
        "task_count": payload["heldout_task_count"],
        "artifact_sha256": payload["artifact_sha256"],
        "grant_fingerprint": verified["grant_fingerprint"],
        "request_fingerprint": request_fingerprint,
        "provider_signed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="e13_openreward_games_local_runner",
        description="Validate OpenReward game split manifests and emit a fail-closed receipt.",
    )
    parser.add_argument(
        "--train-manifest", required=True, help="JSON split manifest for the train split"
    )
    parser.add_argument(
        "--eval-manifest", required=True, help="JSON split manifest for the held-out split"
    )
    parser.add_argument("--out", required=True, help="path to write the receipt JSON")
    parser.add_argument("--lane", default="E13 openreward_games_eval")
    parser.add_argument("--suite", default="openreward_games_eval")
    parser.add_argument("--mode", choices=("local", "request", "collect"), default="local")
    parser.add_argument("--provider-grant", help="signed OpenReward suite grant JSON")
    parser.add_argument("--trust-root", help="lane-scoped OpenReward trust-root JSON")
    parser.add_argument("--model-revision", help="immutable 40-hex model revision")
    parser.add_argument("--request", help="canonical provider execution request JSON")
    parser.add_argument("--result", help="provider-signed native result JSON")
    args = parser.parse_args(argv)

    train = _load_manifest(args.train_manifest)
    evaluation = _load_manifest(args.eval_manifest)
    if args.mode in {"request", "collect"}:
        if not args.provider_grant or not args.trust_root:
            parser.error(f"{args.mode} mode requires --provider-grant and --trust-root")
        grant = json.loads(Path(args.provider_grant).read_text(encoding="utf-8"))
        if args.mode == "request":
            if not args.model_revision:
                parser.error("request mode requires --model-revision")
            payload = build_e13_provider_execution_request(
                grant,
                train,
                evaluation,
                model_revision=args.model_revision,
                trust_root=args.trust_root,
            )
        else:
            if not args.request or not args.result:
                parser.error("collect mode requires --request and --result")
            payload = collect_e13_provider_signed_result(
                grant,
                json.loads(Path(args.request).read_text(encoding="utf-8")),
                json.loads(Path(args.result).read_text(encoding="utf-8")),
                train,
                evaluation,
                trust_root=args.trust_root,
            )
        Path(args.out).write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    separation = prove_seed_separation(train, evaluation)

    receipt = build_receipt(
        lane=args.lane,
        suite=args.suite,
        status="PARTIAL" if separation.holds else "BLOCKED",
        separation=separation,
        outcomes=(),
        run_kind="manifest_validation",
        is_model_score=False,
        synthetic=train.synthetic or evaluation.synthetic,
        evidence={
            "train_manifest": args.train_manifest,
            "eval_manifest": args.eval_manifest,
        },
        blockers=list(separation.violations),
    )
    emit_receipt(args.out, receipt)
    print(
        json.dumps(
            {
                "seed_separation_holds": separation.holds,
                "violations": list(separation.violations),
                "receipt": args.out,
                "score": receipt["score"],
            },
            indent=2,
        )
    )
    return 0 if separation.holds else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

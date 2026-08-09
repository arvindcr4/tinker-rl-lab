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
4. fail-closed receipt emission whose ``score`` is ``None`` unless a real
   model rollout is supplied *and* every gate below passes.

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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

SCHEMA_VERSION = "e13-openreward-games-receipt-v1"

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

    _require(isinstance(task_id, str) and task_id.strip() != "", f"task[{index}].id must be a non-empty string")
    _require(isinstance(env_id, str) and env_id.strip() != "", f"task[{index}].env_id must be a non-empty string")
    # bool is a subclass of int; a boolean seed is a bug, not a seed.
    _require(isinstance(seed, int) and not isinstance(seed, bool), f"task[{index}].seed must be an int")
    _require(seed >= 0, f"task[{index}].seed must be non-negative")
    _require(isinstance(variant, str) and variant.strip() != "", f"task[{index}].variant must be a non-empty string")

    return GameTaskSpec(id=task_id, env_id=env_id, seed=seed, variant=variant)


def parse_split_manifest(payload: Mapping[str, Any]) -> SplitManifest:
    """Parse a split manifest. Duplicate task ids or duplicate seeds are rejected."""

    _require(isinstance(payload, Mapping), "manifest must be a mapping")

    environment = payload.get("environment")
    split = payload.get("split")
    revision = payload.get("source_revision")
    raw_tasks = payload.get("tasks")
    synthetic = bool(payload.get("synthetic", False))

    _require(isinstance(environment, str) and _ENV_NAME.fullmatch(environment) is not None,
             "manifest.environment must look like 'Owner/Name'")
    _require(isinstance(split, str) and split.strip() != "", "manifest.split must be a non-empty string")
    _require(isinstance(revision, str) and _HEX40.fullmatch(revision) is not None,
             "manifest.source_revision must be a 40-hex git commit")
    _require(isinstance(raw_tasks, Sequence) and not isinstance(raw_tasks, (str, bytes)),
             "manifest.tasks must be a list")
    _require(len(raw_tasks) > 0, "manifest.tasks cannot be empty")

    tasks = tuple(parse_task_spec(row, index=i) for i, row in enumerate(raw_tasks))

    ids = [t.id for t in tasks]
    _require(len(ids) == len(set(ids)), "manifest.tasks contains duplicate task ids")
    # A seed is only unique within a variant: upstream emits the same seed index
    # for every variant of a game. The instance identity is (variant, seed).
    keys = [(t.variant, t.seed) for t in tasks]
    _require(len(keys) == len(set(keys)), "manifest.tasks contains duplicate (variant, seed) instances")

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
        violations.append(f"{len(shared_ids)} task id(s) appear in both splits: {list(shared_ids[:10])}")

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
    """Assemble a receipt whose ``score`` is ``None`` unless every gate passes.

    A score survives only when all of the following hold:
      * ``run_kind == "model_rollout"`` and ``is_model_score`` is true,
      * the run is not synthetic,
      * a seed-separation proof is present and holds,
      * at least one episode was verified and every outcome was accepted,
      * ``status`` is not ``BLOCKED``.

    Any other combination yields ``score: None``. There is no override.
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

    if gate_failures:
        score: float | None = None
    else:
        rewards = [o.reward for o in outcomes if o.reward is not None]
        score = sum(rewards) / len(rewards)

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
        if not receipt.get("is_model_score"):
            raise ReceiptIntegrityError("receipt carries a score but is_model_score is false")
        if receipt.get("synthetic_fixture"):
            raise ReceiptIntegrityError("receipt carries a score but is marked synthetic")
        if receipt.get("score_withheld_because"):
            raise ReceiptIntegrityError("receipt carries a score while listing withholding reasons")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_manifest(path: str) -> SplitManifest:
    return parse_split_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="e13_openreward_games_local_runner",
        description="Validate OpenReward game split manifests and emit a fail-closed receipt.",
    )
    parser.add_argument("--train-manifest", required=True, help="JSON split manifest for the train split")
    parser.add_argument("--eval-manifest", required=True, help="JSON split manifest for the held-out split")
    parser.add_argument("--out", required=True, help="path to write the receipt JSON")
    parser.add_argument("--lane", default="E13 openreward_games_eval")
    parser.add_argument("--suite", default="openreward_games_eval")
    args = parser.parse_args(argv)

    train = _load_manifest(args.train_manifest)
    evaluation = _load_manifest(args.eval_manifest)
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
    print(json.dumps({
        "seed_separation_holds": separation.holds,
        "violations": list(separation.violations),
        "receipt": args.out,
        "score": receipt["score"],
    }, indent=2))
    return 0 if separation.holds else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

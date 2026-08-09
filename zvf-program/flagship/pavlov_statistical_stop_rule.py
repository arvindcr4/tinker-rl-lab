"""Deterministic, offline stopping rules for Pavlov successive halving.

The module is deliberately a decision layer, rather than an evaluator.  It
accepts paired per-example metrics or already materialised arm summaries and
returns JSON-safe selection/stop decisions.  All scientific quantities are
computed with :mod:`flagship.pavlov_statistics`; no network, W&B, Hugging
Face, Tinker, or paid call is reachable from this module.

The selection contract is intentionally narrow:

* arms are ranked by perfect-call count, exact (unrounded) mean reward,
  estimated cost, learning rate, and lexical arm id;
* an arm must beat the frozen base on the first two quality keys and pass both
  per-domain no-regression and safety gates;
* selection metrics must be on a selection split, disjoint from the final
  split; and
* every result is scoped to the xLAM component.  It cannot be turned into a
  held-out, portfolio, all-company, or production claim.

Missing gates, missing costs, malformed paired observations, and ambiguous
split/claim metadata fail closed with :class:`StopRuleInputError`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from flagship.pavlov_statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    StatisticsInputError,
    exact_mcnemar_two_sided,
    mcnemar_discordant_counts,
    newcombe_paired_risk_difference_interval,
    paired_bootstrap_mean_difference,
    portfolio_domain_gates,
    wilson_interval,
)


COMPONENT_ONLY_CLAIM = "xlam_component_only"
DEFAULT_SELECTION_SPLIT_ID = "selection"
DEFAULT_FINAL_SPLIT_ID = "final"
DEFAULT_OPERATIONAL_CAP_USD = 16.50
DEFAULT_HARD_MAX_USD = 18.00
DEFAULT_RESERVE_USD = 1.50

_MISSING = object()
_ALLOWED_CLAIM_ALIASES = {
    "xlam",
    "xlam_eval",
    "xlam_component",
    "xlam_component_only",
    "xlam_function_calling",
}
_FORBIDDEN_CLAIM_WORDS = {
    "portfolio",
    "all_company",
    "company",
    "production",
    "heldout",
    "held_out",
    "final",
}


class StopRuleInputError(ValueError):
    """Raised when a stop-rule input cannot be safely interpreted."""


# A descriptive alias is useful to callers that use the statistical naming.
StatisticalStopRuleError = StopRuleInputError


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StopRuleInputError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise StopRuleInputError(f"{name} must be finite")
    return result


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    result = _finite(name, value)
    if not result.is_integer() or int(result) < minimum:
        raise StopRuleInputError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StopRuleInputError(f"{name} must be a non-empty string")
    return value.strip()


def _materialize(name: str, values: Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise StopRuleInputError(f"{name} must be an iterable of observations")
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise StopRuleInputError(f"{name} must be an iterable of observations") from exc
    if not materialized:
        raise StopRuleInputError(f"{name} must be non-empty")
    return materialized


def _paired_finite(
    base_scores: Iterable[Any], candidate_scores: Iterable[Any]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    base = _materialize("base_scores", base_scores)
    candidate = _materialize("candidate_scores", candidate_scores)
    if len(base) != len(candidate):
        raise StopRuleInputError("base_scores and candidate_scores must have identical lengths")
    normalized_base = tuple(_finite("base_scores", value) for value in base)
    normalized_candidate = tuple(_finite("candidate_scores", value) for value in candidate)
    return normalized_base, normalized_candidate


def _perfect_indicators(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(int(value == 1.0) for value in values)


def _normalise_claim_scope(value: Any) -> str:
    if value is _MISSING or value is None:
        return COMPONENT_ONLY_CLAIM
    if not isinstance(value, str):
        raise StopRuleInputError("claim_scope must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _FORBIDDEN_CLAIM_WORDS or any(
        word in normalized for word in ("portfolio", "company", "production", "heldout", "held_out")
    ):
        raise StopRuleInputError(
            f"claim_scope {value!r} is outside the xLAM component-only boundary"
        )
    if normalized not in _ALLOWED_CLAIM_ALIASES:
        raise StopRuleInputError(f"unsupported claim_scope {value!r}")
    return COMPONENT_ONLY_CLAIM


def _json_value(value: Any) -> Any:
    """Convert the small result objects used here into JSON-safe values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StopRuleInputError("result contains a non-finite number")
        return value
    if hasattr(value, "as_dict"):
        return _json_value(value.as_dict())
    return value


def _as_gate_result(gates: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compute or validate a domain-gate result.

    The preferred input is the four raw mappings consumed by
    ``portfolio_domain_gates``.  A previously computed result is accepted only
    when it contains both component results and their boolean conjunction;
    a lone ``{"passed": true}`` is not sufficient evidence.
    """

    if gates is None:
        return {
            "status": "missing",
            "passed": False,
            "failures": ["missing_domain_gates"],
        }
    if not isinstance(gates, Mapping):
        raise StopRuleInputError("domain_gates must be a mapping or None")

    direct_keys = {"deltas", "delta_intervals", "safety_increases", "safety_intervals"}
    if direct_keys.issubset(gates):
        try:
            result = portfolio_domain_gates(
                gates["deltas"],
                gates["delta_intervals"],
                gates["safety_increases"],
                gates["safety_intervals"],
                no_regression_tolerance=gates.get("no_regression_tolerance", 0.05),
                maximum_safety_increase=gates.get("maximum_safety_increase", 0.02),
            )
        except StatisticsInputError as exc:
            raise StopRuleInputError(str(exc)) from exc
        result = _json_value(result)
        result["status"] = "computed"
        return result

    # Also accept the natural nested representation used by receipts.
    no_regression = gates.get("no_regression")
    safety = gates.get("safety")
    if isinstance(no_regression, Mapping) and isinstance(safety, Mapping):
        if not isinstance(no_regression.get("passed"), bool) or not isinstance(
            safety.get("passed"), bool
        ):
            raise StopRuleInputError(
                "domain_gates.no_regression and domain_gates.safety need boolean passed fields"
            )
        if not isinstance(gates.get("passed"), bool):
            raise StopRuleInputError("domain_gates needs a boolean passed field")
        result = _json_value(dict(gates))
        result["passed"] = bool(
            result["passed"] and no_regression["passed"] and safety["passed"]
        )
        result["status"] = "reported"
        return result

    raise StopRuleInputError(
        "domain_gates must contain raw deltas/intervals or complete no_regression and safety results"
    )


def evaluate_domain_gates(
    deltas: Mapping[str, Any],
    delta_intervals: Mapping[str, Sequence[Any]],
    safety_increases: Mapping[str, Any],
    safety_intervals: Mapping[str, Sequence[Any]],
    *,
    no_regression_tolerance: float = 0.05,
    maximum_safety_increase: float = 0.02,
) -> dict[str, Any]:
    """Run both conjunctive Pavlov no-regression and safety gates."""

    return _as_gate_result(
        {
            "deltas": deltas,
            "delta_intervals": delta_intervals,
            "safety_increases": safety_increases,
            "safety_intervals": safety_intervals,
            "no_regression_tolerance": no_regression_tolerance,
            "maximum_safety_increase": maximum_safety_increase,
        }
    )


@dataclass(frozen=True)
class PairedArmMetrics:
    """Validated paired statistics for one selection arm."""

    arm_id: str
    learning_rate: float
    estimated_cost_usd: float
    split_id: str
    split_role: str
    claim_scope: str
    base_scores: tuple[float, ...]
    candidate_scores: tuple[float, ...]
    perfect_call_count: int
    trials: int
    mean_strict_reward: float
    perfect_call_rate: float
    wilson_interval: tuple[float, float]
    mcnemar_counts: tuple[int, int]
    exact_mcnemar_pvalue: float
    newcombe_interval: tuple[float, float]
    paired_bootstrap: Mapping[str, Any]
    domain_gates: Mapping[str, Any]

    @property
    def domain_gates_passed(self) -> bool:
        return self.domain_gates.get("passed") is True

    @property
    def beats_quality_base(self) -> bool:
        # Count is the primary quality key; mean is the exact tie-break.
        base_perfect = sum(value == 1.0 for value in self.base_scores)
        base_mean = math.fsum(self.base_scores) / self.trials
        return self.perfect_call_count > base_perfect or (
            self.perfect_call_count == base_perfect and self.mean_strict_reward > base_mean
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "learning_rate": self.learning_rate,
            "estimated_cost_usd": self.estimated_cost_usd,
            "split_id": self.split_id,
            "split_role": self.split_role,
            "claim_scope": COMPONENT_ONLY_CLAIM,
            "trials": self.trials,
            "perfect_call_count": self.perfect_call_count,
            "perfect_call_rate": self.perfect_call_rate,
            "mean_strict_reward": self.mean_strict_reward,
            "wilson_interval": list(self.wilson_interval),
            "mcnemar": {
                "base_fail_candidate_success": self.mcnemar_counts[0],
                "base_success_candidate_fail": self.mcnemar_counts[1],
                "exact_two_sided_p": self.exact_mcnemar_pvalue,
            },
            "newcombe_interval": list(self.newcombe_interval),
            "paired_bootstrap_mean_difference": _json_value(self.paired_bootstrap),
            "domain_gates": _json_value(self.domain_gates),
            "domain_gates_passed": self.domain_gates_passed,
            "paired_metrics_validated": True,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def build_paired_arm_metrics(
    arm_id: str,
    base_scores: Iterable[Any],
    candidate_scores: Iterable[Any],
    *,
    learning_rate: Any,
    estimated_cost_usd: Any,
    split_id: str = DEFAULT_SELECTION_SPLIT_ID,
    split_role: str = "selection",
    claim_scope: Any = COMPONENT_ONLY_CLAIM,
    domain_gates: Mapping[str, Any] | None = None,
    final_split_id: str | None = DEFAULT_FINAL_SPLIT_ID,
    bootstrap_resamples: Any = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: Any = DEFAULT_BOOTSTRAP_SEED,
) -> PairedArmMetrics:
    """Validate paired scores and compute all arm-level statistics.

    ``base_scores`` and ``candidate_scores`` are paired by position.  Scores
    must be finite and non-empty; perfect calls are exactly values equal to
    ``1.0``.  The final split is named only for separation checks and is never
    consumed here.
    """

    arm_id = _string("arm_id", arm_id)
    learning_rate = _finite("learning_rate", learning_rate)
    if learning_rate <= 0.0:
        raise StopRuleInputError("learning_rate must be positive")
    estimated_cost_usd = _finite("estimated_cost_usd", estimated_cost_usd)
    if estimated_cost_usd < 0.0:
        raise StopRuleInputError("estimated_cost_usd must be non-negative")
    split_id = _string("split_id", split_id)
    split_role = _string("split_role", split_role).lower().replace("-", "_")
    if split_role not in {"selection", "selection_slice"}:
        raise StopRuleInputError("paired arm metrics must use the selection split only")
    if final_split_id is not None:
        final_split_id = _string("final_split_id", final_split_id)
        if final_split_id == split_id:
            raise StopRuleInputError("selection split and final split must be different")
    claim_scope = _normalise_claim_scope(claim_scope)
    base, candidate = _paired_finite(base_scores, candidate_scores)
    base_perfect = _perfect_indicators(base)
    candidate_perfect = _perfect_indicators(candidate)
    trials = len(base)
    perfect_call_count = sum(candidate_perfect)
    mean_strict_reward = math.fsum(candidate) / trials
    bootstrap = paired_bootstrap_mean_difference(
        base,
        candidate,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    try:
        mcnemar_counts = mcnemar_discordant_counts(base_perfect, candidate_perfect)
        mcnemar_pvalue = exact_mcnemar_two_sided(base_perfect, candidate_perfect)
        newcombe = newcombe_paired_risk_difference_interval(base_perfect, candidate_perfect)
        wilson = wilson_interval(perfect_call_count, trials)
    except StatisticsInputError as exc:
        raise StopRuleInputError(str(exc)) from exc
    return PairedArmMetrics(
        arm_id=arm_id,
        learning_rate=learning_rate,
        estimated_cost_usd=estimated_cost_usd,
        split_id=split_id,
        split_role="selection",
        claim_scope=claim_scope,
        base_scores=base,
        candidate_scores=candidate,
        perfect_call_count=perfect_call_count,
        trials=trials,
        mean_strict_reward=mean_strict_reward,
        perfect_call_rate=perfect_call_count / trials,
        wilson_interval=wilson,
        mcnemar_counts=mcnemar_counts,
        exact_mcnemar_pvalue=mcnemar_pvalue,
        newcombe_interval=newcombe,
        paired_bootstrap=bootstrap.as_dict(),
        domain_gates=_as_gate_result(domain_gates),
    )


# Short aliases make the construction API easy to discover.
make_arm_metrics = build_paired_arm_metrics
paired_arm_metrics = build_paired_arm_metrics


@dataclass(frozen=True)
class _ArmView:
    arm_id: str
    learning_rate: float
    estimated_cost_usd: float
    split_id: str
    split_role: str
    claim_scope: str
    perfect_call_count: int
    trials: int
    mean_strict_reward: float
    domain_gates: Mapping[str, Any]
    paired_metrics_validated: bool

    @property
    def perfect_call_rate(self) -> float:
        return self.perfect_call_count / self.trials

    @property
    def gates_passed(self) -> bool:
        return self.domain_gates.get("passed") is True

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "learning_rate": self.learning_rate,
            "estimated_cost_usd": self.estimated_cost_usd,
            "split_id": self.split_id,
            "split_role": self.split_role,
            "claim_scope": COMPONENT_ONLY_CLAIM,
            "perfect_call_count": self.perfect_call_count,
            "trials": self.trials,
            "perfect_call_rate": self.perfect_call_rate,
            "mean_strict_reward": self.mean_strict_reward,
            "domain_gates": _json_value(self.domain_gates),
            "domain_gates_passed": self.gates_passed,
            "paired_metrics_validated": self.paired_metrics_validated,
        }


def _field(record: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def _normalise_arm(record: Mapping[str, Any] | PairedArmMetrics) -> _ArmView:
    if isinstance(record, _ArmView):
        return record
    if isinstance(record, PairedArmMetrics):
        return _ArmView(
            arm_id=record.arm_id,
            learning_rate=record.learning_rate,
            estimated_cost_usd=record.estimated_cost_usd,
            split_id=record.split_id,
            split_role=record.split_role,
            claim_scope=record.claim_scope,
            perfect_call_count=record.perfect_call_count,
            trials=record.trials,
            mean_strict_reward=record.mean_strict_reward,
            domain_gates=record.domain_gates,
            paired_metrics_validated=True,
        )
    if not isinstance(record, Mapping):
        raise StopRuleInputError("each arm must be a mapping or PairedArmMetrics")

    # If paired rows are present, derive the summary rather than trusting
    # caller-provided counts.  This prevents a stale summary from being used
    # for selection.
    raw_base = _field(record, "base_scores", "base_metrics", default=_MISSING)
    raw_candidate = _field(record, "candidate_scores", "arm_scores", "trained_scores", default=_MISSING)
    if raw_base is not _MISSING or raw_candidate is not _MISSING:
        if raw_base is _MISSING or raw_candidate is _MISSING:
            raise StopRuleInputError("both base_scores and candidate_scores are required")
        derived = build_paired_arm_metrics(
            _field(record, "arm_id", "id"),
            raw_base,
            raw_candidate,
            learning_rate=_field(record, "learning_rate", "lr"),
            estimated_cost_usd=_field(record, "estimated_cost_usd", "cost_usd", "cost"),
            split_id=_field(record, "split_id", "selection_split_id", default=DEFAULT_SELECTION_SPLIT_ID),
            split_role=_field(record, "split_role", default="selection"),
            claim_scope=_field(record, "claim_scope", default=COMPONENT_ONLY_CLAIM),
            domain_gates=_field(record, "domain_gates", default=None),
            final_split_id=_field(record, "final_split_id", default=DEFAULT_FINAL_SPLIT_ID),
            bootstrap_resamples=_field(record, "bootstrap_resamples", default=DEFAULT_BOOTSTRAP_RESAMPLES),
            bootstrap_seed=_field(record, "bootstrap_seed", default=DEFAULT_BOOTSTRAP_SEED),
        )
        return _normalise_arm(derived)

    arm_id = _field(record, "arm_id", "id")
    learning_rate = _field(record, "learning_rate", "lr")
    cost = _field(record, "estimated_cost_usd", "cost_usd", "cost")
    count = _field(record, "perfect_call_count", "perfect_calls", "successes")
    trials = _field(record, "trials", "examples", "n")
    mean = _field(record, "mean_strict_reward", "mean_reward", "mean")
    if _MISSING in (arm_id, learning_rate, cost, count, trials, mean):
        raise StopRuleInputError(
            "arm needs arm_id, learning_rate, estimated_cost_usd, perfect_call_count, trials, and mean_strict_reward"
        )
    arm_id = _string("arm_id", arm_id)
    learning_rate = _finite("learning_rate", learning_rate)
    if learning_rate <= 0.0:
        raise StopRuleInputError("learning_rate must be positive")
    cost = _finite("estimated_cost_usd", cost)
    if cost < 0.0:
        raise StopRuleInputError("estimated_cost_usd must be non-negative")
    trials = _integer("trials", trials, minimum=1)
    count = _integer("perfect_call_count", count, minimum=0)
    if count > trials:
        raise StopRuleInputError("perfect_call_count cannot exceed trials")
    mean = _finite("mean_strict_reward", mean)
    supplied_rate = _field(record, "perfect_call_rate", default=_MISSING)
    if supplied_rate is not _MISSING:
        supplied_rate = _finite("perfect_call_rate", supplied_rate)
        if not math.isclose(supplied_rate, count / trials, rel_tol=0.0, abs_tol=1e-15):
            raise StopRuleInputError("perfect_call_rate does not match perfect_call_count/trials")

    split_id = _string(
        "split_id", _field(record, "split_id", "selection_split_id", default=DEFAULT_SELECTION_SPLIT_ID)
    )
    split_role = _string("split_role", _field(record, "split_role", default="selection"))
    split_role_normalized = split_role.lower().replace("-", "_")
    if split_role_normalized not in {"selection", "selection_slice"}:
        raise StopRuleInputError("arm metrics must be on the selection split, never the final split")
    final_split = _field(record, "final_split_id", default=DEFAULT_FINAL_SPLIT_ID)
    if final_split is not None and _string("final_split_id", final_split) == split_id:
        raise StopRuleInputError("selection split and final split must be different")
    if record.get("final_used_for_selection") is True:
        raise StopRuleInputError("final split metrics must not be used for selection")
    claim_scope = _normalise_claim_scope(_field(record, "claim_scope", default=COMPONENT_ONLY_CLAIM))
    gates = _field(record, "domain_gates", default=None)
    if gates is None:
        direct_gate_keys = {"deltas", "delta_intervals", "safety_increases", "safety_intervals"}
        if direct_gate_keys.intersection(record):
            gates = {key: record[key] for key in direct_gate_keys if key in record}
            for key in ("no_regression_tolerance", "maximum_safety_increase"):
                if key in record:
                    gates[key] = record[key]
    normalized_gates = _as_gate_result(gates)
    return _ArmView(
        arm_id=arm_id,
        learning_rate=learning_rate,
        estimated_cost_usd=cost,
        split_id=split_id,
        split_role="selection",
        claim_scope=claim_scope,
        perfect_call_count=count,
        trials=trials,
        mean_strict_reward=mean,
        domain_gates=normalized_gates,
        paired_metrics_validated=False,
    )


@dataclass(frozen=True)
class _BaseView:
    perfect_call_count: int
    trials: int
    mean_strict_reward: float


def _normalise_base(base: Mapping[str, Any] | PairedArmMetrics) -> _BaseView:
    if isinstance(base, PairedArmMetrics):
        return _BaseView(
            perfect_call_count=sum(value == 1.0 for value in base.base_scores),
            trials=len(base.base_scores),
            mean_strict_reward=math.fsum(base.base_scores) / len(base.base_scores),
        )
    if not isinstance(base, Mapping):
        raise StopRuleInputError("base must be a mapping or PairedArmMetrics")
    count = _field(base, "perfect_call_count", "perfect_calls", "successes")
    trials = _field(base, "trials", "examples", "n")
    mean = _field(base, "mean_strict_reward", "mean_reward", "mean")
    if _MISSING in (count, trials, mean):
        raise StopRuleInputError("base needs perfect_call_count, trials, and mean_strict_reward")
    trials = _integer("base.trials", trials, minimum=1)
    count = _integer("base.perfect_call_count", count, minimum=0)
    if count > trials:
        raise StopRuleInputError("base.perfect_call_count cannot exceed base.trials")
    mean = _finite("base.mean_strict_reward", mean)
    supplied_rate = _field(base, "perfect_call_rate", default=_MISSING)
    if supplied_rate is not _MISSING:
        supplied_rate = _finite("base.perfect_call_rate", supplied_rate)
        if not math.isclose(supplied_rate, count / trials, rel_tol=0.0, abs_tol=1e-15):
            raise StopRuleInputError("base.perfect_call_rate does not match count/trials")
    return _BaseView(count, trials, mean)


def strictly_beats_base(
    arm: Mapping[str, Any] | PairedArmMetrics,
    base: Mapping[str, Any] | PairedArmMetrics,
) -> bool:
    """Return whether an arm beats base on count, then exact mean reward."""

    normalized_arm = _normalise_arm(arm)
    normalized_base = _normalise_base(base)
    if normalized_arm.trials != normalized_base.trials:
        raise StopRuleInputError("arm and base must use the same number of paired examples")
    return normalized_arm.perfect_call_count > normalized_base.perfect_call_count or (
        normalized_arm.perfect_call_count == normalized_base.perfect_call_count
        and normalized_arm.mean_strict_reward > normalized_base.mean_strict_reward
    )


arm_beats_base = strictly_beats_base


def rank_arms(
    arms: Iterable[Mapping[str, Any] | PairedArmMetrics],
    *,
    selection_split_id: str = DEFAULT_SELECTION_SPLIT_ID,
) -> list[dict[str, Any]]:
    """Return arms in the protocol's exact deterministic ranking order."""

    selection_split_id = _string("selection_split_id", selection_split_id)
    if isinstance(arms, (str, bytes, bytearray)):
        raise StopRuleInputError("arms must be a non-empty iterable")
    try:
        records = tuple(arms)
    except TypeError as exc:
        raise StopRuleInputError("arms must be a non-empty iterable") from exc
    if not records:
        raise StopRuleInputError("arms must be non-empty")
    normalized = [_normalise_arm(record) for record in records]
    arm_ids = [arm.arm_id for arm in normalized]
    if len(set(arm_ids)) != len(arm_ids):
        raise StopRuleInputError("arm IDs must be unique within a selection round")
    for arm in normalized:
        if arm.split_id != selection_split_id:
            raise StopRuleInputError(
                f"arm {arm.arm_id!r} is on {arm.split_id!r}, expected selection split {selection_split_id!r}"
            )
    ordered = sorted(
        normalized,
        key=lambda arm: (
            -arm.perfect_call_count,
            -arm.mean_strict_reward,
            arm.estimated_cost_usd,
            arm.learning_rate,
            arm.arm_id,
        ),
    )
    return [arm.as_dict() for arm in ordered]


def validate_selection_vs_final_split(
    selection_split_id: str,
    final_split_id: str,
    *,
    selection_example_ids: Iterable[Any] | None = None,
    final_example_ids: Iterable[Any] | None = None,
    final_used_for_selection: bool = False,
) -> dict[str, Any]:
    """Prove that selection and final manifests are separate and untouched."""

    selection_split_id = _string("selection_split_id", selection_split_id)
    final_split_id = _string("final_split_id", final_split_id)
    if selection_split_id == final_split_id:
        raise StopRuleInputError("selection and final split IDs must differ")
    if not isinstance(final_used_for_selection, bool):
        raise StopRuleInputError("final_used_for_selection must be boolean")
    if final_used_for_selection:
        raise StopRuleInputError("final split metrics must not be used for selection")

    def _ids(name: str, values: Iterable[Any] | None) -> tuple[Any, ...] | None:
        if values is None:
            return None
        materialized = _materialize(name, values)
        try:
            if len(set(materialized)) != len(materialized):
                raise StopRuleInputError(f"{name} must not contain duplicate example identities")
        except TypeError as exc:
            raise StopRuleInputError(f"{name} identities must be hashable") from exc
        return materialized

    selection_ids = _ids("selection_example_ids", selection_example_ids)
    final_ids = _ids("final_example_ids", final_example_ids)
    overlap: set[Any] = set()
    if selection_ids is not None and final_ids is not None:
        overlap = set(selection_ids).intersection(final_ids)
        if overlap:
            raise StopRuleInputError(
                f"selection and final example identities overlap ({len(overlap)} rows)"
            )
    return {
        "separate": True,
        "selection_split_id": selection_split_id,
        "final_split_id": final_split_id,
        "selection_examples": None if selection_ids is None else len(selection_ids),
        "final_examples": None if final_ids is None else len(final_ids),
        "overlap_examples": len(overlap),
        "final_used_for_selection": False,
        "selection_only": True,
        "final_evaluation_required": True,
    }


def budget_aware_stop(
    spent_usd: Any,
    next_cost_usd: Any,
    *,
    operational_cap_usd: Any = DEFAULT_OPERATIONAL_CAP_USD,
    hard_max_usd: Any = DEFAULT_HARD_MAX_USD,
    reserve_usd: Any = DEFAULT_RESERVE_USD,
) -> dict[str, Any]:
    """Decide whether another tracked round fits the authorized budget.

    The operational cap is the spend ceiling for work.  The hard maximum and
    reserve are retained in the returned receipt so a caller cannot silently
    reinterpret reserve dollars as available work budget.
    """

    spent = _finite("spent_usd", spent_usd)
    next_cost = _finite("next_cost_usd", next_cost_usd)
    cap = _finite("operational_cap_usd", operational_cap_usd)
    hard = _finite("hard_max_usd", hard_max_usd)
    reserve = _finite("reserve_usd", reserve_usd)
    if min(spent, next_cost, cap, hard, reserve) < 0.0:
        raise StopRuleInputError("budget values must be non-negative")
    if cap > hard:
        raise StopRuleInputError("operational_cap_usd cannot exceed hard_max_usd")
    if reserve > hard - cap:
        raise StopRuleInputError("reserve_usd must fit between the operational cap and hard maximum")
    projected = spent + next_cost
    if spent > hard or projected > hard:
        stop = True
        reason = "hard_max_exceeded"
    elif spent > cap or projected > cap:
        stop = True
        reason = "operational_cap_exceeded"
    else:
        stop = False
        reason = "within_operational_cap"
    return {
        "stop": stop,
        "decision": "stop" if stop else "continue",
        "reason": reason,
        "spent_usd": spent,
        "next_cost_usd": next_cost,
        "projected_total_usd": projected,
        "remaining_operational_cap_usd": cap - spent,
        "remaining_hard_max_usd": hard - spent,
        "operational_cap_usd": cap,
        "hard_max_usd": hard,
        "reserve_usd": reserve,
    }


should_stop_for_budget = budget_aware_stop


def _eligibility(
    arms: Sequence[_ArmView], base: _BaseView
) -> tuple[list[_ArmView], list[dict[str, Any]]]:
    eligible: list[_ArmView] = []
    details: list[dict[str, Any]] = []
    for arm in arms:
        reasons: list[str] = []
        if arm.trials != base.trials:
            reasons.append("paired_trial_count_differs_from_base")
        quality = arm.trials == base.trials and (
            arm.perfect_call_count > base.perfect_call_count
            or (
                arm.perfect_call_count == base.perfect_call_count
                and arm.mean_strict_reward > base.mean_strict_reward
            )
        )
        if not quality:
            reasons.append("does_not_beat_frozen_base_quality")
        if not arm.gates_passed:
            status = arm.domain_gates.get("status", "missing")
            reasons.append("domain_gates_missing" if status == "missing" else "domain_gates_failed")
        is_eligible = not reasons
        if is_eligible:
            eligible.append(arm)
        details.append(
            {
                "arm_id": arm.arm_id,
                "beats_base_quality": quality,
                "domain_gates_passed": arm.gates_passed,
                "eligible": is_eligible,
                "reasons": reasons,
            }
        )
    return eligible, details


def _decision_envelope(
    *,
    status: str,
    reason: str,
    ranked: Sequence[dict[str, Any]],
    eligibility: Sequence[dict[str, Any]],
    budget: Mapping[str, Any],
    split: Mapping[str, Any],
    winner: dict[str, Any] | None = None,
    retained: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "status": status,
        "stop": status != "selected" and status != "winner_selected",
        "reason": reason,
        "ranked": [_json_value(item) for item in ranked],
        "eligibility": [_json_value(item) for item in eligibility],
        "eligible_arm_ids": [item["arm_id"] for item in eligibility if item["eligible"]],
        "retained": [_json_value(item) for item in retained],
        "retained_arm_ids": [item["arm_id"] for item in retained],
        "winner": _json_value(winner),
        "budget": _json_value(budget),
        "split": _json_value(split),
        "claim_scope": COMPONENT_ONLY_CLAIM,
        "analysis_scope": COMPONENT_ONLY_CLAIM,
        "portfolio_claim_permitted": False,
        "company_claim_permitted": False,
        "heldout_claim_permitted": False,
        "final_evaluation_required": True,
    }


def select_winner(
    arms: Iterable[Mapping[str, Any] | PairedArmMetrics],
    base: Mapping[str, Any] | PairedArmMetrics,
    *,
    spent_usd: Any = 0.0,
    next_cost_usd: Any | None = None,
    operational_cap_usd: Any = DEFAULT_OPERATIONAL_CAP_USD,
    hard_max_usd: Any = DEFAULT_HARD_MAX_USD,
    reserve_usd: Any = DEFAULT_RESERVE_USD,
    selection_split_id: str = DEFAULT_SELECTION_SPLIT_ID,
    final_split_id: str = DEFAULT_FINAL_SPLIT_ID,
    selection_example_ids: Iterable[Any] | None = None,
    final_example_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Select one eligible arm or return a deterministic stop decision."""

    split = validate_selection_vs_final_split(
        selection_split_id,
        final_split_id,
        selection_example_ids=selection_example_ids,
        final_example_ids=final_example_ids,
    )
    base_view = _normalise_base(base)
    if isinstance(arms, (str, bytes, bytearray)):
        raise StopRuleInputError("arms must be a non-empty iterable")
    try:
        raw = tuple(arms)
    except TypeError as exc:
        raise StopRuleInputError("arms must be a non-empty iterable") from exc
    if not raw:
        raise StopRuleInputError("arms must be non-empty")
    views = [_normalise_arm(record) for record in raw]
    if len({arm.arm_id for arm in views}) != len(views):
        raise StopRuleInputError("arm IDs must be unique within a selection round")
    if any(arm.split_id != selection_split_id for arm in views):
        raise StopRuleInputError("all arms must be evaluated on the declared selection split")
    ranked = rank_arms(views, selection_split_id=selection_split_id)
    by_id = {arm.arm_id: arm for arm in views}
    ordered_views = [by_id[item["arm_id"]] for item in ranked]
    eligible_views, eligibility = _eligibility(ordered_views, base_view)
    if next_cost_usd is None:
        next_cost_usd = math.fsum(arm.estimated_cost_usd for arm in views)
    budget = budget_aware_stop(
        spent_usd,
        next_cost_usd,
        operational_cap_usd=operational_cap_usd,
        hard_max_usd=hard_max_usd,
        reserve_usd=reserve_usd,
    )
    if budget["stop"]:
        return _decision_envelope(
            status="stopped_budget",
            reason=budget["reason"],
            ranked=ranked,
            eligibility=eligibility,
            budget=budget,
            split=split,
        )
    if not eligible_views:
        return _decision_envelope(
            status="stopped_no_eligible_arm",
            reason="no_arm_beats_frozen_base_and_passes_domain_gates",
            ranked=ranked,
            eligibility=eligibility,
            budget=budget,
            split=split,
        )
    winner = eligible_views[0].as_dict()
    return _decision_envelope(
        status="selected",
        reason="eligible_arm_selected_by_protocol_tie_breaks",
        ranked=ranked,
        eligibility=eligibility,
        budget=budget,
        split=split,
        winner=winner,
        retained=(winner,),
    )


def successive_halving(
    rounds: Sequence[Iterable[Mapping[str, Any] | PairedArmMetrics]],
    base: Mapping[str, Any] | PairedArmMetrics,
    *,
    keep_counts: Sequence[Any] | None = None,
    projected_round_costs: Sequence[Any] | None = None,
    spent_usd: Any = 0.0,
    operational_cap_usd: Any = DEFAULT_OPERATIONAL_CAP_USD,
    hard_max_usd: Any = DEFAULT_HARD_MAX_USD,
    reserve_usd: Any = DEFAULT_RESERVE_USD,
    selection_split_id: str = DEFAULT_SELECTION_SPLIT_ID,
    final_split_id: str = DEFAULT_FINAL_SPLIT_ID,
    selection_example_ids: Iterable[Any] | None = None,
    final_example_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic successive halving over precomputed round receipts.

    A round's cost is the sum of its active arms unless an explicit projected
    cost is supplied.  Extra arms in later round receipts are ignored after
    halving; a missing retained arm simply leaves fewer active candidates and
    is reported in that round's receipt.  No final metrics are consumed.
    """

    split = validate_selection_vs_final_split(
        selection_split_id,
        final_split_id,
        selection_example_ids=selection_example_ids,
        final_example_ids=final_example_ids,
    )
    if isinstance(rounds, (str, bytes, bytearray)):
        raise StopRuleInputError("rounds must be a non-empty sequence")
    try:
        round_values = tuple(rounds)
    except TypeError as exc:
        raise StopRuleInputError("rounds must be a non-empty sequence") from exc
    if not round_values:
        raise StopRuleInputError("rounds must be non-empty")
    if keep_counts is not None:
        try:
            keep_values = tuple(keep_counts)
        except TypeError as exc:
            raise StopRuleInputError("keep_counts must be a sequence") from exc
        if len(keep_values) != len(round_values):
            raise StopRuleInputError("keep_counts must have one value per round")
    else:
        keep_values = None
    if projected_round_costs is not None:
        try:
            cost_values = tuple(projected_round_costs)
        except TypeError as exc:
            raise StopRuleInputError("projected_round_costs must be a sequence") from exc
        if len(cost_values) != len(round_values):
            raise StopRuleInputError("projected_round_costs must have one value per round")
    else:
        cost_values = None

    base_view = _normalise_base(base)
    active_ids: set[str] | None = None
    spent = _finite("spent_usd", spent_usd)
    if spent < 0.0:
        raise StopRuleInputError("spent_usd must be non-negative")
    round_receipts: list[dict[str, Any]] = []

    for round_index, raw_round in enumerate(round_values):
        if isinstance(raw_round, (str, bytes, bytearray)):
            raise StopRuleInputError(f"round {round_index} must be an iterable of arms")
        try:
            raw_arms = tuple(raw_round)
        except TypeError as exc:
            raise StopRuleInputError(f"round {round_index} must be an iterable of arms") from exc
        if not raw_arms:
            raise StopRuleInputError(f"round {round_index} must contain at least one arm")
        views = [_normalise_arm(record) for record in raw_arms]
        if len({arm.arm_id for arm in views}) != len(views):
            raise StopRuleInputError(f"round {round_index} has duplicate arm IDs")
        if any(arm.split_id != selection_split_id for arm in views):
            raise StopRuleInputError("all arms must be evaluated on the declared selection split")
        if active_ids is not None:
            views = [arm for arm in views if arm.arm_id in active_ids]
        if not views:
            return {
                "status": "stopped_missing_retained_arm",
                "stop": True,
                "reason": "no_retained_arm_receipt_in_next_round",
                "rounds": round_receipts,
                "winner": None,
                "split": split,
                "claim_scope": COMPONENT_ONLY_CLAIM,
                "analysis_scope": COMPONENT_ONLY_CLAIM,
                "portfolio_claim_permitted": False,
                "company_claim_permitted": False,
                "heldout_claim_permitted": False,
                "final_evaluation_required": True,
                "spent_usd": spent,
            }

        arm_ids = {arm.arm_id for arm in views}
        if cost_values is None:
            next_cost = math.fsum(arm.estimated_cost_usd for arm in views)
        else:
            next_cost = cost_values[round_index]
        budget = budget_aware_stop(
            spent,
            next_cost,
            operational_cap_usd=operational_cap_usd,
            hard_max_usd=hard_max_usd,
            reserve_usd=reserve_usd,
        )
        ranked = rank_arms(views, selection_split_id=selection_split_id)
        by_id = {arm.arm_id: arm for arm in views}
        ordered_views = [by_id[item["arm_id"]] for item in ranked]
        eligible_views, eligibility = _eligibility(ordered_views, base_view)
        if keep_values is None:
            keep_count = max(1, math.ceil(len(ordered_views) / 2))
        else:
            keep_count = _integer(f"keep_counts[{round_index}]", keep_values[round_index], minimum=1)
        if budget["stop"]:
            round_receipts.append(
                {
                    "round_index": round_index,
                    "active_arm_ids": sorted(arm_ids),
                    "ranked_arm_ids": [item["arm_id"] for item in ranked],
                    "eligible_arm_ids": [arm.arm_id for arm in eligible_views],
                    "retained_arm_ids": [],
                    "eligibility": eligibility,
                    "budget": budget,
                }
            )
            return {
                "status": "stopped_budget",
                "stop": True,
                "reason": budget["reason"],
                "rounds": _json_value(round_receipts),
                "winner": None,
                "split": split,
                "claim_scope": COMPONENT_ONLY_CLAIM,
                "analysis_scope": COMPONENT_ONLY_CLAIM,
                "portfolio_claim_permitted": False,
                "company_claim_permitted": False,
                "heldout_claim_permitted": False,
                "final_evaluation_required": True,
                "spent_usd": spent,
            }
        spent += next_cost
        if not eligible_views:
            round_receipts.append(
                {
                    "round_index": round_index,
                    "active_arm_ids": sorted(arm_ids),
                    "ranked_arm_ids": [item["arm_id"] for item in ranked],
                    "eligible_arm_ids": [],
                    "retained_arm_ids": [],
                    "eligibility": eligibility,
                    "budget": budget,
                }
            )
            return {
                "status": "stopped_no_eligible_arm",
                "stop": True,
                "reason": "no_arm_beats_frozen_base_and_passes_domain_gates",
                "rounds": _json_value(round_receipts),
                "winner": None,
                "split": split,
                "claim_scope": COMPONENT_ONLY_CLAIM,
                "analysis_scope": COMPONENT_ONLY_CLAIM,
                "portfolio_claim_permitted": False,
                "company_claim_permitted": False,
                "heldout_claim_permitted": False,
                "final_evaluation_required": True,
                "spent_usd": spent,
            }
        retained = eligible_views[: min(keep_count, len(eligible_views))]
        round_receipts.append(
            {
                "round_index": round_index,
                "active_arm_ids": sorted(arm_ids),
                "ranked_arm_ids": [item["arm_id"] for item in ranked],
                "eligible_arm_ids": [arm.arm_id for arm in eligible_views],
                "retained_arm_ids": [arm.arm_id for arm in retained],
                "eligibility": eligibility,
                "budget": budget,
            }
        )
        if round_index == len(round_values) - 1:
            winner = retained[0].as_dict()
            return {
                "status": "winner_selected",
                "stop": False,
                "reason": "successive_halving_winner_selected_on_selection_split",
                "rounds": _json_value(round_receipts),
                "winner": _json_value(winner),
                "split": split,
                "claim_scope": COMPONENT_ONLY_CLAIM,
                "analysis_scope": COMPONENT_ONLY_CLAIM,
                "portfolio_claim_permitted": False,
                "company_claim_permitted": False,
                "heldout_claim_permitted": False,
                "final_evaluation_required": True,
                "spent_usd": spent,
            }
        active_ids = {arm.arm_id for arm in retained}

    # The loop always returns; this is defensive for static checkers.
    raise StopRuleInputError("successive halving did not produce a decision")


successive_halving_decision = successive_halving
run_successive_halving = successive_halving


__all__ = [
    "COMPONENT_ONLY_CLAIM",
    "DEFAULT_FINAL_SPLIT_ID",
    "DEFAULT_HARD_MAX_USD",
    "DEFAULT_OPERATIONAL_CAP_USD",
    "DEFAULT_RESERVE_USD",
    "DEFAULT_SELECTION_SPLIT_ID",
    "PairedArmMetrics",
    "StatisticalStopRuleError",
    "StopRuleInputError",
    "arm_beats_base",
    "budget_aware_stop",
    "build_paired_arm_metrics",
    "evaluate_domain_gates",
    "make_arm_metrics",
    "paired_arm_metrics",
    "rank_arms",
    "run_successive_halving",
    "select_winner",
    "should_stop_for_budget",
    "strictly_beats_base",
    "successive_halving",
    "successive_halving_decision",
    "validate_selection_vs_final_split",
]

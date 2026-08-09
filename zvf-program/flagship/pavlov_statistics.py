"""Dependency-light statistics for the Pavlov evaluation protocol.

The functions in this module are intentionally small and deterministic.  They
operate on already-validated receipt rows; they do not read files, contact
services, or make scientific claims.  The paired-difference orientation used
throughout is ``candidate - base``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import random
import statistics
from numbers import Real
from typing import Any


DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 809


class StatisticsInputError(ValueError):
    """Raised when an input violates the statistical contract."""


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StatisticsInputError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise StatisticsInputError(f"{name} must be finite")
    return result


def _integer(name: str, value: Any, *, minimum: int | None = None) -> int:
    result = _finite_real(name, value)
    if not result.is_integer():
        raise StatisticsInputError(f"{name} must be an integer")
    integer = int(result)
    if minimum is not None and integer < minimum:
        raise StatisticsInputError(f"{name} must be at least {minimum}")
    return integer


def _confidence(confidence: float) -> float:
    result = _finite_real("confidence", confidence)
    if not 0.0 < result < 1.0:
        raise StatisticsInputError("confidence must be strictly between 0 and 1")
    return result


def _materialize(name: str, values: Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise StatisticsInputError(f"{name} must be an iterable of observations")
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise StatisticsInputError(f"{name} must be an iterable of observations") from exc
    if not materialized:
        raise StatisticsInputError(f"{name} must be non-empty")
    return materialized


def _binary(name: str, value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, Real) or not math.isfinite(float(value)):
        raise StatisticsInputError(f"{name} must contain only 0/1 or boolean values")
    numeric = float(value)
    if numeric not in (0.0, 1.0):
        raise StatisticsInputError(f"{name} must contain only 0/1 or boolean values")
    return int(numeric)


def _paired_binary(
    base: Iterable[Any], candidate: Iterable[Any]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    base_values = _materialize("base", base)
    candidate_values = _materialize("candidate", candidate)
    if len(base_values) != len(candidate_values):
        raise StatisticsInputError("base and candidate must have the same length")
    return (
        tuple(_binary("base", value) for value in base_values),
        tuple(_binary("candidate", value) for value in candidate_values),
    )


def _paired_real(
    base: Iterable[Any], candidate: Iterable[Any]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    base_values = _materialize("base", base)
    candidate_values = _materialize("candidate", candidate)
    if len(base_values) != len(candidate_values):
        raise StatisticsInputError("base and candidate must have the same length")
    return (
        tuple(_finite_real("base", value) for value in base_values),
        tuple(_finite_real("candidate", value) for value in candidate_values),
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise StatisticsInputError("cannot compute a percentile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise StatisticsInputError("percentile probability must be in [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    successes = _integer("successes", successes, minimum=0)
    trials = _integer("trials", trials, minimum=1)
    if successes > trials:
        raise StatisticsInputError("successes cannot exceed trials")
    confidence = _confidence(confidence)

    alpha = 1.0 - confidence
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_discordant_counts(base: Iterable[Any], candidate: Iterable[Any]) -> tuple[int, int]:
    """Return ``(b, c)`` for base-fail/candidate-pass and base-pass/candidate-fail."""

    base_values, candidate_values = _paired_binary(base, candidate)
    base_fail_candidate_pass = sum(
        base_value == 0 and candidate_value == 1
        for base_value, candidate_value in zip(base_values, candidate_values)
    )
    base_pass_candidate_fail = sum(
        base_value == 1 and candidate_value == 0
        for base_value, candidate_value in zip(base_values, candidate_values)
    )
    return base_fail_candidate_pass, base_pass_candidate_fail


def exact_mcnemar_two_sided(base: Iterable[Any], candidate: Iterable[Any]) -> float:
    """Return the exact two-sided McNemar p-value without continuity correction."""

    b, c = mcnemar_discordant_counts(base, candidate)
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1))
    upper_tail = sum(math.comb(discordant, k) for k in range(max(b, c), discordant + 1))
    # Keep the denominator integral: converting ``2**discordant`` to a float
    # first overflows for perfectly valid (albeit unusually large) inputs.
    p_value = 2.0 * (min(lower_tail, upper_tail) / (1 << discordant))
    return min(1.0, p_value)


def newcombe_paired_risk_difference_interval(
    base: Iterable[Any],
    candidate: Iterable[Any],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Return Newcombe's hybrid-score interval for ``candidate - base``.

    This is the standard hybrid construction: Wilson score intervals are
    calculated for both paired marginal proportions and combined as
    ``d ± sqrt(marginal half-widths squared)``.  The paired rows determine the
    risk-difference orientation and sample size; the exact McNemar test remains
    the paired hypothesis test reported alongside this interval.
    """

    base_values, candidate_values = _paired_binary(base, candidate)
    confidence = _confidence(confidence)
    trials = len(base_values)
    base_successes = sum(base_values)
    candidate_successes = sum(candidate_values)
    base_proportion = base_successes / trials
    candidate_proportion = candidate_successes / trials
    base_low, base_high = wilson_interval(base_successes, trials, confidence=confidence)
    candidate_low, candidate_high = wilson_interval(
        candidate_successes, trials, confidence=confidence
    )
    difference = candidate_proportion - base_proportion
    lower = difference - math.sqrt(
        (candidate_proportion - candidate_low) ** 2 + (base_high - base_proportion) ** 2
    )
    upper = difference + math.sqrt(
        (candidate_high - candidate_proportion) ** 2 + (base_proportion - base_low) ** 2
    )
    return max(-1.0, lower), min(1.0, upper)


@dataclass(frozen=True)
class BootstrapMeanDifference:
    """A percentile bootstrap interval with its reproducibility metadata."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    seed: int
    resamples: int
    sample_size: int

    @property
    def interval(self) -> tuple[float, float]:
        return self.lower, self.upper

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "seed": self.seed,
            "resamples": self.resamples,
            "sample_size": self.sample_size,
            "method": "paired_percentile_bootstrap",
        }

    def __iter__(self):
        yield self.lower
        yield self.upper


def paired_bootstrap_mean_difference(
    base: Iterable[Any],
    candidate: Iterable[Any],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> BootstrapMeanDifference:
    """Return a deterministic paired percentile-bootstrap mean-difference CI."""

    base_values, candidate_values = _paired_real(base, candidate)
    confidence = _confidence(confidence)
    resamples = _integer("resamples", resamples, minimum=1)
    seed = _integer("seed", seed)
    differences_list: list[float] = []
    for base_value, candidate_value in zip(base_values, candidate_values):
        difference = candidate_value - base_value
        if not math.isfinite(difference):
            raise StatisticsInputError("paired differences must be finite")
        differences_list.append(difference)
    differences = tuple(differences_list)
    sample_size = len(differences)
    estimate = statistics.fmean(differences)
    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(resamples):
        total = sum(differences[rng.randrange(sample_size)] for _ in range(sample_size))
        bootstrap_means.append(total / sample_size)
    alpha = 1.0 - confidence
    return BootstrapMeanDifference(
        estimate=estimate,
        lower=_percentile(bootstrap_means, alpha / 2.0),
        upper=_percentile(bootstrap_means, 1.0 - alpha / 2.0),
        confidence=confidence,
        seed=seed,
        resamples=resamples,
        sample_size=sample_size,
    )


def _labels(name: str, values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise StatisticsInputError(f"{name} must be an iterable of labels")
    try:
        labels = tuple(values)
    except TypeError as exc:
        raise StatisticsInputError(f"{name} must be an iterable of labels") from exc
    if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise StatisticsInputError(f"{name} must contain non-empty string labels")
    if len(set(labels)) != len(labels):
        raise StatisticsInputError(f"{name} must not contain duplicate labels")
    return labels


def equal_domain_macro_aggregate(
    suite_scores: Mapping[str, Any],
    suite_domains: Mapping[str, Iterable[str]],
    *,
    domains: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Aggregate suite scores with equal weight per suite within each domain."""

    if not isinstance(suite_scores, Mapping) or not suite_scores:
        raise StatisticsInputError("suite_scores must be a non-empty mapping")
    if not isinstance(suite_domains, Mapping) or not suite_domains:
        raise StatisticsInputError("suite_domains must be a non-empty mapping")
    score_keys = set(suite_scores)
    domain_keys = set(suite_domains)
    if score_keys != domain_keys:
        raise StatisticsInputError("suite_scores and suite_domains must have identical keys")

    scores: dict[str, float] = {}
    normalized_domains: dict[str, tuple[str, ...]] = {}
    for suite_id, score in suite_scores.items():
        if not isinstance(suite_id, str) or not suite_id.strip():
            raise StatisticsInputError("suite IDs must be non-empty strings")
        scores[suite_id] = _finite_real(f"suite_scores[{suite_id!r}]", score)
        normalized_domains[suite_id] = _labels(
            f"suite_domains[{suite_id!r}]", suite_domains[suite_id]
        )

    union = tuple(sorted({domain for tags in normalized_domains.values() for domain in tags}))
    selected_domains = union if domains is None else tuple(sorted(_labels("domains", domains)))
    if set(selected_domains) != set(union):
        raise StatisticsInputError("domains must exactly match the suite domain union")

    domain_means: dict[str, float] = {}
    suite_counts: dict[str, int] = {}
    for domain in selected_domains:
        values = [
            scores[suite_id] for suite_id, tags in normalized_domains.items() if domain in tags
        ]
        if not values:
            raise StatisticsInputError(f"domain {domain!r} has no suite score")
        suite_counts[domain] = len(values)
        domain_means[domain] = statistics.fmean(values)

    return {
        "domain_means": domain_means,
        "suite_counts": suite_counts,
        "macro_score": statistics.fmean(tuple(domain_means.values())),
        "domains": list(selected_domains),
    }


def equal_domain_macro_score(
    suite_scores: Mapping[str, Any],
    suite_domains: Mapping[str, Iterable[str]],
    *,
    domains: Iterable[str] | None = None,
) -> float:
    """Return only the equal-domain macro score."""

    return float(
        equal_domain_macro_aggregate(suite_scores, suite_domains, domains=domains)["macro_score"]
    )


@dataclass(frozen=True)
class DomainGateResult:
    """Per-domain gate details and an overall conjunction result."""

    passed: bool
    failures: tuple[str, ...]
    by_domain: Mapping[str, Mapping[str, Any]]
    rule: str

    def __bool__(self) -> bool:
        return self.passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "by_domain": {key: dict(value) for key, value in self.by_domain.items()},
            "rule": self.rule,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _gate_inputs(
    deltas: Mapping[str, Any], intervals: Mapping[str, Sequence[Any]]
) -> tuple[tuple[str, ...], dict[str, float], dict[str, tuple[float, float]]]:
    if not isinstance(deltas, Mapping) or not deltas:
        raise StatisticsInputError("deltas must be a non-empty mapping")
    if not isinstance(intervals, Mapping) or not intervals:
        raise StatisticsInputError("intervals must be a non-empty mapping")
    if set(deltas) != set(intervals):
        raise StatisticsInputError("deltas and intervals must have identical domain keys")
    if any(not isinstance(domain, str) or not domain.strip() for domain in deltas):
        raise StatisticsInputError("domain keys must be non-empty strings")
    ordered = tuple(sorted(deltas))
    finite_deltas = {
        domain: _finite_real(f"deltas[{domain!r}]", deltas[domain]) for domain in ordered
    }
    finite_intervals: dict[str, tuple[float, float]] = {}
    for domain in ordered:
        interval = intervals[domain]
        if isinstance(interval, (str, bytes, bytearray)):
            raise StatisticsInputError(f"interval for {domain!r} must contain two bounds")
        try:
            bounds = tuple(interval)
        except TypeError as exc:
            raise StatisticsInputError(f"interval for {domain!r} must contain two bounds") from exc
        if len(bounds) != 2:
            raise StatisticsInputError(f"interval for {domain!r} must contain two bounds")
        lower = _finite_real(f"interval[{domain!r}][0]", bounds[0])
        upper = _finite_real(f"interval[{domain!r}][1]", bounds[1])
        if lower > upper:
            raise StatisticsInputError(f"interval for {domain!r} has reversed bounds")
        finite_intervals[domain] = (lower, upper)
    return ordered, finite_deltas, finite_intervals


def domain_no_regression_gate(
    deltas: Mapping[str, Any],
    intervals: Mapping[str, Sequence[Any]],
    *,
    tolerance: float = 0.05,
) -> DomainGateResult:
    """Apply the protocol's point and lower-bound no-regression guard."""

    tolerance = _finite_real("tolerance", tolerance)
    if tolerance < 0.0:
        raise StatisticsInputError("tolerance must be non-negative")
    domains, finite_deltas, finite_intervals = _gate_inputs(deltas, intervals)
    by_domain: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    threshold = -tolerance
    for domain in domains:
        delta = finite_deltas[domain]
        lower = finite_intervals[domain][0]
        point_ok = delta >= threshold
        interval_ok = lower > threshold
        passed = point_ok and interval_ok
        by_domain[domain] = {
            "delta": delta,
            "lower_bound": lower,
            "threshold": threshold,
            "point_ok": point_ok,
            "interval_ok": interval_ok,
            "passed": passed,
        }
        if not passed:
            failures.append(domain)
    return DomainGateResult(
        passed=not failures,
        failures=tuple(failures),
        by_domain=by_domain,
        rule="delta >= -tolerance and lower_bound > -tolerance",
    )


def domain_safety_gate(
    safety_increases: Mapping[str, Any],
    intervals: Mapping[str, Sequence[Any]],
    *,
    maximum_increase: float = 0.02,
) -> DomainGateResult:
    """Apply the protocol's point and one-sided upper safety guard."""

    maximum_increase = _finite_real("maximum_increase", maximum_increase)
    if maximum_increase < 0.0:
        raise StatisticsInputError("maximum_increase must be non-negative")
    domains, finite_increases, finite_intervals = _gate_inputs(safety_increases, intervals)
    by_domain: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for domain in domains:
        increase = finite_increases[domain]
        upper = finite_intervals[domain][1]
        point_ok = increase <= maximum_increase
        interval_ok = upper <= maximum_increase
        passed = point_ok and interval_ok
        by_domain[domain] = {
            "increase": increase,
            "upper_bound": upper,
            "threshold": maximum_increase,
            "point_ok": point_ok,
            "interval_ok": interval_ok,
            "passed": passed,
        }
        if not passed:
            failures.append(domain)
    return DomainGateResult(
        passed=not failures,
        failures=tuple(failures),
        by_domain=by_domain,
        rule="increase <= maximum_increase and upper_bound <= maximum_increase",
    )


def portfolio_domain_gates(
    deltas: Mapping[str, Any],
    delta_intervals: Mapping[str, Sequence[Any]],
    safety_increases: Mapping[str, Any],
    safety_intervals: Mapping[str, Sequence[Any]],
    *,
    no_regression_tolerance: float = 0.05,
    maximum_safety_increase: float = 0.02,
) -> dict[str, Any]:
    """Evaluate both conjunctive per-domain guards and return JSON-safe details."""

    if not all(
        isinstance(value, Mapping)
        for value in (deltas, delta_intervals, safety_increases, safety_intervals)
    ):
        raise StatisticsInputError("all portfolio gate inputs must be mappings")
    if not (set(deltas) == set(delta_intervals) == set(safety_increases) == set(safety_intervals)):
        raise StatisticsInputError("all portfolio gate inputs must have identical domain keys")
    no_regression = domain_no_regression_gate(
        deltas,
        delta_intervals,
        tolerance=no_regression_tolerance,
    )
    safety = domain_safety_gate(
        safety_increases,
        safety_intervals,
        maximum_increase=maximum_safety_increase,
    )
    return {
        "passed": no_regression.passed and safety.passed,
        "no_regression": no_regression.as_dict(),
        "safety": safety.as_dict(),
    }


# Descriptive aliases make the protocol terminology easy to discover without
# duplicating implementations.
wilson_score_interval = wilson_interval
exact_mcnemar_pvalue = exact_mcnemar_two_sided
newcombe_hybrid_score_interval = newcombe_paired_risk_difference_interval


__all__ = [
    "BootstrapMeanDifference",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE",
    "DomainGateResult",
    "StatisticsInputError",
    "domain_no_regression_gate",
    "domain_safety_gate",
    "equal_domain_macro_aggregate",
    "equal_domain_macro_score",
    "exact_mcnemar_pvalue",
    "exact_mcnemar_two_sided",
    "mcnemar_discordant_counts",
    "newcombe_hybrid_score_interval",
    "newcombe_paired_risk_difference_interval",
    "paired_bootstrap_mean_difference",
    "portfolio_domain_gates",
    "wilson_interval",
    "wilson_score_interval",
]

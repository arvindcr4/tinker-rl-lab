from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import CONDITION_ORDER, REGIME_ORDER


SCREENING_SEEDS = (11, 23, 37)
EVALUATION_STEPS = (0, 20, 40, 60, 80, 100)


class AnalysisContractError(RuntimeError):
    """Accepted screening evidence is incomplete or violates the frozen analysis schema."""


def _full_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("schema_version") == "flagship-pilot-acceptance-v1":
        if record.get("status") != "accepted":
            raise AnalysisContractError("screening receipt is not accepted")
        return record["full_record"]
    if record.get("schema_version") == "flagship-pilot-unit-v1":
        return record
    raise AnalysisContractError("unknown screening record schema")


def _record_map(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    mapped: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for source in records:
        record = _full_record(source)
        key = (str(record["condition"]), str(record["regime"]), int(record["seed"]))
        if key in mapped:
            raise AnalysisContractError(f"duplicate screening unit: {key}")
        mapped[key] = record
    expected = {
        (condition, regime, seed)
        for condition in CONDITION_ORDER
        for regime in REGIME_ORDER
        for seed in SCREENING_SEEDS
    }
    missing = expected - set(mapped)
    extra = set(mapped) - expected
    if missing or extra:
        raise AnalysisContractError(
            f"screening matrix mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    return mapped


def _receipts(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    receipts = record["manifest"]["gradient_receipts"]
    if len(receipts) != 100 or [item["step"] for item in receipts] != list(range(1, 101)):
        raise AnalysisContractError("unit gradient receipts are not contiguous 1..100")
    return receipts


def normalized_auc(record: Mapping[str, Any]) -> float:
    evaluations = record["evaluations"]
    steps = tuple(int(item["step"]) for item in evaluations)
    if steps != EVALUATION_STEPS:
        raise AnalysisContractError(f"unit evaluation steps are not frozen: {steps}")
    values = [float(item["accuracy"]) for item in evaluations]
    area = sum(
        (steps[index + 1] - steps[index]) * (values[index] + values[index + 1]) / 2.0
        for index in range(len(steps) - 1)
    )
    return area / 100.0


def _standardized_mean(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    if len(values) < 2:
        raise AnalysisContractError("paired standardized effect requires at least two seeds")
    sd = statistics.stdev(values)
    if sd == 0:
        return math.inf if mean != 0 else 0.0
    return abs(mean) / sd


def _comparison(
    receipt: Mapping[str, Any], *, selected_vs_intended: bool = False
) -> tuple[str, float | None, float | None]:
    prefix = "selected_vs_intended_" if selected_vs_intended else "gradient_"
    relation = receipt.get(f"{prefix}relation")
    allowed = (
        {"nonzero", "joint_zero", "selected_zero", "intended_zero"}
        if selected_vs_intended
        else {"nonzero", "joint_zero", "intended_zero", "native_zero"}
    )
    if relation not in allowed:
        raise AnalysisContractError(f"invalid gradient relation: {relation}")
    cosine = receipt.get(f"{prefix}cosine")
    relative_l2 = receipt.get(f"{prefix}relative_l2")
    if relation == "nonzero":
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in (cosine, relative_l2)
        ):
            raise AnalysisContractError("nonzero gradient comparison lacks finite diagnostics")
        return relation, float(cosine), float(relative_l2)
    if cosine is not None or relative_l2 is not None:
        raise AnalysisContractError("zero-vector gradient comparison has fabricated diagnostics")
    return str(relation), None, None


def _comparison_effect(receipt: Mapping[str, Any], *, selected_vs_intended: bool = False) -> float:
    relation, cosine, _ = _comparison(receipt, selected_vs_intended=selected_vs_intended)
    if relation == "joint_zero":
        return 0.0
    if relation == "nonzero":
        assert cosine is not None
        return 1.0 - cosine
    return 1.0


def _comparison_equivalent(
    receipt: Mapping[str, Any], *, selected_vs_intended: bool = False
) -> bool:
    relation, cosine, relative_l2 = _comparison(receipt, selected_vs_intended=selected_vs_intended)
    if relation == "joint_zero":
        return True
    if relation != "nonzero":
        return False
    assert cosine is not None and relative_l2 is not None
    return cosine >= 0.999 and relative_l2 <= 0.01


def _comparison_diverges(receipt: Mapping[str, Any]) -> bool:
    relation, cosine, relative_l2 = _comparison(receipt)
    if relation == "joint_zero":
        return False
    if relation != "nonzero":
        return True
    assert cosine is not None and relative_l2 is not None
    return cosine <= 0.99 and relative_l2 >= 0.05


def _relation_counts(
    receipts: Sequence[Mapping[str, Any]], *, selected_vs_intended: bool = False
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for receipt in receipts:
        relation, _, _ = _comparison(receipt, selected_vs_intended=selected_vs_intended)
        counts[relation] = counts.get(relation, 0) + 1
    return dict(sorted(counts.items()))


def screening_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped = _record_map(records)
    mechanism: dict[str, Any] = {}
    mechanism_pass = True
    for regime in REGIME_ORDER:
        mechanism[regime] = {}
        for seed in SCREENING_SEEDS:
            reference = mapped[("intended_full", regime, seed)]
            receipts = _receipts(reference)
            if regime == "filtered_variable_length":
                qualifying = sum(_comparison_diverges(item) for item in receipts)
                passed = qualifying >= 20
            else:
                qualifying = sum(_comparison_equivalent(item) for item in receipts)
                passed = qualifying >= 95
            mechanism[regime][str(seed)] = {
                "qualifying_steps": qualifying,
                "required_steps": 20 if regime == "filtered_variable_length" else 95,
                "gradient_relation_counts": _relation_counts(receipts),
                "pass": passed,
            }
            mechanism_pass &= passed

    attribution: dict[str, Any] = {}
    attribution_pass = True
    for seed in SCREENING_SEEDS:
        intended_receipts = _receipts(mapped[("intended_full", "filtered_variable_length", seed)])
        reduction_receipts = _receipts(mapped[("reduction_only", "filtered_variable_length", seed)])
        epsilon_receipts = _receipts(mapped[("epsilon_only", "filtered_variable_length", seed)])
        native_effect = statistics.fmean(_comparison_effect(item) for item in intended_receipts)
        reduction_effect = statistics.fmean(
            _comparison_effect(item, selected_vs_intended=True) for item in reduction_receipts
        )
        ratio = reduction_effect / native_effect if native_effect > 1e-12 else 0.0
        epsilon_equivalent = sum(
            _comparison_equivalent(item, selected_vs_intended=True) for item in epsilon_receipts
        )
        passed = ratio >= 0.80 and epsilon_equivalent >= 95
        attribution[str(seed)] = {
            "native_relation_effect": native_effect,
            "reduction_relation_effect": reduction_effect,
            "reproduction_ratio": ratio,
            "epsilon_equivalent_steps": epsilon_equivalent,
            "native_gradient_relation_counts": _relation_counts(intended_receipts),
            "reduction_gradient_relation_counts": _relation_counts(
                reduction_receipts, selected_vs_intended=True
            ),
            "epsilon_gradient_relation_counts": _relation_counts(
                epsilon_receipts, selected_vs_intended=True
            ),
            "pass": passed,
        }
        attribution_pass &= passed

    auc_differences: dict[str, list[float]] = {regime: [] for regime in REGIME_ORDER}
    final_differences: dict[str, list[float]] = {regime: [] for regime in REGIME_ORDER}
    matched_compute_pass = True
    for regime in REGIME_ORDER:
        for seed in SCREENING_SEEDS:
            block = [mapped[(condition, regime, seed)] for condition in CONDITION_ORDER]
            corpus_fingerprints = {record["corpus_fingerprint"] for record in block}
            token_charges = {
                int(record["token_flop_ledger"]["charged_generated_tokens"]) for record in block
            }
            matched_compute_pass &= len(corpus_fingerprints) == 1 and len(token_charges) == 1
            intended = mapped[("intended_full", regime, seed)]
            native = mapped[("native_trl", regime, seed)]
            auc_differences[regime].append(normalized_auc(intended) - normalized_auc(native))
            final_differences[regime].append(
                float(intended["evaluations"][-1]["accuracy"])
                - float(native["evaluations"][-1]["accuracy"])
            )

    filtered_auc = auc_differences["filtered_variable_length"]
    signs = {math.copysign(1.0, value) if value != 0 else 0.0 for value in filtered_auc}
    learning_direction = len(signs) == 1 and 0.0 not in signs
    standardized = _standardized_mean(filtered_auc)
    filtered_learning_pass = learning_direction and standardized >= 0.5
    balanced_equivalence = all(
        abs(value) <= 0.02 for value in auc_differences["balanced_equal_length"]
    ) and all(abs(value) <= 0.01 for value in final_differences["balanced_equal_length"])
    final_noninferiority = all(
        value >= -0.01 for regime in REGIME_ORDER for value in final_differences[regime]
    )
    learning_pass = filtered_learning_pass and balanced_equivalence and final_noninferiority

    go = mechanism_pass and attribution_pass and learning_pass and matched_compute_pass
    return {
        "schema_version": "flagship-pilot-screening-gate-v2",
        "verdict": "GO" if go else "KILL",
        "mechanism": mechanism,
        "mechanism_pass": mechanism_pass,
        "causal_attribution": attribution,
        "causal_attribution_pass": attribution_pass,
        "learning": {
            "auc_differences": auc_differences,
            "final_differences": final_differences,
            "filtered_sign_consistent": learning_direction,
            "filtered_standardized_mean": standardized,
            "balanced_equivalence": balanced_equivalence,
            "final_noninferiority": final_noninferiority,
            "pass": learning_pass,
        },
        "matched_compute_pass": matched_compute_pass,
    }


@dataclass(frozen=True, slots=True)
class PowerEstimate:
    effect_power: float
    tost_power: float | None


def _monte_carlo(
    values: Sequence[float],
    *,
    n: int,
    draws: int,
    alpha: float,
    seed: int,
    equivalence_margin: float | None,
) -> PowerEstimate:
    if n != 5 or not math.isclose(alpha, 0.0125, rel_tol=0.0, abs_tol=1e-15):
        raise AnalysisContractError("power gate is frozen to n=5 and alpha=0.0125")
    two_sided_critical = 4.314656048909503
    one_sided_critical = 3.495405932516977
    observed = np.asarray(values, dtype=np.float64)
    if observed.shape != (3,) or not np.isfinite(observed).all():
        raise AnalysisContractError("power input must contain three finite screening effects")
    mean = float(observed.mean())
    sd = float(observed.std(ddof=1))
    if sd == 0.0:
        effect_power = float(mean != 0.0)
        tost_power = None
        if equivalence_margin is not None:
            tost_power = float(abs(mean) < equivalence_margin)
        return PowerEstimate(effect_power=effect_power, tost_power=tost_power)
    generator = np.random.Generator(np.random.PCG64(seed))
    simulations = generator.normal(loc=mean, scale=sd, size=(draws, n))
    simulated_mean = simulations.mean(axis=1)
    simulated_sd = simulations.std(axis=1, ddof=1)
    standard_error = simulated_sd / math.sqrt(n)
    valid = standard_error > 0
    effect_t = np.zeros(draws, dtype=np.float64)
    effect_t[valid] = np.abs(simulated_mean[valid] / standard_error[valid])
    effect_power = float(np.mean(valid & (effect_t > two_sided_critical)))
    tost_power = None
    if equivalence_margin is not None:
        lower_t = np.zeros(draws, dtype=np.float64)
        upper_t = np.zeros(draws, dtype=np.float64)
        lower_t[valid] = (simulated_mean[valid] + equivalence_margin) / standard_error[valid]
        upper_t[valid] = (simulated_mean[valid] - equivalence_margin) / standard_error[valid]
        tost_power = float(
            np.mean(valid & (lower_t > one_sided_critical) & (upper_t < -one_sided_critical))
        )
    return PowerEstimate(effect_power=effect_power, tost_power=tost_power)


def confirmatory_power_gate(
    screening_report: Mapping[str, Any],
    *,
    draws: int = 100_000,
    n: int = 5,
    alpha: float = 0.0125,
    seed: int = 20_260_720,
) -> dict[str, Any]:
    if screening_report.get("verdict") != "GO":
        raise AnalysisContractError("confirmatory power cannot be estimated after a KILL gate")
    auc = screening_report["learning"]["auc_differences"]
    pooled = [
        statistics.fmean(
            [auc["balanced_equal_length"][index], auc["filtered_variable_length"][index]]
        )
        for index in range(3)
    ]
    auc_power = _monte_carlo(
        pooled,
        n=n,
        draws=draws,
        alpha=alpha,
        seed=seed,
        equivalence_margin=None,
    )
    final = screening_report["learning"]["final_differences"]
    tost: dict[str, float] = {}
    for offset, regime in enumerate(REGIME_ORDER, start=1):
        estimate = _monte_carlo(
            final[regime],
            n=n,
            draws=draws,
            alpha=alpha,
            seed=seed + offset,
            equivalence_margin=0.01,
        )
        assert estimate.tost_power is not None
        tost[regime] = estimate.tost_power
    passed = auc_power.effect_power >= 0.80 and all(value >= 0.80 for value in tost.values())
    return {
        "schema_version": "flagship-pilot-confirmatory-power-v1",
        "verdict": "GO" if passed else "STOP_UNDERPOWERED",
        "draws": draws,
        "n": n,
        "alpha": alpha,
        "seed": seed,
        "pooled_auc_effects": pooled,
        "pooled_auc_power": auc_power.effect_power,
        "final_quality_tost_power": tost,
    }

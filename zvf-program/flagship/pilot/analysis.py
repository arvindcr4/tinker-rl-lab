from __future__ import annotations

import math
import statistics
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import CONDITION_ORDER, REGIME_ORDER, canonical_fingerprint


SCREENING_SEEDS = (11, 23, 37)
EVALUATION_STEPS = (0, 20, 40, 60, 80, 100)


class AnalysisContractError(RuntimeError):
    """Accepted screening evidence is incomplete or violates the frozen analysis schema."""


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _full_record(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Consume verifier acceptance receipts, never bare unit mappings.

    Checks integrity and identity, not remote authenticity. Receipt files must be
    obtained from the independent verifier; hashes alone are not signatures.
    """
    try:
        if receipt.get("schema_version") != "flagship-pilot-acceptance-v1" or receipt.get("status") != "accepted":
            raise AnalysisContractError("independently verified accepted receipt required")
        record = receipt["full_record"]
        if record.get("schema_version") != "flagship-pilot-unit-v1" or record.get("status") != "completed":
            raise AnalysisContractError("completed full record required")
        payload = dict(record)
        fingerprint = payload.pop("fingerprint", None)
        if not _digest(fingerprint) or canonical_fingerprint(payload) != fingerprint:
            raise AnalysisContractError("full record fingerprint mismatch")
        plan = record["manifest"]["plan"]
        plan_payload = dict(plan)
        plan_fingerprint = plan_payload.pop("fingerprint", None)
        if not _digest(plan_fingerprint) or canonical_fingerprint(plan_payload) != plan_fingerprint:
            raise AnalysisContractError("plan fingerprint mismatch")
        for field in ("condition", "regime", "seed"):
            if receipt["unit"][field] != record[field] or plan["unit"][field] != record[field]:
                raise AnalysisContractError(f"acceptance identity mismatch: {field}")
        for field in ("unit_fingerprint", "corpus_fingerprint"):
            if not _digest(record[field]) or receipt[field] != record[field]:
                raise AnalysisContractError(f"acceptance {field} mismatch")
        if record["unit_fingerprint"] != plan_fingerprint:
            raise AnalysisContractError("unit plan identity mismatch")
        sources = record["manifest"]["source_manifest"]
        if not sources or not all(_digest(v) for v in sources.values()):
            raise AnalysisContractError("invalid source identity")
        if any(plan["source_bindings"].get(k) != v for k, v in sources.items()):
            raise AnalysisContractError("source identity does not match plan")
        charge = record["token_flop_ledger"]["charged_generated_tokens"]
        if type(charge) is not int or charge <= 0:
            raise AnalysisContractError("invalid generated token charge")
        normalized_auc(record)
        for evaluation in record["evaluations"]:
            if evaluation.get("heldout_n") != 128 or type(evaluation.get("correct")) is not int:
                raise AnalysisContractError("evaluation requires heldout_n=128 and integer correct count")
            if not 0 <= evaluation["correct"] <= 128 or not math.isclose(evaluation["accuracy"], evaluation["correct"]/128, abs_tol=1e-12):
                raise AnalysisContractError("evaluation correct count/accuracy mismatch")
        _receipts(record)
        return record
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisContractError(f"malformed acceptance evidence: {exc}") from exc


def _record_map(
    records: Sequence[Mapping[str, Any]],
    *, regimes: Sequence[str] = REGIME_ORDER,
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    mapped: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    identities = set()
    for source in records:
        record = _full_record(source)
        identities.add(canonical_fingerprint(record["manifest"]["plan"]["source_bindings"]))
        if type(record["seed"]) is not int:
            raise AnalysisContractError("seed must be an integer")
        key = (str(record["condition"]), str(record["regime"]), int(record["seed"]))
        if key in mapped:
            raise AnalysisContractError(f"duplicate screening unit: {key}")
        mapped[key] = record
    expected = {
        (condition, regime, seed)
        for condition in CONDITION_ORDER
        for regime in regimes
        for seed in SCREENING_SEEDS
    }
    if len(identities) != 1:
        raise AnalysisContractError("mixed protocol/source identities")
    missing = expected - set(mapped)
    extra = set(mapped) - expected
    if missing or extra:
        raise AnalysisContractError(
            f"screening matrix mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    for regime in regimes:
        for seed in SCREENING_SEEDS:
            groups = {tuple(r["group_fingerprint"] for r in _receipts(mapped[(c, regime, seed)])) for c in CONDITION_ORDER}
            if len(groups) != 1:
                raise AnalysisContractError("paired conditions have different replay group identities")
    return mapped


def _receipts(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    receipts = record["manifest"]["gradient_receipts"]
    if len(receipts) != 100 or any(type(item.get("step")) is not int for item in receipts) or [item["step"] for item in receipts] != list(range(1, 101)):
        raise AnalysisContractError("unit gradient receipts are not contiguous 1..100")
    for item in receipts:
        if item.get("condition") != record["condition"] or not _digest(item.get("group_fingerprint")):
            raise AnalysisContractError("gradient receipt condition/group identity mismatch")
        for key in ("intended_gradient_norm", "native_gradient_norm", "selected_gradient_norm"):
            value = item.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise AnalysisContractError("gradient norm must be finite and nonnegative")
        _comparison(item)
        _comparison(item, selected_vs_intended=True)
    return receipts


def normalized_auc(record: Mapping[str, Any]) -> float:
    evaluations = record["evaluations"]
    if any(type(item.get("step")) is not int for item in evaluations):
        raise AnalysisContractError("evaluation steps must be integers")
    steps = tuple(item["step"] for item in evaluations)
    if steps != EVALUATION_STEPS:
        raise AnalysisContractError(f"unit evaluation steps are not frozen: {steps}")
    values = [item["accuracy"] for item in evaluations]
    if not all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise AnalysisContractError("invalid evaluation accuracy")
    area = sum(
        (steps[index + 1] - steps[index]) * (values[index] + values[index + 1]) / 2.0
        for index in range(len(steps) - 1)
    )
    return area / 100.0


def _standardized_mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(v) for v in values):
        raise AnalysisContractError("finite paired effects required")
    mean = statistics.fmean(values)
    if len(values) < 2:
        raise AnalysisContractError("paired standardized effect requires at least two seeds")
    sd = statistics.stdev(values)
    if sd == 0:
        return math.copysign(math.inf, mean) if mean != 0 else 0.0
    return mean / sd


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
        if not -1 <= cosine <= 1 or relative_l2 < 0:
            raise AnalysisContractError("gradient diagnostics outside valid domain")
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
            "attribution_metric": "angular_discrepancy_1_minus_cosine_with_zero_relation_convention",
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
    learning_direction = all(value > 0 for value in filtered_auc)
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
        "schema_version": "flagship-pilot-screening-gate-v3-corrected-direction",
        "limitations": "Three-seed screening heuristics, not confirmatory inference; practical margins require scientific justification. Frozen v2 stored outputs are not rewritten.",
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
        "absolute_evaluations": [{"condition": key[0], "regime": key[1], "seed": key[2],
                                  "evaluations": value["evaluations"]} for key, value in mapped.items()],
        "auc_seed_summaries": {regime: {"mean": statistics.fmean(values),
            "sd": statistics.stdev(values), "n": len(values),
            "standard_error": statistics.stdev(values)/math.sqrt(len(values)),
            "interpretation": "descriptive independent-seed uncertainty, not a formal efficacy test"}
            for regime, values in auc_differences.items()},
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
    if type(draws) is not int or draws <= 0:
        raise AnalysisContractError("draws must be a positive integer")
    if equivalence_margin is not None and (not math.isfinite(equivalence_margin) or equivalence_margin <= 0):
        raise AnalysisContractError("equivalence margin must be positive and finite")
    from scipy.stats import t
    two_sided_critical = float(t.ppf(1-alpha/2, df=n-1))
    one_sided_critical = float(t.ppf(1-alpha, df=n-1))
    observed = np.asarray(values, dtype=np.float64)
    if observed.shape != (3,) or not np.isfinite(observed).all():
        raise AnalysisContractError("power input must contain three finite screening effects")
    mean = float(observed.mean())
    sd = float(observed.std(ddof=1))
    if sd == 0.0:
        raise AnalysisContractError("DEGENERATE_VARIANCE: plug-in studentized power is not estimable")
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
    if screening_report.get("schema_version") != "flagship-pilot-screening-gate-v3-corrected-direction":
        raise AnalysisContractError("power sensitivity requires corrected two-regime screening report")
    # Never average scientifically different regimes into one effect.
    results = {}
    from scipy.stats import chi2
    for offset, regime in enumerate(REGIME_ORDER):
        values = screening_report["learning"]["auc_differences"][regime]
        array = np.asarray(values, dtype=float)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise AnalysisContractError("three finite effects required per regime")
        sd = float(array.std(ddof=1))
        variance_interval = [2*sd**2/float(chi2.ppf(.975, 2)), 2*sd**2/float(chi2.ppf(.025, 2))]
        scenarios = []
        # Sensitivity grid, not a selected scientifically meaningful alternative.
        for mean_scale in (0.0, 0.5, 1.0):
            for sd_scale in (1.0, 2.0):
                mean = float(array.mean())*mean_scale
                centered = (array-array.mean())*sd_scale + mean
                try:
                    estimate = _monte_carlo(centered, n=n, draws=draws, alpha=alpha,
                        seed=seed+offset, equivalence_margin=None)
                    scenarios.append({"mean": mean, "sd": sd*sd_scale, "effect_power": estimate.effect_power})
                except AnalysisContractError as exc:
                    if "DEGENERATE_VARIANCE" not in str(exc):
                        raise
                    scenarios.append({"mean": mean, "sd": sd*sd_scale, "effect_power": None,
                                      "status": "DEGENERATE_VARIANCE"})
        results[regime] = {"observed_effects": list(values), "normal_variance_ci95": variance_interval,
                           "sensitivity": scenarios}
    return {"schema_version": "flagship-pilot-power-sensitivity-v2",
        "verdict": "EXPLORATORY_ONLY", "regime_specific": results,
        "n": n, "draws": draws, "alpha": alpha, "seed": seed,
        "final_quality_equivalence": "NOT_APPROVED: prespecified margin and variance sensitivity required",
        "limitations": "Three-seed normal plug-in sensitivity, not confirmatory approval. Effects/margins and heldout sample size must be prespecified independently. Zero variance is not estimable; no nonexistent regime pooling."}


BALANCED_CANDIDATE_PROTOCOL = {
    "schema_version": "flagship-balanced-analysis-candidate-v1",
    "status": "candidate_not_authorized_for_execution",
    "regimes": ["balanced_equal_length"],
    "seeds": list(SCREENING_SEEDS),
    "conditions": list(CONDITION_ORDER),
    "estimand": "paired intended-minus-native normalized AUC on balanced_equal_length",
    "decision": "descriptive_only_no_original_pilot_GO",
}


def balanced_candidate_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """New analysis identity; cannot authorize training or substitute for frozen v2."""
    regime = "balanced_equal_length"
    mapped = _record_map(records, regimes=(regime,))
    effects = []
    finals = []
    diagnostics = []
    for seed in SCREENING_SEEDS:
        block = [mapped[(c, regime, seed)] for c in CONDITION_ORDER]
        if len({r["corpus_fingerprint"] for r in block}) != 1:
            raise AnalysisContractError("unmatched balanced corpus")
        if len({r["token_flop_ledger"]["charged_generated_tokens"] for r in block}) != 1:
            raise AnalysisContractError("unmatched balanced compute")
        intended, native = (mapped[(c, regime, seed)] for c in ("intended_full", "native_trl"))
        effects.append(normalized_auc(intended)-normalized_auc(native))
        finals.append({"seed": seed, "intended": intended["evaluations"][-1],
                       "native": native["evaluations"][-1]})
        receipts = _receipts(intended)
        diagnostics.append({"seed": seed, "relations": _relation_counts(receipts),
            "cosines": [r["gradient_cosine"] for r in receipts],
            "relative_l2": [r["gradient_relative_l2"] for r in receipts],
            "intended_norms": [r.get("intended_gradient_norm") for r in receipts],
            "native_norms": [r.get("native_gradient_norm") for r in receipts]})
    from scipy.stats import t
    mean = statistics.fmean(effects)
    half = float(t.ppf(.975, len(effects)-1))*statistics.stdev(effects)/math.sqrt(len(effects))
    return {"schema_version": BALANCED_CANDIDATE_PROTOCOL["schema_version"],
        "analysis_protocol_fingerprint": canonical_fingerprint(BALANCED_CANDIDATE_PROTOCOL),
        "verdict": "DESCRIPTIVE_ONLY_PENDING_PROTOCOL_AUTHORIZATION",
        "regimes": [regime], "paired_seed_auc_differences": effects,
        "mean_auc_difference": mean, "exploratory_normal_seed_mean_ci95": [mean-half, mean+half],
        "final_evaluations": finals, "gradient_diagnostics": diagnostics,
        "limitations": ["Three seeds cannot establish broad generalization; interval assumes independent normal seed effects.",
            "128 shared heldout examples have 1/128 accuracy resolution; paired example evidence is required for paired uncertainty.",
            "Repeated checkpoint evaluation is not a separate final test set.",
            "No filtered-regime claim, power approval, formal equivalence or original pilot GO."]}

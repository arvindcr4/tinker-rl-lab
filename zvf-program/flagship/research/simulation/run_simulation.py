#!/usr/bin/env python3
"""Synthetic, matched-token stress test for frozen ZVF flagship policies.

This script intentionally models no real optimizer, task, or future outcome.
It is a decision aid: a latent Bernoulli-success task model plus configurable
verifier noise makes the cost/signal consequences of allocation rules visible.
See the generated report for the assumptions and limits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
E1_PATH = ROOT / "zvf-program/colab-experiments/results/e1_grad_signal.json"
SCENARIOS_PATH = HERE / "scenarios.json"
POLICIES = (
    "static_g8",
    "static_g16",
    "symmetric_zvf",
    "failure_only",
    "boundary_aware",
    "full_triage",
    "stateful_bandit_diagnostic",
)
REGISTERED_POLICIES = POLICIES[:6]


@dataclass
class Outcome:
    policy: str
    regime: str
    replicate: int
    auc_per_token: float
    final_quality: float
    useful_gradient_fraction: float
    useful_groups_per_1k_rollouts: float
    observed_contrast_fraction: float
    false_contrast_fraction: float
    all_wrong_fraction: float
    all_correct_fraction: float
    expanded_groups: int
    retired_groups: int
    rollout_count: int
    generated_tokens: int
    training_flops: float
    updates: int


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def sample_difficulty(
    rng: np.random.Generator, spec: dict[str, Any], size: int, shift: float
) -> np.ndarray:
    if spec["kind"] == "normal":
        return rng.normal(spec["mean"] + shift, spec["std"], size=size)
    choices = rng.choice(len(spec["weights"]), size=size, p=spec["weights"])
    means = np.asarray(spec["means"], dtype=float)[choices] + shift
    stds = np.asarray(spec["stds"], dtype=float)[choices]
    return rng.normal(means, stds)


def difficulty_bin(difficulty: np.ndarray) -> np.ndarray:
    return np.where(difficulty < -0.5, 0, np.where(difficulty > 0.8, 2, 1))


def observe(
    rng: np.random.Generator,
    p: np.ndarray,
    draws: np.ndarray,
    sensitivity: float,
    specificity: float,
) -> tuple[np.ndarray, np.ndarray]:
    truth = rng.random((len(p), draws.shape[1])) < p[:, None]
    becomes_positive = rng.random(truth.shape)
    observed = np.where(truth, becomes_positive < sensitivity, becomes_positive > specificity)
    return truth, observed


def wilson_upper(successes: np.ndarray, totals: np.ndarray, z: float = 1.96) -> np.ndarray:
    totals = np.maximum(totals.astype(float), 1.0)
    p = successes / totals
    denom = 1.0 + z * z / totals
    centre = p + z * z / (2.0 * totals)
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * totals)) / totals)
    return np.clip((centre + half) / denom, 0.0, 1.0)


def policy_scores(
    policy: str,
    all_wrong: np.ndarray,
    group_bins: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return which all-wrong groups to expand and their priority.

    The registered full_triage rule is intentionally simple: a bin-level
    Wilson upper bound, seeded with a weak E1-derived prior, estimates whether
    another eight rollouts have enough expected latent contrast to justify the
    compute. The bandit is a non-registered diagnostic, not an oracle.
    """
    eligible = all_wrong.copy()
    if policy == "symmetric_zvf":
        return eligible, eligible.astype(float)
    if policy in ("failure_only", "boundary_aware"):
        return eligible, eligible.astype(float)
    if policy == "full_triage":
        successes = alpha[group_bins] - 1.0
        totals = alpha[group_bins] + beta[group_bins] - 2.0
        p_upper = wilson_upper(successes, totals)
        # Expected *valid* extra-group yield. Near p=0, no amount of G=16
        # will create useful contrast; near p=1 this action is ineligible.
        contrast = 1.0 - (1.0 - p_upper) ** 8
        signal = 4.0 * p_upper * (1.0 - p_upper)
        score = contrast * signal
        return eligible & (score >= 0.065), score
    if policy == "stateful_bandit_diagnostic":
        p_draw = rng.beta(alpha[group_bins], beta[group_bins])
        contrast = 1.0 - (1.0 - p_draw) ** 8
        score = contrast * (4.0 * p_draw * (1.0 - p_draw))
        # A small exploration probability makes the policy stateful rather
        # than a fixed posterior threshold. It remains implementable only if
        # difficulty strata are available and stable, hence diagnostic status.
        explore = rng.random(len(score)) < 0.08
        return eligible & ((score >= 0.055) | explore), score
    raise ValueError(f"unknown policy: {policy}")


def one_run(
    policy: str,
    regime: str,
    cfg: dict[str, Any],
    e1: dict[str, Any],
    replicate: int,
    seed: int,
    prior_strength: float = 4.0,
    learning_rate_multiplier: float = 1.0,
) -> Outcome:
    rng = np.random.default_rng(seed)
    n_batch = int(cfg["batch_prompts"])
    budget_rollouts = int(n_batch * 16 * cfg["updates_at_g16"])
    completion_tokens = int(cfg["completion_tokens"])
    sensitivity = float(cfg["verifier_sensitivity"])
    specificity = float(cfg["verifier_specificity"])

    # E1 is used only as a weak, declared simulation calibration convention:
    # its mean reported p is the centre of a low-strength Beta prior, and the
    # two signal correlations scale a synthetic learning-rate constant. The
    # resulting effect sizes are never treated as empirical E1 estimates.
    e1_mean_p = float(np.mean([row["mean_p"] for row in e1["by_difficulty"].values()]))
    alpha = np.full(3, 1.0 + max(1e-3, e1_mean_p * prior_strength))
    beta = np.full(3, 1.0 + max(1e-3, (1.0 - e1_mean_p) * prior_strength))
    e1_signal_anchor = 0.5 * (
        float(e1["pearson_gradnorm_vs_p1mp"]) + float(e1["pearson_gradnorm_vs_GU"])
    )
    learning_rate = 0.15 * e1_signal_anchor * float(cfg["learning_rate_scale"]) * learning_rate_multiplier

    active_difficulty = sample_difficulty(rng, cfg["difficulty"], int(cfg["pool_size"]), 0.0)
    heldout_difficulty = sample_difficulty(
        np.random.default_rng(seed + 8_000_000), cfg["difficulty"], 2_000, 0.0
    )
    skill = float(cfg["initial_skill"])
    rollouts = 0
    useful_groups = observed_mixed_groups = false_contrast_groups = 0
    all_wrong_count = all_correct_count = expanded_groups = retired_groups = 0
    curve_x = [0]
    curve_y = [float(np.mean(sigmoid(skill - heldout_difficulty)))]
    update = 0
    shifted = False

    while rollouts + 8 <= budget_rollouts:
        if (
            cfg.get("shift_at_update") is not None
            and not shifted
            and update >= int(cfg["shift_at_update"])
        ):
            delta = float(cfg["difficulty_shift"])
            active_difficulty += delta
            heldout_difficulty += delta
            shifted = True

        remaining = budget_rollouts - rollouts
        base_g = 16 if policy == "static_g16" else 8
        groups = min(n_batch, remaining // base_g)
        if groups == 0:
            break
        indices = rng.choice(len(active_difficulty), size=groups, replace=False)
        difficulty = active_difficulty[indices]
        p = sigmoid(skill - difficulty)
        truth, observed = observe(
            rng, p, np.empty((groups, base_g)), sensitivity=sensitivity, specificity=specificity
        )
        rollouts += groups * base_g
        truth_sum = truth.sum(axis=1).astype(int)
        observed_sum = observed.sum(axis=1).astype(int)
        error_sum = (truth != observed).sum(axis=1).astype(float)
        group_sizes = np.full(groups, base_g, dtype=int)
        bins = difficulty_bin(difficulty)

        if policy not in ("static_g8", "static_g16"):
            all_wrong = observed_sum == 0
            all_correct = observed_sum == group_sizes
            if policy == "symmetric_zvf":
                eligible = all_wrong | all_correct
                scores = eligible.astype(float)
            else:
                eligible, scores = policy_scores(policy, all_wrong, bins, alpha, beta, rng)
            available_expansions = max(0, (budget_rollouts - rollouts) // 8)
            selected = np.zeros(groups, dtype=bool)
            if available_expansions:
                candidates = np.flatnonzero(eligible)
                if len(candidates):
                    order = candidates[np.argsort(scores[candidates])[::-1]]
                    selected[order[:available_expansions]] = True
            if selected.any():
                extra_truth, extra_observed = observe(
                    rng, p[selected], np.empty((int(selected.sum()), 8)), sensitivity, specificity
                )
                truth_sum[selected] += extra_truth.sum(axis=1).astype(int)
                observed_sum[selected] += extra_observed.sum(axis=1).astype(int)
                error_sum[selected] += (extra_truth != extra_observed).sum(axis=1)
                group_sizes[selected] += 8
                rollouts += int(selected.sum()) * 8
                expanded_groups += int(selected.sum())
            # Retirement is deliberately a selection intervention, not free
            # learning: replace a mastered all-correct prompt in the active
            # pool with a fresh draw from the same task distribution.
            if policy in ("boundary_aware", "full_triage", "stateful_bandit_diagnostic"):
                retire = all_correct
                if retire.any():
                    active_difficulty[indices[retire]] = sample_difficulty(
                        rng, cfg["difficulty"], int(retire.sum()), float(cfg["difficulty_shift"]) if shifted else 0.0
                    )
                    retired_groups += int(retire.sum())

        latent_mixed = (truth_sum > 0) & (truth_sum < group_sizes)
        observed_mixed = (observed_sum > 0) & (observed_sum < group_sizes)
        valid_gradient = latent_mixed & observed_mixed
        false_contrast = observed_mixed & ~latent_mixed
        all_wrong_count += int((observed_sum == 0).sum())
        all_correct_count += int((observed_sum == group_sizes).sum())
        useful_groups += int(valid_gradient.sum())
        observed_mixed_groups += int(observed_mixed.sum())
        false_contrast_groups += int(false_contrast.sum())

        # A simulated group updates global skill only if observed contrast is
        # backed by latent contrast. Label flips attenuate rather than reverse
        # the update; their adverse allocation effect is reported separately.
        true_signal = 4.0 * p * (1.0 - p)
        label_reliability = np.maximum(0.0, 1.0 - 2.0 * error_sum / group_sizes)
        gain = learning_rate * float(np.mean(valid_gradient * true_signal * label_reliability))
        skill += gain
        alpha_bins = np.bincount(bins, weights=observed_sum, minlength=3)
        total_bins = np.bincount(bins, weights=group_sizes, minlength=3)
        alpha += alpha_bins
        beta += total_bins - alpha_bins
        update += 1
        curve_x.append(rollouts)
        curve_y.append(float(np.mean(sigmoid(skill - heldout_difficulty))))

    x = np.asarray(curve_x, dtype=float)
    y = np.asarray(curve_y, dtype=float)
    auc_per_token = float(np.trapezoid(y, x) / max(x[-1], 1.0))
    total_groups = max(useful_groups + (observed_mixed_groups - useful_groups) + all_wrong_count + all_correct_count, 1)
    flop_per_generated_token = 8.0 * float(cfg["parameter_billions"]) * 1e9
    return Outcome(
        policy=policy,
        regime=regime,
        replicate=replicate,
        auc_per_token=auc_per_token,
        final_quality=float(y[-1]),
        useful_gradient_fraction=useful_groups / total_groups,
        useful_groups_per_1k_rollouts=1000.0 * useful_groups / max(rollouts, 1),
        observed_contrast_fraction=observed_mixed_groups / total_groups,
        false_contrast_fraction=false_contrast_groups / max(observed_mixed_groups, 1),
        all_wrong_fraction=all_wrong_count / total_groups,
        all_correct_fraction=all_correct_count / total_groups,
        expanded_groups=expanded_groups,
        retired_groups=retired_groups,
        rollout_count=rollouts,
        generated_tokens=rollouts * completion_tokens,
        training_flops=rollouts * completion_tokens * flop_per_generated_token,
        updates=update,
    )


def percentile_interval(values: list[float]) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def summarize(outcomes: list[Outcome]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[Outcome]] = defaultdict(list)
    for item in outcomes:
        buckets[(item.regime, item.policy)].append(item)
    rows: list[dict[str, Any]] = []
    metrics = (
        "auc_per_token",
        "final_quality",
        "useful_gradient_fraction",
        "useful_groups_per_1k_rollouts",
        "observed_contrast_fraction",
        "false_contrast_fraction",
        "all_wrong_fraction",
        "all_correct_fraction",
        "generated_tokens",
        "training_flops",
        "updates",
        "expanded_groups",
        "retired_groups",
    )
    for (regime, policy), values in sorted(buckets.items()):
        row: dict[str, Any] = {"regime": regime, "policy": policy, "n": len(values)}
        for metric in metrics:
            mean, lo, hi = percentile_interval([float(getattr(v, metric)) for v in values])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_mc95_lo"] = lo
            row[f"{metric}_mc95_hi"] = hi
        rows.append(row)
    for regime in sorted({r["regime"] for r in rows}):
        candidates = [r for r in rows if r["regime"] == regime]
        best_auc = max(r["auc_per_token_mean"] for r in candidates)
        best_final = max(r["final_quality_mean"] for r in candidates)
        for row in candidates:
            row["auc_regret_vs_best_tested"] = best_auc - row["auc_per_token_mean"]
            row["final_quality_regret_vs_best_tested"] = best_final - row["final_quality_mean"]
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=120)
    parser.add_argument("--sensitivity-replicates", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.replicates < 4 or args.sensitivity_replicates < 4:
        raise SystemExit("replicate counts must be at least 4")

    e1_raw = json.loads(E1_PATH.read_text())
    e1 = {
        "experiment": e1_raw["experiment"],
        "model": e1_raw["model"],
        "seeds": e1_raw["seeds"],
        "n_steps": e1_raw["n_steps"],
        "pearson_gradnorm_vs_p1mp": e1_raw["pearson_gradnorm_vs_p1mp"],
        "pearson_gradnorm_vs_GU": e1_raw["pearson_gradnorm_vs_GU"],
        "pearson_gradnorm_vs_ERF": e1_raw["pearson_gradnorm_vs_ERF"],
        "by_difficulty": e1_raw["by_difficulty"],
        "source_path": str(E1_PATH.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(E1_PATH.read_bytes()).hexdigest(),
        "use_constraint": "synthetic calibration input only; not pooled with or counted as confirmation for this simulation",
    }
    (HERE / "e1_frozen_inputs.json").write_text(json.dumps(e1, indent=2) + "\n")
    scenario_doc = json.loads(SCENARIOS_PATH.read_text())
    scenario_configs = {
        name: merge(scenario_doc["base"], spec)
        for name, spec in scenario_doc["regimes"].items()
    }
    scenario_configs.update(
        {
            f"scaling::{name}": merge(scenario_doc["base"], spec)
            for name, spec in scenario_doc["scaling_cells"].items()
        }
    )

    outcomes: list[Outcome] = []
    for scenario_index, (regime, cfg) in enumerate(scenario_configs.items()):
        for policy_index, policy in enumerate(POLICIES):
            for replicate in range(args.replicates):
                seed = args.seed + scenario_index * 1_000_000 + policy_index * 10_000 + replicate
                outcomes.append(one_run(policy, regime, cfg, e1, replicate, seed))
    summary = summarize(outcomes)
    write_csv(HERE / "policy_regime_summary.csv", summary)

    # Sensitivity is intentionally limited and separated from the main table:
    # it asks whether ranking survives perturbing the model's least defensible
    # knobs, not whether a hypothesis has been confirmed.
    sensitivity_rows: list[dict[str, Any]] = []
    probes = [
        ("low_learning_gain", "transitional", 4.0, 0.5, {}),
        ("high_learning_gain", "transitional", 4.0, 1.5, {}),
        ("uninformative_prior", "mostly_wrong", 0.1, 1.0, {}),
        ("strong_e1_prior", "mostly_wrong", 16.0, 1.0, {}),
        ("clean_verifier", "noisy_verifier", 4.0, 1.0, {"verifier_sensitivity": 0.995, "verifier_specificity": 0.995}),
        ("severe_verifier_noise", "noisy_verifier", 4.0, 1.0, {"verifier_sensitivity": 0.70, "verifier_specificity": 0.70}),
        ("harder_shift", "distribution_shift", 4.0, 1.0, {"difficulty_shift": 2.0}),
    ]
    for probe_index, (probe, regime, prior, lr_mult, overrides) in enumerate(probes):
        cfg = merge(scenario_configs[regime], overrides)
        probe_outcomes = []
        for policy_index, policy in enumerate(POLICIES):
            for replicate in range(args.sensitivity_replicates):
                seed = args.seed + 50_000_000 + probe_index * 1_000_000 + policy_index * 10_000 + replicate
                probe_outcomes.append(one_run(policy, regime, cfg, e1, replicate, seed, prior, lr_mult))
        by_policy = defaultdict(list)
        for outcome in probe_outcomes:
            by_policy[outcome.policy].append(outcome.auc_per_token)
        ranked = sorted(((float(np.mean(v)), p) for p, v in by_policy.items()), reverse=True)
        for rank, (mean_auc, policy) in enumerate(ranked, start=1):
            sensitivity_rows.append(
                {
                    "probe": probe,
                    "base_regime": regime,
                    "policy": policy,
                    "rank_by_auc": rank,
                    "auc_per_token_mean": mean_auc,
                    "prior_strength": prior,
                    "learning_rate_multiplier": lr_mult,
                }
            )
    write_csv(HERE / "sensitivity_summary.csv", sensitivity_rows)

    result = {
        "simulation_status": "synthetic_decision_analysis_not_empirical_confirmation",
        "registered_policies": list(REGISTERED_POLICIES),
        "nonregistered_diagnostic": "stateful_bandit_diagnostic",
        "replicates": args.replicates,
        "seed": args.seed,
        "assumption_summary": {
            "matching": "same generated-token ceiling within each synthetic scenario",
            "latent_task": "Bernoulli success from sigmoid(skill - difficulty)",
            "gradient": "only observed contrast agreeing with latent contrast yields positive simulated learning",
            "verifier": "independent sensitivity/specificity flip channel",
            "regret": "difference from best tested policy in a scenario, not an oracle or empirical loss",
        },
        "summary": summary,
    }
    (HERE / "simulation_results.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "command": "python3 zvf-program/flagship/research/simulation/run_simulation.py "
        f"--replicates {args.replicates} --sensitivity-replicates {args.sensitivity_replicates} --seed {args.seed}",
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "e1_source_sha256": e1["source_sha256"],
        "scenarios_sha256": hashlib.sha256(SCENARIOS_PATH.read_bytes()).hexdigest(),
        "outputs": ["e1_frozen_inputs.json", "simulation_results.json", "policy_regime_summary.csv", "sensitivity_summary.csv"],
    }
    (HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(outcomes)} main synthetic outcomes and {len(sensitivity_rows)} sensitivity rows to {HERE}")


if __name__ == "__main__":
    main()

"""
Statistical Analysis Tooling for RL Experiments
=================================================
Implements rliable-based aggregate metrics, bootstrap confidence intervals,
and proper statistical testing following:

- Colas et al., "A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms" (2019)
  https://arxiv.org/abs/1904.06979
- Agarwal et al., "Deep RL at the Edge of the Statistical Precipice" (2021)
  https://arxiv.org/abs/2108.13264
- Patterson et al., "Empirical Design in Reinforcement Learning" (2024)
  https://arxiv.org/abs/2304.01315

Usage:
    python utils/stats.py --results-dir results/ --output-dir paper/figures/
"""

import os
import json
import glob
import argparse
import math
import re
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from numbers import Integral, Real
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _seed_from_directory(folder: str) -> int:
    name = os.path.basename(folder)
    if not re.fullmatch(r"seed_[0-9]+", name):
        raise ValueError(f"Invalid seed directory name: {folder}")
    return int(name.removeprefix("seed_"))


def load_multi_seed_results(results_dir: str, experiment: str) -> Dict[int, List[dict]]:
    """
    Load results from multiple seeds for a given experiment.

    Expected directory structure:
        results/<experiment>/seed_<N>/metrics.jsonl

    Returns:
        Dict mapping seed -> list of metric records over training steps.
    """
    seed_results = {}
    sources: Dict[int, str] = {}
    pattern = os.path.join(results_dir, experiment, "seed_*", "*.jsonl")
    for filepath in sorted(glob.glob(pattern)):
        seed = _seed_from_directory(os.path.dirname(filepath))
        if seed in sources:
            raise ValueError(
                f"Ambiguous metric sources for seed {seed}: {sources[seed]} and {filepath}"
            )
        sources[seed] = filepath
        metrics = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError(f"Metric record must be an object: {filepath}")
                metrics.append(data)
        if not metrics:
            raise ValueError(f"Empty metric source for seed {seed}: {filepath}")
        seed_results[seed] = metrics
    for folder in sorted(glob.glob(os.path.join(results_dir, experiment, "seed_*"))):
        seed = _seed_from_directory(folder)
        if not os.path.isdir(folder):
            raise ValueError(f"Seed source is not a directory: {folder}")
        if not glob.glob(os.path.join(folder, "*.jsonl")):
            raise ValueError(f"Missing metric source for seed {seed}: {folder}")
    return seed_results


def _finite_mean(values: np.ndarray) -> float:
    """NumPy's existing mean, with exact-range fallback on sum overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.mean(values)
    if np.isfinite(result):
        return result
    context = Context(
        prec=1600 + len(str(len(values))), rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999
    )
    with localcontext(context):
        return float(sum((Decimal.from_float(float(v)) for v in values), Decimal(0)) / len(values))


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    """Preserve NumPy's linear quantile, avoiding overflowing endpoint differences."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.percentile(values, percentile)
    if np.isfinite(result):
        return result
    ordered = np.sort(values)
    index = (len(ordered) - 1) * (percentile / 100)
    left = math.floor(index)
    right = math.ceil(index)
    context = Context(prec=1600, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
    with localcontext(context):
        weight = Decimal.from_float(index - left)
        a, b = Decimal.from_float(float(ordered[left])), Decimal.from_float(float(ordered[right]))
        return float(a + (b - a) * weight)


def compute_bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        scores: Array of scores (one per seed/run).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        rng: NumPy random generator for reproducibility.

    Returns:
        (mean, lower_ci, upper_ci). With one score, the bounds are NaN:
        sampling uncertainty cannot be estimated from one seed/run.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("compute_bootstrap_ci requires one-dimensional observations")
    if not np.all(np.isfinite(scores)):
        raise ValueError("compute_bootstrap_ci requires finite scores")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if scores.size == 0:
        raise ValueError("compute_bootstrap_ci requires at least one score")
    if not np.all(np.isfinite(scores)):
        raise ValueError("compute_bootstrap_ci requires finite scores")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")

    if rng is None:
        rng = np.random.default_rng(42)

    n = len(scores)
    if n == 1:
        return float(scores[0]), float("nan"), float("nan")
    bootstrap_means = np.array(
        [_finite_mean(rng.choice(scores, size=n, replace=True)) for _ in range(n_bootstrap)]
    )

    alpha = (1 - confidence) / 2
    lower = _finite_percentile(bootstrap_means, 100 * alpha)
    upper = _finite_percentile(bootstrap_means, 100 * (1 - alpha))
    mean = _finite_mean(scores)

    return mean, lower, upper


def standard_error(scores: np.ndarray, axis: int = 0):
    """
    Standard error of the mean; one observation has unestimable uncertainty.

    ``std(ddof=1)`` is undefined for one observation, so n=1 returns NaN,
    never zero spread. Empty/nonfinite inputs are errors. Returns a scalar
    for 1-D input, an array along ``axis`` otherwise.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("standard_error requires at least one score")
    if not np.all(np.isfinite(arr)):
        raise ValueError("standard_error requires finite scores")
    if arr.ndim == 0:
        return float("nan")
    n = arr.shape[axis]
    if n == 1:
        unavailable = np.full_like(np.mean(arr, axis=axis), np.nan)
        return unavailable.item() if unavailable.ndim == 0 else unavailable
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(n)


def welch_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    Welch's t-test for comparing two algorithms.
    Recommended over Student's t-test when variances may differ.
    Cohen's d uses the sample-size-weighted pooled sample variance.
    Identical point masses report d=0; distinct point masses report signed infinity.

    Reference: Colas et al. (2019), Section 4.1
    """
    from scipy import stats

    scores_a = np.asarray(scores_a, dtype=float).ravel()
    scores_b = np.asarray(scores_b, dtype=float).ravel()
    if scores_a.size < 2 or scores_b.size < 2:
        raise ValueError("welch_ttest requires at least two scores per group")
    if not np.all(np.isfinite(scores_a)) or not np.all(np.isfinite(scores_b)):
        raise ValueError("welch_ttest requires finite scores")

    constant_a = np.all(scores_a == scores_a[0])
    constant_b = np.all(scores_b == scores_b[0])
    if constant_a and constant_b:
        # Avoid a 0/0 statistic for identical point masses. Distinct point
        # masses have a signed infinite standardized effect, not zero effect.
        difference = scores_a[0] - scores_b[0]
        t_stat = effect_size = float(np.copysign(np.inf, difference)) if difference else 0.0
        p_value = 0.0 if difference else 1.0
    else:
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
        pooled = np.sqrt(
            (
                (len(scores_a) - 1) * np.var(scores_a, ddof=1)
                + (len(scores_b) - 1) * np.var(scores_b, ddof=1)
            )
            / (len(scores_a) + len(scores_b) - 2)
        )
        effect_size = (np.mean(scores_a) - np.mean(scores_b)) / pooled

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "effect_size_cohens_d": float(effect_size),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "std_a": float(np.std(scores_a, ddof=1)),
        "std_b": float(np.std(scores_b, ddof=1)),
        "n_a": len(scores_a),
        "n_b": len(scores_b),
    }


def mann_whitney_u(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).
    Use when distributions may be non-normal.

    Reference: Colas et al. (2019), Section 4.2
    """
    from scipy import stats

    scores_a = np.asarray(scores_a, dtype=float).ravel()
    scores_b = np.asarray(scores_b, dtype=float).ravel()
    if scores_a.size == 0 or scores_b.size == 0:
        raise ValueError("mann_whitney_u requires at least one score per group")
    if not np.all(np.isfinite(scores_a)) or not np.all(np.isfinite(scores_b)):
        raise ValueError("mann_whitney_u requires finite scores")
    if np.all(scores_a == scores_a[0]) and np.all(scores_b == scores_a[0]):
        # All ranks tie: every permutation has the same U statistic.
        u_stat, p_value = scores_a.size * scores_b.size / 2, 1.0
    else:
        u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")

    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "significant_at_005": bool(p_value < 0.05),
        "median_a": float(np.median(scores_a)),
        "median_b": float(np.median(scores_b)),
    }


def plot_learning_curves_with_ci(
    results: Dict[str, Dict[int, List[dict]]],
    metric_key: str = "reward/mean",
    output_path: str = "learning_curves.pdf",
    title: str = "Learning Curves with 95% Confidence Intervals",
):
    """
    Plot pointwise 95% Student-t mean bands (independent, approximately normal seeds).

    No uncertainty band is drawn for a single seed; bands are not simultaneous.

    Seeds must share strictly increasing step grids. Missing metrics and unequal
    schedules are rejected. Legacy step-less data uses observation indices.

    Args:
        results: Dict[algorithm_name -> Dict[seed -> List[step_metrics]]]
        metric_key: Which metric to plot
        output_path: Where to save the figure
        title: Plot title
    """
    # Validate before allocating a figure or writing any output. Step-less legacy
    # curves remain supported, but only on identical observation-index grids.
    prepared = []
    modes = set()
    for algo_name, seed_data in results.items():
        if not seed_data:
            raise ValueError(f"No curves to plot for algorithm {algo_name!r}")
        curves = []
        grid: list = []
        for metrics_list in seed_data.values():
            if not metrics_list:
                raise ValueError(f"Empty curves for algorithm {algo_name!r}")
            has_steps = ["step" in m for m in metrics_list]
            if any(has_steps) and not all(has_steps):
                raise ValueError("Every observation must have a step, or none may have one")
            explicit = all(has_steps)
            modes.add(explicit)
            values = [m.get(metric_key) for m in metrics_list]
            steps = (
                [m["step"] for m in metrics_list] if explicit else list(range(1, len(values) + 1))
            )
            for label, numbers in (("metrics", values), ("steps", steps)):
                if any(
                    isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                    for v in numbers
                ):
                    raise ValueError(f"Plot {label} must be finite numeric values")
            if any(b <= a for a, b in zip(steps, steps[1:])):
                raise ValueError(
                    "Plot steps must be strictly increasing; duplicate or reordered steps are invalid"
                )
            if grid and steps != grid:
                raise ValueError(
                    "All seeds must use the same step grid; no truncation or interpolation"
                )
            grid = steps
            curves.append(values)
        prepared.append((algo_name, grid, np.asarray(curves, dtype=float)))
    if len(modes) > 1:
        raise ValueError("Cannot mix explicit training steps and observation indices")
    if not prepared:
        raise ValueError("No curves to plot")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = sns.color_palette("colorblind", n_colors=len(prepared))
    for idx, (algo_name, steps, aligned_curves) in enumerate(prepared):
        mean = np.mean(aligned_curves, axis=0)
        se = standard_error(aligned_curves, axis=0)
        n_seeds = len(aligned_curves)
        label = algo_name if n_seeds >= 2 else f"{algo_name} (n=1: CI not estimable)"
        ax.plot(steps, mean, label=label, color=colors[idx], linewidth=2)
        if len(aligned_curves) >= 2:
            from scipy.stats import t

            critical = t.ppf(0.975, df=len(aligned_curves) - 1)
            ax.fill_between(
                steps, mean - critical * se, mean + critical * se, alpha=0.2, color=colors[idx]
            )
    ax.set_xlabel("Training Step" if True in modes else "Observation index", fontsize=12)
    ax.set_ylabel(metric_key, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves to {output_path}")


def generate_results_table(
    results: Dict[str, np.ndarray],
    output_path: str = "results_table.tex",
    metric_name: str = "Accuracy",
):
    """
    Generate a LaTeX table with mean ± SE and bootstrap CIs.

    Args:
        results: Dict[algorithm_name -> array of final scores across seeds]
        output_path: Where to save the LaTeX table
        metric_name: Column header for the metric
    """
    rows = []
    for algo_name, scores in results.items():
        scores = np.asarray(scores, dtype=float).ravel()
        mean, ci_lower, ci_upper = compute_bootstrap_ci(scores)
        se = standard_error(scores)
        mean_se = f"{mean:.3f} ± {se:.3f}"
        interval = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        if len(scores) == 1:
            mean_se = f"{mean:.3f} (SE not estimable)"
            interval = "not estimable (n=1)"
        rows.append(
            {
                "Algorithm": algo_name,
                f"{metric_name} (mean ± SE)": mean_se,
                "95% CI": interval,
                "Seeds": len(scores),
            }
        )

    df = pd.DataFrame(rows)

    # Save as LaTeX
    latex = df.to_latex(index=False, escape=False)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved results table to {output_path}")

    # Also save as CSV
    csv_path = output_path.replace(".tex", ".csv")
    df.to_csv(csv_path, index=False)

    return df


def try_rliable_analysis(results: Dict[str, np.ndarray], output_dir: str):
    """
    Run rliable aggregate metrics if the library is available.

    Reference: Agarwal et al. (2021)
    https://arxiv.org/abs/2108.13264
    """
    try:
        from rliable import library as rly
        from rliable import metrics as rly_metrics

        # ``plot_utils`` is imported to verify the full rliable install is
        # present (the caller later builds rliable plots via helper scripts);
        # we assign ``_`` to signal the availability check to ruff/linters.
        from rliable import plot_utils as _  # noqa: F401

        print("Running rliable analysis...")

        # Prepare score dictionaries
        score_dict = {}
        for algo, scores in results.items():
            # rliable expects (n_runs, n_tasks) array
            score_dict[algo] = scores.reshape(-1, 1) if scores.ndim == 1 else scores

        # Compute aggregate metrics with CIs
        aggregate_func = lambda x: np.array(
            [
                rly_metrics.aggregate_median(x),
                rly_metrics.aggregate_iqm(x),
                rly_metrics.aggregate_mean(x),
                rly_metrics.aggregate_optimality_gap(x),
            ]
        )

        aggregate_scores, aggregate_cis = rly.get_interval_estimates(
            score_dict, aggregate_func, reps=50000
        )

        # Save rliable results
        rliable_results = {
            "aggregate_scores": {k: v.tolist() for k, v in aggregate_scores.items()},
            "aggregate_cis": {k: v.tolist() for k, v in aggregate_cis.items()},
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "rliable_results.json"), "w") as f:
            json.dump(rliable_results, f, indent=2)

        print(f"rliable results saved to {output_dir}/rliable_results.json")

    except ImportError:
        print("rliable not installed. Install with: pip install rliable")
        print("Falling back to bootstrap CI analysis.")


METRIC_UNITS = {"accuracy": "fraction", "reward/mean": "reward", "eval/percent_correct": "percent"}


def validate_contract(contract):
    required = {
        "schema",
        "metric",
        "unit",
        "split",
        "endpoint_step",
        "expected_seeds",
        "protocol_id",
    }
    if not isinstance(contract, dict) or not required <= contract.keys():
        raise ValueError(f"contract requires {sorted(required)}")
    if contract["schema"] != "stats-measurement-v2":
        raise ValueError("unsupported measurement schema")
    if (
        contract["metric"] not in METRIC_UNITS
        or contract["unit"] != METRIC_UNITS[contract["metric"]]
    ):
        raise ValueError("unsupported metric/unit combination")
    if any(not isinstance(contract[k], str) or not contract[k] for k in ("split", "protocol_id")):
        raise ValueError("split and protocol_id must be nonempty strings")
    if type(contract["endpoint_step"]) is not int or contract["endpoint_step"] < 0:
        raise ValueError("endpoint_step must be a nonnegative integer")
    seeds = contract["expected_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(x) is not int or x < 0 for x in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("expected_seeds must be distinct nonnegative integers")


def contract_scores(seed_results, contract=None, metric=None):
    """Validate the population, measurement and horizon before reducing any seed."""
    if not seed_results:
        raise ValueError("no seed results")
    if contract is not None:
        validate_contract(contract)
        if set(seed_results) != set(contract["expected_seeds"]):
            raise ValueError("discovered seeds do not match expected seed roster")
    if contract is not None and metric is not None and metric != contract["metric"]:
        raise ValueError("selected metric differs from measurement contract")
    if metric is not None and metric not in METRIC_UNITS:
        raise ValueError("unsupported metric")
    common_keys = set.intersection(*(set(rows[-1]) if rows else set() for rows in seed_results.values()))
    legacy_metric = metric or next((key for key in METRIC_UNITS if key in common_keys), None)
    selected, keys, endpoints, modes = [], set(), set(), set()
    for seed, rows in sorted(seed_results.items()):
        if not rows:
            raise ValueError(f"seed {seed}: empty/incomplete seed file")
        if any({"global_step", "_step", "optimizer_step"}.intersection(row) for row in rows):
            raise ValueError(f"seed {seed}: unsupported step metadata; normalize explicitly to step")
        present = ["step" in row for row in rows]
        if any(present) != all(present):
            raise ValueError(f"seed {seed}: missing steps")
        explicit = all(present)
        modes.add(explicit)
        steps = [row["step"] for row in rows] if explicit else list(range(len(rows)))
        if any(type(step) is not int or step < 0 for step in steps) or any(
            b <= a for a, b in zip(steps, steps[1:])
        ):
            raise ValueError(f"seed {seed}: steps must be strictly increasing integers")
        endpoints.add(steps[-1])
        if contract is not None:
            if not explicit or steps[-1] != contract["endpoint_step"]:
                raise ValueError(f"seed {seed}: incomplete or incorrect endpoint")
            key = contract["metric"]
            for row in rows:
                if any(row.get(k) != contract[k] for k in ("unit", "split", "protocol_id")):
                    raise ValueError(f"seed {seed}: measurement metadata differs from contract")
        else:
            key = legacy_metric
            if key is None:
                raise ValueError(f"seed {seed}: final score must identify one common finite numeric metric")
        keys.add(key)
        for row in rows:
            score = row.get(key)
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(f"seed {seed}: final score must be a finite numeric metric")
            if key in {"accuracy", "eval/percent_correct"} and not 0 <= score <= (
                1 if key == "accuracy" else 100
            ):
                raise ValueError(f"seed {seed}: metric outside declared unit domain")
        selected.append(rows[-1][key])
    if len(keys) != 1 or len(endpoints) != 1 or len(modes) != 1:
        raise ValueError("mixed metric measurements, units or endpoint steps/observation-index horizons across seeds")
    metric = next(iter(keys))
    return selected, {
        "analysis_schema": "stats-measurement-v2",
        "metric": metric,
        "unit": METRIC_UNITS[metric],
        "split": contract["split"] if contract else "unknown_legacy",
        "protocol_id": contract["protocol_id"] if contract else "legacy_exploratory",
        "endpoint_step": next(iter(endpoints)),
        "step_mode": "explicit" if next(iter(modes)) else "observation_index",
        "final_step": next(iter(endpoints)) if next(iter(modes)) else next(iter(endpoints)) + 1,
        "expected_seeds": json.dumps(
            contract["expected_seeds"] if contract else sorted(seed_results)
        ),
        "included_seeds": json.dumps(sorted(seed_results)),
        "excluded_seeds": "[]",
        "population_status": "contract_complete" if contract else "observed_only_exploratory",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Statistical analysis for RL experiments")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument(
        "--experiment", type=str, default=None, help="Specific experiment to analyze"
    )
    parser.add_argument("--output-dir", type=str, default="paper/figures/")
    parser.add_argument("--format", type=str, choices=["latex", "csv", "both"], default="both")
    parser.add_argument(
        "--metric", help="Metric key shared by every seed; otherwise infer a common legacy key"
    )
    parser.add_argument("--rliable", action="store_true", help="Run rliable aggregate analysis")
    parser.add_argument(
        "--bootstrap-samples", type=int, default=10000, help="Bootstrap resamples for the CI"
    )
    parser.add_argument("--contract", help="JSON measurement contract (stats-measurement-v2)")
    parser.add_argument(
        "--legacy-exploratory",
        action="store_true",
        help="Explicitly analyze observed legacy logs; no completeness claim",
    )
    args = parser.parse_args(argv)
    if args.rliable:
        parser.error(
            "--rliable is not supported by this CLI: its per-experiment seed files "
            "do not define a normalized multi-task score matrix. Use the explicit "
            "try_rliable_analysis API with appropriately prepared scores."
        )
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if not os.path.isdir(args.results_dir):
        parser.error("--results-dir must name an existing directory")
    contract = None
    if args.contract:
        try:
            with open(args.contract) as source:
                contract = json.load(source)
            validate_contract(contract)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(f"invalid measurement contract: {exc}")
    elif not args.legacy_exploratory:
        parser.error(
            "--contract is required; use --legacy-exploratory only for observed legacy inputs"
        )
    summary_rows = []

    # Discover experiments
    if args.experiment:
        experiments = [args.experiment]
    else:
        experiments = [
            d
            for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d))
        ]

    print(f"Found experiments: {experiments}")

    for exp in sorted(experiments):
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {exp}")
        print(f"{'=' * 60}")

        try:
            seed_results = load_multi_seed_results(args.results_dir, exp)
        except ValueError as exc:
            parser.error(str(exc))
        if not seed_results:
            parser.error(f"{exp}: no seed results; cannot silently omit an experiment")

        print(f"  Found {len(seed_results)} seeds: {list(seed_results.keys())}")

        try:
            final_scores, measurement = contract_scores(seed_results, contract, metric=args.metric)
        except ValueError as exc:
            parser.error(f"{exp}: {exc}")

        if final_scores:
            scores_arr = np.array(final_scores)
            mean, ci_lower, ci_upper = compute_bootstrap_ci(
                scores_arr, n_bootstrap=args.bootstrap_samples
            )
            se = standard_error(scores_arr)
            if len(final_scores) < 2:
                se = ci_lower = ci_upper = float("nan")
            summary_rows.append(
                {
                    "experiment": exp,
                    "seeds": len(final_scores),
                    "mean": float(mean),
                    "standard_error": float(se),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "bootstrap_samples": args.bootstrap_samples,
                    **measurement,
                    "uncertainty_status": "available"
                    if len(final_scores) >= 2
                    else "unavailable_n_lt_2",
                    "interval_method": "pointwise_percentile_bootstrap_mean_95pct",
                }
            )
            if len(final_scores) == 1:
                print(f"  Final score: {mean:.4f} (SE and 95% CI not estimable: n=1)")
            else:
                print(
                    f"  Final score: {mean:.4f} ± {se:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )

    if not summary_rows:
        parser.error("no nonempty seed results found; no analysis was produced")
    os.makedirs(args.output_dir, exist_ok=True)
    table = pd.DataFrame(summary_rows)
    if args.format in {"csv", "both"}:
        table.to_csv(os.path.join(args.output_dir, "results_table.csv"), index=False)
    if args.format in {"latex", "both"}:
        table.to_latex(os.path.join(args.output_dir, "results_table.tex"), index=False, escape=True)
    print("\nStatistical analysis complete.")


# ----------------------------------------------------------------------------
# Compatibility aliases
# ----------------------------------------------------------------------------
# Some downstream tooling / CI imports the bootstrap CI under a shorter name.
bootstrap_ci = compute_bootstrap_ci


def compute_iqm(scores: np.ndarray, tau: float = 0.25) -> float:
    """Compute the Interquartile Mean (IQM) of a score array.

    Sorts observations and trims floor(n * tau) values from each tail,
    matching scipy.stats.trim_mean and rliable's rank-trimming convention.
    Ties are retained according to rank, not quantile-value membership.
    Default tau=0.25 gives the usual interquartile trimmed mean.
    Empty, nonfinite, and multidimensional inputs are rejected.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1 or not arr.size or not np.all(np.isfinite(arr)):
        raise ValueError("IQM requires nonempty finite one-dimensional scores")
    if (
        isinstance(tau, bool)
        or not isinstance(tau, Real)
        or not math.isfinite(tau)
        or not 0 <= tau < 0.5
    ):
        raise ValueError("tau must be finite and in [0, 0.5)")
    # Protocol rank-trim-floor-v2: remove floor(n*tau) observations per tail.
    trim = int(len(arr) * tau)
    ranked = np.sort(arr)
    return float(np.mean(ranked[trim : len(ranked) - trim]))


if __name__ == "__main__":
    main()

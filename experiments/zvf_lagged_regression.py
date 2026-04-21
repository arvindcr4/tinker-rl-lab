#!/usr/bin/env python3
"""Lagged ZVF regression: test whether ZVF predicts FUTURE reward
controlling for CURRENT reward.

Addresses Reviewer Objection O8: "ZVF is tautologically correlated with
performance because on binary-reward tasks ZVF = p^G + (1-p)^G is a
deterministic function of accuracy p."

If ZVF is merely a restatement of accuracy, then after controlling for
current reward, ZVF should have NO additional predictive power for future
reward. If ZVF provides independent early-warning signal, then the
coefficient on lagged ZVF should be significant after controlling for
current reward.

Models estimated (per-step, across all runs with step-level ZVF data):
  1. R_{t+k} = a + b * R_t                      (reward-only baseline)
  2. R_{t+k} = a + b * R_t + c * ZVF_t          (ZVF incremental)
  3. R_{t+k} = a + b * R_t + c * ZVF_t + d * R_t * ZVF_t  (interaction)

Also: within-run analysis (does step-t ZVF predict step t+k reward,
controlling for step-t reward, within the same training trajectory?)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    ROOT / "experiments" / "master_results.json",
    ROOT / "experiments" / "all_results_consolidated.json",
]
SOURCE_FILES += sorted((ROOT / "experiments" / "tinker-runs" / "results").glob("*.json"))
SOURCE_FILES += sorted((ROOT / "experiments" / "results").glob("*.json"))

OUT_JSON = ROOT / "experiments" / "zvf_lagged_regression_results.json"
OUT_MD = ROOT / "experiments" / "zvf_lagged_regression.md"


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def walk_records(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        if isinstance(obj.get("step_log"), list):
            yield path, obj
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            yield from walk_records(value, next_path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from walk_records(value, f"{path}[{index}]")


def extract_step_data(rec: dict) -> list[dict] | None:
    """Extract (step, reward, zvf, gu) tuples from a step_log."""
    step_log = rec.get("step_log")
    if not isinstance(step_log, list) or len(step_log) < 5:
        return None

    rows = []
    for step_entry in step_log:
        if not isinstance(step_entry, dict):
            continue
        reward = as_float(step_entry.get("reward"))
        zvf = as_float(step_entry.get("zvf"))
        gu = as_float(step_entry.get("gu"))
        step_num = as_float(step_entry.get("step"), default=0)

        if math.isnan(reward):
            continue
        if math.isnan(zvf) and not math.isnan(gu):
            zvf = 1.0 - gu
        if math.isnan(gu) and not math.isnan(zvf):
            gu = 1.0 - zvf
        if math.isnan(zvf):
            continue

        rows.append({
            "step": int(step_num),
            "reward": float(reward),
            "zvf": float(zvf),
            "gu": float(gu),
        })

    if len(rows) < 5:
        return None

    # Sort by step number
    rows.sort(key=lambda r: r["step"])
    return rows


@dataclass
class RunData:
    key: str
    model: str
    task: str
    step_data: list[dict]  # sorted list of {step, reward, zvf, gu}


def load_runs() -> list[RunData]:
    runs = []
    seen = set()
    for source in SOURCE_FILES:
        if not source.exists():
            continue
        try:
            data = json.loads(source.read_text())
        except json.JSONDecodeError:
            continue

        for subpath, rec in walk_records(data):
            step_data = extract_step_data(rec)
            if step_data is None:
                continue

            model = str(rec.get("model_short") or rec.get("model") or "unknown")
            task = str(rec.get("task") or "unknown")
            run_id = str(rec.get("run_id") or "")
            tag = str(rec.get("tag") or rec.get("experiment") or "")
            key = f"{run_id}|{tag}|{model}|{task}|{len(step_data)}"

            if key in seen:
                continue
            seen.add(key)

            runs.append(RunData(
                key=key,
                model=model,
                task=task,
                step_data=step_data,
            ))

    return runs


def ols_regression(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS with standard errors, t-statistics, p-values, and R²."""
    n, k = X.shape
    if n <= k:
        return {"error": "insufficient observations"}

    # Coefficients
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - sse / sst if sst > 0 else 0.0
    adj_r_squared = 1.0 - (1 - r_squared) * (n - 1) / (n - k) if n > k else 0.0

    # Standard errors
    mse = sse / (n - k)
    try:
        var_beta = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    t_stats = beta / se if not np.any(np.isnan(se)) and not np.any(se == 0) else np.full(k, np.nan)

    # Two-tailed p-values using normal approximation (large sample)
    from scipy import stats as sp_stats
    p_values = np.array([2.0 * (1.0 - sp_stats.t.cdf(abs(t), df=n - k)) if not np.isnan(t) else np.nan
                         for t in t_stats])

    # F-test for overall significance
    if k > 1 and sst > 0:
        f_stat = ((sst - sse) / (k - 1)) / mse if mse > 0 else 0.0
        f_pvalue = 1.0 - sp_stats.f.cdf(f_stat, k - 1, n - k)
    else:
        f_stat = np.nan
        f_pvalue = np.nan

    # AIC / BIC
    aic = n * np.log(sse / n) + 2 * k if sse > 0 else np.nan
    bic = n * np.log(sse / n) + k * np.log(n) if sse > 0 else np.nan

    return {
        "n": n,
        "k": k,
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stats": t_stats.tolist(),
        "p_values": p_values.tolist(),
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj_r_squared),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
        "aic": float(aic),
        "bic": float(bic),
        "sse": float(sse),
    }


def cross_run_lagged_regression(runs: list[RunData], lag_k: int = 5) -> dict:
    """Cross-run analysis: for each (run, step t), predict R_{t+k} from R_t and ZVF_t.

    This pools all (t, t+k) pairs across all runs, including the run identity
    as a potential confound.
    """
    R_t_list = []
    R_tk_list = []
    ZVF_t_list = []
    run_ids = []

    for run_idx, run in enumerate(runs):
        sd = run.step_data
        for i in range(len(sd) - lag_k):
            r_t = sd[i]["reward"]
            r_tk = sd[i + lag_k]["reward"]
            zvf_t = sd[i]["zvf"]
            R_t_list.append(r_t)
            R_tk_list.append(r_tk)
            ZVF_t_list.append(zvf_t)
            run_ids.append(run_idx)

    R_t = np.array(R_t_list)
    R_tk = np.array(R_tk_list)
    ZVF_t = np.array(ZVF_t_list)
    run_ids = np.array(run_ids)
    n = len(R_t)

    if n < 20:
        return {"error": f"too few observations: {n}"}

    # Model 1: R_{t+k} = a + b * R_t
    X1 = np.column_stack([np.ones(n), R_t])
    m1 = ols_regression(X1, R_tk)

    # Model 2: R_{t+k} = a + b * R_t + c * ZVF_t
    X2 = np.column_stack([np.ones(n), R_t, ZVF_t])
    m2 = ols_regression(X2, R_tk)

    # Model 3: R_{t+k} = a + b * R_t + c * ZVF_t + d * R_t * ZVF_t
    X3 = np.column_stack([np.ones(n), R_t, ZVF_t, R_t * ZVF_t])
    m3 = ols_regression(X3, R_tk)

    # Model 4: R_{t+k} = a + b * R_t + c * ZVF_t + run fixed effects
    # Use run dummies (drop one to avoid multicollinearity)
    unique_runs = np.unique(run_ids)
    if len(unique_runs) > 1 and len(unique_runs) < n // 3:
        run_dummies = np.zeros((n, len(unique_runs) - 1))
        for i, rid in enumerate(unique_runs[:-1]):
            run_dummies[:, i] = (run_ids == rid).astype(float)
        X4 = np.column_stack([np.ones(n), R_t, ZVF_t, run_dummies])
        m4 = ols_regression(X4, R_tk)
    else:
        m4 = {"error": "too many runs or too few obs for fixed effects"}

    # Incremental F-test: Model 2 vs Model 1
    if "sse" in m1 and "sse" in m2 and not m2.get("error"):
        sse_restricted = m1["sse"]
        sse_unrestricted = m2["sse"]
        q = 1  # one additional restriction
        from scipy import stats as sp_stats
        f_inc = ((sse_restricted - sse_unrestricted) / q) / (sse_unrestricted / (n - 3))
        f_inc_pvalue = 1.0 - sp_stats.f.cdf(f_inc, q, n - 3)
    else:
        f_inc = np.nan
        f_inc_pvalue = np.nan

    return {
        "lag_k": lag_k,
        "n_observations": n,
        "n_runs": len(set(run_ids.tolist())),
        "model1_reward_only": m1,
        "model2_reward_plus_zvf": m2,
        "model3_with_interaction": m3,
        "model4_with_run_fe": m4,
        "incremental_f_test": {
            "description": "H0: ZVF coefficient = 0 in Model 2 vs Model 1",
            "f_stat": float(f_inc),
            "p_value": float(f_inc_pvalue),
        },
        "delta_r_squared": float(m2.get("r_squared", 0) - m1.get("r_squared", 0)) if not m2.get("error") and not m1.get("error") else np.nan,
        "delta_adj_r_squared": float(m2.get("adj_r_squared", 0) - m1.get("adj_r_squared", 0)) if not m2.get("error") and not m1.get("error") else np.nan,
    }


def within_run_lagged_regression(runs: list[RunData], lag_k: int = 5) -> dict:
    """Within-run analysis: for each run, fit R_{t+k} ~ R_t + ZVF_t,
    then pool coefficients. This controls for run-level confounders
    (model, task, hardware) by estimating within each run.
    """
    coefs_zvf = []
    coefs_reward = []
    r2_model1 = []
    r2_model2 = []
    n_per_run = []

    for run in runs:
        sd = run.step_data
        if len(sd) < lag_k + 5:
            continue

        R_t = np.array([sd[i]["reward"] for i in range(len(sd) - lag_k)])
        R_tk = np.array([sd[i + lag_k]["reward"] for i in range(len(sd) - lag_k)])
        ZVF_t = np.array([sd[i]["zvf"] for i in range(len(sd) - lag_k)])
        n = len(R_t)

        if n < 5:
            continue

        # Model 1: R_{t+k} = a + b * R_t
        X1 = np.column_stack([np.ones(n), R_t])
        m1 = ols_regression(X1, R_tk)

        # Model 2: R_{t+k} = a + b * R_t + c * ZVF_t
        X2 = np.column_stack([np.ones(n), R_t, ZVF_t])
        m2 = ols_regression(X2, R_tk)

        if m1.get("error") or m2.get("error"):
            continue

        coefs_reward.append(m2["beta"][1])
        coefs_zvf.append(m2["beta"][2])
        r2_model1.append(m1["r_squared"])
        r2_model2.append(m2["r_squared"])
        n_per_run.append(n)

    if not coefs_zvf:
        return {"error": "no runs with sufficient data"}

    coefs_zvf = np.array(coefs_zvf)
    coefs_reward = np.array(coefs_reward)
    r2_model1 = np.array(r2_model1)
    r2_model2 = np.array(r2_model2)

    # One-sample t-test: is the mean ZVF coefficient significantly different from 0?
    from scipy import stats as sp_stats
    t_stat_zvf, p_value_zvf = sp_stats.ttest_1samp(coefs_zvf, 0)

    return {
        "lag_k": lag_k,
        "n_runs": len(coefs_zvf),
        "mean_zvf_coef": float(np.mean(coefs_zvf)),
        "std_zvf_coef": float(np.std(coefs_zvf, ddof=1)),
        "t_stat_zvf": float(t_stat_zvf),
        "p_value_zvf": float(p_value_zvf),
        "fraction_negative_zvf_coef": float(np.mean(coefs_zvf < 0)),
        "mean_reward_coef": float(np.mean(coefs_reward)),
        "mean_r2_model1": float(np.mean(r2_model1)),
        "mean_r2_model2": float(np.mean(r2_model2)),
        "mean_delta_r2": float(np.mean(r2_model2 - r2_model1)),
        "per_run_zvf_coefs": coefs_zvf.tolist(),
        "per_run_r2_m1": r2_model1.tolist(),
        "per_run_r2_m2": r2_model2.tolist(),
    }


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Partial correlation between x and y controlling for z."""
    from scipy import stats as sp_stats

    # Regress x on z, get residuals
    X_z = np.column_stack([np.ones(len(z)), z])
    res_x = x - X_z @ np.linalg.lstsq(X_z, x, rcond=None)[0]
    res_y = y - X_z @ np.linalg.lstsq(X_z, y, rcond=None)[0]

    n = len(res_x)
    r = float(np.corrcoef(res_x, res_y)[0, 1])
    # t-test for partial correlation
    t_stat = r * np.sqrt((n - 3) / (1 - r**2)) if abs(r) < 1 else 0.0
    p_value = 2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=n - 3))

    return r, p_value


def zvf_partial_correlation_analysis(runs: list[RunData], lag_k: int = 5) -> dict:
    """Compute partial correlation between ZVF_t and R_{t+k}, controlling for R_t."""
    R_t_list = []
    R_tk_list = []
    ZVF_t_list = []

    for run in runs:
        sd = run.step_data
        for i in range(len(sd) - lag_k):
            R_t_list.append(sd[i]["reward"])
            R_tk_list.append(sd[i + lag_k]["reward"])
            ZVF_t_list.append(sd[i]["zvf"])

    R_t = np.array(R_t_list)
    R_tk = np.array(R_tk_list)
    ZVF_t = np.array(ZVF_t_list)

    n = len(R_t)
    if n < 20:
        return {"error": "too few observations"}

    # Zero-order correlations
    r_zvf_reward_future = float(np.corrcoef(ZVF_t, R_tk)[0, 1])
    r_zvf_reward_current = float(np.corrcoef(ZVF_t, R_t)[0, 1])
    r_reward_current_future = float(np.corrcoef(R_t, R_tk)[0, 1])

    # Partial correlations
    pr_zvf_future_controlling_current, pr_pval = partial_correlation(ZVF_t, R_tk, R_t)
    pr_reward_future_controlling_zvf, pr2_pval = partial_correlation(R_t, R_tk, ZVF_t)

    return {
        "lag_k": lag_k,
        "n_observations": n,
        "zero_order": {
            "zvf_vs_future_reward": r_zvf_reward_future,
            "zvf_vs_current_reward": r_zvf_reward_current,
            "current_vs_future_reward": r_reward_current_future,
        },
        "partial": {
            "zvf_vs_future_reward_controlling_current": {
                "r": pr_zvf_future_controlling_current,
                "p_value": pr_pval,
            },
            "current_vs_future_reward_controlling_zvf": {
                "r": pr_reward_future_controlling_zvf,
                "p_value": pr2_pval,
            },
        },
    }


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> None:
    runs = load_runs()
    print(f"Loaded {len(runs)} runs with step-level ZVF data")
    for run in runs:
        print(f"  {run.model:30s} {run.task:15s} {len(run.step_data)} steps")

    results = {}

    # Run lagged regressions at multiple lag horizons
    for lag_k in [1, 3, 5, 10]:
        print(f"\n{'='*60}")
        print(f"Lag k = {lag_k}")
        print(f"{'='*60}")

        # Cross-run analysis
        cross = cross_run_lagged_regression(runs, lag_k=lag_k)
        results[f"cross_run_lag{lag_k}"] = cross

        print(f"  N observations: {cross.get('n_observations', 'NA')}")
        m1 = cross.get("model1_reward_only", {})
        m2 = cross.get("model2_reward_plus_zvf", {})
        print(f"  Model 1 (reward only) R²: {fmt(m1.get('r_squared'))}")
        print(f"  Model 2 (reward + ZVF) R²: {fmt(m2.get('r_squared'))}")
        print(f"  Delta R²: {fmt(cross.get('delta_r_squared'))}")
        if not m2.get("error"):
            zvf_coef = m2.get("beta", [None, None, None])[2]
            zvf_se = m2.get("se", [None, None, None])[2]
            zvf_pval = m2.get("p_values", [None, None, None])[2]
            print(f"  ZVF coefficient: {fmt(zvf_coef)} (SE={fmt(zvf_se)}, p={fmt(zvf_pval, 4)})")
        print(f"  Incremental F-test: F={fmt(cross.get('incremental_f_test', {}).get('f_stat'))}, p={fmt(cross.get('incremental_f_test', {}).get('p_value'), 4)}")

        # Within-run analysis
        within = within_run_lagged_regression(runs, lag_k=lag_k)
        results[f"within_run_lag{lag_k}"] = within

        print(f"  Within-run: {within.get('n_runs', 'NA')} runs")
        print(f"  Mean ZVF coef: {fmt(within.get('mean_zvf_coef'))} (t={fmt(within.get('t_stat_zvf'))}, p={fmt(within.get('p_value_zvf'), 4)})")
        print(f"  Fraction negative ZVF coef: {fmt(within.get('fraction_negative_zvf_coef'))}")
        print(f"  Mean delta R²: {fmt(within.get('mean_delta_r2'))}")

        # Partial correlation analysis
        partial = zvf_partial_correlation_analysis(runs, lag_k=lag_k)
        results[f"partial_corr_lag{lag_k}"] = partial

        print(f"  Zero-order r(ZVF_t, R_{{t+{lag_k}}}): {fmt(partial.get('zero_order', {}).get('zvf_vs_future_reward'))}")
        print(f"  Partial r(ZVF_t, R_{{t+{lag_k}}} | R_t): {fmt(partial.get('partial', {}).get('zvf_vs_future_reward_controlling_current', {}).get('r'))} (p={fmt(partial.get('partial', {}).get('zvf_vs_future_reward_controlling_current', {}).get('p_value'), 4)})")

    # Save results
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))

    # Write markdown report
    write_markdown_report(runs, results)

    print(f"\nResults saved to {OUT_JSON}")
    print(f"Report saved to {OUT_MD}")


def write_markdown_report(runs: list[RunData], results: dict) -> None:
    lines = [
        "# ZVF Lagged Regression Analysis",
        "",
        "## Purpose",
        "",
        "This analysis addresses the reviewer objection that ZVF is tautologically",
        "correlated with reward on binary-reward tasks (since ZVF = p^G + (1-p)^G",
        "is a deterministic function of accuracy p).",
        "",
        "If ZVF is merely a restatement of accuracy, then after controlling for",
        "current reward R_t, ZVF_t should have NO additional predictive power for",
        "future reward R_{t+k}. If ZVF provides independent early-warning signal,",
        "then the coefficient on ZVF_t should remain significant after controlling",
        "for R_t.",
        "",
        f"## Data: {len(runs)} runs with step-level ZVF/GU telemetry",
        "",
        "| Model | Task | Steps |",
        "|---|---|---:|",
    ]
    for run in runs:
        lines.append(f"| {run.model} | {run.task} | {len(run.step_data)} |")

    lines.append("")

    for lag_k in [1, 3, 5, 10]:
        cross = results.get(f"cross_run_lag{lag_k}", {})
        within = results.get(f"within_run_lag{lag_k}", {})
        partial = results.get(f"partial_corr_lag{lag_k}", {})

        lines += [
            f"## Lag k = {lag_k}",
            "",
            f"### Cross-Run Pooled Regression (N = {cross.get('n_observations', 'NA')})",
            "",
            "| Model | R² | Adj. R² | ZVF coef | ZVF SE | ZVF p-value |",
            "|---|---|---|---|---|---|",
        ]

        m1 = cross.get("model1_reward_only", {})
        m2 = cross.get("model2_reward_plus_zvf", {})
        m3 = cross.get("model3_with_interaction", {})

        lines.append(f"| Reward only | {fmt(m1.get('r_squared'))} | {fmt(m1.get('adj_r_squared'))} | — | — | — |")
        if not m2.get("error"):
            zvf_coef = m2.get("beta", [0, 0, 0])[2]
            zvf_se = m2.get("se", [0, 0, 0])[2]
            zvf_pval = m2.get("p_values", [0, 0, 0])[2]
            lines.append(f"| Reward + ZVF | {fmt(m2.get('r_squared'))} | {fmt(m2.get('adj_r_squared'))} | {fmt(zvf_coef)} | {fmt(zvf_se)} | {fmt(zvf_pval, 4)} |")
        if not m3.get("error"):
            zvf_coef = m3.get("beta", [0, 0, 0, 0])[2]
            int_coef = m3.get("beta", [0, 0, 0, 0])[3]
            zvf_pval = m3.get("p_values", [0, 0, 0, 0])[2]
            lines.append(f"| Reward + ZVF + Interaction | {fmt(m3.get('r_squared'))} | {fmt(m3.get('adj_r_squared'))} | {fmt(zvf_coef)} | — | {fmt(zvf_pval, 4)} |")

        lines += [
            "",
            f"**Delta R² from adding ZVF:** {fmt(cross.get('delta_r_squared'))}",
            f"**Delta Adj. R²:** {fmt(cross.get('delta_adj_r_squared'))}",
            f"**Incremental F-test (H0: ZVF coef = 0):** F = {fmt(cross.get('incremental_f_test', {}).get('f_stat'))}, p = {fmt(cross.get('incremental_f_test', {}).get('p_value'), 4)}",
            "",
            "### Within-Run Regression (controls for run-level confounders)",
            "",
            f"- Runs with sufficient data: {within.get('n_runs', 'NA')}",
            f"- Mean ZVF coefficient: {fmt(within.get('mean_zvf_coef'))}",
            f"- Std ZVF coefficient: {fmt(within.get('std_zvf_coef'))}",
            f"- t-test (H0: mean ZVF coef = 0): t = {fmt(within.get('t_stat_zvf'))}, p = {fmt(within.get('p_value_zvf'), 4)}",
            f"- Fraction of runs with negative ZVF coefficient: {fmt(within.get('fraction_negative_zvf_coef'))}",
            f"- Mean ΔR² from adding ZVF: {fmt(within.get('mean_delta_r2'))}",
            "",
            "### Partial Correlation Analysis",
            "",
            f"- Zero-order r(ZVF_t, R_{{t+{lag_k}}}): {fmt(partial.get('zero_order', {}).get('zvf_vs_future_reward'))}",
            f"- Zero-order r(ZVF_t, R_t): {fmt(partial.get('zero_order', {}).get('zvf_vs_current_reward'))}",
            f"- Partial r(ZVF_t, R_{{t+{lag_k}}} | R_t): {fmt(partial.get('partial', {}).get('zvf_vs_future_reward_controlling_current', {}).get('r'))} (p = {fmt(partial.get('partial', {}).get('zvf_vs_future_reward_controlling_current', {}).get('p_value'), 4)})",
            "",
        ]

    # Summary / verdict
    lines += [
        "## Verdict",
        "",
    ]

    # Check if ZVF is independently predictive at lag 5
    lag5_cross = results.get("cross_run_lag5", {})
    lag5_within = results.get("within_run_lag5", {})
    lag5_partial = results.get("partial_corr_lag5", {})

    delta_r2 = lag5_cross.get("delta_r_squared", 0)
    inc_p = lag5_cross.get("incremental_f_test", {}).get("p_value", 1.0)
    within_p = lag5_within.get("p_value_zvf", 1.0)
    partial_r = lag5_partial.get("partial", {}).get("zvf_vs_future_reward_controlling_current", {}).get("r", 0)
    partial_p = lag5_partial.get("partial", {}).get("zvf_vs_future_reward_controlling_current", {}).get("p_value", 1.0)

    if delta_r2 > 0.01 and inc_p < 0.05:
        lines.append("**ZVF provides statistically significant incremental predictive power** beyond current reward.")
        lines.append(f"Adding ZVF to the model increases R² by {fmt(delta_r2)} (incremental F-test p = {fmt(inc_p, 4)}).")
    else:
        lines.append("**ZVF does NOT provide statistically significant incremental predictive power** beyond current reward in the cross-run pooled analysis.")
        lines.append(f"Adding ZVF to the model increases R² by only {fmt(delta_r2)} (incremental F-test p = {fmt(inc_p, 4)}).")

    lines.append("")
    if within_p < 0.05:
        lines.append(f"Within-run analysis confirms: mean ZVF coefficient is significantly non-zero (p = {fmt(within_p, 4)}), supporting ZVF as an independent early-warning signal within training trajectories.")
    else:
        lines.append(f"Within-run analysis: mean ZVF coefficient is NOT significantly non-zero (p = {fmt(within_p, 4)}), suggesting ZVF's predictive power may be driven by cross-run confounders rather than within-trajectory dynamics.")

    lines.append("")
    if abs(partial_r) > 0.1 and partial_p < 0.05:
        lines.append(f"Partial correlation analysis: r(ZVF_t, R_{{t+5}} | R_t) = {fmt(partial_r)} (p = {fmt(partial_p, 4)}), confirming ZVF carries signal beyond current reward.")
    else:
        lines.append(f"Partial correlation analysis: r(ZVF_t, R_{{t+5}} | R_t) = {fmt(partial_r)} (p = {fmt(partial_p, 4)}), suggesting ZVF's zero-order correlation with future reward is largely explained by current reward.")

    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
A4 — CLMP length-mediation estimator validation.

Re-analysis of existing rollout data to estimate how much of the group-size
effect on reward is mediated by completion length.

Data source
-----------
experiments/results/quick_20260704/qp3_group_tensors.jsonl
Paired-phase GRPO run on Qwen/Qwen3.5-4B / GSM8K:
  - phase A: group_size G=4, 16 steps, 16 prompts/step, 64 completions/step
  - phase B: group_size G=8, 16 steps,  8 prompts/step, 64 completions/step
Same prompt pool, matched completion budget, same hyperparameters.

Causal model
------------
T : treatment = 1 if G=8, 0 if G=4
M : mediator  = completion length (tokens)
Y : outcome   = binary reward (correctness)

Estimands
---------
NDE  = E[Y(1, M(0)) - Y(0, M(0))]   natural direct effect
NIE  = E[Y(1, M(1)) - Y(1, M(0))]   natural indirect effect
TE   = E[Y(1, M(1)) - Y(0, M(0))]   total effect
GER  = NIE / TE                       proportion of total effect mediated by length
PM   = NIE / TE                       (same quantity, labelled proportion mediated)

Identification uses standard parametric mediation:
  M|T ~ Normal(gamma0 + gamma1*T, sigma^2)
  Y|T,M ~ Bernoulli(sigmoid(beta0 + beta1*T + beta2*M + beta3*T*M))
Counterfactuals are drawn by Monte Carlo integration.

Outputs
-------
mediation_estimates.json / .tsv
mediation_records.tsv
length_by_group_size.png
reward_by_group_size.png
mediation_bars.png
effect_surface.png
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path("experiments/results/quick_20260704/qp3_group_tensors.jsonl")

SEED = 42
N_BOOT = 2000
N_MC = 50_000


def load_records(path: Path) -> pd.DataFrame:
    """Flatten per-(step,prompt,group) reward/length tensors to records."""
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            g = int(row["group_size"])
            step = int(row["step"])
            pids = row["prompt_indices"]
            rewards = row["rewards"]
            lengths = row["lengths"]
            for pid, rv, lv in zip(pids, rewards, lengths):
                for r, l in zip(rv, lv):
                    records.append(
                        {
                            "group_size": g,
                            "step": step,
                            "prompt_id": pid,
                            "reward": float(r),
                            "length": float(l),
                        }
                    )
    return pd.DataFrame.from_records(records)


def fit_mediator(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """OLS: length ~ 1 + T.  Returns params [intercept, slope], residual sd."""
    X = np.column_stack((np.ones(len(df)), df["T"].values))
    y = df["length_z"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sigma = float(np.sqrt(np.mean(resid**2)))
    return beta, sigma


def neg_loglik_outcome(params: np.ndarray, T: np.ndarray, M: np.ndarray, Y: np.ndarray) -> float:
    """Log-likelihood for logistic regression Y ~ 1 + T + M + T*M."""
    b0, b1, b2, b3 = params
    eta = b0 + b1 * T + b2 * M + b3 * T * M
    # stable log-likelihood
    ll = np.where(
        Y == 1,
        -np.log1p(np.exp(-eta)),
        -np.log1p(np.exp(eta)),
    )
    return -float(np.sum(ll))


def fit_outcome(df: pd.DataFrame) -> np.ndarray:
    """Logistic regression with T*M interaction."""
    T = df["T"].values
    M = df["length_z"].values
    Y = df["reward"].values
    init = np.zeros(4)
    res = minimize(
        neg_loglik_outcome,
        init,
        args=(T, M, Y),
        method="L-BFGS-B",
    )
    if not res.success:
        raise RuntimeError(f"Outcome model failed to converge: {res.message}")
    return res.x


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def estimate_effects(
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: float,
    n_mc: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Monte-Carlo NDE/NIE/TE/GER given fitted mediator + outcome models."""
    b0, b1, b2, b3 = beta
    g0, g1 = gamma

    # Draw counterfactual mediators from their distributions under T=0 and T=1.
    M0 = rng.normal(g0, sigma, size=n_mc)
    M1 = rng.normal(g0 + g1, sigma, size=n_mc)

    def mu_y(t: np.ndarray, m: np.ndarray) -> np.ndarray:
        return sigmoid(b0 + b1 * t + b2 * m + b3 * t * m)

    # Controlled direct effect path: fix mediator, vary treatment.
    y_t1_m0 = mu_y(np.ones_like(M0), M0)
    y_t0_m0 = mu_y(np.zeros_like(M0), M0)
    nde = float(np.mean(y_t1_m0 - y_t0_m0))

    # Indirect effect path: fix treatment=1, vary mediator.
    y_t1_m1 = mu_y(np.ones_like(M1), M1)
    y_t1_m0 = mu_y(np.ones_like(M0), M0)
    nie = float(np.mean(y_t1_m1 - y_t1_m0))

    te = float(np.mean(y_t1_m1 - y_t0_m0))
    ger = nie / te if abs(te) > 1e-12 else float("nan")
    pm = ger

    return {
        "NDE": nde,
        "NIE": nie,
        "TE": te,
        "GER": ger,
        "PM": pm,
        "NDE_plus_NIE": nde + nie,
    }


def bootstrap_effects(
    df: pd.DataFrame,
    n_boot: int,
    n_mc: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Non-parametric bootstrap of mediation estimands."""
    rows = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bdf = df.iloc[idx].copy()
        gamma, sigma = fit_mediator(bdf)
        beta = fit_outcome(bdf)
        rows.append(estimate_effects(beta, gamma, sigma, n_mc, rng))
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame, point: dict, boot: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Render diagnostic and result figures."""
    paths: dict[str, Path] = {}

    # 1. Length distribution by group size
    fig, ax = plt.subplots(figsize=(6, 4))
    for g, color in [(4, "#4c78a8"), (8, "#f58518")]:
        subset = df[df["group_size"] == g]["length"]
        ax.hist(subset, bins=30, alpha=0.6, label=f"G={g}", color=color, density=True)
    ax.set_xlabel("Completion length (tokens)")
    ax.set_ylabel("Density")
    ax.set_title("A4: Completion length by group size")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "length_by_group_size.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["length_by_group_size"] = p

    # 2. Reward distribution by group size
    fig, ax = plt.subplots(figsize=(6, 4))
    means = df.groupby("group_size")["reward"].mean()
    ax.bar(means.index.astype(str), means.values, color=["#4c78a8", "#f58518"])
    ax.set_xlabel("Group size")
    ax.set_ylabel("Mean reward")
    ax.set_title("A4: Mean reward by group size")
    ax.set_ylim(0, 1)
    for x, y in zip(means.index.astype(str), means.values):
        ax.text(x, y + 0.02, f"{y:.3f}", ha="center")
    fig.tight_layout()
    p = out_dir / "reward_by_group_size.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["reward_by_group_size"] = p

    # 3. Mediation effect bar plot with bootstrap CI
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["NDE", "NIE", "TE"]
    vals = [point["NDE"], point["NIE"], point["TE"]]
    ci_low = [boot["NDE"].quantile(0.025), boot["NIE"].quantile(0.025), boot["TE"].quantile(0.025)]
    ci_high = [boot["NDE"].quantile(0.975), boot["NIE"].quantile(0.975), boot["TE"].quantile(0.975)]
    errs = [[v - l for v, l in zip(vals, ci_low)], [h - v for v, h in zip(vals, ci_high)]]
    colors = ["#54a24b", "#eeca3b", "#b279a2"]
    bars = ax.bar(labels, vals, yerr=errs, color=colors, capsize=5, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Effect on reward (probability scale)")
    ax.set_title("A4: CLMP length-mediation estimates (G=8 vs G=4)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.015 if v >= 0 else -0.025), f"{v:.4f}", ha="center", va="bottom" if v >= 0 else "top")
    fig.tight_layout()
    p = out_dir / "mediation_bars.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["mediation_bars"] = p

    # 4. Effect surface: predicted reward vs length for both group sizes
    fig, ax = plt.subplots(figsize=(7, 4.5))
    length_grid = np.linspace(df["length"].min(), df["length"].max(), 200)
    # Recover standardized values
    mu_len = df["length"].mean()
    sd_len = df["length"].std()
    m_z = (length_grid - mu_len) / sd_len

    beta = point["_beta"]
    b0, b1, b2, b3 = beta
    y_t0 = sigmoid(b0 + b2 * m_z)
    y_t1 = sigmoid(b0 + b1 + (b2 + b3) * m_z)
    ax.plot(length_grid, y_t0, color="#4c78a8", label="G=4 (T=0)")
    ax.plot(length_grid, y_t1, color="#f58518", label="G=8 (T=1)")
    ax.set_xlabel("Completion length (tokens)")
    ax.set_ylabel("Predicted reward")
    ax.set_title("A4: Outcome model P(reward=1 | group, length)")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "effect_surface.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["effect_surface"] = p

    return paths


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    df = load_records(DATA_PATH)
    if df.empty:
        raise RuntimeError(f"No data loaded from {DATA_PATH}")

    # Encode treatment and standardize mediator.
    df["T"] = (df["group_size"] == 8).astype(int)
    df["length_z"] = (df["length"] - df["length"].mean()) / df["length"].std()

    # Observed contrasts.
    obs = (
        df.groupby("group_size")
        .agg(n=("reward", "size"), mean_reward=("reward", "mean"), mean_length=("length", "mean"))
        .reset_index()
    )

    # Fit models.
    gamma, sigma = fit_mediator(df)
    beta = fit_outcome(df)

    # Point estimates.
    point = estimate_effects(beta, gamma, sigma, N_MC, rng)
    point["_beta"] = beta  # stash for plotting

    # Bootstrap inference.
    boot = bootstrap_effects(df, N_BOOT, N_MC, rng)

    # Confidence intervals and p-values (percentile + two-sided p).
    ci = {}
    pvals = {}
    for key in ["NDE", "NIE", "TE", "GER", "PM"]:
        s = boot[key].dropna()
        ci[key] = (float(s.quantile(0.025)), float(s.quantile(0.975)))
        # two-sided bootstrap p-value: 2 * min(P<0, P>0)
        p_neg = (s < 0).mean()
        p_pos = (s > 0).mean()
        pvals[key] = float(2 * min(p_neg, p_pos))

    # Summary JSON.
    summary = {
        "experiment": "A4_CLMP_length_mediation",
        "data_source": str(DATA_PATH),
        "n_rollouts": int(len(df)),
        "n_rollouts_G4": int((df["group_size"] == 4).sum()),
        "n_rollouts_G8": int((df["group_size"] == 8).sum()),
        "observed_group_means": obs.to_dict(orient="records"),
        "mediator_model": {
            "formula": "length_z ~ 1 + T",
            "params": {"gamma0": float(gamma[0]), "gamma1": float(gamma[1])},
            "sigma": sigma,
            "interpretation": "G=8 completions are %.3f sd longer than G=4 completions" % float(gamma[1]),
        },
        "outcome_model": {
            "formula": "reward ~ 1 + T + length_z + T:length_z",
            "params": {"beta0": float(beta[0]), "beta1": float(beta[1]), "beta2": float(beta[2]), "beta3": float(beta[3])},
        },
        "estimates": {
            key: {
                "point": float(point[key]),
                "ci_95": [float(ci[key][0]), float(ci[key][1])],
                "p_value": float(pvals[key]),
            }
            for key in ["NDE", "NIE", "TE", "GER", "PM"]
        },
        "headline": "GER=%.3f (%.1f%% of the G=8-vs-G=4 reward effect is mediated by completion length)" % (point["GER"], point["GER"] * 100),
        "method_note": "Parametric causal mediation with Monte Carlo counterfactuals; bootstrap CI based on %d resamples." % N_BOOT,
    }

    json_path = out_dir / "mediation_estimates.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Summary TSV.
    est_rows = []
    for key in ["NDE", "NIE", "TE", "GER", "PM"]:
        est_rows.append(
            {
                "estimand": key,
                "point": point[key],
                "ci_lower": ci[key][0],
                "ci_upper": ci[key][1],
                "p_value": pvals[key],
            }
        )
    est_df = pd.DataFrame(est_rows)
    est_df.to_csv(out_dir / "mediation_estimates.tsv", sep="\t", index=False, float_format="%.6f")

    # Raw records TSV.
    df[["group_size", "step", "prompt_id", "reward", "length"]].to_csv(
        out_dir / "mediation_records.tsv", sep="\t", index=False
    )

    # Figures.
    fig_paths = make_figures(df, point, boot, out_dir)

    # Print concise report.
    print("A4 CLMP length-mediation analysis complete")
    print(f"  rollouts: {len(df)} (G=4: {(df['group_size']==4).sum()}, G=8: {(df['group_size']==8).sum()})")
    print(f"  NDE = {point['NDE']:.4f} [{ci['NDE'][0]:.4f}, {ci['NDE'][1]:.4f}], p={pvals['NDE']:.3f}")
    print(f"  NIE = {point['NIE']:.4f} [{ci['NIE'][0]:.4f}, {ci['NIE'][1]:.4f}], p={pvals['NIE']:.3f}")
    print(f"  TE  = {point['TE']:.4f}  [{ci['TE'][0]:.4f}, {ci['TE'][1]:.4f}], p={pvals['TE']:.3f}")
    print(f"  GER = {point['GER']:.3f} [{ci['GER'][0]:.3f}, {ci['GER'][1]:.3f}], p={pvals['GER']:.3f}")
    print(f"  outputs: {out_dir}")
    for name, p in fig_paths.items():
        print(f"    {name}: {p.name}")


if __name__ == "__main__":
    main()

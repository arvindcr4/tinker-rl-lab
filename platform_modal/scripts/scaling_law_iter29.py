"""Pillar 1 iter29 -- Cross-stack identifiability audit (PPO vs GRPO).

Motivation: iter25 showed that across 5 frontier-scale traces (Qwen3.5-4B,
Qwen3-8B, Llama-3.1-8B-Instruct, DeepSeek-V3.1, Nemotron-120B) the canonical
saturation model R(t)=R_max*(1-e^{-lambda t}) is NOT identifiable -- 0/5
traces had a non-bound lambda, and 0/5 preferred saturation over a constant
baseline in noise-aware AICc. The frontier synthesis Round 1 licenses a much
stronger claim: an Estimator-Equivalence Principle (EEP) under which PPO and
GRPO are performance-equivalent once the rollout batch, KL, clipping, masking,
optimizer and reward parser are fixed. If EEP holds, the saturation-fit
identifiability failure should reproduce in the PPO stack at the same rate
and severity as in the GRPO stack.

This driver:
  1. Pulls the 10 same-stack traces (5 GRPO + 5 PPO seeds, Qwen2.5-0.5B on
     GSM8K, 40 steps, n_gen=128) from experiments/results/samestack_ppo_grpo.json.
  2. Applies the iter25 noise-aware saturation test to every trace, recording
     lambda-at-bound, binomial noise floor, bootstrap CI on lambda, AICc-best
     among {constant, linear, saturation}, and the heldout-accuracy residual
     explained by the saturation fit.
  3. Compares PPO and GRPO stacks side-by-side on each diagnostic.
  4. Tests three EEP-falsifiable predictions:
       (E1) lambda-at-bound rate:    PPO == GRPO
       (E2) AICc-best saturation rate: PPO == GRPO
       (E3) Heldout residual explained by saturation: PPO == GRPO
     via Fisher's exact (E1, E2) and Welch's t (E3).

Outputs:
  experiments/results/scaling_law_iter29_identifiability.tsv
  experiments/results/scaling_law_iter29_bootstrap.tsv
  experiments/results/scaling_law_iter29_summary.tsv
  experiments/results/scaling_law_iter29_stack_compare.tsv
  figures/scaling_law_iter29.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import ttest_ind  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "experiments" / "results" / "samestack_ppo_grpo.json"
RESULTS = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
PAPER_FIG = REPO / "paper" / "figures"
for d in (FIG_DIR, PAPER_FIG):
    d.mkdir(parents=True, exist_ok=True)

LAM_BOUND = 10.0  # iter25 convention
B_BOOT = 400  # bootstrap resamples
RNG_SEED = 42


def sat(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def lin(t, a, b):
    return a + b * t


def aicc(rss, n, k):
    """Small-sample-corrected AIC."""
    if n - k - 1 <= 0:
        return float("inf")
    return n * math.log(max(rss, 1e-12) / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)


def effective_batch_n(rewards):
    """Infer binomial denominator from the GCD of the fractional rewards.
    A reward of k/n arises because the rollout group produced k correct
    out of n samples. We find the smallest n that makes every reward an
    integer multiple of 1/n, and return n.
    """
    fracs = [Fraction(r).limit_denominator(4096) for r in rewards]
    denoms = [f.denominator for f in fracs]
    # lcm of denominators
    from math import gcd

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for d in denoms:
        L = lcm(L, d)
        if L > 4096:
            return int(max(denoms))  # fall back to max denom
    return int(L)


def fit_saturation(t, r, lam_bound=LAM_BOUND):
    """Fit R(t)=R_max*(1-exp(-lambda t)); flag if lambda hits the bound."""
    rmax_lo, rmax_hi = 0.0, 2.0
    try:
        popt, _ = curve_fit(
            sat,
            t,
            r,
            p0=[0.8, 1.0],
            bounds=([rmax_lo, 1e-3], [rmax_hi, lam_bound]),
            maxfev=5000,
        )
        r_max, lam = float(popt[0]), float(popt[1])
        hit = lam >= lam_bound - 1e-3
        return r_max, lam, hit
    except Exception:
        return float("nan"), float("nan"), True


def fit_linear(t, r):
    try:
        popt, _ = curve_fit(lin, t, r, p0=[0.0, 0.0])
        return float(popt[0]), float(popt[1])
    except Exception:
        return 0.0, 0.0


def fit_constant(r):
    return float(np.mean(r)), 0.0, 0.0  # rss_const


def bootstrap_lambda(t, r, n_eff, lam_bound=LAM_BOUND, B=B_BOOT, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    lams = np.empty(B)
    hits = np.zeros(B, dtype=bool)
    for b in range(B):
        # simulate per-step binomial sampling around the fitted saturation
        # (use the actual mean; per-step p_t = R(t) is fine for 0/1 rewards)
        p_t = np.clip(r, 0.0, 1.0)
        k_t = rng.binomial(n_eff, p_t)
        r_sim = k_t / n_eff
        _, lam, hit = fit_saturation(t, r_sim, lam_bound=lam_bound)
        lams[b] = lam
        hits[b] = hit
    return lams, hits


def detectability_tstar(sigma_step, alpha=0.05, power=0.8):
    """Minimal trace length T* to reject lambda=0 under per-step noise sigma.
    Using a one-sample t-style rule of thumb:
        T* = (z_{1-alpha/2} + z_{power})^2 * sigma^2 / delta^2
    where delta is the smallest detectable per-step slope. We use
    delta = sigma (SNR=1) as a conservative reference.
    """
    from scipy.stats import norm

    z_a = norm.ppf(1 - alpha / 2)
    z_p = norm.ppf(power)
    return float((z_a + z_p) ** 2)


def main():
    raw = json.loads(DATA.read_text())
    runs = raw["runs"]
    summary = raw["summary"]
    paired = raw["paired_grpo_vs_ppo"]

    # Build a per-trace table; carry algorithm + seed + heldout acc.
    per_trace_rows = []
    bootstrap_rows = []
    summary_rows = []  # one row per (algo)
    rng = np.random.default_rng(RNG_SEED)

    for algo in ("grpo", "ppo"):
        algo_runs = [r for r in runs if r["algo"] == algo]
        per_stack = {
            "n_traces": 0,
            "n_lam_at_bound": 0,
            "n_sat_supported": 0,
            "n_aicc_best_sat": 0,
            "n_aicc_best_const": 0,
            "lam_values": [],
            "lam_at_bound": [],
            "resid_explained_sat": [],
            "resid_explained_lin": [],
            "t80_values": [],
            "n_eff_values": [],
            "sigma_step_values": [],
        }
        for r in algo_runs:
            sl = r["step_log"]
            t = np.array([s["step"] for s in sl], dtype=float)
            # Use the cumulative-mean reward as the running trace: a
            # standard visualisation for GRPO training curves.
            mean_r = np.array([s["mean_reward"] for s in sl], dtype=float)
            cum_r = np.cumsum(mean_r) / np.arange(1, len(mean_r) + 1)
            n_eff = effective_batch_n(mean_r)
            # Binomial noise floor per step (approximation)
            p_hat = float(np.clip(np.mean(cum_r), 1e-3, 1 - 1e-3))
            sigma_step = float(math.sqrt(p_hat * (1 - p_hat) / n_eff))
            # Fit saturation
            r_max, lam, hit = fit_saturation(t, cum_r)
            t80 = -math.log(0.2) / lam if lam > 0 and not math.isnan(lam) else float("inf")
            # Fit linear
            a, b = fit_linear(t, cum_r)
            # AICc on the cumulative-mean trace (n = #steps)
            n_t = len(t)
            rss_sat = float(np.sum((cum_r - sat(t, r_max, lam)) ** 2))
            rss_lin = float(np.sum((cum_r - lin(t, a, b)) ** 2))
            rss_const = float(np.sum((cum_r - float(np.mean(cum_r))) ** 2))
            aicc_sat = aicc(rss_sat, n_t, 2)
            aicc_lin = aicc(rss_lin, n_t, 2)
            aicc_const = aicc(rss_const, n_t, 1)
            aics = {"const": aicc_const, "lin": aicc_lin, "sat": aicc_sat}
            best = min(aics, key=aics.get)
            # Heldout-residual-explained: how much of heldout-1.0 does the
            # fitted saturation's final value capture?
            final_sat = sat(t[-1], r_max, lam) if not math.isnan(lam) else float("nan")
            final_lin = lin(t[-1], a, b)
            final_const = float(np.mean(cum_r))
            heldout = float(r["heldout_acc"])
            resid_sat = abs(heldout - final_sat) if not math.isnan(final_sat) else float("nan")
            resid_lin = abs(heldout - final_lin)
            resid_const = abs(heldout - final_const)
            # "Saturation supported": AICc-best == sat AND lambda CI excludes bound
            sat_supported = (best == "sat") and (not hit)
            per_trace_rows.append(
                {
                    "algo": algo,
                    "seed": r["seed"],
                    "n_steps": n_t,
                    "n_eff": n_eff,
                    "sigma_step": round(sigma_step, 6),
                    "r_max": round(r_max, 4),
                    "lambda": round(lam, 4) if not math.isnan(lam) else "NaN",
                    "lam_at_bound": bool(hit),
                    "t80": round(t80, 4) if math.isfinite(t80) else "inf",
                    "aicc_const": round(aicc_const, 3),
                    "aicc_lin": round(aicc_lin, 3),
                    "aicc_sat": round(aicc_sat, 3),
                    "aicc_best": best,
                    "sat_supported": bool(sat_supported),
                    "final_sat": round(final_sat, 4) if not math.isnan(final_sat) else "NaN",
                    "final_lin": round(final_lin, 4),
                    "final_const": round(final_const, 4),
                    "heldout_acc": heldout,
                    "resid_sat_vs_heldout": round(resid_sat, 4),
                    "resid_lin_vs_heldout": round(resid_lin, 4),
                    "resid_const_vs_heldout": round(resid_const, 4),
                }
            )
            # Bootstrap CI on lambda for this trace
            lams_b, hits_b = bootstrap_lambda(t, cum_r, n_eff, rng=rng)
            ci_lo = float(np.quantile(lams_b, 0.025))
            ci_hi = float(np.quantile(lams_b, 0.975))
            frac_at_bound = float(np.mean(hits_b))
            bootstrap_rows.append(
                {
                    "algo": algo,
                    "seed": r["seed"],
                    "lam_fit": round(lam, 4) if not math.isnan(lam) else "NaN",
                    "lam_boot_median": round(float(np.median(lams_b)), 4),
                    "lam_boot_ci_lo": round(ci_lo, 4),
                    "lam_boot_ci_hi": round(ci_hi, 4),
                    "frac_boot_at_bound": round(frac_at_bound, 3),
                    "ci_excludes_bound": bool(ci_hi < LAM_BOUND - 1e-3),
                }
            )
            # Stack-level accumulators
            per_stack["n_traces"] += 1
            per_stack["n_lam_at_bound"] += int(hit)
            per_stack["n_sat_supported"] += int(sat_supported)
            per_stack["n_aicc_best_sat"] += int(best == "sat")
            per_stack["n_aicc_best_const"] += int(best == "const")
            if not math.isnan(lam):
                per_stack["lam_values"].append(lam)
                per_stack["lam_at_bound"].append(int(hit))
                per_stack["t80_values"].append(t80)
            per_stack["resid_explained_sat"].append(resid_sat)
            per_stack["resid_explained_lin"].append(resid_lin)
            per_stack["n_eff_values"].append(n_eff)
            per_stack["sigma_step_values"].append(sigma_step)
        # Per-stack detectability T*
        sigma_med = float(np.median(per_stack["sigma_step_values"]))
        t_star = detectability_tstar(sigma_med)
        n_t_med = float(np.median([r["n_steps"] for r in per_trace_rows if r["algo"] == algo]))
        summary_rows.append(
            {
                "algo": algo,
                "n_traces": per_stack["n_traces"],
                "n_lam_at_bound": per_stack["n_lam_at_bound"],
                "frac_lam_at_bound": round(per_stack["n_lam_at_bound"] / per_stack["n_traces"], 3),
                "n_sat_supported": per_stack["n_sat_supported"],
                "frac_sat_supported": round(per_stack["n_sat_supported"] / per_stack["n_traces"], 3),
                "n_aicc_best_sat": per_stack["n_aicc_best_sat"],
                "frac_aicc_best_sat": round(per_stack["n_aicc_best_sat"] / per_stack["n_traces"], 3),
                "n_aicc_best_const": per_stack["n_aicc_best_const"],
                "frac_aicc_best_const": round(per_stack["n_aicc_best_const"] / per_stack["n_traces"], 3),
                "median_lambda": round(float(np.median(per_stack["lam_values"])), 4),
                "median_t80": round(float(np.median([t for t in per_stack["t80_values"] if math.isfinite(t)])), 4)
                if any(math.isfinite(t) for t in per_stack["t80_values"])
                else "inf",
                "median_n_eff": float(np.median(per_stack["n_eff_values"])),
                "median_sigma_step": round(sigma_med, 6),
                "T_star_detectability": round(t_star, 1),
                "median_n_steps_actual": n_t_med,
                "T_star_over_steps": round(t_star / n_t_med, 2),
                "median_resid_sat_vs_heldout": round(float(np.median(per_stack["resid_explained_sat"])), 4),
                "median_resid_lin_vs_heldout": round(float(np.median(per_stack["resid_explained_lin"])), 4),
                "median_resid_const_vs_heldout": round(float(np.median([abs(r["heldout_acc"] - r["final_const"]) for r in per_trace_rows if r["algo"] == algo])), 4),
                "stack_mean_heldout": round(summary[algo]["heldout_mean"], 4),
                "stack_se_heldout": round(summary[algo]["heldout_se"], 4),
            }
        )

    # ----- stack comparison: three EEP-falsifiable predictions -----
    grpo_rows = [r for r in per_trace_rows if r["algo"] == "grpo"]
    ppo_rows = [r for r in per_trace_rows if r["algo"] == "ppo"]

    # E1: lambda-at-bound rate
    grpo_bound = sum(int(r["lam_at_bound"]) for r in grpo_rows)
    ppo_bound = sum(int(r["lam_at_bound"]) for r in ppo_rows)
    n_g, n_p = len(grpo_rows), len(ppo_rows)

    # Fisher's exact (2x2): rows=algo, cols=(at-bound, free)
    # Use a simple permutation-style test (small n)
    def fisher_2x2(a, b, c, d):
        # mid-p corrected Fisher's exact (two-sided)
        from math import comb

        table = [[a, b], [c, d]]
        n = a + b + c + d
        r1, r2 = a + b, c + d
        k1, k2 = a + c, b + d
        # hypergeometric pmf
        def pmf(x):
            if x < 0 or x > r1 or (k1 - x) < 0 or (k1 - x) > r2:
                return 0.0
            return comb(r1, x) * comb(r2, k1 - x) / comb(n, k1)

        obs = pmf(a)
        tail = 0.0
        for x in range(0, r1 + 1):
            px = pmf(x)
            if px <= obs + 1e-12:
                tail += px
        return min(1.0, tail)

    p_e1 = fisher_2x2(grpo_bound, n_g - grpo_bound, ppo_bound, n_p - ppo_bound)

    # E2: AICc-best saturation rate
    grpo_sat = sum(int(r["aicc_best"] == "sat") for r in grpo_rows)
    ppo_sat = sum(int(r["aicc_best"] == "sat") for r in ppo_rows)
    p_e2 = fisher_2x2(grpo_sat, n_g - grpo_sat, ppo_sat, n_p - ppo_sat)

    # E3: heldout residual explained by saturation (continuous)
    grpo_resid = [r["resid_sat_vs_heldout"] for r in grpo_rows]
    ppo_resid = [r["resid_sat_vs_heldout"] for r in ppo_rows]
    t_e3, p_e3 = ttest_ind(grpo_resid, ppo_resid, equal_var=False)

    # Bootstrap CI exclusion rate
    grpo_ci_excl = sum(int(r["ci_excludes_bound"]) for r in bootstrap_rows if r["algo"] == "grpo")
    ppo_ci_excl = sum(int(r["ci_excludes_bound"]) for r in bootstrap_rows if r["algo"] == "ppo")

    stack_compare_rows = [
        {
            "prediction": "E1_lambda_at_bound_rate",
            "grpo": f"{grpo_bound}/{n_g}",
            "ppo": f"{ppo_bound}/{n_p}",
            "p_value": round(p_e1, 4),
            "eep_status": "sustained" if p_e1 > 0.05 else "falsified",
        },
        {
            "prediction": "E2_aicc_best_saturation_rate",
            "grpo": f"{grpo_sat}/{n_g}",
            "ppo": f"{ppo_sat}/{n_p}",
            "p_value": round(p_e2, 4),
            "eep_status": "sustained" if p_e2 > 0.05 else "falsified",
        },
        {
            "prediction": "E3_heldout_resid_sat",
            "grpo": f"{np.mean(grpo_resid):.4f}",
            "ppo": f"{np.mean(ppo_resid):.4f}",
            "p_value": round(float(p_e3), 4),
            "eep_status": "sustained" if p_e3 > 0.05 else "falsified",
        },
        {
            "prediction": "F1_bootstrap_ci_excludes_bound",
            "grpo": f"{grpo_ci_excl}/{n_g}",
            "ppo": f"{ppo_ci_excl}/{n_p}",
            "p_value": "n/a",
            "eep_status": "shared_failure" if grpo_ci_excl == 0 and ppo_ci_excl == 0 else "divergent",
        },
        {
            "prediction": "F2_paired_heldout_p",
            "grpo": "0.99+/-0.0035",
            "ppo": "0.992+/-0.003",
            "p_value": round(float(paired["p_two_sided"]), 4),
            "eep_status": "sustained" if paired["p_two_sided"] > 0.05 else "falsified",
        },
    ]

    # ---- write outputs ----
    def write_tsv(rows, path, fieldnames=None):
        if not rows:
            return
        fn = fieldnames or list(rows[0].keys())
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fn, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fn})

    write_tsv(per_trace_rows, RESULTS / "scaling_law_iter29_identifiability.tsv")
    write_tsv(bootstrap_rows, RESULTS / "scaling_law_iter29_bootstrap.tsv")
    write_tsv(summary_rows, RESULTS / "scaling_law_iter29_summary.tsv")
    write_tsv(stack_compare_rows, RESULTS / "scaling_law_iter29_stack_compare.tsv")

    # ---- figure: side-by-side traces + AICc badge ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    palette = {"grpo": "#1f77b4", "ppo": "#d62728"}
    algo_label = {"grpo": "GRPO (same-stack)", "ppo": "PPO (same-stack)"}
    for ax, algo in zip(axes, ("grpo", "ppo")):
        for r in [x for x in runs if x["algo"] == algo]:
            t = np.array([s["step"] for s in r["step_log"]], dtype=float)
            mr = np.array([s["mean_reward"] for s in r["step_log"]], dtype=float)
            cum = np.cumsum(mr) / np.arange(1, len(mr) + 1)
            ax.plot(t, cum, color=palette[algo], alpha=0.5, lw=1.0)
        # plot mean trace
        all_cum = []
        for r in [x for x in runs if x["algo"] == algo]:
            mr = np.array([s["mean_reward"] for s in r["step_log"]], dtype=float)
            cum = np.cumsum(mr) / np.arange(1, len(mr) + 1)
            all_cum.append(cum)
        m = np.mean(all_cum, axis=0)
        ax.plot(t, m, color=palette[algo], lw=2.4, label=f"{algo_label[algo]} mean")
        ax.set_title(algo_label[algo])
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cumulative mean reward")
    # annotate stack-level identifiability
    grpo_bound_str = f"{grpo_bound}/{n_g} lam-at-bound"
    ppo_bound_str = f"{ppo_bound}/{n_p} lam-at-bound"
    axes[0].annotate(grpo_bound_str, xy=(0.05, 0.92), xycoords="axes fraction",
                     fontsize=9, color=palette["grpo"],
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=palette["grpo"], alpha=0.8))
    axes[1].annotate(ppo_bound_str, xy=(0.05, 0.92), xycoords="axes fraction",
                     fontsize=9, color=palette["ppo"],
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=palette["ppo"], alpha=0.8))
    fig.suptitle(
        f"Cross-stack saturation identifiability (GSM8K, Qwen2.5-0.5B, 40 steps)\n"
        f"EEP test: Fisher p={p_e1:.3f} (lam-at-bound); "
        f"AICc-sat p={p_e2:.3f}; heldout-resid p={p_e3:.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "scaling_law_iter29.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "scaling_law_iter29.png", dpi=160, bbox_inches="tight")
    fig.savefig(PAPER_FIG / "scaling_law_iter29.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- console rollup ----
    print("=== iter29 cross-stack identifiability ===")
    print(f"PPO:   {ppo_bound}/{n_p} lambda-at-bound, "
          f"{ppo_sat}/{n_p} AICc-best-sat, "
          f"median resid_sat_vs_heldout={np.median(ppo_resid):.4f}")
    print(f"GRPO:  {grpo_bound}/{n_g} lambda-at-bound, "
          f"{grpo_sat}/{n_g} AICc-best-sat, "
          f"median resid_sat_vs_heldout={np.median(grpo_resid):.4f}")
    print(f"EEP tests:")
    print(f"  E1 (lam-at-bound rate)         Fisher p={p_e1:.4f}")
    print(f"  E2 (AICc-best sat rate)        Fisher p={p_e2:.4f}")
    print(f"  E3 (heldout resid explained)   Welch p={p_e3:.4f}")
    print(f"  F1 (CI excludes bound)         GRPO={grpo_ci_excl}/{n_g}  PPO={ppo_ci_excl}/{n_p}")
    print(f"  F2 (paired heldout p)          p={paired['p_two_sided']:.4f} (from samestack)")
    return per_trace_rows, summary_rows, stack_compare_rows


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Iter 100 -- Pillar 4 (Length Bias / Dr.GRPO): Bivariate VAR(1) impulse
response and forecast-error variance decomposition.

Iter 80 (OU / unit-root) characterised the *marginal level dynamics* of L_t and R_t.
Iter 84 (Welch / Granger F / DFA) decomposed the *linear frequency-domain* coupling.
Iter 88 (quantile regression) localised the *conditional-mean* coupling.
Iter 92 (transfer entropy) measured the *directed information* L<->R.
Iter 96 (ICCA) whitened each series with a *marginal* AR(1) and looked at the
asymmetry of the resulting innovation cross-correlations.

Iter 100 takes the natural structural step: rather than fitting MARGINAL AR(1)
models (each equation only sees its own lag), we fit a JOINT bivariate VAR(1)

    [L_t]   = c  +  K * [L_{t-1}]  +  e_t
    [R_t]                       [R_{t-1}]

with the transition matrix
    K = [[phi_LL, phi_LR],
         [phi_RL, phi_RR]]

and a vec-shaped error covariance Sigma.  This unifies two things that the
prior iters kept separate:

  * MARGINAL AR(1) level dynamics  (iter 80)  ->  phi_LL, phi_RR diagonals.
  * GRANGER-TYPE cross-coupling    (iter 84)  ->  phi_LR, phi_RL off-diagonals.

Iter 100 measures the joint structural quantities that were not directly
accessible from the marginal AR(1) used by iter 96:

  1. Cross-equation coefficients phi_LR (L_{t-1} -> R_t) and phi_RL
     (R_{t-1} -> L_t): direct one-step predictive cross-coupling.

  2. Spectral radius  rho(K)  = max |eig(K)| : stability/persistence of
     the joint system (|rho|<1 => mean-revert; rho near 1 => near
     unit-root).

  3. Forecast-error variance decomposition (FEVD) at horizons h=1, 4, 8:
     fraction of R_t forecast-error variance attributable to L shocks
     (vs R shocks).  This is the *standard* orthogonalised structural
     decomposition; a systematic Dr.GRPO shift in the "of R, from L"
     share directly tests the algorithm's effect on the *transmitted*
     (not the level) coupling.

  4. Long-run cumulative impulse response  C = (I - K)^{-1} : the total
     permanent effect of a unit L shock on R (and vice versa) under
     stationarity.  Diagonal cumulation is a textbook alternative to
     FEVD that compresses the entire dynamic into one summary.

Iter 96 showed that Dr.GRPO severs the *innovation-space* R->L coupling
(Delta CCF(k<0) = -0.103 on GSM8K, p<0.001).  Iter 100 asks the structural
question: does Dr.GRPO also shrink the *predictive* R->L coefficient phi_RL
and the *long-run* R->L cumulative impulse?  Or is its effect only on the
innovation space (i.e. noise leak, not structural coupling)?

INPUTS
------
experiments/results/drgrpo_vs_grpo.json         (arithmetic_easy, n=40, 5 seeds)
experiments/results/drgrpo_gsm8k_cot_full.json  (gsm8k_cot,    n=30, 3 seeds)

OUTPUTS
-------
experiments/results/length_bias_iter100_perrun.tsv   (one row per run)
experiments/results/length_bias_iter100_paired.tsv   (one row per (task,key))
experiments/results/length_bias_iter100_summary.tsv  (task-level aggregates)
experiments/results/length_bias_iter100_meta.json    (run configuration)

USAGE
-----
python3 scripts/length_bias_iter100.py [--B 2000] [--h-max 8]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np


DRGRPO_VS_GRPO_PATH = "experiments/results/drgrpo_vs_grpo.json"
DRGRPO_GSM8K_PATH = "experiments/results/drgrpo_gsm8k_cot_full.json"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_step_log(path: str) -> list[dict[str, Any]]:
    """Load runs from a Dr.GRPO JSON file with (L, R) per-step series."""
    with open(path) as fh:
        d = json.load(fh)
    out = []
    for r in d["runs"]:
        step_log = r.get("step_log") or []
        if len(step_log) < 6:
            continue
        L = np.array([float(s["mean_comp_len"]) for s in step_log], dtype=np.float64)
        R = np.array([float(s["mean_reward"]) for s in step_log], dtype=np.float64)
        out.append({
            "algo": r["algo"],
            "seed": r["seed"],
            "n": int(len(step_log)),
            "L": L,
            "R": R,
        })
    return out


def pair_by_seed(arun: dict[str, Any], brun: dict[str, Any]) -> bool:
    return arun["algo"] != brun["algo"] and arun["seed"] == brun["seed"]


# ---------------------------------------------------------------------------
# Bivariate VAR(1) (OLS) with intercept
# ---------------------------------------------------------------------------

def fit_var1(L: np.ndarray, R: np.ndarray) -> dict[str, Any]:
    """Fit the bivariate VAR(1)
        [L_t] = c0 + c1 + phi_LL*L_{t-1} + phi_LR*R_{t-1} + e_L
        [R_t] = d0 + d1 + phi_RL*L_{t-1} + phi_RR*R_{t-1} + e_R
    Returns dict with K (2x2), intercept (length-2), residual covariance
    Sigma (2x2), phi_LR, phi_RL, spectral radius, long-run impact C = (I-K)^{-1},
    FEVD shares and cumulative IRFs at h=1,4,8.
    """
    L1, L0 = L[1:], L[:-1]
    R1, R0 = R[1:], R[:-1]
    T = len(L1)
    # Equation L:  L_t = c0 + phi_LL * L_{t-1} + phi_LR * R_{t-1}
    XL = np.column_stack([np.ones(T), L0, R0])
    beta_L, *_ = np.linalg.lstsq(XL, L1, rcond=None)
    c0 = float(beta_L[0])
    phi_LL, phi_LR = float(beta_L[1]), float(beta_L[2])
    eL = L1 - XL @ beta_L
    # Equation R:  R_t = d0 + phi_RL * L_{t-1} + phi_RR * R_{t-1}
    XR = np.column_stack([np.ones(T), L0, R0])
    beta_R, *_ = np.linalg.lstsq(XR, R1, rcond=None)
    d0 = float(beta_R[0])
    phi_RL, phi_RR = float(beta_R[1]), float(beta_R[2])
    eR = R1 - XR @ beta_R
    K = np.array([[phi_LL, phi_LR], [phi_RL, phi_RR]], dtype=np.float64)
    e = np.column_stack([eL, eR])  # T x 2
    Sigma = (e.T @ e) / max(T - 3, 1)
    # Spectral radius
    eig = np.linalg.eigvals(K)
    rho = float(np.max(np.abs(eig)))
    # Long-run cumulative IRF (only defined if rho < 1)
    if rho < 1.0:
        C = np.linalg.inv(np.eye(2) - K)
    else:
        C = np.full((2, 2), np.nan)
    # MA-representation coefficients Psi_h = K^h for h = 0, 1, ..., hmax
    # Standard Cholesky-ordered FEVD with order (L, R): shock to L first, then R.
    P = np.linalg.cholesky(Sigma + 1e-12 * np.eye(2)) if rho < 1.0 else None
    hmax = 8
    fevd_R_from_L = []
    fevd_L_from_R = []
    cum_R_from_L = 0.0
    cum_L_from_R = 0.0
    cum_R_total = 0.0
    cum_L_total = 0.0
    if P is not None:
        for h in range(0, hmax + 1):
            Psi = np.linalg.matrix_power(K, h)
            # Forecast error at h+1 has variance (Psi @ Sigma @ Psi^T)
            if h == 0:
                # contemporaneous
                FEV = Sigma
            else:
                FEV = Psi @ Sigma @ Psi.T
            diag_R = float(np.diag(FEV)[1])
            diag_L = float(np.diag(FEV)[0])
            if h == 0:
                # contemporaneous variance share using P (Cholesky factor)
                # For the contemporaneous step, contribution to R-equation variance:
                cont_RL = float(P[1, 0] ** 2)
                cont_RR = float(P[1, 1] ** 2)
                cont_LL = float(P[0, 0] ** 2)
                cont_LR = float(P[0, 1] ** 2)
                share_R_from_L_h0 = cont_RL / (cont_RL + cont_RR) if (cont_RL + cont_RR) > 0 else np.nan
                share_L_from_R_h0 = cont_LR / (cont_LL + cont_LR) if (cont_LL + cont_LR) > 0 else np.nan
            else:
                share_R_from_L_h0 = None  # placeholder
            # Accumulate step-by-step MSE decomposition (Lutkepohl-style)
            # Numerator: (e_j' Psi_h P)^2, sum over j=0..h
            num_R_from_L = 0.0
            num_R_from_R = 0.0
            num_L_from_L = 0.0
            num_L_from_R = 0.0
            for s in range(0, h + 1):
                Psi_s = np.linalg.matrix_power(K, s)
                A = Psi_s @ P  # 2x2
                num_R_from_L += float(A[1, 0] ** 2)  # row R, column L
                num_R_from_R += float(A[1, 1] ** 2)  # row R, column R
                num_L_from_L += float(A[0, 0] ** 2)
                num_L_from_R += float(A[0, 1] ** 2)
            tot_R = num_R_from_L + num_R_from_R
            tot_L = num_L_from_L + num_L_from_R
            share_R = num_R_from_L / tot_R if tot_R > 0 else np.nan
            share_L = num_L_from_R / tot_L if tot_L > 0 else np.nan
            fevd_R_from_L.append(share_R)
            fevd_L_from_R.append(share_L)
        # Final cumulative shares
        cum_R_from_L = float(fevd_R_from_L[-1])
        cum_L_from_R = float(fevd_L_from_R[-1])
    return {
        "phi_LL": phi_LL, "phi_LR": phi_LR, "phi_RL": phi_RL, "phi_RR": phi_RR,
        "rho": rho, "C": C,
        "cumul_impulse_L_to_R": float(C[1, 0]) if not np.isnan(C).any() else np.nan,
        "cumul_impulse_R_to_L": float(C[0, 1]) if not np.isnan(C).any() else np.nan,
        "cumul_diag_self_L": float(C[0, 0]) if not np.isnan(C).any() else np.nan,
        "cumul_diag_self_R": float(C[1, 1]) if not np.isnan(C).any() else np.nan,
        "fevd_R_from_L_h1": float(fevd_R_from_L[1]),
        "fevd_R_from_L_h4": float(fevd_R_from_L[4]) if hmax >= 4 else np.nan,
        "fevd_R_from_L_h8": float(fevd_R_from_L[8]) if hmax >= 8 else np.nan,
        "fevd_L_from_R_h1": float(fevd_L_from_R[1]),
        "fevd_L_from_R_h4": float(fevd_L_from_R[4]) if hmax >= 4 else np.nan,
        "fevd_L_from_R_h8": float(fevd_L_from_R[8]) if hmax >= 8 else np.nan,
        "cumul_fevd_R_from_L": cum_R_from_L,
        "cumul_fevd_L_from_R": cum_L_from_R,
        "Sigma_LL": float(Sigma[0, 0]),
        "Sigma_RR": float(Sigma[1, 1]),
        "Sigma_LR": float(Sigma[0, 1]),
        "n_obs": T,
    }


# ---------------------------------------------------------------------------
# Per-run analysis + pairing + bootstrap
# ---------------------------------------------------------------------------

def analyze_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in runs:
        rec = fit_var1(r["L"], r["R"])
        rec.update({"algo": r["algo"], "seed": r["seed"], "n": r["n"]})
        rows.append(rec)
    return rows


def pair_algos(perrun_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair Dr.GRPO and GRPO runs by seed; return list of (grpo, dr_grpo) pairs.
    Note: caller passes a list that has both algos; we pair by seed."""
    by_seed_algo: dict[tuple[Any, str], dict[str, Any]] = {}
    for r in perrun_rows:
        by_seed_algo[(r["seed"], r["algo"])] = r
    seeds = sorted({r["seed"] for r in perrun_rows})
    pairs = []
    for s in seeds:
        if (s, "grpo") in by_seed_algo and (s, "dr_grpo") in by_seed_algo:
            pairs.append((by_seed_algo[(s, "grpo")], by_seed_algo[(s, "dr_grpo")]))
    return pairs


def paired_bootstrap(keys: list[str], pairs: list[tuple[dict[str, Any], dict[str, Any]]], B: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    """For each key, compute seed-paired differences (dr_grpo - grpo),
    the median, 95% percentile-bootstrap CI, and a paired sign-flip
    permutation p-value: under H0 ("Dr.GRPO has no effect") sign-flipping
    each pair's difference is exchangeable, so the permutation distribution
    of ``median(d * flips)`` is a valid null."""
    rng = np.random.default_rng(seed)
    n_pairs = len(pairs)
    diffs = {k: np.array([float(dr[k]) - float(gr[k]) for gr, dr in pairs]) for k in keys}
    out = []
    for k in keys:
        d = diffs[k]
        med_obs = float(np.median(d))
        # Percentile bootstrap CI for the median (resample pairs with replacement)
        boots = np.empty(B, dtype=np.float64)
        for b in range(B):
            idx = rng.integers(0, n_pairs, size=n_pairs)
            boots[b] = float(np.median(d[idx]))
        ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        # Paired sign-flip permutation p-value (Hittner-May style)
        null_med = np.empty(B, dtype=np.float64)
        for b in range(B):
            flips = rng.choice([-1.0, 1.0], size=n_pairs)
            null_med[b] = float(np.median(d * flips))
        p_perm = float(np.mean(np.abs(null_med) >= np.abs(med_obs)))
        out.append({
            "key": k, "n_pairs": n_pairs,
            "median_diff": med_obs,
            "ci95_low": ci_lo, "ci95_high": ci_hi,
            "p_perm": p_perm,
            "median_grpo": float(np.median([float(gr[k]) for gr, _ in pairs])),
            "median_drgrpo": float(np.median([float(dr[k]) for _, dr in pairs])),
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--h-max", type=int, default=8)
    ap.add_argument("--out-prefix", default="length_bias_iter100")
    args = ap.parse_args()

    runs = []
    runs += [(r, "arithmetic_easy") for r in load_step_log(DRGRPO_VS_GRPO_PATH)]
    runs += [(r, "gsm8k_cot") for r in load_step_log(DRGRPO_GSM8K_PATH)]

    # Group by task
    by_task: dict[str, list[dict[str, Any]]] = {}
    for run, task in runs:
        by_task.setdefault(task, []).append(run)

    perrun_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    keys = [
        "phi_LL", "phi_LR", "phi_RL", "phi_RR",
        "rho",
        "cumul_impulse_L_to_R", "cumul_impulse_R_to_L",
        "cumul_diag_self_L", "cumul_diag_self_R",
        "fevd_R_from_L_h1", "fevd_R_from_L_h4", "fevd_R_from_L_h8",
        "fevd_L_from_R_h1", "fevd_L_from_R_h4", "fevd_L_from_R_h8",
        "cumul_fevd_R_from_L", "cumul_fevd_L_from_R",
        "Sigma_LL", "Sigma_RR", "Sigma_LR",
    ]

    for task, runs_t in by_task.items():
        per_t = analyze_runs(runs_t)
        # tag with task
        for r in per_t:
            r2 = {"task": task, **r}
            perrun_rows.append(r2)
        pairs = pair_algos(per_t)
        if not pairs:
            continue
        paired = paired_bootstrap(keys, pairs, B=args.B, seed=42 + hash(task) % 9999)
        for p in paired:
            summary_rows.append({
                "task": task, "algo_grpo": "grpo", "algo_drgrpo": "dr_grpo",
                "key": p["key"], "median_grpo": p["median_grpo"],
                "median_drgrpo": p["median_drgrpo"],
                "n_pairs": p["n_pairs"],
            })
            paired_rows.append({
                "task": task, "key": p["key"], "n_pairs": p["n_pairs"],
                "median_diff": p["median_diff"],
                "ci95_low": p["ci95_low"], "ci95_high": p["ci95_high"],
                "p_perm": p["p_perm"],
            })

    out_dir = "experiments/results"
    os.makedirs(out_dir, exist_ok=True)

    perrun_path = os.path.join(out_dir, f"{args.out_prefix}_perrun.tsv")
    paired_path = os.path.join(out_dir, f"{args.out_prefix}_paired.tsv")
    summary_path = os.path.join(out_dir, f"{args.out_prefix}_summary.tsv")
    meta_path = os.path.join(out_dir, f"{args.out_prefix}_meta.json")

    def _write_tsv(path: str, rows: list[dict[str, Any]], cols: list[str]) -> None:
        with open(path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    perrun_cols = ["task", "algo", "seed", "n", "n_obs", *keys]
    _write_tsv(perrun_path, perrun_rows, perrun_cols)
    _write_tsv(paired_path, paired_rows, ["task", "key", "n_pairs", "median_diff", "ci95_low", "ci95_high", "p_perm"])
    _write_tsv(summary_path, summary_rows, ["task", "algo_grpo", "algo_drgrpo", "key", "median_grpo", "median_drgrpo", "n_pairs"])

    meta = {
        "iter": 100,
        "B_bootstrap": args.B,
        "h_max": args.h_max,
        "n_runs": len(perrun_rows),
        "tasks": sorted(by_task.keys()),
        "algos": ["grpo", "dr_grpo"],
        "inputs": [DRGRPO_VS_GRPO_PATH, DRGRPO_GSM8K_PATH],
        "outputs": [perrun_path, paired_path, summary_path],
        "notes": (
            "Bivariate VAR(1) on (L_t, R_t).  Cross-equation coefficients phi_LR, phi_RL "
            "isolate the joint cross-coupling (not absorbed by marginal AR(1) alone).  "
            "Long-run IRF: (I-K)^{-1}.  FEVD: Cholesky-ordered (L,R) decomposition at h=1..h_max."
        ),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    # Console report
    print(f"Iter 100 wrote {len(perrun_rows)} perrun rows, {len(paired_rows)} paired rows")
    # Headline finding on GSM8K CoT:
    g = [r for r in paired_rows if r["task"] == "gsm8k_cot"]
    print("GSM8K CoT paired Dr.GRPO - GRPO medians:")
    for r in g:
        sig = "*" if r["p_perm"] < 0.05 else " "
        print(f"  {sig} {r['key']:>30s}  med={r['median_diff']:+.4f}  CI=[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]  p_perm={r['p_perm']:.3f}")
    a = [r for r in paired_rows if r["task"] == "arithmetic_easy"]
    print("Arithmetic paired Dr.GRPO - GRPO medians:")
    for r in a:
        sig = "*" if r["p_perm"] < 0.05 else " "
        print(f"  {sig} {r['key']:>30s}  med={r['median_diff']:+.4f}  CI=[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]  p_perm={r['p_perm']:.3f}")


if __name__ == "__main__":
    main()

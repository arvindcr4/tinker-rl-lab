"""length_bias_iter72.py — Iter 72 Pillar 4 (Length Bias / Dr.GRPO).

Length-shock persistence: AR(1) decomposition of per-step ΔL trajectories.

For each paired (algo, seed) run we fit an AR(1) model that residualises the
reward-direction effect, then read off the persistence half-life. Iter68 showed
Dr.GRPO has 14 pp MORE reversal on reward-up steps -- a binary flip metric. This
iter asks the CONTINUOUS analogue: holding the reward step fixed, does a length
shock persist longer (higher φ, longer half-life) under Dr.GRPO than GRPO?

If the clipped advantage truncates the magnitude of compression shocks but
preserves their sign, Dr.GRPO should show a SHORTER persistence half-life on
GSM8K CoT -- the same effect iter68 saw in reversal structure, now measured on
the continuous ΔL series. The arithmetic-easy task should give clean null (both
algos converge to a single mean length and ΔL ≈ 0).

Three paired diagnostics:
  1. AR(1) persistence φ with reward-driven regression:
       ΔL_t = c * ΔR_t + φ * ΔL_{t-1} + ε_t
     φ > 0 means shocks persist (momentum); φ < 0 means shocks mean-revert
     (oscillate). Half-life = -log(2) / log(φ) when |φ| < 1 and φ > 0.
  2. Detrended residual variance ratio: σ²(residual under Dr.GRPO) /
     σ²(residual under GRPO), paired bootstrap.
  3. Cross-lag correlation r(dL_t, dR_{t-k}) at k = 0, 1, 2: which lags show
     significant lead/lag between reward and length response.

Negative control: arithmetic_easy has ΔL ≈ 0 by step 5; both algos should give
near-zero φ on the trailing window.

Outputs:
  platform_hybrid/experiments/results/length_bias_iter72_persistence.tsv
  platform_hybrid/experiments/results/length_bias_iter72_residvar.tsv
  platform_hybrid/experiments/results/length_bias_iter72_lagcorr.tsv
  platform_hybrid/experiments/results/length_bias_iter72_summary.tsv
  platform_hybrid/experiments/results/length_bias_iter72_meta.json
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSM_PATH = os.path.join(ROOT, "experiments", "results", "drgrpo_gsm8k_cot_full.json")
ARITH_PATH = os.path.join(ROOT, "experiments", "results", "drgrpo_vs_grpo.json")
OUT_DIR = os.path.join(ROOT, "experiments", "results")

# bootstrap
B = 2000
RNG = random.Random(20260703)
Z975 = 1.959963984540054

LAGS = (0, 1, 2)
TAIL_START_FRAC = 0.20  # use steps from 20% onward (steady-state of length drift)


def load_runs(path: str) -> Dict[Tuple[str, int], List[dict]]:
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    for r in d.get("runs", []):
        algo = r["algo"]
        if algo == "dr_grpo":
            algo = "drgrpo"
        out[(algo, r["seed"])] = list(r["step_log"])
    return out


def deltas(steps: List[dict]) -> Tuple[List[float], List[float]]:
    """Return (dL, dR) lists of length T-1."""
    L = [s["mean_comp_len"] for s in steps]
    R = [s["mean_reward"] for s in steps]
    dL = [L[t] - L[t - 1] for t in range(1, len(L))]
    dR = [R[t] - R[t - 1] for t in range(1, len(R))]
    return dL, dR


def tail_window(xs: List[float], frac: float = TAIL_START_FRAC) -> List[float]:
    n = len(xs)
    if n == 0:
        return xs
    start = int(n * frac)
    return xs[start:]


def ols(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Fit y = a + b*x. Return (a, b)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    if sxx == 0:
        return my, 0.0
    b = sxy / sxx
    a = my - b * mx
    return a, b


def fit_ar1(dL: List[float], dR: List[float]) -> dict:
    """Fit ΔL_t = α + c·ΔR_t + φ·ΔL_{t-1} + ε. Returns φ, half_life, etc."""
    T = len(dL)
    if T < 4:
        return dict(phi=float("nan"), half_life=float("nan"), c=float("nan"),
                    resid_var=float("nan"), n_steps=T)
    # Build pairs (ΔL_t as target, ΔL_{t-1}, ΔR_t as predictors)
    y = dL[1:]
    x_lag = dL[:-1]
    x_dR = dR[1:]
    # Two-predictor OLS via grid: y = α + c·x_dR + φ·x_lag
    n = len(y)
    def fit(coeffs):
        a, c, phi = coeffs
        return sum((y[i] - (a + c * x_dR[i] + phi * x_lag[i])) ** 2 for i in range(n))
    # closed form: matrix X = [1, x_dR, x_lag], beta = (X'X)^-1 X'y
    # compute by explicit formulas
    S00 = n
    S01 = sum(x_dR)
    S02 = sum(x_lag)
    S11 = sum(x_dR[i] * x_dR[i] for i in range(n))
    S12 = sum(x_dR[i] * x_lag[i] for i in range(n))
    S22 = sum(x_lag[i] * x_lag[i] for i in range(n))
    # y side
    T0 = sum(y)
    T1 = sum(x_dR[i] * y[i] for i in range(n))
    T2 = sum(x_lag[i] * y[i] for i in range(n))
    # determinant of X'X (3x3 with row/col order 0,1,2 for intercept, dR, lag)
    M = [[S00, S01, S02], [S01, S11, S12], [S02, S12, S22]]
    def det3(m):
        a, b, c = m[0]
        d, e, f = m[1]
        g, h, i = m[2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    D = det3(M)
    if D == 0:
        return dict(phi=float("nan"), half_life=float("nan"), c=float("nan"),
                    resid_var=float("nan"), n_steps=T)
    # Solve via Cramer's rule
    def replace_col(m, col, v):
        out = [list(r) for r in m]
        for r in range(3):
            out[r][col] = v[r]
        return out
    rhs = [T0, T1, T2]
    def inv_beta(col):
        return det3(replace_col(M, col, rhs)) / D
    a_hat = inv_beta(0)
    c_hat = inv_beta(1)
    phi_hat = inv_beta(2)
    # residuals
    resid = [y[i] - (a_hat + c_hat * x_dR[i] + phi_hat * x_lag[i]) for i in range(n)]
    rmean = sum(resid) / n
    rv = sum((r - rmean) ** 2 for r in resid) / max(n - 1, 1)
    # half-life only meaningful when |phi| < 1 and phi > 0
    if 0 < phi_hat < 1:
        hl = -math.log(2) / math.log(phi_hat)
    elif phi_hat >= 1:
        hl = float("inf")  # explosive
    elif phi_hat <= 0:
        hl = 0.0  # anti-persistent
    else:
        hl = float("nan")
    return dict(phi=phi_hat, half_life=hl, c=c_hat, alpha=a_hat,
                resid_var=rv, n_steps=T, n_pairs=n)


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def lag_corr(dL: List[float], dR: List[float], lag: int) -> float:
    """Pearson r between dL_t and dR_{t-lag}. lag=0: contemporaneous; lag>0: dR leads dL."""
    n = min(len(dL), len(dR))
    if n - abs(lag) < 3:
        return 0.0
    if lag >= 0:
        xs = dL[lag:n]
        ys = dR[:n - lag]
    else:
        xs = dL[:n + lag]
        ys = dR[-lag:n]
    return pearson(xs, ys)


def paired_bootstrap_diff(gr_vals: List[float], dr_vals: List[float],
                          fn=lambda a, b: sum(b) - sum(a)) -> Tuple[float, float, float]:
    """Returns (diff_point, ci_lo, ci_hi). Both lists are paired over seeds (same length)."""
    n = len(gr_vals)
    assert n == len(dr_vals)
    diffs = [dr_vals[i] - gr_vals[i] for i in range(n)]
    point = sum(diffs) / n
    boots = []
    for _ in range(B):
        idx = [RNG.randrange(n) for _ in range(n)]
        s = sum(diffs[i] for i in idx) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[int(0.975 * B)]
    # one-sided p (sign): p_le0 = fraction of boots <= 0
    p_le0 = sum(1 for b in boots if b <= 0) / B
    return point, lo, hi, p_le0


def paired_bootstrap_log_ratio(gr_vals: List[float], dr_vals: List[float]) -> Tuple[float, float, float, float]:
    """Bootstrap CI on log ratio, return (log_ratio, lo, hi, p_le0 where p is for ratio<=1)."""
    n = len(gr_vals)
    assert n == len(dr_vals)
    log_diffs = [math.log(max(dr_vals[i], 1e-9)) - math.log(max(gr_vals[i], 1e-9)) for i in range(n)]
    point = sum(log_diffs) / n
    boots = []
    for _ in range(B):
        idx = [RNG.randrange(n) for _ in range(n)]
        s = sum(log_diffs[i] for i in idx) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[int(0.975 * B)]
    # one-sided p (sign): p_le0 = fraction of boots <= 0
    p_le0 = sum(1 for b in boots if b <= 0) / B
    return point, lo, hi, p_le0


def main():
    gsm = load_runs(GSM_PATH)
    arith = load_runs(ARITH_PATH)
    # group by seed and algo
    rows_persist = []  # raw persistence per run
    rows_resid = []  # resid_var per run
    rows_lag = []  # lag correlations per run
    summaries = []  # per-experiment summary

    # Use n_steps discriminant for label assignment later
    rows_persist = []  # raw persistence per run

    for exp_name, exp_runs in [("drgrpo_gsm8k_cot", gsm), ("drgrpo_vs_grpo", arith)]:
        per_seed = defaultdict(dict)
        for (algo, seed), steps in exp_runs.items():
            dL_full, dR_full = deltas(steps)
            # use TAIL window
            dL = tail_window(dL_full)
            dR = tail_window(dR_full)
            if len(dL) < 4:
                continue
            fit = fit_ar1(dL, dR)
            lc = {lag: lag_corr(dL, dR, lag) for lag in LAGS}
            per_seed[seed][algo] = dict(
                algo=algo, seed=seed, n_steps=fit["n_steps"], n_pairs=fit["n_pairs"],
                phi=fit["phi"], half_life=fit["half_life"], c_dR=fit["c"], alpha=fit["alpha"],
                resid_var=fit["resid_var"],
                lc0=lc[0], lc1=lc[1], lc2=lc[2],
                mean_dL=sum(dL) / len(dL), std_dL=math.sqrt(
                    sum((x - sum(dL) / len(dL)) ** 2 for x in dL) / max(len(dL) - 1, 1)),
                # exact-length for context
                mean_L=sum(s["mean_comp_len"] for s in steps) / len(steps),
            )

        # write per-run rows
        for seed, algos in per_seed.items():
            for algo, r in algos.items():
                rows_persist.append(r)
                rows_resid.append(dict(experiment=exp_name, seed=seed, algo=algo,
                                       resid_var=r["resid_var"], phi=r["phi"],
                                       half_life=r["half_life"]))
                for lag in LAGS:
                    rows_lag.append(dict(experiment=exp_name, seed=seed, algo=algo, lag=lag,
                                         r_corr=r[f"lc{lag}"]))

        # paired bootstrap on phi
        gr_phis = [algos["grpo"]["phi"] for seed, algos in per_seed.items()
                   if "grpo" in algos and "drgrpo" in algos]
        dr_phis = [algos["drgrpo"]["phi"] for seed, algos in per_seed.items()
                   if "grpo" in algos and "drgrpo" in algos]
        if not gr_phis:
            continue
        # paired diff: drgrpo - grpo
        diff_phi, lo_phi, hi_phi, p_phi = paired_bootstrap_diff(gr_phis, dr_phis)

        # half-life diff
        gr_hl = [algos["grpo"]["half_life"] for seed, algos in per_seed.items()
                 if "grpo" in algos and "drgrpo" in algos]
        dr_hl = [algos["drgrpo"]["half_life"] for seed, algos in per_seed.items()
                 if "grpo" in algos and "drgrpo" in algos]
        # truncate inf to large finite for bootstrap
        def finitize(xs):
            return [min(x, 100.0) if math.isfinite(x) else 100.0 for x in xs]
        diff_hl, lo_hl, hi_hl, p_hl = paired_bootstrap_diff(finitize(gr_hl), finitize(dr_hl))

        # residual variance log-ratio
        gr_rv = [max(algos["grpo"]["resid_var"], 1e-9) for seed, algos in per_seed.items()
                 if "grpo" in algos and "drgrpo" in algos]
        dr_rv = [max(algos["drgrpo"]["resid_var"], 1e-9) for seed, algos in per_seed.items()
                 if "grpo" in algos and "drgrpo" in algos]
        log_rv, lo_rv, hi_rv, p_rv = paired_bootstrap_log_ratio(gr_rv, dr_rv)
        rv_ratio = math.exp(log_rv)

        # lag correlations
        lag_results = {}
        for lag in LAGS:
            gr_l = [algos["grpo"][f"lc{lag}"] for seed, algos in per_seed.items()
                    if "grpo" in algos and "drgrpo" in algos]
            dr_l = [algos["drgrpo"][f"lc{lag}"] for seed, algos in per_seed.items()
                    if "grpo" in algos and "drgrpo" in algos]
            d_lc, lo_lc, hi_lc, p_lc = paired_bootstrap_diff(gr_l, dr_l)
            lag_results[lag] = dict(diff=d_lc, lo=lo_lc, hi=hi_lc, p_le0=p_lc,
                                    gr_mean=sum(gr_l) / len(gr_l),
                                    dr_mean=sum(dr_l) / len(dr_l))

        # mean phi each
        m_gr_phi = sum(gr_phis) / len(gr_phis)
        m_dr_phi = sum(dr_phis) / len(dr_phis)
        summaries.append(dict(
            experiment=exp_name,
            n_pairs=len(gr_phis),
            phi_grpo=m_gr_phi, phi_drgrpo=m_dr_phi,
            phi_diff=diff_phi, phi_ci_lo=lo_phi, phi_ci_hi=hi_phi, phi_p_le0=p_phi,
            half_life_grpo=sum(finitize(gr_hl)) / len(finitize(gr_hl)),
            half_life_drgrpo=sum(finitize(dr_hl)) / len(finitize(dr_hl)),
            half_life_diff=diff_hl, half_life_ci_lo=lo_hl, half_life_ci_hi=hi_hl, half_life_p_le0=p_hl,
            resid_var_ratio=rv_ratio, resid_var_log=log_rv,
            resid_var_lo=lo_rv, resid_var_hi=hi_rv, resid_var_p_le0=p_rv,
            lag0_diff=lag_results[0]["diff"], lag0_lo=lag_results[0]["lo"],
            lag0_hi=lag_results[0]["hi"], lag0_p=lag_results[0]["p_le0"],
            lag1_diff=lag_results[1]["diff"], lag1_lo=lag_results[1]["lo"],
            lag1_hi=lag_results[1]["hi"], lag1_p=lag_results[1]["p_le0"],
            lag2_diff=lag_results[2]["diff"], lag2_lo=lag_results[2]["lo"],
            lag2_hi=lag_results[2]["hi"], lag2_p=lag_results[2]["p_le0"],
        ))

    # write TSVs
    def write_tsv(path, rows, cols):
        with open(path, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                line = []
                for c in cols:
                    v = r.get(c, "")
                    if isinstance(v, float):
                        if not math.isfinite(v):
                            line.append("nan" if math.isnan(v) else "inf")
                        else:
                            line.append(f"{v:.6f}")
                    else:
                        line.append(str(v))
                f.write("\t".join(line) + "\n")

    persist_cols = ["experiment", "algo", "seed", "n_steps", "n_pairs", "phi", "half_life",
                    "c_dR", "alpha", "resid_var", "mean_dL", "std_dL", "mean_L",
                    "lc0", "lc1", "lc2"]
    rows_per_run = []
    for r in rows_persist:
        n_steps = r.get("n_steps", 0)
        # GSM has 30 steps total, ARITH has 40. Tail 20% -> 24 vs 32.
        exp = "drgrpo_gsm8k_cot" if n_steps <= 28 else "drgrpo_vs_grpo"
        rows_per_run.append({"experiment": exp, **{k: r[k] for k in persist_cols if k != "experiment"}})
    write_tsv(os.path.join(OUT_DIR, "length_bias_iter72_persistence.tsv"), rows_per_run, persist_cols)

    write_tsv(os.path.join(OUT_DIR, "length_bias_iter72_residvar.tsv"), rows_resid,
              ["experiment", "seed", "algo", "phi", "half_life", "resid_var"])

    write_tsv(os.path.join(OUT_DIR, "length_bias_iter72_lagcorr.tsv"), rows_lag,
              ["experiment", "seed", "algo", "lag", "r_corr"])

    # summary as flat row
    summary_cols = ["experiment", "n_pairs",
                    "phi_grpo", "phi_drgrpo", "phi_diff", "phi_ci_lo", "phi_ci_hi", "phi_p_le0",
                    "half_life_grpo", "half_life_drgrpo", "half_life_diff",
                    "half_life_ci_lo", "half_life_ci_hi", "half_life_p_le0",
                    "resid_var_ratio", "resid_var_log", "resid_var_lo", "resid_var_hi", "resid_var_p_le0",
                    "lag0_diff", "lag0_lo", "lag0_hi", "lag0_p",
                    "lag1_diff", "lag1_lo", "lag1_hi", "lag1_p",
                    "lag2_diff", "lag2_lo", "lag2_hi", "lag2_p"]
    write_tsv(os.path.join(OUT_DIR, "length_bias_iter72_summary.tsv"), summaries, summary_cols)

    # meta
    meta = dict(
        experiments=[
            dict(name="drgrpo_gsm8k_cot", n_seeds=len(gsm) // 2),
            dict(name="drgrpo_vs_grpo", n_seeds=len(arith) // 2),
        ],
        primary_metric="phi_diff (AR(1) coefficient, drgrpo - grpo)",
        interpretation=dict(
            phi="AR(1) coefficient of ΔL regressed on (intercept, ΔR_t, ΔL_{t-1}); positive=shock persists, negative=mean-revert",
            half_life="-log(2)/log(phi); steps for a length shock to halve",
            resid_var_ratio="var(residual after AR1+dR fit) ratio Dr.GRPO / GRPO; >1 = noisier residual",
            lag_correlation="pearson r between dL_t and dR_{t-lag}; lag=0 contemporaneous, lag>0 dR leads dL",
        ),
        outputs={
            "persistence": os.path.join(OUT_DIR, "length_bias_iter72_persistence.tsv"),
            "residvar": os.path.join(OUT_DIR, "length_bias_iter72_residvar.tsv"),
            "lagcorr": os.path.join(OUT_DIR, "length_bias_iter72_lagcorr.tsv"),
            "summary": os.path.join(OUT_DIR, "length_bias_iter72_summary.tsv"),
        },
        summary_rows=summaries,
    )
    with open(os.path.join(OUT_DIR, "length_bias_iter72_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=== Iter 72 Length Persistence Summary ===")
    for s in summaries:
        print(f"exp={s['experiment']} n_pairs={s['n_pairs']}")
        print(f"  phi: GRPO={s['phi_grpo']:+.3f} Dr.GRPO={s['phi_drgrpo']:+.3f} diff={s['phi_diff']:+.3f} "
              f"CI=[{s['phi_ci_lo']:+.3f},{s['phi_ci_hi']:+.3f}] p_le0={s['phi_p_le0']:.3f}")
        print(f"  half-life: GRPO={s['half_life_grpo']:.2f} Dr.GRPO={s['half_life_drgrpo']:.2f} "
              f"diff={s['half_life_diff']:+.2f} CI=[{s['half_life_ci_lo']:+.2f},{s['half_life_ci_hi']:+.2f}] p_le0={s['half_life_p_le0']:.3f}")
        print(f"  resid_var ratio (Dr/GR): {s['resid_var_ratio']:.3f} "
              f"(log={s['resid_var_log']:+.3f} CI=[{s['resid_var_lo']:+.3f},{s['resid_var_hi']:+.3f}] p_le0={s['resid_var_p_le0']:.3f})")
        print(f"  lag0 corr diff: {s['lag0_diff']:+.3f} CI=[{s['lag0_lo']:+.3f},{s['lag0_hi']:+.3f}] p_le0={s['lag0_p']:.3f}")
        print(f"  lag1 corr diff (dR leads dL): {s['lag1_diff']:+.3f} CI=[{s['lag1_lo']:+.3f},{s['lag1_hi']:+.3f}] p_le0={s['lag1_p']:.3f}")
        print(f"  lag2 corr diff: {s['lag2_diff']:+.3f} CI=[{s['lag2_lo']:+.3f},{s['lag2_hi']:+.3f}] p_le0={s['lag2_p']:.3f}")


if __name__ == "__main__":
    main()

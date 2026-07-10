#!/usr/bin/env python3
"""Iter 88 -- Pillar 4 (Length Bias / Dr.GRPO): Quantile-regression coupling decomposition.

Iter 76 settled level dynamics (half-life, damping, loop area).
Iter 80 settled boundedness (OU / unit-root).
Iter 84 settled linear frequency-domain (Hurst, coherence, Granger).
Iter 88 attacks the orthogonal question:

    *Where in the conditional R distribution* does length couple to reward?

For every run (algorithm x seed x task) we have per-step trajectories
(L_t, R_t, zvf_t) of length 30 (GSM8K CoT) or 40 (arithmetic_easy).

New measurements per run:

  (A) Quantile-regression slope at q in {.10, .25, .50, .75, .90}
      Beta_q = argmin_beta  sum_t  rho_q( R_t - beta * L_t )
      where rho_q(r) = max(q*r, (q-1)*r) is the pinball loss.
      The intercept is centred by an empirical q-quantile pivot so that
      the model is identifiable at q != .50 (median regression would absorb
      it into the intercept otherwise).

  (B) IQR_Q = beta_.75 - beta_.25  (heteroscedasticity proxy:
        how much does the regression slope depend on the conditional
        quantile of R?  Larger IQR -> more heteroscedastic coupling.)

  (C) Tail ratio T = beta_.90 / beta_.50  (asymmetry: does the high-R tail
        couple to length more strongly than median R?)

  (D) Partial Spearman rho(L_t, R_t | t)  (does L-R coupling survive
        controlling for training step?  Computed as the Spearman rho of
        the residuals from a lowess-of-step detrending.)

  (E) Spearman rank-correlation between {q} and {beta_q} -- monotone-Q
        index.  +1 means the slope grows monotonically with quantile,
        -1 means it decreases.

Sharpest falsifiable Dr.GRPO prediction (Liu et al. 2025, arXiv:2503.20783):

    Removing the length normalisation should sharpen the L-R coupling
    into a *tail-anchored* pattern (only the high-R tail couples to length).
    Therefore Dr.GRPO should have larger IQR_Q and a higher tail ratio
    than GRPO on the GSM8K CoT task (where the signal is strong), with
    the magnitude of the delta scaling with the q quantile at the
    upper end.

Inputs : drgrpo_vs_grpo.json          (arithmetic_easy, Qwen2.5-0.5B,
                                       5 seeds x 2 algos, n=40 step log)
         drgrpo_gsm8k_cot_full.json   (GSM8K CoT, Qwen2.5-1.5B,
                                       3 seeds x 2 algos, n=30 step log)
Outputs: platform_hybrid/experiments/results/length_bias_iter88_{perrun,quantile,
         paired,summary}.tsv + meta.json
Stdlib + numpy + scipy.stats + scipy.optimize.
"""
import json, os, math
from collections import defaultdict
import numpy as np
from scipy import stats as scs
from scipy import optimize as sco

W = "/home/claude/tinker-rl-lab-minimax"
RES = os.path.join(W, "experiments", "results")
FIG = os.path.join(W, "figures")
PAPERFIG = os.path.join(W, "paper", "figures")
B_BOOT = 2000
RNG_SEED = 88
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


# ------------------------------------------------------ quantile regression ----
def quantreg_slope(R: np.ndarray, L: np.ndarray, q: float) -> float:
    """Slope-only quantile regression with empirical-intercept pivot.

    Beta_q = argmin_beta  mean_t  rho_q( (R_t - c_q(beta)) - beta * L_t )

    Implemented as 1-D bounded Brent minimisation on the centred pinball
    loss.  We pin c_q(beta) to the empirical q-quantile of (R - beta*L)
    at each beta-evaluation so the loss is identifiable.

    Returns the optimal slope (NaN if R/L too short or degenerate).
    """
    R = np.asarray(R, float)
    L = np.asarray(L, float)
    if len(R) < 8 or L.std() < 1e-12:
        return float("nan")
    rng_lo = -10.0 * np.std(R) / max(np.std(L), 1e-6)
    rng_hi = -rng_lo

    def loss(b: float) -> float:
        residual = R - b * L
        c = np.quantile(residual, q)
        r = residual - c
        return float(np.mean(np.maximum(q * r, (q - 1.0) * r)))

    res = sco.minimize_scalar(loss, bounds=(rng_lo, rng_hi),
                              method="bounded",
                              options={"xatol": 1e-6})
    return float(res.x) if np.isfinite(res.x) else float("nan")


def quantreg_profile(R: np.ndarray, L: np.ndarray,
                     qs=QUANTILES) -> dict:
    """Return the full {q: beta_q} profile plus IQR / tail-ratio / monotonicity."""
    out = {q: quantreg_slope(R, L, q) for q in qs}
    arr = np.array([out[q] for q in qs], float)
    if not np.all(np.isfinite(arr)):
        return {**out, "iqr": float("nan"), "tail_ratio": float("nan"),
                "monotone": float("nan")}
    return {
        **out,
        "iqr": float(arr[3] - arr[1]),
        "tail_ratio": float(arr[4] / arr[2]) if abs(arr[2]) > 1e-6 else float("nan"),
        "monotone": float(scs.spearmanr(qs, arr).statistic)
                       if np.isfinite(scs.spearmanr(qs, arr).statistic) else 0.0,
    }


# ----------------------------------------------- partial Spearman rho ----
def partial_spearman(R: np.ndarray, L: np.ndarray, t: np.ndarray) -> float:
    """Partial Spearman rho(L, R | t) via rank-detrending.

    1) regress rank(R) on t with simple OLS, take residuals
    2) regress rank(L) on t, take residuals
    3) return Spearman rho of the two residual series
    """
    R = np.asarray(R, float)
    L = np.asarray(L, float)
    t = np.asarray(t, float)
    if len(R) < 8:
        return float("nan")

    def detrend(y):
        a, b = np.polyfit(t, y, 1)
        return y - (a * t + b)

    rR = scs.rankdata(R)
    rL = scs.rankdata(L)
    eR = detrend(rR)
    eL = detrend(rL)
    rho, _ = scs.spearmanr(eL, eR)
    return float(rho) if np.isfinite(rho) else float("nan")


# ----------------------------------------------------- data loading ----
def load_runs() -> list:
    runs = []

    # arithmetic_easy
    d1 = json.load(open(os.path.join(RES, "drgrpo_vs_grpo.json")))
    for r in d1["runs"]:
        sl = r.get("step_log", [])
        if len(sl) < 8:
            continue
        L = np.array([s["mean_comp_len"] for s in sl], float)
        R = np.array([s["mean_reward"]  for s in sl], float)
        Z = np.array([s.get("zvf", 0.0) for s in sl], float)
        runs.append({"task": "arithmetic_easy", "algo": r["algo"],
                     "seed": r["seed"], "L": L, "R": R, "Z": Z,
                     "last10_acc": r.get("last10_avg", float("nan"))})

    # gsm8k_cot
    d2 = json.load(open(os.path.join(RES, "drgrpo_gsm8k_cot_full.json")))
    for r in d2["runs"]:
        sl = r.get("step_log", [])
        if len(sl) < 8:
            continue
        L = np.array([s["mean_comp_len"] for s in sl], float)
        R = np.array([s["mean_reward"]  for s in sl], float)
        Z = np.array([s.get("zvf", 0.0) for s in sl], float)
        runs.append({"task": "gsm8k_cot", "algo": r["algo"],
                     "seed": r["seed"], "L": L, "R": R, "Z": Z,
                     "last10_acc": r.get("last10_avg", float("nan"))})
    return runs


# ----------------------------------------------------- main analysis ----
def analyse() -> dict:
    runs = load_runs()
    perrun = []
    for r in runs:
        L, R, Z, t = r["L"], r["R"], r["Z"], np.arange(len(r["L"]))
        prof = quantreg_profile(R, L)
        psp = partial_spearman(R, L, t)
        row = {
            "task": r["task"],
            "algo": r["algo"],
            "seed": r["seed"],
            "n": int(len(L)),
            "last10_acc": r["last10_acc"],
            "L_mean": float(L.mean()),
            "L_std":  float(L.std()),
            "R_mean": float(R.mean()),
            "R_std":  float(R.std()),
            "partial_spear_rho": psp,
            "q10": prof[0.10], "q25": prof[0.25],
            "q50": prof[0.50], "q75": prof[0.75], "q90": prof[0.90],
            "iqr_q": prof["iqr"], "tail_ratio": prof["tail_ratio"],
            "monotone_q": prof["monotone"],
        }
        perrun.append(row)
    return {"perrun": perrun, "runs": runs}


# ----------------------------------------------------- paired bootstrap ----
def paired_diff(perrun: list, task: str, key: str) -> dict:
    """Seed-paired diff = Dr.GRPO - GRPO on `key` for `task`."""
    by_key = defaultdict(dict)
    for row in perrun:
        if row["task"] != task:
            continue
        by_key[row["seed"]][row["algo"]] = row[key]
    pairs = []
    for s, d in by_key.items():
        if "grpo" in d and "dr_grpo" in d:
            a = d["dr_grpo"]
            b = d["grpo"]
            if not (math.isnan(a) or math.isnan(b)):
                pairs.append((a, b))
    if len(pairs) < 2:
        return {"task": task, "key": key, "n_pairs": len(pairs),
                "mean_diff": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p": float("nan")}
    diffs = np.array([p[0] - p[1] for p in pairs], float)
    obs = float(diffs.mean())
    rng = np.random.default_rng(RNG_SEED)
    n = len(diffs)
    # 2000-iter paired bootstrap on the *mean of diff*
    means = np.empty(B_BOOT, float)
    for b in range(B_BOOT):
        idx = rng.integers(0, n, size=n)
        means[b] = diffs[idx].mean()
    means.sort()
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    # two-sided p against H0: mean(diff) = 0, via t on n pairs
    if diffs.std(ddof=1) > 0:
        t = obs / (diffs.std(ddof=1) / np.sqrt(n))
        p = float(2.0 * (1.0 - scs.t.cdf(abs(t), df=n - 1)))
    else:
        p = 1.0
    return {"task": task, "key": key, "n_pairs": n,
            "mean_diff": obs, "ci_lo": lo, "ci_hi": hi, "p": p}


# ----------------------------------------------------- writers ----
def write_tsv(path: str, rows: list, header: list):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            line = []
            for h in header:
                v = r.get(h, "")
                if isinstance(v, float):
                    line.append(f"{v:.6f}" if math.isfinite(v) else "nan")
                else:
                    line.append(str(v))
            f.write("\t".join(line) + "\n")


def main():
    out = analyse()
    runs = out["runs"]
    perrun = out["perrun"]
    base = os.path.join(RES, "length_bias_iter88_")

    write_tsv(base + "perrun.tsv", perrun,
              ["task", "algo", "seed", "n", "last10_acc", "L_mean", "L_std",
               "R_mean", "R_std", "partial_spear_rho",
               "q10", "q25", "q50", "q75", "q90",
               "iqr_q", "tail_ratio", "monotone_q"])

    # per-task per-key paired diff
    paired_rows = []
    summary_rows = []
    keys_for_paired = ["partial_spear_rho", "q10", "q25", "q50", "q75", "q90",
                       "iqr_q", "tail_ratio", "monotone_q"]
    for task in {"arithmetic_easy", "gsm8k_cot"}:
        # collect means per algo for the summary tab
        for algo in ("grpo", "dr_grpo"):
            vals_by_key = defaultdict(list)
            for r in perrun:
                if r["task"] == task and r["algo"] == algo:
                    for k in keys_for_paired:
                        if math.isfinite(r[k]):
                            vals_by_key[k].append(r[k])
            for k, vs in vals_by_key.items():
                summary_rows.append({
                    "task": task, "algo": algo, "key": k,
                    "n_seeds": len(vs), "mean": float(np.mean(vs)),
                    "std":  float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
                })
        for k in keys_for_paired:
            d = paired_diff(perrun, task, k)
            paired_rows.append(d)

    write_tsv(base + "paired.tsv", paired_rows,
              ["task", "key", "n_pairs", "mean_diff", "ci_lo", "ci_hi", "p"])
    write_tsv(base + "summary.tsv", summary_rows,
              ["task", "algo", "key", "n_seeds", "mean", "std"])

    meta = {
        "iter": 88,
        "pillar": "P4-LengthBias",
        "task": "Pillar 4 (Length Bias / Dr.GRPO): Quantile-regression coupling decomposition",
        "stats": ["quantile_slopes", "iqr_q", "tail_ratio",
                  "partial_spear_rho", "monotone_q"],
        "quantiles": list(QUANTILES),
        "inputs": ["drgrpo_vs_grpo.json", "drgrpo_gsm8k_cot_full.json"],
        "n_runs": len(runs),
        "n_paired_tests": len(paired_rows),
    }
    with open(base + "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"iter88 wrote {len(perrun)} per-run rows; "
          f"{len(paired_rows)} paired-bootstrap tests")
    # print the most diagnostic paired rows
    for d in paired_rows:
        if d["key"] in ("iqr_q", "tail_ratio", "partial_spear_rho", "q90"):
            print(f"  {d['task']:>15s}  {d['key']:>18s}  "
                  f"diff={d['mean_diff']:+.4f}  "
                  f"CI=[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]  "
                  f"p={d['p']:.4f}  n={d['n_pairs']}")


if __name__ == "__main__":
    main()

"""Iter 44: Conditional Quantile Regression on R|L for length-bias / Dr.GRPO.

Novel angle vs iter28/32/36/40: those iters all used OLS on the conditional mean
E[R|L]. Iter44 fits CONDITIONAL QUANTILES q_tau(R|L) for tau in {0.1, 0.25, 0.5,
0.75, 0.9}, plus the conditional standard deviation Var(R|L)^0.5, to detect
asymmetric length-response that OLS would average out.

Verbosity-trap prediction: if longer responses tend to come from confident but
short-on-substance rollouts, then high-L should DECREASE the upper quantiles of
R (less upside) while leaving the lower quantiles unchanged -> negative skew in
q_tau(R|L) - E[R|L] as a function of tau. We test this on both tasks, both
algorithms, and stratify by ZVF bin (cross-pillar with iter34's ZVF proxy).

Outputs (5 TSVs + 1 fig driver):
  platform_hybrid/experiments/results/length_bias_iter44_quantile_slopes.tsv  per (task, algo,
      seed, tau) with linear-fit slope and intercept of q_tau(R) on L
  platform_hybrid/experiments/results/length_bias_iter44_condvar.tsv          per (task, algo,
      seed, L_bin) with std(R|L) and residuals
  platform_hybrid/experiments/results/length_bias_iter44_asymmetry.tsv        per (task, algo,
      seed) with delta_slope = slope_q90 - slope_q10 and CI
  platform_hybrid/experiments/results/length_bias_iter44_grpo_vs_drgrpo.tsv   paired bootstrap
      on the asymmetry score
  platform_hybrid/experiments/results/length_bias_iter44_zvf_binned.tsv       per (task, algo,
      zvf_bin) with quantiles and asymmetry
  platform_hybrid/experiments/results/length_bias_iter44_summary.tsv          one-rollup table

Implements an in-house pinball-loss quantile regression (no sklearn dep) on
per-step (R, L) pairs pooled across seeds within (task, algo).
"""
from __future__ import annotations
import csv
import json
import math
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
DRGRPO_JSON = RES / "drgrpo_vs_grpo.json"
GSM_JSON = RES / "drgrpo_gsm8k_cot_full.json"

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
SEEDS_ARITH = (42, 123, 456, 789, 1024)
SEEDS_GSM = (42, 123, 456)
RNG_SEED = 44
N_BOOT = 4000
ZVF_BINS = ("low", "mid", "high")  # per-step zvf: low<0.2, mid in [0.2,0.5), high>=0.5


def load_runs(path: Path, task: str) -> list[dict]:
    """Yield rows {task, algo, seed, step, R, L, zvf}."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d["runs"]:
        algo = r["algo"]
        seed = r["seed"]
        for s in r["step_log"]:
            out.append({
                "task": task,
                "algo": algo,
                "seed": seed,
                "step": s["step"],
                "R": float(s["mean_reward"]),
                "L": float(s["mean_comp_len"]),
                "zvf": float(s.get("zvf", float("nan"))),
            })
    return out


def pinball_fit(X: list[float], y: list[float], tau: float, lr: float = 1e-3,
                n_iter: int = 4000, init: tuple[float, float] = (0.0, 0.0)
                ) -> tuple[float, float]:
    """Linear quantile regression by SGD on pinball loss."""
    a, b = init
    n = len(X)
    if n < 3:
        return a, b
    rx = max(X) - min(X) or 1.0
    ry = max(y) - min(y) or 1.0
    xs = [(x - min(X)) / rx for x in X]
    ys = [(yi - min(y)) / ry for yi in y]
    for _ in range(n_iter):
        idx = random.randrange(n)
        xi, yi = xs[idx], ys[idx]
        yhat = a * xi + b
        resid = yi - yhat
        if resid > 0:
            grad_a = -tau * xi
            grad_b = -tau
        else:
            grad_a = (1 - tau) * xi
            grad_b = (1 - tau)
        a -= lr * grad_a
        b -= lr * grad_b
    slope = a * ry / rx
    intercept = b * ry + (min(y) - slope * min(X))
    return slope, intercept


def ols(X: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return slope, intercept, pearson_r."""
    n = len(X)
    if n < 3:
        return 0.0, 0.0, 0.0
    mx = sum(X) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in X)
    syy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((X[i] - mx) * (y[i] - my) for i in range(n))
    if sxx == 0 or syy == 0:
        return 0.0, my, 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    r = sxy / math.sqrt(sxx * syy)
    return slope, intercept, r


def std_within_bins(X: list[float], y: list[float], n_bins: int = 5
                    ) -> list[tuple[float, float, float]]:
    """Bin X into n_bins quantile-bins and compute std(y) per bin."""
    n = len(X)
    if n < 3:
        return []
    order = sorted(range(n), key=lambda i: X[i])
    bins = []
    chunk = max(1, n // n_bins)
    for b in range(n_bins):
        idx = order[b * chunk: (b + 1) * chunk] if b < n_bins - 1 else order[b * chunk:]
        if not idx:
            continue
        ys = [y[i] for i in idx]
        xs = [X[i] for i in idx]
        m = sum(ys) / len(ys)
        var = sum((yi - m) ** 2 for yi in ys) / max(1, len(ys) - 1)
        bins.append((sum(xs) / len(xs), var ** 0.5, len(ys)))
    return bins


def zvf_bin(z: float) -> str:
    if z < 0.2:
        return "low"
    if z < 0.5:
        return "mid"
    return "high"


def bootstrap_paired_diff(values_a: list[float], values_b: list[float],
                          n_boot: int = N_BOOT) -> tuple[float, float, float, float]:
    """Return mean_diff, ci_lo, ci_hi, p(Delta<=0)."""
    n = len(values_a)
    assert n == len(values_b)
    diffs = [values_a[i] - values_b[i] for i in range(n)]
    mean_d = sum(diffs) / n
    rng = random.Random(RNG_SEED)
    boot = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bd = sum(diffs[i] for i in idx) / n
        boot.append(bd)
    boot.sort()
    ci_lo = boot[int(0.025 * n_boot)]
    ci_hi = boot[int(0.975 * n_boot)]
    p_le0 = sum(1 for b in boot if b <= 0) / n_boot
    return mean_d, ci_lo, ci_hi, p_le0


def main() -> None:
    random.seed(RNG_SEED)
    runs = load_runs(DRGRPO_JSON, "arithmetic_easy") + load_runs(GSM_JSON, "gsm8k_cot")

    # ---- 1. Per (task, algo, seed, tau) quantile slopes ----
    quantile_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for algo in ("grpo", "dr_grpo"):
            for seed in (SEEDS_ARITH if task == "arithmetic_easy" else SEEDS_GSM):
                sub = [r for r in runs if r["task"] == task
                       and r["algo"] == algo and r["seed"] == seed]
                if not sub:
                    continue
                sub.sort(key=lambda r: r["step"])
                L = [r["L"] for r in sub]
                R = [r["R"] for r in sub]
                mean_slope, mean_intercept, pearson_r = ols(L, R)
                for tau in QUANTILES:
                    slope, intercept = pinball_fit(L, R, tau)
                    quantile_rows.append({
                        "task": task, "algo": algo, "seed": seed,
                        "tau": tau, "slope": round(slope, 6),
                        "intercept": round(intercept, 6),
                        "ols_slope": round(mean_slope, 6),
                        "ols_pearson_r": round(pearson_r, 4),
                        "n_steps": len(sub),
                    })

    # ---- 2. Conditional std(R|L) by L-bin ----
    condvar_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for algo in ("grpo", "dr_grpo"):
            for seed in (SEEDS_ARITH if task == "arithmetic_easy" else SEEDS_GSM):
                sub = [r for r in runs if r["task"] == task
                       and r["algo"] == algo and r["seed"] == seed]
                if not sub:
                    continue
                L = [r["L"] for r in sub]
                R = [r["R"] for r in sub]
                for lcent, std, npts in std_within_bins(L, R, n_bins=5):
                    condvar_rows.append({
                        "task": task, "algo": algo, "seed": seed,
                        "L_bin_center": round(lcent, 3),
                        "std_R_given_L": round(std, 6),
                        "n_pts": npts,
                    })

    # ---- 3. Asymmetry score: slope_q90 - slope_q10 per (task, algo, seed) ----
    asym_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for algo in ("grpo", "dr_grpo"):
            for seed in (SEEDS_ARITH if task == "arithmetic_easy" else SEEDS_GSM):
                sub_q = [q for q in quantile_rows if q["task"] == task
                         and q["algo"] == algo and q["seed"] == seed]
                if not sub_q:
                    continue
                s10 = next(q["slope"] for q in sub_q if q["tau"] == 0.1)
                s90 = next(q["slope"] for q in sub_q if q["tau"] == 0.9)
                asym_rows.append({
                    "task": task, "algo": algo, "seed": seed,
                    "slope_q10": round(s10, 6),
                    "slope_q50": round(next(q["slope"] for q in sub_q
                                            if q["tau"] == 0.5), 6),
                    "slope_q90": round(s90, 6),
                    "asymmetry_delta": round(s90 - s10, 6),
                })

    # ---- 4. Paired bootstrap: Dr.GRPO vs GRPO on asymmetry_delta ----
    paired_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        g = [a for a in asym_rows if a["task"] == task and a["algo"] == "grpo"]
        d = [a for a in asym_rows if a["task"] == task and a["algo"] == "dr_grpo"]
        # match by seed
        for ga, da in zip(g, d):
            assert ga["seed"] == da["seed"]
        diffs_a = [da["asymmetry_delta"] - ga["asymmetry_delta"]
                   for ga, da in zip(g, d)]
        mean_d, ci_lo, ci_hi, p_le0 = bootstrap_paired_diff(
            [da["asymmetry_delta"] for da in d],
            [ga["asymmetry_delta"] for ga in g])
        paired_rows.append({
            "task": task,
            "metric": "asymmetry_delta = slope_q90 - slope_q10",
            "n_pairs": len(g),
            "mean_grpo": round(sum(a["asymmetry_delta"] for a in g) / len(g), 6),
            "mean_drgrpo": round(sum(a["asymmetry_delta"] for a in d) / len(d), 6),
            "mean_diff": round(mean_d, 6),
            "ci_lo": round(ci_lo, 6),
            "ci_hi": round(ci_hi, 6),
            "p_le0": round(p_le0, 4),
        })

    # ---- 5. ZVF-binned asymmetry ----
    zvf_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for algo in ("grpo", "dr_grpo"):
            for zb in ZVF_BINS:
                sub = [r for r in runs if r["task"] == task and r["algo"] == algo
                       and zvf_bin(r["zvf"]) == zb]
                if len(sub) < 5:
                    zvf_rows.append({
                        "task": task, "algo": algo, "zvf_bin": zb,
                        "n_steps": len(sub),
                        "ols_slope": float("nan"),
                        "asymmetry_delta": float("nan"),
                        "median_L": float("nan"),
                        "median_R": float("nan"),
                    })
                    continue
                L = [r["L"] for r in sub]
                R = [r["R"] for r in sub]
                slope, _, _ = ols(L, R)
                s10, _ = pinball_fit(L, R, 0.1)
                s90, _ = pinball_fit(L, R, 0.9)
                medL = sorted(L)[len(L) // 2]
                medR = sorted(R)[len(R) // 2]
                zvf_rows.append({
                    "task": task, "algo": algo, "zvf_bin": zb,
                    "n_steps": len(sub),
                    "ols_slope": round(slope, 6),
                    "asymmetry_delta": round(s90 - s10, 6),
                    "median_L": round(medL, 4),
                    "median_R": round(medR, 6),
                })

    # ---- write outputs ----
    def w(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
            wri.writeheader()
            for row in rows:
                wri.writerow(row)
        print(f"  wrote {len(rows)} rows -> {path.relative_to(ROOT)}")

    print("Iter 44 length-bias / quantile regression artifacts:")
    w(RES / "length_bias_iter44_quantile_slopes.tsv", quantile_rows)
    w(RES / "length_bias_iter44_condvar.tsv", condvar_rows)
    w(RES / "length_bias_iter44_asymmetry.tsv", asym_rows)
    w(RES / "length_bias_iter44_grpo_vs_drgrpo.tsv", paired_rows)
    w(RES / "length_bias_iter44_zvf_binned.tsv", zvf_rows)

    # summary
    summary_rows = []
    for pr in paired_rows:
        summary_rows.append(pr)
    for vr in zvf_rows:
        if vr["n_steps"] >= 5:
            summary_rows.append({
                "task": vr["task"], "metric": f"zvf_bin={vr['zvf_bin']}",
                "n_pairs": vr["n_steps"],
                "mean_grpo": vr["ols_slope"] if vr["algo"] == "grpo" else float("nan"),
                "mean_drgrpo": vr["ols_slope"] if vr["algo"] == "dr_grpo" else float("nan"),
                "mean_diff": vr["asymmetry_delta"],
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "p_le0": float("nan"),
            })
    w(RES / "length_bias_iter44_summary.tsv", summary_rows)
    print("DONE")


if __name__ == "__main__":
    main()
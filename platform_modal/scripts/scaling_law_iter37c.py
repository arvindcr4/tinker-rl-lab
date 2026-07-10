"""Pillar 1 iter37c -- Raw per-step model selection cross-check.

Iter37/iter37b fit candidate reward-curve models to SYNTHETIC traces
reconstructed from the 12-anchor summary statistics. The result is that
linear/null dominates (7/10 dynamic anchors). The natural concern is that
the synthetic trace lacks the curvature signal of the real per-step data.

This cross-check fits the same 5 candidate models to the REAL per-step
reward trace from the 10 same-stack runs in samestack_ppo_grpo.json
(5 GRPO seeds + 5 PPO seeds on Qwen/Qwen2.5-0.5B, 40 steps each).

The cross-check is direct: if the saturation model is correct, it should
win on these 40-step per-step traces too. If linear wins here as well,
the synthetic-vs-raw gap is closed and the falsification stands.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "experiments" / "results" / "samestack_ppo_grpo.json"
RESULTS = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
PAPER_FIG = REPO / "paper" / "figures"
for d in (FIG_DIR, PAPER_FIG):
    d.mkdir(parents=True, exist_ok=True)


def model_saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def model_michaelis_menten(t, r_max, t_half):
    return r_max * t / (t_half + t)


def model_hill(t, r_max, k):
    return r_max * (t * t) / (k * k + t * t)


def model_power(t, r_max, alpha):
    return r_max * (1.0 - np.power(1.0 + t, -alpha))


def model_linear(t, a, b):
    return a + b * t


CANDIDATES = [
    {"name": "A_saturation_exp", "fn": model_saturation,
     "p0": [0.8, 0.3], "bounds": ([0.0, 1e-3], [2.0, 5.0]), "n_params": 2},
    {"name": "B_michaelis_menten", "fn": model_michaelis_menten,
     "p0": [0.9, 5.0], "bounds": ([0.0, 1e-3], [2.0, 1e4]), "n_params": 2},
    {"name": "C_hill_n2", "fn": model_hill,
     "p0": [0.9, 5.0], "bounds": ([0.0, 1e-3], [2.0, 1e4]), "n_params": 2},
    {"name": "D_power_law", "fn": model_power,
     "p0": [0.9, 0.4], "bounds": ([0.0, 1e-3], [2.0, 5.0]), "n_params": 2},
    {"name": "E_linear", "fn": model_linear,
     "p0": [0.3, 0.01], "bounds": ([-1.0, -1.0], [2.0, 2.0]), "n_params": 2},
]


def aic_bic(n, k, ss_res):
    if ss_res <= 0 or not math.isfinite(ss_res):
        return float("inf"), float("inf")
    log_lik = -0.5 * n * (1.0 + math.log(2.0 * math.pi * ss_res / n))
    return -2.0 * log_lik + 2 * k, -2.0 * log_lik + k * math.log(n)


def fit_candidate(t, y, cand):
    try:
        popt, _ = curve_fit(cand["fn"], t, y, p0=cand["p0"],
                            bounds=cand["bounds"], maxfev=8000)
    except Exception:
        return None, float("inf"), float("inf"), float("inf"), -math.inf, True
    y_hat = cand["fn"](t, *popt)
    resid = y - y_hat
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    aic, bic = aic_bic(len(y), cand["n_params"], ss_res)
    return popt, ss_res, aic, bic, r2, False


def main() -> None:
    with open(SRC) as f:
        d = json.load(f)
    runs = d["runs"]

    fits_rows = []
    summary_rows = []
    for r in runs:
        algo = r["algo"]
        seed = r["seed"]
        sl = r.get("step_log", [])
        if not sl:
            continue
        steps = np.array([s["step"] for s in sl], dtype=float) + 1.0
        rewards = np.array([s["mean_reward"] for s in sl], dtype=float)
        per = {"algo": algo, "seed": seed, "model": r["model"],
               "n_steps": len(sl),
               "r_first": float(rewards[0]),
               "r_last": float(rewards[-1]),
               "r_mean": float(np.mean(rewards))}
        for cand in CANDIDATES:
            popt, ss_res, aic, bic, r2, hit = fit_candidate(steps, rewards, cand)
            params = {f"param_{k_}": float(p) for k_, p in zip(["p1", "p2"], popt)} \
                if popt is not None else {"param_p1": float("nan"), "param_p2": float("nan")}
            row = dict(per)
            row["model_name"] = cand["name"]
            row.update(params)
            row["ss_res"] = round(ss_res, 6)
            row["r2"] = round(r2, 4) if math.isfinite(r2) else float("nan")
            row["aic"] = round(aic, 4) if math.isfinite(aic) else float("inf")
            row["bic"] = round(bic, 4) if math.isfinite(bic) else float("inf")
            row["hit_bound"] = hit
            fits_rows.append(row)
        # per-run best model
        run_fits = [f for f in fits_rows if f["algo"] == algo and f["seed"] == seed]
        aics = np.array([f["aic"] for f in run_fits], dtype=float)
        finite = np.isfinite(aics)
        if not finite.any():
            best = "NONE"
        else:
            best = run_fits[int(np.argmin(np.where(finite, aics, np.inf)))]["model_name"]
        # Akaike weights
        delta = aics - np.where(finite, aics, np.inf).min()
        delta_clip = np.clip(delta, 0, 700)
        exp_neg_half = np.exp(-0.5 * delta_clip)
        exp_neg_half[~finite] = 0.0
        w = exp_neg_half / max(exp_neg_half.sum(), 1e-12)
        summary_rows.append({
            "algo": algo,
            "seed": seed,
            "n_steps": len(sl),
            "r_first": float(rewards[0]),
            "r_last": float(rewards[-1]),
            "best_aic_model": best,
            "w_saturation": round(float(w[0]), 4),
            "w_michaelis_menten": round(float(w[1]), 4),
            "w_hill_n2": round(float(w[2]), 4),
            "w_power_law": round(float(w[3]), 4),
            "w_linear": round(float(w[4]), 4),
        })

    # Aggregate by algo
    summary_by_algo = {}
    for algo in ("grpo", "ppo"):
        sub = [r for r in summary_rows if r["algo"] == algo]
        if not sub:
            continue
        winner_count = {}
        for s in sub:
            winner_count[s["best_aic_model"]] = winner_count.get(s["best_aic_model"], 0) + 1
        summary_by_algo[algo] = {
            "n_runs": len(sub),
            "winner_count": winner_count,
            "mean_w_sat": round(float(np.mean([s["w_saturation"] for s in sub])), 4),
            "mean_w_mm": round(float(np.mean([s["w_michaelis_menten"] for s in sub])), 4),
            "mean_w_hill": round(float(np.mean([s["w_hill_n2"] for s in sub])), 4),
            "mean_w_power": round(float(np.mean([s["w_power_law"] for s in sub])), 4),
            "mean_w_linear": round(float(np.mean([s["w_linear"] for s in sub])), 4),
        }

    # write
    with open(RESULTS / "scaling_law_iter37c_fits.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(fits_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(fits_rows)
    print(f"wrote {RESULTS / 'scaling_law_iter37c_fits.tsv'}  ({len(fits_rows)} rows)")
    with open(RESULTS / "scaling_law_iter37c_summary.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t")
        for k, v in summary_by_algo.items():
            for kk, vv in v.items():
                w.writerow([f"{k}_{kk}", vv])
    print(f"wrote {RESULTS / 'scaling_law_iter37c_summary.tsv'}")

    # print to console
    print("\n=== Iter 37c summary ===")
    for algo, s in summary_by_algo.items():
        print(f"\n{algo} (n={s['n_runs']}):")
        print(f"  winner_count: {s['winner_count']}")
        print(f"  mean_w_sat={s['mean_w_sat']:.3f} mm={s['mean_w_mm']:.3f} "
              f"hill={s['mean_w_hill']:.3f} power={s['mean_w_power']:.3f} "
              f"linear={s['mean_w_linear']:.3f}")

    # figure: per-run mean w stacked by algo
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = ["#2b8cbe", "#fdae61", "#7fcdbb", "#edf8b1", "#636363"]
    short_names = [c["name"].split("_", 1)[1] for c in CANDIDATES]
    for ax_i, algo in enumerate(("grpo", "ppo")):
        sub = [s for s in summary_rows if s["algo"] == algo]
        x = np.arange(len(sub))
        bot = np.zeros(len(sub))
        w_mat = np.array([[s["w_saturation"], s["w_michaelis_menten"],
                            s["w_hill_n2"], s["w_power_law"], s["w_linear"]] for s in sub])
        for i, c in enumerate(colors):
            axes[ax_i].bar(x, w_mat[:, i], bottom=bot, color=c, width=0.7,
                            label=short_names[i] if ax_i == 0 else None,
                            edgecolor="white", lw=0.3)
            bot += w_mat[:, i]
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels([f"seed={s['seed']}" for s in sub],
                                   rotation=0, fontsize=8)
        axes[ax_i].set_ylim(0, 1.0)
        axes[ax_i].set_ylabel("Akaike weight")
        axes[ax_i].set_title(f"Raw per-step reward curve, {algo.upper()} (Qwen2.5-0.5B, 40 steps)")
        if ax_i == 0:
            axes[ax_i].legend(loc="upper right", ncol=2, fontsize=7.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"scaling_law_iter37c.{ext}", bbox_inches="tight")
        fig.savefig(PAPER_FIG / f"scaling_law_iter37c.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/scaling_law_iter37c.{{pdf,png}}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P8 JOB A (iter 88): LLM-sensor cost-vs-recall frontier under extraction-noise
with paired bootstrap CIs.

Fresh vein (not in any of the 103 prior P8 rows). Iter-84 (#84) measured
CALIBRATION under 9 noise levels on sigma_multiplier 0.05..2.00. Iter-72
measured the K*-sweep at CLEAN conditions. Iter-52 measured decision-regret.
The combined reviewer question:

  "If my LLM-as-scribe extracts V_mean/V_std/V_max/V_min with sigma noise of
   X%, at what alert-volume K does the LLM-bearing model (XGB-24full)
   still Pareto-dominate XGB-20raw on (cost, recall)?"

is UNANSWERED. We sweep 5 noise levels x 4 trees x 11 K-alert-budgets,
compute paired bootstrap CIs on (cost_ratio, recall_delta, prec_delta),
and answer the headline.

The noise model: V_agg_i observed = V_agg_i_true + N(0, sigma^2), where
sigma = sigma_mult x sigma of the V20 distribution (per-feature scaled).
This iter only injects noise into the TEST set (training is clean), so the
gap isolates the **inference robustness** of each model.

Cost model: matches iter-72 cost-adj curve.
  cost(model, K) = c_sense(model) + [C_inv * (TP+FP) + L * FN] / N_test
  C_inv = $0.50/alert ; L = $100/fraud-missed (rho=200, the iter-28 default
    high-cost regime)

Pareto metric: cost_at_recall(R) = min over K { cost(model, K) : recall(K) >= R }

Outputs
-------
experiments/results/p5p8/p8_iter88_noise_cost.tsv          (4 models x 5 noise x 11 K)
experiments/results/p5p8/p8_iter88_noise_cost_boot.tsv     (paired bootstrap on (24full-cost, 20raw-cost) at matched-K)
experiments/results/p5p8/p8_iter88_noise_pareto.tsv        (cost_at_recall for each (model, noise) at recall in {0.5,0.7,0.8,0.9,0.95})
experiments/results/p5p8/p8_iter88_noise_summary.json
experiments/results/p5p8/figures/p8_iter88_noise_pareto.{png,pdf}

Stdlib + numpy + pandas + xgboost + sklearn + matplotlib. <=300 lines.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
FIG = RES / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"

V20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]

# Trees:
#   XGB-20raw       : V20 only                    c_sense = 0
#   XGB-24full      : V20 + clean AGG4            c_sense = 0.0035/decision (LLM extracting 4 aggs)
#   XGB-24noisy     : XGB-24 full but TEST injected with sigma_mult x sd of AGG4  c_sense = 0.0035
#   XGB-4sensor_only: only 4 AGG4                 c_sense = 0.0035
FEATS = {
    "XGB-20raw": V20,
    "XGB-24full": V20 + AGG4,
    "XGB-4sensor": AGG4,
}
SENSE = {
    "XGB-20raw": 0.0,
    "XGB-24full": 0.0035,
    "XGB-4sensor": 0.0035,
}

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.25, 0.50]   # sigma_mult applied to AGG4 at TEST time
K_PCT = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]   # alert budgets (% of test)

C_INV = 0.50      # $/alert
L_MISS = 100.0    # $/missed-fraud (rho = 200, high-cost regime from iter-28)

SEED = 20260705
N_BOOT = 800


def load(path: Path, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    with path.open() as f:
        header = f.readline().strip().split(",")
    cols = [header.index(c) for c in feats]
    yi = header.index("Class")
    X = df[:, cols]
    y = df[:, yi].astype(int)
    return X, y


def fit_predict(X_tr, y_tr, X_te) -> np.ndarray:
    import xgboost as xgb

    m = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=7.0,
        eval_metric="logloss",
        random_state=SEED,
        tree_method="hist",
        n_jobs=4,
    )
    m.fit(X_tr, y_tr, verbose=False)
    return m.predict_proba(X_te)[:, 1]


def add_noise(X_te: np.ndarray, agg_idx: list[int], sigma_mult: float, sd_per: np.ndarray, rng: random.Random) -> np.ndarray:
    out = X_te.copy()
    if sigma_mult == 0.0:
        return out
    for j, col in enumerate(agg_idx):
        out[:, col] = X_te[:, col] + rng.gauss(0.0, sigma_mult * sd_per[j])
    return out


def cost_at_k(y: np.ndarray, p: np.ndarray, k_pct: float, c_sense: float) -> dict:
    n = len(y)
    k = max(1, int(round(k_pct / 100.0 * n)))
    topk_idx = np.argpartition(-p, k - 1)[:k]
    pred = np.zeros(n, dtype=bool)
    pred[topk_idx] = True
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    cost = c_sense + (C_INV * (tp + fp) + L_MISS * fn) / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(int(np.sum(y == 1)), 1)
    return dict(
        k=k,
        tp=tp,
        fp=fp,
        fn=fn,
        cost=cost,
        precision=precision,
        recall=recall,
    )


def pareto(y: np.ndarray, p: np.ndarray, c_sense: float, k_grid: list[float], recall_targets: list[float]) -> dict:
    """For each recall target R, return the minimum cost such that recall >= R."""
    rows = [cost_at_k(y, p, k, c_sense) for k in k_grid]
    out = {}
    for R in recall_targets:
        feasible = [r for r in rows if r["recall"] >= R]
        if not feasible:
            out[R] = None
            continue
        out[R] = min(r["cost"] for r in feasible)
    return out


def paired_bootstrap_cost(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, k_pct: float, c_a: float, c_b: float,
    n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    rng = random.Random(seed)
    n = len(y)
    deltas = {"cost": [], "tp": [], "fp": [], "fn": []}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yv = y[idx]
        pa = p_a[idx]
        pb = p_b[idx]
        # K picked from full n to avoid variance from k-fluctuation across bootstraps
        k = max(1, int(round(k_pct / 100.0 * n)))
        order_a = np.argsort(-pa)[:k]
        order_b = np.argsort(-pb)[:k]
        pred_a = np.zeros(n, dtype=bool); pred_a[order_a] = True
        pred_b = np.zeros(n, dtype=bool); pred_b[order_b] = True
        tpa = int(np.sum(pred_a & (yv == 1))); fpa = int(np.sum(pred_a & (yv == 0))); fna = int(np.sum(~pred_a & (yv == 1)))
        tpb = int(np.sum(pred_b & (yv == 1))); fpb = int(np.sum(pred_b & (yv == 0))); fnb = int(np.sum(~pred_b & (yv == 1)))
        cost_a = c_a + (C_INV * (tpa + fpa) + L_MISS * fna) / n
        cost_b = c_b + (C_INV * (tpb + fpb) + L_MISS * fnb) / n
        deltas["cost"].append(cost_a - cost_b)
        deltas["tp"].append(tpa - tpb)
        deltas["fp"].append(fpa -fpb)
        deltas["fn"].append(fna - fnb)
    out = {}
    for k_, xs in deltas.items():
        xs.sort()
        lo = xs[int(0.025 * len(xs))]
        hi = xs[int(0.975 * len(xs))]
        out[k_] = dict(median=xs[len(xs) // 2], lo=lo, hi=hi, excludes_zero=(lo > 0) or (hi < 0))
    return out


def main() -> None:
    print("[iter88] loading ...")
    Xtr_24, y_tr = load(TRAIN, V20 + AGG4)
    Xtr_20, _ = load(TRAIN, V20)
    Xte_24, y_te = load(TEST, V20 + AGG4)
    Xte_20, _ = load(TEST, V20)

    # Both load helpers use the same V20+AGG4 column order:
    #   columns 0..19 = V1..V20, columns 20..23 = V_mean, V_std, V_max, V_min
    agg_idx = [20, 21, 22, 23]
    sd_per = np.std(Xte_24[:, agg_idx], axis=0)
    print(f"  AGG4 sd_per = {sd_per}")

    print("[iter88] fitting trees on CLEAN training ...")
    p_20 = fit_predict(Xtr_20, y_tr, Xte_20)
    p_24_clean = fit_predict(Xtr_24, y_tr, Xte_24)
    Xtr_agg = Xtr_24[:, agg_idx]
    Xte_agg = Xte_24[:, agg_idx]
    p_4 = fit_predict(Xtr_agg, y_tr, Xte_agg)

    print("[iter88] sweep noise x K ...")
    rows = []
    rng = random.Random(SEED)
    p_24_noisy_cache = {}
    p_4_noisy_cache = {}
    for sigma in NOISE_LEVELS:
        if sigma not in p_24_noisy_cache:
            Xte_24_n = add_noise(Xte_24, [20, 21, 22, 23], sigma, sd_per, rng)
            p_24_noisy = fit_predict(Xtr_24, y_tr, Xte_24_n)
            p_24_noisy_cache[sigma] = (Xte_24_n, p_24_noisy)
            # also 4-sensor at this noise
            p_4_noisy = fit_predict(Xtr_agg, y_tr, Xte_24_n[:, 20:24])
            p_4_noisy_cache[sigma] = p_4_noisy
        Xte_24_n, p_24_noisy = p_24_noisy_cache[sigma]
        p_4_noisy = p_4_noisy_cache[sigma]
        for model, p_pred, c_sense in (
            ("XGB-20raw", p_20, SENSE["XGB-20raw"]),
            ("XGB-24full", p_24_noisy if sigma > 0 else p_24_clean, SENSE["XGB-24full"]),
            ("XGB-4sensor", p_4_noisy if sigma > 0 else p_4, SENSE["XGB-4sensor"]),
        ):
            for k in K_PCT:
                m = cost_at_k(y_te, p_pred, k, c_sense)
                rows.append(dict(
                    noise_sigma=sigma,
                    model=model,
                    k_pct=k,
                    k_n=m["k"],
                    tp=m["tp"], fp=m["fp"], fn=m["fn"],
                    cost=m["cost"],
                    precision=m["precision"],
                    recall=m["recall"],
                ))

    with (RES / "p8_iter88_noise_cost.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- Paired bootstrap on (XGB-24full, XGB-20raw) at each (noise, K) ----
    print("[iter88] paired bootstrap ...")
    boot_rows = []
    for sigma in NOISE_LEVELS:
        _, p_24_n = p_24_noisy_cache[sigma] if sigma > 0 else (None, p_24_clean)
        for k in K_PCT:
            ci = paired_bootstrap_cost(y_te, p_24_n, p_20, k, SENSE["XGB-24full"], SENSE["XGB-20raw"])
            boot_rows.append(dict(
                noise_sigma=sigma, k_pct=k,
                cost_delta_median=ci["cost"]["median"],
                cost_delta_lo=ci["cost"]["lo"], cost_delta_hi=ci["cost"]["hi"],
                cost_ci_excludes_zero=ci["cost"]["excludes_zero"],
                tp_delta_median=ci["tp"]["median"],
                fn_delta_median=ci["fn"]["median"],
                fp_delta_median=ci["fp"]["median"],
            ))
    with (RES / "p8_iter88_noise_cost_boot.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in boot_rows:
            w.writerow(r)

    # ---- Pareto frontier (cost at fixed recall) ----
    print("[iter88] pareto ...")
    recall_targets = [0.50, 0.70, 0.80, 0.90, 0.95]
    pareto_rows = []
    for sigma in NOISE_LEVELS:
        _, p_24_n = p_24_noisy_cache[sigma] if sigma > 0 else (None, p_24_clean)
        for model, p_pred, c_sense in (
            ("XGB-20raw", p_20, SENSE["XGB-20raw"]),
            ("XGB-24full", p_24_n, SENSE["XGB-24full"]),
            ("XGB-4sensor", p_4_noisy_cache[sigma] if sigma > 0 else p_4, SENSE["XGB-4sensor"]),
        ):
            front = pareto(y_te, p_pred, c_sense, K_PCT, recall_targets)
            for R in recall_targets:
                pareto_rows.append(dict(
                    noise_sigma=sigma, model=model, recall_target=R,
                    cost_at_recall=front[R] if front[R] is not None else -1.0,
                    infeasible=front[R] is None,
                ))
    with (RES / "p8_iter88_noise_pareto.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(pareto_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in pareto_rows:
            w.writerow(r)

    # ---- Figure: per-noise, the (recall, cost) curve for each model ----
    fig, axes = plt.subplots(1, len(NOISE_LEVELS), figsize=(4.0 * len(NOISE_LEVELS), 3.6), sharey=True)
    if len(NOISE_LEVELS) == 1:
        axes = [axes]
    colors = {"XGB-20raw": "tab:blue", "XGB-24full": "tab:red", "XGB-4sensor": "tab:green"}
    markers = {"XGB-20raw": "s", "XGB-24full": "o", "XGB-4sensor": "^"}
    for ax, sigma in zip(axes, NOISE_LEVELS):
        for model in ("XGB-20raw", "XGB-24full", "XGB-4sensor"):
            sub = [r for r in rows if r["noise_sigma"] == sigma and r["model"] == model]
            sub.sort(key=lambda r: r["k_pct"])
            ax.plot(
                [r["recall"] for r in sub],
                [r["cost"] for r in sub],
                marker=markers[model], color=colors[model], label=model, linewidth=1.2, markersize=4,
            )
        ax.set_title(f"sigma = {sigma}")
        ax.set_xlabel("recall")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("expected cost ($/decision)")
    axes[0].legend(fontsize=8)
    fig.suptitle("P8 iter-88 — noise × cost-vs-recall frontier (rho=200, $0.50/alert, $100/miss)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "p8_iter88_noise_pareto.png", dpi=140)
    fig.savefig(FIG / "p8_iter88_noise_pareto.pdf")
    plt.close(fig)

    # ---- headline summary ----
    # Headline 1: at matched K=1% (mid-range operational budget), cost-delta sign across noise levels.
    headline_rows = []
    for sigma in NOISE_LEVELS:
        row = next(r for r in boot_rows if r["noise_sigma"] == sigma and r["k_pct"] == 1.0)
        headline_rows.append(dict(
            sigma=sigma,
            cost_delta_median=row["cost_delta_median"],
            ci_lo=row["cost_delta_lo"], ci_hi=row["cost_delta_hi"],
            sig=(row["cost_ci_excludes_zero"]),
        ))
    # Headline 2: at noise=0.10 (typical LLM extraction noise), the K* at which XGB-24
    # Pareto-dominates XGB-20 on cost_at_recall(0.80)
    sig10 = [r for r in pareto_rows if r["noise_sigma"] == 0.10 and r["recall_target"] == 0.80]
    sig10_lookup = {(r["model"]): r["cost_at_recall"] for r in sig10}
    # Headline 3: at what noise level does XGB-24 cost_at_recall(0.80) stop beating XGB-20?
    noise_cross = []
    for sigma in NOISE_LEVELS:
        sub = [r for r in pareto_rows if r["noise_sigma"] == sigma and r["recall_target"] == 0.80]
        c20 = next((r["cost_at_recall"] for r in sub if r["model"] == "XGB-20raw"), None)
        c24 = next((r["cost_at_recall"] for r in sub if r["model"] == "XGB-24full"), None)
        c04 = next((r["cost_at_recall"] for r in sub if r["model"] == "XGB-4sensor"), None)
        if c24 is not None and c20 is not None:
            noise_cross.append((sigma, c24, c20, c04, c24 - c20))

    summary = {
        "n_test": int(len(y_te)),
        "n_pos_test": int(np.sum(y_te == 1)),
        "noise_levels": NOISE_LEVELS,
        "k_grid_pct": K_PCT,
        "C_inv": C_INV,
        "L_miss": L_MISS,
        "rho": L_MISS / C_INV,
        "headline_k1pct_noise": headline_rows,
        "headline_noise_cross_at_recall_0p80": [
            dict(sigma=s, cost_24full=c24, cost_20raw=c20, cost_4sensor=c04, delta_24_vs_20=c24 - c20)
            for s, c24, c20, c04, _ in noise_cross
        ],
        "n_boot": N_BOOT,
    }
    with (RES / "p8_iter88_noise_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P8 JOB A (iter 96): per-cohort noise × cost-vs-recall frontier with calibration CIs.

Fresh vein (not in 111 prior rows). Closes iter-88's open question: when iter-88
(#104) finds sigma=0.10 is the operational noise threshold for the GLOBAL cost
flip from XGB-24full Pareto-dominant to XGB-20raw dominant, it does NOT break
that threshold down by COHORT. Iter-99 (#99) measured per-cohort ECE WITHOUT a
noise sweep. Together these leave open: does any SINGLE cohort flip the cost
ordering before sigma=0.10 (a per-cohort noise fragility)?

Method:
- Cohort axes already standard in P8: V_mean quintile (0-4), Amount quintile (0-4),
  Time-of-day tercile (T0/T1/T2).
- For each (cohort-axis, stratum, noise-sigma, K, model) cell: compute the cost
  on the model output restricted to that stratum (test-set observations only).
- Paired bootstrap B=800, seed 20260705, on cost-delta (24-20) per stratum.
- The headline: which (cohort, noise) cells have cost-delta CI EXCLUDING zero
  in the unfavorable (24 - 20 > 0) direction? Those are the per-cohort
  fragility hot-spots that the global H1 of iter-88 misses.
- Cross-couples with iter-99 cohort-calibration, iter-88 noise-frontier.

Outputs:
- experiments/results/p5p8/p8_iter96_per_cohort_noise.tsv
- experiments/results/p5p8/p8_iter96_per_cohort_noise_boot.tsv
- experiments/results/p5p8/p8_iter96_per_cohort_summary.json
- paper/sections/p8_iter96_per_cohort_noise.tex
- docs/p5p8_improvements/112_p8_per_cohort_noise.md
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
FEATS_20 = V20
FEATS_24 = V20 + AGG4

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.25, 0.50]
K_PCT = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]

C_INV = 0.50
L_MISS = 100.0

SEED = 20260705
N_BOOT = 800

# cohorts: V_mean real, Amount = synth from V_std, Time = synth from V_max
COHORTS = [
    ("V_mean_q", "real"),
    ("Amount_q", "synth_V_std"),
    ("Time_t", "synth_V_max"),
]


def load(path: Path) -> tuple[np.ndarray, list[str]]:
    df = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    with path.open() as f:
        header = f.readline().strip().split(",")
    return df, header


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


def cohort_strata(values: np.ndarray, axis_name: str) -> np.ndarray:
    """Return integer stratum labels: 5 quintiles for V_mean/Amount, 3 terciles for Time."""
    if axis_name in ("V_mean_q", "Amount_q"):
        q = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
        out = np.zeros(len(values), dtype=int)
        for i, t in enumerate(q, start=1):
            out[values > t] = i
        return out
    elif axis_name == "Time_t":
        q = np.quantile(values, [1 / 3, 2 / 3])
        out = np.zeros(len(values), dtype=int)
        for i, t in enumerate(q, start=1):
            out[values > t] = i
        return out
    raise ValueError(axis_name)


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
    if k >= n:
        k = n - 1
    topk_idx = np.argpartition(-p, k - 1)[:k]
    pred = np.zeros(n, dtype=bool)
    pred[topk_idx] = True
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    cost = c_sense + (C_INV * (tp + fp) + L_MISS * fn) / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(int(np.sum(y == 1)), 1)
    return dict(k=k, tp=tp, fp=fp, fn=fn, cost=cost, precision=precision, recall=recall)


def paired_bootstrap_cost_cohort(
    y_strat: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, k_pct: float, c_a: float, c_b: float,
    n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    """Paired bootstrap on per-row predictions, stratified by cohort label.

    For each replicate, take n_stratum rows from EACH stratum (preserving
    stratum membership), concatenate, compute top-K on the full concatenated
    score vectors. Returns cost-delta statistic on the SUBSET (stratum-only).
    """
    rng = random.Random(seed)
    n = len(y_strat)
    strata = np.unique(y_strat)
    # We resample WITHIN each stratum, preserving the per-stratum count.
    deltas = []
    strata_indices = {s: np.where(y_strat == s)[0] for s in strata}
    sizes = {s: len(idxs) for s, idxs in strata_indices.items()}
    for _ in range(n_boot):
        pa_b = np.empty(n)
        pb_b = np.empty(n)
        yv_b = np.empty(n, dtype=int)
        cursor = 0
        for s in strata:
            idxs = strata_indices[s]
            sampled = [idxs[rng.randrange(sizes[s])] for _ in range(sizes[s])]
            for j, ix in enumerate(sampled):
                pa_b[cursor] = p_a[ix]
                pb_b[cursor] = p_b[ix]
                yv_b[cursor] = y_strat[ix]  # holding stratum index as label
                cursor += 1
        # cost at top-K on the bootstrapped set
        k = max(1, int(round(k_pct / 100.0 * n)))
        order_a = np.argsort(-pa_b)[:k]
        order_b = np.argsort(-pb_b)[:k]
        pred_a = np.zeros(n, dtype=bool); pred_a[order_a] = True
        pred_b = np.zeros(n, dtype=bool); pred_b[order_b] = True
        # We need binary CLASS labels (1/0), not stratum int. Re-read from caller.
        # In this version we collect CLASS labels via a closure; see main()
        deltas.append((None, None, None))  # placeholder; main() computes fully
    return deltas


def stratified_pair_bootstrap(
    y: np.ndarray, y_strat: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
    k_pct: float, c_a: float, c_b: float, n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    """Stratum-preserving paired bootstrap on cost-delta."""
    rng = random.Random(seed)
    n = len(y)
    strata = np.unique(y_strat)
    strata_idx = {s: np.where(y_strat == s)[0] for s in strata}
    sizes = {s: len(idxs) for s, idxs in strata_idx.items()}
    deltas_cost = []
    for _ in range(n_boot):
        pa_b = np.empty(n); pb_b = np.empty(n); yv_b = np.empty(n, dtype=int)
        cursor = 0
        for s in strata:
            idxs = strata_idx[s]
            samp = [idxs[rng.randrange(sizes[s])] for _ in range(sizes[s])]
            for ix in samp:
                pa_b[cursor] = p_a[ix]; pb_b[cursor] = p_b[ix]; yv_b[cursor] = y[ix]; cursor += 1
        k = max(1, int(round(k_pct / 100.0 * n)))
        if k >= n: k = n - 1
        order_a = np.argsort(-pa_b)[:k]; order_b = np.argsort(-pb_b)[:k]
        pred_a = np.zeros(n, dtype=bool); pred_a[order_a] = True
        pred_b = np.zeros(n, dtype=bool); pred_b[order_b] = True
        tpa = int(np.sum(pred_a & (yv_b == 1))); fpa = int(np.sum(pred_a & (yv_b == 0))); fna = int(np.sum(~pred_a & (yv_b == 1)))
        tpb = int(np.sum(pred_b & (yv_b == 1))); fpb = int(np.sum(pred_b & (yv_b == 0))); fnb = int(np.sum(~pred_b & (yv_b == 1)))
        cost_a = c_a + (C_INV * (tpa + fpa) + L_MISS * fna) / n
        cost_b = c_b + (C_INV * (tpb + fpb) + L_MISS * fnb) / n
        deltas_cost.append(cost_a - cost_b)
    deltas_cost.sort()
    lo = deltas_cost[int(0.025 * len(deltas_cost))]
    hi = deltas_cost[int(0.975 * len(deltas_cost))]
    med = deltas_cost[len(deltas_cost) // 2]
    return dict(median=med, lo=lo, hi=hi, excludes_zero=(lo > 0) or (hi < 0))


def main() -> None:
    print("[iter96] loading ...")
    train, hdr_tr = load(TRAIN)
    test, hdr_te = load(TEST)

    def col(name): return hdr_tr.index(name)

    # V column indices in test (and train, with class-jitter handled separately)
    v20_idx = [col(f"V{i}") for i in range(1, 21)]
    yi_train = col("Class")
    # AGG4 indices RELATIVE to the 25-col header (or 25-col test header)
    agg_header_idx = [col(c) for c in AGG4]
    # In the 24-col feature matrix X_te_24 (V20+AGG4), AGG4 lives at cols 20,21,22,23.
    agg_idx = [20, 21, 22, 23]

    X_tr_20 = train[:, v20_idx]; y_tr = train[:, yi_train].astype(int)
    X_tr_24 = train[:, v20_idx + agg_header_idx]
    X_te_20 = test[:, v20_idx]; y_te = test[:, col("Class")].astype(int)
    X_te_24 = test[:, v20_idx + agg_header_idx]
    # Test split has only V1..V20 + V_mean..V_min + Class. Synthesize
    # Amount/Time from V_std and V_max ranks (per iter-99).
    amount_syn = test[:, agg_header_idx[1]]
    time_syn = test[:, agg_header_idx[2]]

    # cohort strata (using clean-test aggregates)
    cohorts = {}
    cohorts["V_mean_q"] = cohort_strata(test[:, agg_header_idx[0]], "V_mean_q")
    cohorts["Amount_q"] = cohort_strata(amount_syn, "Amount_q")
    cohorts["Time_t"] = cohort_strata(time_syn, "Time_t")

    print("[iter96] fitting trees on CLEAN training ...")
    p_20 = fit_predict(X_tr_20, y_tr, X_te_20)
    p_24_clean = fit_predict(X_tr_24, y_tr, X_te_24)
    sd_per = np.std(X_te_24[:, agg_idx], axis=0)
    print(f"[iter96] sd_per (AGG4) = {sd_per}")

    p_24_noisy_cache = {}
    rng = random.Random(SEED)
    for sigma in NOISE_LEVELS:
        if sigma == 0.0:
            p_24_noisy_cache[sigma] = p_24_clean
            continue
        Xte_n = add_noise(X_te_24, agg_idx, sigma, sd_per, rng)
        p_n = fit_predict(X_tr_24, y_tr, Xte_n)
        p_24_noisy_cache[sigma] = p_n

    print("[iter96] sweep per-cohort ...")
    rows = []
    for axis_name, _ in COHORTS:
        strata = cohorts[axis_name]
        n_strata = int(strata.max()) + 1
        for s in range(n_strata):
            mask = strata == s
            n_stratum = int(mask.sum())
            pos_stratum = int(np.sum(y_te[mask] == 1))
            if n_stratum < 50 or pos_stratum < 2:
                continue
            for sigma in NOISE_LEVELS:
                p24 = p_24_noisy_cache[sigma]
                for k_pct in K_PCT:
                    # Compute cost per model on this stratum only.
                    ca = cost_at_k(y_te[mask], p_20[mask], k_pct, c_sense=0.0)
                    cb = cost_at_k(y_te[mask], p24[mask], k_pct, c_sense=0.0035)
                    rows.append(dict(
                        axis=axis_name, stratum=int(s),
                        n_stratum=n_stratum, pos_stratum=pos_stratum,
                        noise_sigma=sigma, k_pct=k_pct,
                        cost_20raw=ca["cost"], cost_24full=cb["cost"],
                        delta_cost_24_minus_20=cb["cost"] - ca["cost"],
                        recall_20raw=ca["recall"], recall_24full=cb["recall"],
                        tp_20raw=ca["tp"], fp_20raw=ca["fp"], fn_20raw=ca["fn"],
                        tp_24full=cb["tp"], fp_24full=cb["fp"], fn_24full=cb["fn"],
                    ))
    with (RES / "p8_iter96_per_cohort_noise.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); [w.writerow(r) for r in rows]

    # ---- Stratified paired bootstrap at canonical (k=1.5%, axis=V_mean_q) ----
    print("[iter96] stratified bootstrap ...")
    boot_rows = []
    for axis_name, _ in COHORTS:
        strata = cohorts[axis_name]
        n_strata = int(strata.max()) + 1
        for s in range(n_strata):
            mask = strata == s
            if mask.sum() < 50 or int(np.sum(y_te[mask] == 1)) < 2:
                continue
            y_st = y_te[mask]; st = strata[mask]
            for sigma in NOISE_LEVELS:
                p24 = p_24_noisy_cache[sigma]
                for k_pct in (1.5, 2.0, 3.0):
                    ci = stratified_pair_bootstrap(
                        y_st, st, p_20[mask], p24[mask], k_pct, 0.0, 0.0035,
                    )
                    boot_rows.append(dict(
                        axis=axis_name, stratum=int(s),
                        n_stratum=int(mask.sum()), noise_sigma=sigma, k_pct=k_pct,
                        delta_median=ci["median"], delta_lo=ci["lo"], delta_hi=ci["hi"],
                        sig_unfavorable=ci["excludes_zero"] and ci["median"] > 0,
                    ))
    with (RES / "p8_iter96_per_cohort_noise_boot.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()), delimiter="\t")
        w.writeheader(); [w.writerow(r) for r in boot_rows]

    # ---- Headline ----
    # Count, across all (axis, stratum, sigma, k_pct) boot rows, how many have
    # unfavorable CI (24 - 20 > 0 excluding zero).
    n_unfav = sum(1 for r in boot_rows if r["sig_unfavorable"])
    n_total = len(boot_rows)
    # Identify the FIRST noise level per (axis, stratum) where unfavorable appears.
    first_unfav = {}
    for r in boot_rows:
        if r["sig_unfavorable"]:
            key = (r["axis"], int(r["stratum"]))
            if key not in first_unfav or r["noise_sigma"] < first_unfav[key]:
                first_unfav[key] = r["noise_sigma"]

    # Cross-coupling: amount-Q0 had iter-99 worst ECE on XGB-20raw — does it
    # also flip the cost frontier earliest?
    amount_q0_unfav_sigma = [s for (a, s), sig in first_unfav.items() if a == "Amount_q" and s == 0]
    vmean_q2_unfav_sigma = [s for (a, s), sig in first_unfav.items() if a == "V_mean_q" and s == 2]

    summary = {
        "n_test": int(len(y_te)),
        "n_pos_test": int(np.sum(y_te == 1)),
        "noise_levels": NOISE_LEVELS,
        "k_grid_pct": K_PCT,
        "C_inv": C_INV, "L_miss": L_MISS, "rho": L_MISS / C_INV,
        "cohorts": [c[0] for c in COHORTS],
        "n_cohort_cells": len(rows),
        "n_boot": N_BOOT,
        "n_boot_cells_evaluated": n_total,
        "n_unfavorable_cells": n_unfav,
        "first_unfavorable_sigma_per_cohort": {
            f"{a}|stratum={s}": float(sig) for (a, s), sig in sorted(first_unfav.items())
        },
        "iter99_amount_q0_first_flip_sigma": (amount_q0_unfav_sigma[0] if amount_q0_unfav_sigma else None),
        "iter99_vmean_q2_first_flip_sigma": (vmean_q2_unfav_sigma[0] if vmean_q2_unfav_sigma else None),
    }
    (RES / "p8_iter96_per_cohort_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- Figure: per-axis, per-stratum first-unfavorable-sigma ----
    fig, axes = plt.subplots(1, len(COHORTS), figsize=(4.0 * len(COHORTS), 3.6), sharey=True)
    for ax, (axis_name, _) in zip(axes, COHORTS):
        sigmas_per_stratum = []
        labels = []
        for s in range(int(cohorts[axis_name].max()) + 1):
            mask = cohorts[axis_name] == s
            if mask.sum() < 50:
                continue
            key = (axis_name, s)
            sigmas_per_stratum.append(first_unfav.get(key, np.nan))
            labels.append(f"{s} (n={int(mask.sum())})")
        ax.bar(range(len(sigmas_per_stratum)), [0.5 if np.isnan(s) else s for s in sigmas_per_stratum],
               color=["tab:gray" if np.isnan(s) else "tab:red" for s in sigmas_per_stratum])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_title(axis_name)
        ax.set_ylabel("min σ with unfavorable CI")
        ax.axhline(0.10, color="tab:green", linestyle="--", linewidth=1.0, label="σ=0.10 (iter-88 H1)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle("P8 iter-96 — first-unfavorable σ per cohort stratum (rho=200)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "p8_iter96_per_cohort_flip.png", dpi=140)
    fig.savefig(FIG / "p8_iter96_per_cohort_flip.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

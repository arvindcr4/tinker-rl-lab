#!/usr/bin/env python3
"""P8 V_mean threshold sweep -- precision-recall Pareto frontier (iter 168 JOB A).

Fresh vein, not in 172 prior P8 rows. Closes the iter-156 H1 REFUTED
(escalation precision 1% at tau=0.0, far below the 0.05 bar) and the
iter-156 operational recommendation "TUNE V_mean threshold to balance
precision against recall lift -- iter-156 measures the unconstrained
upper bound".

For each (seed, rate, fset) cell, fit XGB once (same as iter-156) and
then evaluate the 5-way disagreement counts at 7 V_mean thresholds:
  TAU_VMEAN ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}.

The LLM fire condition changes:
    iter-156:  llm_fire = (V_mean > 0.0)        <- everything fires
    iter-168:  llm_fire = (V_mean > tau)        <- stricter gates

Hypotheses
----------
H1 (DECISIVE) -- stricter thresholds achieve esc_prec >= 0.10 on >=
     50% of (seed x rate x fset x tau >= 1.0) cells. The Pareto
     frontier exists.
H2 -- Pareto frontier cell exists: there is some tau such that
     esc_prec >= 0.10 AND value_rate >= 0.30 simultaneously on at
     least one (seed, rate, fset) cell.
H3 -- value_rate monotone non-increasing in tau on >= 80% of cells
     (fewer fires -> fewer lifts, basic monotonicity).
H4 -- breakeven rate monotone non-decreasing in tau on >= 50% of
     cells (precision lifts reduce cost_per_lift, restoring the
     breakeven property).

Stdlib + numpy + xgboost.  <= 300 lines.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
SEEDS = [20260706, 20260708, 20260710, 20260712, 20260714]
K_PCT = 2.0
COST_XGB = 0.0001
VALUE_PER_CATCH = 50.0

LLM_PRICE_TIERS = [
    ("cheap_heuristic", 0.0001),
    ("small_open",      0.0006),
    ("iter120_default", 0.0010),
    ("mid_tier",        0.0050),
    ("frontier_gpt4",   0.0300),
]
RATES_PCT = [1.44, 1.00, 0.50, 0.10, 0.05]

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
FEATURE_SETS = {
    "24full":       ALL24,
    "20raw":        RAW20,
    "20raw+minmax": RAW20 + ["V_min", "V_max"],
    "20raw+stat":   RAW20 + ["V_mean", "V_std"],
}

TAUS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load(path):
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        X, y = [], []
        for line in rdr:
            X.append([float(line[idx[c]]) for c in ALL24])
            y.append(int(float(line[idx["Class"]])))
    return np.array(X), np.array(y)


def fit_predict(Xtr, ytr, Xte, feats, seed):
    cols = [ALL24.index(c) for c in feats]
    Xtr_s = Xtr[:, cols]
    Xte_s = Xte[:, cols]
    n_pos = max(1, int(ytr.sum()))
    n_neg = max(1, len(ytr) - n_pos)
    spw = n_neg / n_pos
    m = xgb.XGBClassifier(
        n_estimators=180, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=seed, n_jobs=4,
    )
    m.fit(Xtr_s, ytr)
    return m.predict_proba(Xte_s)[:, 1]


def downsample_positives(Xte, yte, target_rate_pct, rng):
    n_te = len(yte)
    n_target_pos = max(1, int(round(n_te * target_rate_pct / 100.0)))
    pos_idx = np.where(yte == 1)[0]
    neg_idx = np.where(yte == 0)[0]
    keep_pos = pos_idx if len(pos_idx) < n_target_pos else rng.choice(
        pos_idx, size=n_target_pos, replace=False)
    keep = np.concatenate([keep_pos, neg_idx])
    keep.sort()
    return Xte[keep], yte[keep]


def cell_counters_at_tau(scores, v_mean, y, tau):
    """Compute the 5-way disagreement counts at a single V_mean threshold."""
    n = len(y)
    k = max(1, int(round(n * K_PCT / 100.0)))
    top_k_idx = np.argsort(-scores)[:k]
    xgb_fire = np.zeros(n, dtype=bool)
    xgb_fire[top_k_idx] = True
    llm_fire = v_mean > tau
    n_xgb_pos = int(np.sum(xgb_fire & (y == 1)))
    n_xgb_missed_pos = int(np.sum(~xgb_fire & (y == 1)))
    n_lift = int(np.sum(~xgb_fire & llm_fire & (y == 1)))
    n_waste = int(np.sum(~xgb_fire & llm_fire & (y == 0)))
    n_xgb_only = int(np.sum(xgb_fire & ~llm_fire))
    n_both = int(np.sum(xgb_fire & llm_fire))
    n_llm_only = int(np.sum(~xgb_fire & llm_fire))
    return {
        "n_test": n,
        "n_pos": int(y.sum()),
        "n_xgb_pos": n_xgb_pos,
        "n_xgb_missed_pos": n_xgb_missed_pos,
        "n_lift": n_lift,
        "n_waste": n_waste,
        "n_xgb_only": n_xgb_only,
        "n_both": n_both,
        "n_llm_only": n_llm_only,
    }


def main():
    print(f"[iter168] loading train/test ...")
    Xtr_full, ytr_full = load(ROOT / "train_data.csv")
    Xte_full, yte_full = load(ROOT / "test_data.csv")
    print(f"[iter168] Xtr={Xtr_full.shape} ytr_pos={ytr_full.sum()} | "
          f"Xte={Xte_full.shape} yte_pos={yte_full.sum()}")

    matrix_rows = []  # per-(seed, rate, fset, tau, tier)
    h3_rows = []      # per-(rate, fset, tau) monotonicity verdicts
    pareto_rows = []  # Pareto-frontier cells

    for seed in SEEDS:
        print(f"[iter168] === SEED {seed} ===")
        rng = np.random.default_rng(seed)

        for rate_pct in RATES_PCT:
            Xte, yte = downsample_positives(Xte_full, yte_full, rate_pct, rng)
            n_te = len(yte)
            v_mean = Xte[:, ALL24.index("V_mean")]
            for fset_name, feats in FEATURE_SETS.items():
                scores = fit_predict(Xtr_full, ytr_full, Xte, feats, seed)

                # Per-tau counters (XGB does not change)
                per_tau_counters = {}
                for tau in TAUS:
                    per_tau_counters[tau] = cell_counters_at_tau(scores, v_mean, yte, tau)

                # Per-tau × per-tier escalation metrics
                for tau in TAUS:
                    c = per_tau_counters[tau]
                    for tier_name, cost_llm in LLM_PRICE_TIERS:
                        n_lift = c["n_lift"]
                        n_waste = c["n_waste"]
                        n_llm_only = c["n_llm_only"]
                        value_rate = n_lift / max(1, c["n_xgb_missed_pos"])
                        esc_prec = n_lift / max(1, n_lift + n_waste)
                        esc_cost = (n_lift + n_waste) * cost_llm / max(1, n_lift)
                        esc_value = n_lift * VALUE_PER_CATCH
                        breakeven = esc_cost <= VALUE_PER_CATCH
                        matrix_rows.append({
                            "seed": seed,
                            "rate_pct": rate_pct,
                            "fset": fset_name,
                            "tau_vmean": tau,
                            "tier": tier_name,
                            "cost_llm_per_call": cost_llm,
                            "n_test": c["n_test"],
                            "n_pos": c["n_pos"],
                            "n_xgb_pos": c["n_xgb_pos"],
                            "n_lift": n_lift,
                            "n_waste": n_waste,
                            "n_llm_only": n_llm_only,
                            "value_rate": value_rate,
                            "esc_prec": esc_prec,
                            "esc_cost_per_lift": esc_cost,
                            "esc_value": esc_value,
                            "breakeven": breakeven,
                        })

                # Pareto frontier at tau × cheap tier (the canonical value lens)
                for tau in TAUS:
                    c = per_tau_counters[tau]
                    n_lift = c["n_lift"]
                    n_waste = c["n_waste"]
                    value_rate = n_lift / max(1, c["n_xgb_missed_pos"])
                    esc_prec = n_lift / max(1, n_lift + n_waste)
                    # Pareto candidate: esc_prec >= 0.10 AND value_rate >= 0.30
                    pareto_ok = (esc_prec >= 0.10) and (value_rate >= 0.30)
                    pareto_rows.append({
                        "seed": seed,
                        "rate_pct": rate_pct,
                        "fset": fset_name,
                        "tau_vmean": tau,
                        "value_rate": value_rate,
                        "esc_prec": esc_prec,
                        "n_lift": n_lift,
                        "n_waste": n_waste,
                        "pareto_ok": pareto_ok,
                    })

        # ----------------------------------------------------------------
        # H3: value_rate monotone non-increasing in tau across the 7 levels
        # At each (rate, fset), avg value_rate across 5 tiers -> 7-tuple;
        # check monotone non-increasing.
        # ----------------------------------------------------------------
        for rate_pct in RATES_PCT:
            for fset_name in FEATURE_SETS:
                per_tau_vr = []
                for tau in TAUS:
                    vr_at_tau = [r["value_rate"] for r in matrix_rows
                                 if r["seed"] == seed
                                 and r["rate_pct"] == rate_pct
                                 and r["fset"] == fset_name
                                 and r["tau_vmean"] == tau]
                    if vr_at_tau:
                        per_tau_vr.append(float(np.mean(vr_at_tau)))
                if len(per_tau_vr) == len(TAUS):
                    mono = all(per_tau_vr[i] >= per_tau_vr[i+1] for i in range(len(per_tau_vr)-1))
                    h3_rows.append({
                        "seed": seed, "rate_pct": rate_pct, "fset": fset_name,
                        "tau0_vr": per_tau_vr[0],
                        "tau3_vr": per_tau_vr[-1],
                        "monotone_non_increasing": mono,
                        "ratio_tau0_over_tau3": per_tau_vr[0] / max(1e-9, per_tau_vr[-1]),
                    })

    # ----------------------------------------------------------------
    # H1: esc_prec >= 0.10 on >= 50% of (seed x rate x fset x tau >= 1.0) cells
    # at the cheap tier (canonical value lens).
    # ----------------------------------------------------------------
    cheap_tau_pass = sum(1 for r in matrix_rows
                         if r["tier"] == "cheap_heuristic"
                         and r["tau_vmean"] >= 1.0
                         and r["esc_prec"] >= 0.10)
    cheap_tau_total = sum(1 for r in matrix_rows
                          if r["tier"] == "cheap_heuristic"
                          and r["tau_vmean"] >= 1.0)
    h1_pass_rate = cheap_tau_pass / max(1, cheap_tau_total)
    h1_pass = h1_pass_rate >= 0.50
    print(f"[iter168] H1: {cheap_tau_pass}/{cheap_tau_total} = {h1_pass_rate:.3f} "
          f"esc_prec >= 0.10 at cheap_tau>=1.0; PASS={h1_pass}")

    # ----------------------------------------------------------------
    # H2: Pareto frontier exists -- some cell with esc_prec >= 0.10 AND
    #     value_rate >= 0.30 (at any tier/tau).
    # ----------------------------------------------------------------
    n_pareto = sum(1 for p in pareto_rows if p["pareto_ok"])
    h2_pass = n_pareto >= 1
    print(f"[iter168] H2: {n_pareto}/{len(pareto_rows)} pareto_ok cells; "
          f"PASS={h2_pass}")

    # Best Pareto cell
    if n_pareto >= 1:
        best = max((p for p in pareto_rows if p["pareto_ok"]),
                   key=lambda p: p["value_rate"])
        print(f"[iter168] H2 best Pareto cell: seed={best['seed']} rate={best['rate_pct']} "
              f"fset={best['fset']} tau={best['tau_vmean']} vr={best['value_rate']:.4f} "
              f"prec={best['esc_prec']:.4f} n_lift={best['n_lift']}")
    else:
        # Find the closest-to-Pareto cell (max esc_prec subject to value_rate >= 0.30)
        closest = max((p for p in pareto_rows if p["value_rate"] >= 0.30),
                      key=lambda p: p["esc_prec"], default=None)
        if closest is not None:
            print(f"[iter168] H2 closest-to-Pareto (no strict winner): "
                  f"seed={closest['seed']} rate={closest['rate_pct']} "
                  f"fset={closest['fset']} tau={closest['tau_vmean']} "
                  f"vr={closest['value_rate']:.4f} prec={closest['esc_prec']:.4f}")

    # ----------------------------------------------------------------
    # H3: value_rate monotone non-increasing in tau on >= 80% of cells
    # ----------------------------------------------------------------
    h3_pass_rate = sum(1 for r in h3_rows if r["monotone_non_increasing"]) / max(1, len(h3_rows))
    h3_pass = h3_pass_rate >= 0.80
    print(f"[iter168] H3: {sum(1 for r in h3_rows if r['monotone_non_increasing'])}/{len(h3_rows)} "
          f"= {h3_pass_rate:.3f} cells monotone_non_increasing; PASS={h3_pass}")

    # ----------------------------------------------------------------
    # H4: breakeven rate (cheap tier) monotone non-decreasing in tau on >= 50% of cells
    # ----------------------------------------------------------------
    h4_rows = []
    for rate_pct in RATES_PCT:
        for fset_name in FEATURE_SETS:
            for seed in SEEDS:
                per_tau_be = []
                for tau in TAUS:
                    rows = [r["breakeven"] for r in matrix_rows
                            if r["seed"] == seed
                            and r["rate_pct"] == rate_pct
                            and r["fset"] == fset_name
                            and r["tier"] == "cheap_heuristic"
                            and r["tau_vmean"] == tau]
                    if rows:
                        per_tau_be.append(float(np.mean(rows)))
                if len(per_tau_be) == len(TAUS):
                    mono = all(per_tau_be[i] <= per_tau_be[i+1] for i in range(len(per_tau_be)-1))
                    h4_rows.append({
                        "seed": seed, "rate_pct": rate_pct, "fset": fset_name,
                        "tau0_be": per_tau_be[0],
"tau3_be": per_tau_be[-1],
                        "monotone_non_decreasing": mono,
                    })
    h4_pass_rate = sum(1 for r in h4_rows if r["monotone_non_decreasing"]) / max(1, len(h4_rows))
    h4_pass = h4_pass_rate >= 0.50
    print(f"[iter168] H4: {sum(1 for r in h4_rows if r['monotone_non_decreasing'])}/{len(h4_rows)} "
          f"= {h4_pass_rate:.3f} cells breakeven monotone_non_decreasing in tau; PASS={h4_pass}")

    # ----------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------
    out_matrix = RES / "p8_iter168_threshold_matrix.tsv"
    fieldnames = list(matrix_rows[0].keys())
    with out_matrix.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(matrix_rows)
    print(f"[iter168] wrote {out_matrix} ({len(matrix_rows)} rows)")

    out_pareto = RES / "p8_iter168_pareto_cells.tsv"
    fieldnames2 = list(pareto_rows[0].keys())
    with out_pareto.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames2, delimiter="\t")
        w.writeheader()
        w.writerows(pareto_rows)
    print(f"[iter168] wrote {out_pareto} ({len(pareto_rows)} rows)")

    out_h3 = RES / "p8_iter168_vrate_monotone.tsv"
    fieldnames3 = list(h3_rows[0].keys())
    with out_h3.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames3, delimiter="\t")
        w.writeheader()
        w.writerows(h3_rows)
    print(f"[iter168] wrote {out_h3} ({len(h3_rows)} rows)")

    out_h4 = RES / "p8_iter168_breakeven_monotone.tsv"
    fieldnames4 = list(h4_rows[0].keys())
    with out_h4.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames4, delimiter="\t")
        w.writeheader()
        w.writerows(h4_rows)
    print(f"[iter168] wrote {out_h4} ({len(h4_rows)} rows)")

    summary = {
        "iter": 168,
        "job": "P8 V_mean threshold sweep -- precision-recall Pareto frontier",
        "n_seeds": len(SEEDS),
        "n_taus": len(TAUS),
        "n_total_cells": len(matrix_rows),
        "value_per_catch_dollars": VALUE_PER_CATCH,
        "h1_pass": h1_pass,
        "h1_esc_prec_pass_rate": h1_pass_rate,
        "h1_n_pass": cheap_tau_pass,
        "h1_n_total": cheap_tau_total,
        "h2_pass": h2_pass,
        "h2_n_pareto_cells": n_pareto,
        "h3_pass": h3_pass,
        "h3_pass_rate": h3_pass_rate,
        "h4_pass": h4_pass,
        "h4_pass_rate": h4_pass_rate,
    }
    out_sum = RES / "p8_iter168_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter168] wrote {out_sum}")
    print(f"[iter168] DONE")


if __name__ == "__main__":
    main()
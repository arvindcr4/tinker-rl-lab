#!/usr/bin/env python3
"""P8 JOB A (iter 28): expected-cost-optimal decision threshold.

Prior P8 cost work measures TP-per-dollar at FIXED top-K review budgets
(item 21) or precision/recall/F1 swept over thresholds (item 27). Neither
answers the decision-theoretic question a fraud-ops lead actually faces:

    given a fraud cost matrix (investigation cost C_inv per alert, loss L
    per MISSED fraud), what threshold tau* = argmin_tau E[cost] minimises
    dollars-per-decision -- and does the LLM-as-sensor aggregate block
    (V_mean/std/max/min) LOWER that minimum, net of the LLM sensing cost?

We sweep the cost ratio rho = L / C_inv, find tau* per feature set, and put
paired bootstrap CIs on the cost ADVANTAGE of the sensor feature sets. We
also compute the break-even fraud loss L* above which paying the per-row
LLM sensing cost (c_sense) is net cost-reducing vs the raw-feature tree.

Feature sets (LLM-sensor surrogate = the 4 hand-engineered aggregates, an
oracle LLM that emits one deterministic 4-vector per transaction):
  XGB-20raw    : V1..V20                         (no LLM cost)
  XGB-24full   : V1..V20 + 4 aggregates          (pays c_sense/row)
  XGB-4sensor  : 4 aggregates only               (pays c_sense/row)

Cost per decision (USD): C(m,tau) = c_sense_m + [C_inv*(TP+FP)+L*FN]/N.
tau* is invariant to the constant c_sense_m, so argmin is done on the
detection term; c_sense_m enters the cross-model level comparison.

Stdlib + numpy + pandas + xgboost. <=300 lines. Real data only.
"""
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"
OUT = ROOT / "experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)

V20 = [f"V{i}" for i in range(1, 21)]
V_AGG = ["V_mean", "V_std", "V_max", "V_min"]
FEATS = {"XGB-20raw": V20, "XGB-24full": V20 + V_AGG, "XGB-4sensor": V_AGG}
SENSES = {"XGB-20raw": 0.0, "XGB-24full": 0.0035, "XGB-4sensor": 0.0035}

C_INV = 0.50          # analyst investigation cost / alert (USD), from item 21
RHO_GRID = [2, 5, 10, 20, 50, 100, 200, 500]   # rho = L / C_inv
N_BOOT = 400
BOOT_SEED = 2026
TREE_SEED = 42


def fit_tree(X_tr, y_tr):
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", random_state=TREE_SEED, n_jobs=4,
    )
    clf.fit(X_tr, y_tr)
    return clf


def load_scores():
    df_tr = pd.read_csv(TRAIN)
    df_te = pd.read_csv(TEST)
    y_tr = df_tr["Class"].to_numpy(np.int32)
    y_te = df_te["Class"].to_numpy(np.int32)
    scores = {}
    for name, cols in FEATS.items():
        clf = fit_tree(df_tr[cols].to_numpy(np.float64), y_tr)
        scores[name] = clf.predict_proba(df_te[cols].to_numpy(np.float64))[:, 1]
    return scores, y_te


def optimal_cut(score, y, C_inv, L):
    """argmin over all rank cutoffs of C_inv*alerts + L*missed. Returns dict."""
    N = len(y)
    P = int(y.sum())
    order = np.argsort(-score, kind="mergesort")     # descending score
    ys = y[order]
    ss = score[order]
    cum_tp = np.concatenate([[0], np.cumsum(ys)])     # cum_tp[k] = TP in top-k
    k = np.arange(N + 1)
    alerts = k
    tp = cum_tp
    fn = P - tp
    det_cost = C_inv * alerts + L * fn                # detection term (USD-total)
    kstar = int(np.argmin(det_cost))
    # threshold = score of the last alerted row (kstar-th); k=0 -> tau>max (alert none)
    if kstar == 0:
        tau = float(ss[0]) + 1e-9
    elif kstar >= N:
        tau = float(ss[-1])
    else:
        tau = float(ss[kstar - 1])
    tp_star = int(tp[kstar])
    return {
        "kstar": kstar, "tau": tau, "alert_rate": kstar / N,
        "recall": tp_star / P if P else 0.0,
        "tp": tp_star, "fp": kstar - tp_star, "fn": P - tp_star,
        "det_cost_per_dec": det_cost[kstar] / N,
    }


def cost_at_threshold(score, y, tau, C_inv, L, c_sense):
    """Realised per-decision cost applying fixed policy tau to (score,y)."""
    N = len(y)
    alert = score >= tau
    tp = int(np.sum(alert & (y == 1)))
    fp = int(np.sum(alert & (y == 0)))
    fn = int(np.sum(~alert & (y == 1)))
    return c_sense + (C_inv * (tp + fp) + L * fn) / N


def main():
    scores, y = load_scores()
    N = len(y)
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = [rng.integers(0, N, N) for _ in range(N_BOOT)]

    op_rows = []            # per (rho, model) operating point
    boot_rows = []          # per (rho, contrast) paired-bootstrap CI
    summary = {"C_inv": C_INV, "c_sense": SENSES, "N": N,
               "n_boot": N_BOOT, "rho_grid": RHO_GRID, "contrasts": {}}

    # store per-model optimal policy per rho
    policy = {}
    for rho in RHO_GRID:
        L = rho * C_INV
        policy[rho] = {}
        for m in FEATS:
            oc = optimal_cut(scores[m], y, C_INV, L)
            full_cost = SENSES[m] + oc["det_cost_per_dec"]
            policy[rho][m] = {**oc, "L": L, "full_cost_per_dec": full_cost}
            op_rows.append({
                "rho": rho, "L_usd": L, "model": m, "tau_star": round(oc["tau"], 4),
                "alert_rate": round(oc["alert_rate"], 5), "recall": round(oc["recall"], 4),
                "tp": oc["tp"], "fp": oc["fp"], "fn": oc["fn"],
                "c_sense": SENSES[m],
                "cost_per_dec_usd": round(full_cost, 6),
            })

    # paired bootstrap on the cost advantage at each model's fixed tau*
    CONTRASTS = [("XGB-24full", "XGB-4sensor"), ("XGB-24full", "XGB-20raw"),
                 ("XGB-20raw", "XGB-4sensor")]
    for rho in RHO_GRID:
        L = rho * C_INV
        for a, b in CONTRASTS:
            # advantage of `a` = cost_b - cost_a  (positive => a cheaper)
            ta, tb = policy[rho][a]["tau"], policy[rho][b]["tau"]
            ca, cb = SENSES[a], SENSES[b]
            point = policy[rho][b]["full_cost_per_dec"] - policy[rho][a]["full_cost_per_dec"]
            deltas = np.empty(N_BOOT)
            for j, bi in enumerate(boot_idx):
                sa, sb, yb = scores[a][bi], scores[b][bi], y[bi]
                cost_a = cost_at_threshold(sa, yb, ta, C_INV, L, ca)
                cost_b = cost_at_threshold(sb, yb, tb, C_INV, L, cb)
                deltas[j] = cost_b - cost_a
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            boot_rows.append({
                "rho": rho, "L_usd": L, "advantage_of": a, "vs": b,
                "delta_cost_per_dec_usd": round(point, 6),
                "boot_mean": round(float(deltas.mean()), 6),
                "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "sign": "a_cheaper" if point > 0 else "b_cheaper",
            })

    # break-even fraud loss L* where XGB-24full net-beats XGB-20raw (fine L sweep)
    L_fine = np.round(np.arange(0.5, 400.01, 0.5), 2)
    lstar = None
    breakeven_curve = []
    for L in L_fine:
        c24 = SENSES["XGB-24full"] + optimal_cut(scores["XGB-24full"], y, C_INV, L)["det_cost_per_dec"]
        c20 = SENSES["XGB-20raw"] + optimal_cut(scores["XGB-20raw"], y, C_INV, L)["det_cost_per_dec"]
        adv = c20 - c24
        breakeven_curve.append({"L_usd": float(L), "rho": float(L / C_INV),
                                "cost_24full": round(c24, 6), "cost_20raw": round(c20, 6),
                                "adv_24full": round(adv, 6)})
        if lstar is None and adv > 0:
            lstar = float(L)
    summary["breakeven_L_usd_24full_beats_20raw"] = lstar
    summary["breakeven_rho"] = (lstar / C_INV) if lstar is not None else None

    # headline: at each rho, does 24full's CI vs 4sensor exclude zero (sensor-only insufficient)?
    for a, b in CONTRASTS:
        rows = [r for r in boot_rows if r["advantage_of"] == a and r["vs"] == b]
        summary["contrasts"][f"{a}_vs_{b}"] = {
            "n_rho_ci_excludes_zero": sum(r["excludes_zero"] for r in rows),
            "n_rho": len(rows),
            "a_cheaper_at_all_rho": all(r["delta_cost_per_dec_usd"] > 0 for r in rows),
        }

    # write outputs
    with open(OUT / "p8_cost_optimal.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(op_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(op_rows)
    with open(OUT / "p8_cost_optimal_boot.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(boot_rows)
    with open(OUT / "p8_cost_optimal_breakeven.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(breakeven_curve[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(breakeven_curve)
    with open(OUT / "p8_cost_optimal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # console headlines
    print(f"[p8-cost-optimal] N={N} pos_rate={y.mean():.4f} C_inv=${C_INV} c_sense=${SENSES['XGB-24full']}")
    print(f"break-even L* (24full net-beats 20raw): ${lstar}  (rho*={summary['breakeven_rho']})")
    for k, v in summary["contrasts"].items():
        print(f"  {k}: CI excludes 0 at {v['n_rho_ci_excludes_zero']}/{v['n_rho']} rho; "
              f"a_cheaper_all={v['a_cheaper_at_all_rho']}")
    print("tau* shift with rho (24full):",
          [(r["rho"], r["tau_star"], r["recall"]) for r in op_rows if r["model"] == "XGB-24full"])


if __name__ == "__main__":
    main()

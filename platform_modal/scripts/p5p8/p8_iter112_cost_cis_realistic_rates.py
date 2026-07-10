#!/usr/bin/env python3
"""P8 JOB A (iter 112): paired-row bootstrap CI on cost-per-decision and
cost-per-fraud-caught across 5 REALISTIC POSITIVE RATES × 3 RULES.

Fresh vein -- combines iter-12 (5 realistic fraud ratios) with iter-108
(paired bootstrap CI on $/dec and $/caught across rules) on the SAME
test split. Closes the iter-88 cross-cohort / cross-noise gap at the
realistic-rate level -- the live operational question for a fraud-ops
analyst: "at my deployed base-rate, am I confident the gradient-band
rule Pareto-dominates the absolute-band rule on $/caught?"

For each positive rate (release 1.44% / 1.00% / 0.50% / 0.10% / 0.05%)
and each rule (xgb-only, gradient-band, absolute-band):
  * downsample positives to target_rate (mirrors iter-12)
  * replay the rule at K=2% (recall@K) and K=top-1% (precision@K)
  * paired bootstrap CI on $/dec and $/caught
  * return cost-delta matrix (gradient-band minus absolute-band, etc.)

Falsifiable headlines
---------------------
H1 -- across all 5 rates, no CI excludes zero on $/dec
   gradient-band - absolute-band (the two are within ~ 1e-4 $/dec).
H2 -- across all 5 rates, gradient-band Pareto-dominates absolute-band
   on $/caught with CIs excluding zero for rates >= 0.50%.
H3 -- at the smallest rate (0.05%, top-1% K=100), no rule catches
   any positive reliably: $/caught CIs are vastly wider, parity holds.

Stdlib + numpy + xgboost. <= 300 lines.
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
SEED = 20260705
N_BOOT = 1000
COST_XGB = 0.0001
COST_LLM = 0.0010
G_THR = 1e-4
W_ABS = 0.10
K_PCT = 2.0
RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
RATES = [0.0144, 0.0100, 0.0050, 0.0010, 0.0005]


def load(path):
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        cols = {n: [] for n in header}
        for row in rdr:
            for n in header:
                cols[n].append(float(row[idx[n]]))
    return {n: np.array(cols[n]) for n in header}


def downsample_pos(y, target, rng):
    """Downsample positives so fraction of `y=1` in returned mask == target.
    Negatives are kept whole. Returns (mask, idx_kept)."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_target_pos = max(1, int(round(target * (len(pos) + len(neg)))))
    n_pos = min(len(pos), n_target_pos)
    sel_pos = rng.choice(pos, size=n_pos, replace=False) if n_pos < len(pos) else pos
    return np.concatenate([sel_pos, neg]), np.arange(len(y))


def fit_xgb(X_tr, y_tr, X_te, seed):
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=seed, n_jobs=4,
    )
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def rule_costs(scores, y_true, n_pos_caught, n_llm_calls):
    """Return ($/dec, $/caught) for this rule on this positive set."""
    n = len(y_true)
    cpd = (n * COST_XGB + n_llm_calls * (COST_LLM - COST_XGB)) / n
    cpf = (n * COST_XGB + n_llm_calls * (COST_LLM - COST_XGB)) / max(1, n_pos_caught)
    return cpd, cpf


def apply_rules(scores, y, K_pct):
    """Apply xgb-only / gradient-band / absolute-band (mirrors iter-108 logic).
    Returns dict rule -> tuple (n_llm, pos_at_K, cpd, cpf)."""
    n = len(y)
    K = max(1, int(round(K_pct / 100 * n)))
    # xgb-only
    cpd_x, cpf_x = rule_costs(scores, y, _pos_at_K(scores, y, K), 0)
    # gradient-band: forward-difference on sorted scores; plateau indices
    # (within top-K) invoke the LLM. Mirrors iter-108 apply_gradient.
    order = np.argsort(-scores)
    grads = np.zeros(n)
    grads[1:] = scores[order[:-1]] - scores[order[1:]]
    plateaus = np.zeros(n, dtype=bool)
    plateaus[order[1:]] = grads[1:] <= G_THR
    top_k = np.zeros(n, dtype=bool)
    top_k[order[:K]] = True
    invoke_g = plateaus & top_k
    n_grad = int(invoke_g.sum())
    pos_at_K_g = _pos_at_K(scores, y, K)
    cpd_g, cpf_g = rule_costs(scores, y, pos_at_K_g, n_grad)
    # absolute-band
    band = (scores >= 0.5 - W_ABS / 2) & (scores <= 0.5 + W_ABS / 2)
    n_abs = int(band.sum())
    pos_at_K_a = _pos_at_K(scores, y, K)
    cpd_a, cpf_a = rule_costs(scores, y, pos_at_K_a, min(n_abs, K))
    return {
        "xgb_only": (0, pos_at_K_g, cpd_x, cpf_x),
        "gradient_band": (n_grad, pos_at_K_g, cpd_g, cpf_g),
        "absolute_band": (min(n_abs, K), pos_at_K_a, cpd_a, cpf_a),
    }


def _pos_at_K(scores, y, K):
    order = np.argsort(-scores)
    return int(np.sum(y[order[:K]] == 1))


def paired_boot_ci(y, scores, rules, n_boot, seed):
    """For each pair of rules, paired-row bootstrap on $/dec and $/caught.
    Returns dict (pair -> cpd_delta [ci_lo, ci_hi, excl_zero], cpf_delta ...)."""
    n = len(y)
    rng = np.random.default_rng(seed)
    base = apply_rules(scores, y, K_PCT)
    pairs = [
        ("gradient_band", "absolute_band"),
        ("gradient_band", "xgb_only"),
        ("absolute_band", "xgb_only"),
    ]
    diffs = {p: {"cpd": [], "cpf": []} for p in pairs}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rb = apply_rules(scores[idx], y[idx], K_PCT)
        for p in pairs:
            for metric in ("cpd", "cpf"):
                diffs[p][metric].append(rb[p[0]][2 + (metric == "cpf")]
                                         - rb[p[1]][2 + (metric == "cpf")])
    out = {}
    for p in pairs:
        cpd_arr = np.array(diffs[p]["cpd"])
        cpf_arr = np.array(diffs[p]["cpf"])
        cpd_lo, cpd_hi = np.quantile(cpd_arr, [0.025, 0.975])
        cpf_lo, cpf_hi = np.quantile(cpf_arr, [0.025, 0.975])
        out[p] = {
            "cpd_delta": float(cpd_arr.mean()),
            "cpd_ci_lo": float(cpd_lo),
            "cpd_ci_hi": float(cpd_hi),
            "cpd_excl_zero": bool((cpd_lo > 0) or (cpd_hi < 0)),
            "cpf_delta": float(cpf_arr.mean()),
            "cpf_ci_lo": float(cpf_lo),
            "cpf_ci_hi": float(cpf_hi),
            "cpf_excl_zero": bool((cpf_lo > 0) or (cpf_hi < 0)),
        }
    return out, base


def main():
    print("[iter112] loading ...")
    tr = load(ROOT / "fraud_data.csv")
    te = load(ROOT / "test_data.csv")
    Xtr = np.column_stack([tr[v] for v in ALL24])
    ytr = tr["Class"].astype(int)
    Xte = np.column_stack([te[v] for v in ALL24])
    yte = te["Class"].astype(int)
    print(f"[iter112] n_train={len(ytr)} n_test={len(yte)} pos_rate_release={yte.mean():.4f}")

    print("[iter112] training all 3 trees on released full-rate train ...")
    p20 = fit_xgb(Xtr[:, :20], ytr, Xte[:, :20], SEED)
    p24 = fit_xgb(Xtr, ytr, Xte, SEED)
    p4 = fit_xgb(Xtr[:, 20:], ytr, Xte[:, 20:], SEED)
    print(f"[iter112] p20 AUC={_auc(yte, p20):.5f} p24 AUC={_auc(yte, p24):.5f} p4 AUC={_auc(yte, p4):.5f}")

    rng_down = np.random.default_rng(SEED)
    per_cell_rows = []
    pair_rows = []
    sig_count = {"cpd": 0, "cpf": 0}
    print("[iter112] iterating 5 rates x 3 trees x 3 rule pairs ...")
    for tgt in RATES:
        mask, _ = downsample_pos(yte, tgt, rng_down)
        y_d = yte[mask]
        for tree_name, scores in (("XGB-20raw", p20[mask]),
                                   ("XGB-24full", p24[mask]),
                                   ("XGB-4sensor", p4[mask])):
            pair_ci, base = paired_boot_ci(y_d, scores, None, N_BOOT, SEED + int(tgt * 1e7))
            for p, ci in pair_ci.items():
                cpd_excl = ci["cpd_excl_zero"]
                cpf_excl = ci["cpf_excl_zero"]
                if cpd_excl:
                    sig_count["cpd"] += 1
                if cpf_excl:
                    sig_count["cpf"] += 1
                pair_rows.append(dict(
                    rate=tgt, tree=tree_name, pair="_vs_".join(p),
                    cpd_delta=ci["cpd_delta"],
                    cpd_ci_lo=ci["cpd_ci_lo"], cpd_ci_hi=ci["cpd_ci_hi"],
                    cpd_excl_zero=cpd_excl,
                    cpf_delta=ci["cpf_delta"],
                    cpf_ci_lo=ci["cpf_ci_lo"], cpf_ci_hi=ci["cpf_ci_hi"],
                    cpf_excl_zero=cpf_excl,
                    n_pos=int(y_d.sum()),
                ))
            for rule, vals in base.items():
                n_llm, pos_at_K, cpd, cpf = vals
                per_cell_rows.append(dict(
                    rate=tgt, tree=tree_name, rule=rule,
                    n_pos=int(y_d.sum()),
                    n_pos_caught_at_K=pos_at_K,
                    n_llm_calls=n_llm,
                    cpd_usd=cpd, cpf_usd=cpf,
                ))

    cols_cell = list(per_cell_rows[0].keys())
    with (RES / "p8_iter112_cost_per_rate_cell.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols_cell, delimiter="\t")
        w.writeheader()
        for r in per_cell_rows:
            w.writerow(r)
    cols_pair = list(pair_rows[0].keys())
    with (RES / "p8_iter112_paired_bootstrap_ci.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols_pair, delimiter="\t")
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    # --- H1: gradient vs absolute cpd excl_zero at each rate ---
    h1 = []
    for tgt in RATES:
        sub = [r for r in pair_rows if r["rate"] == tgt
               and r["pair"] == "gradient_band_vs_absolute_band"
               and r["tree"] == "XGB-24full"]
        if sub:
            r = sub[0]
            h1.append(dict(rate=tgt, cpd_delta=r["cpd_delta"],
                            ci_lo=r["cpd_ci_lo"], ci_hi=r["cpd_ci_hi"],
                            excl_zero=r["cpd_excl_zero"]))
    # --- H2: gradient vs absolute cpf excl_zero at each rate ---
    h2 = []
    for tgt in RATES:
        sub = [r for r in pair_rows if r["rate"] == tgt
               and r["pair"] == "gradient_band_vs_absolute_band"
               and r["tree"] == "XGB-24full"]
        if sub:
            r = sub[0]
            h2.append(dict(rate=tgt, cpf_delta=r["cpf_delta"],
                            ci_lo=r["cpf_ci_lo"], ci_hi=r["cpf_ci_hi"],
                            excl_zero=r["cpf_excl_zero"]))
    summary = {
        "iter": 112,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "rates": RATES,
        "rules": ["xgb_only", "gradient_band", "absolute_band"],
        "n_boot": N_BOOT,
        "k_pct": K_PCT,
        "seed": SEED,
        "cost_xgb": COST_XGB,
        "cost_llm": COST_LLM,
        "g_thr": G_THR,
        "w_abs": W_ABS,
        "n_cells": len(per_cell_rows),
        "n_pair_cells": len(pair_rows),
        "n_sig_cpd": sig_count["cpd"],
        "n_sig_cpf": sig_count["cpf"],
        "h1_grad_vs_abs_cpd_by_rate": h1,
        "h2_grad_vs_abs_cpf_by_rate": h2,
    }
    (RES / "p8_iter112_cost_cis_realistic_rates_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[iter112] done. n_pair_cells={len(pair_rows)} "
          f"n_sig_cpd={sig_count['cpd']} n_sig_cpf={sig_count['cpf']}")
    print(f"[iter112] h1 (cpd) exclude_zero per rate:")
    for h in h1:
        print(f"  rate={h['rate']:.4f} delta={h['cpd_delta']:+.6f} CI=[{h['ci_lo']:+.6f},{h['ci_hi']:+.6f}] excl={h['excl_zero']}")
    print(f"[iter112] h2 (cpf) exclude_zero per rate:")
    for h in h2:
        print(f"  rate={h['rate']:.4f} delta={h['cpf_delta']:+.4e} CI=[{h['ci_lo']:+.4e},{h['ci_hi']:+.4e}] excl={h['excl_zero']}")


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


if __name__ == "__main__":
    main()

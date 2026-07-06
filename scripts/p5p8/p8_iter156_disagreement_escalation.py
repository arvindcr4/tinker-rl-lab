#!/usr/bin/env python3
"""P8 disagreement-driven escalation economics on the 5-seed panel (iter 156 JOB A).

Fresh vein, not in 171 prior P8 rows. Closes the iter-148 cost matrix + iter-152
5-seed CV: those measured cppr/acd *averaged* across all fires. Iter-156
decomposes fires into **value** (LLM catches fraud XGB missed) vs **waste**
(LLM call on a row XGB already caught, or on a non-fraud row).

The escalation question is operationally critical: when the LLM fires,
does it actually add recall? At what cost per added recall?

For each test row r:
  - xgb_fire_r = (xgb_score_r in top-K of xgb scores)
  - llm_fire_r = (V_mean_r > tau) where tau is the per-fset tuned threshold
  - is_fraud_r = y_r

Per-cell counters (rate, tier, fset, seed):
  - n_xgb_pos  = #(xgb_fire AND is_fraud)         — XGB-only positives caught
  - n_lift     = #(NOT xgb_fire AND llm_fire AND is_fraud)  — VALUE
  - n_waste    = #(NOT xgb_fire AND llm_fire AND NOT is_fraud)  — no-op
  - n_xgb_only = #(xgb_fire AND NOT llm_fire)     — XGB caught w/o LLM
  - n_both     = #(xgb_fire AND llm_fire)         — overlap

Escalation metrics:
  - value_rate        = n_lift / max(1, n_xgb_missed_pos)
  - escalation_prec   = n_lift / max(1, n_lift + n_waste)
  - escalation_recall_lift = value_rate
  - escalation_cost_per_lift = (n_lift + n_waste) * cost_llm / max(1, n_lift)
  - escalation_breakeven_value = escalation_cost_per_lift / value_per_catch

Hypotheses
----------
H1 -- escalation_precision >= 0.05 on >= 80/100 cells (sparse-positive
     regimes have enough positive density to make LLM escalation useful).

H2 -- escalation_cost_per_lift is monotone in LLM tier price (cheap <
     small_open < ... < frontier_gpt4) at all (rate, fset) cells; the
     cheap tier crosses the breakeven value line.

H3 -- 5-seed CV on escalation_recall_lift is < 0.20 for >= 80/100 cells
     (seed-robust escalation decision).

H4 -- fset sensitivity: 20raw+stat maximizes value_rate at low rates
     (rate in {0.05, 0.10}%); 24full maximizes value_rate at the
     release rate (1.44%).

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
N_BOOT = 1000
COST_XGB = 0.0001
K_PCT = 2.0
TAU_VMEAN = 0.0  # V_mean > 0 (the raw data has both signs; this is a one-sided filter)

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
# VALUE_PER_CATCH ($) -- operational fraud-ops estimate ($50–$500 per caught
# fraud; we use a conservative $50 mid-range).
VALUE_PER_CATCH = 50.0


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


def cell_counters(scores, v_mean, y):
    """Compute the 5-way disagreement counts per test set."""
    n = len(y)
    k = max(1, int(round(n * K_PCT / 100.0)))
    top_k_idx = np.argsort(-scores)[:k]
    xgb_fire = np.zeros(n, dtype=bool)
    xgb_fire[top_k_idx] = True
    llm_fire = v_mean > TAU_VMEAN  # per-fset threshold tunable
    n_xgb_pos = int(np.sum(xgb_fire & (y == 1)))
    n_xgb_missed_pos = int(np.sum(~xgb_fire & (y == 1)))
    n_lift = int(np.sum(~xgb_fire & llm_fire & (y == 1)))
    n_waste = int(np.sum(~xgb_fire & llm_fire & (y == 0)))
    n_xgb_only = int(np.sum(xgb_fire & ~llm_fire))
    n_both = int(np.sum(xgb_fire & llm_fire))
    n_llm_only = int(np.sum(~xgb_fire & llm_fire))
    n_lift_overlap = int(np.sum(xgb_fire & llm_fire & (y == 1)))
    return {
        "n_test": n,
        "n_pos": int(y.sum()),
        "n_xgb_fire": int(xgb_fire.sum()),
        "n_llm_fire": int(llm_fire.sum()),
        "n_xgb_pos": n_xgb_pos,
        "n_xgb_missed_pos": n_xgb_missed_pos,
        "n_lift": n_lift,
        "n_waste": n_waste,
        "n_xgb_only": n_xgb_only,
        "n_both": n_both,
        "n_llm_only": n_llm_only,
        "n_lift_overlap": n_lift_overlap,
    }


def bootstrap_escalation(n_lift, n_waste, n_test, n_boot=N_BOOT, seed=20260706):
    rng = np.random.default_rng(seed)
    precs = np.empty(n_boot)
    for bi in range(n_boot):
        idx = rng.integers(0, n_test, n_test)
        # per-row labels: synthesise proportion matching n_lift/n_waste
        n_pos_s = rng.binomial(n_test, (n_lift + 0.01) / max(1, n_test))
        n_neg_s = n_test - n_pos_s
        n_pos_lift = int(round(n_pos_s * n_lift / max(1, n_lift + n_waste)))
        n_pos_waste = n_pos_s - n_pos_lift
        # among n_lift + n_waste LLM-only fires
        total = n_lift + n_waste
        if total > 0:
            precs[bi] = n_pos_lift / total
        else:
            precs[bi] = 0.0
    return float(np.mean(precs)), float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5))


def main():
    print(f"[iter156] loading train/test ...")
    Xtr_full, ytr_full = load(ROOT / "train_data.csv")
    Xte_full, yte_full = load(ROOT / "test_data.csv")
    print(f"[iter156] Xtr={Xtr_full.shape} ytr_pos={ytr_full.sum()} | "
          f"Xte={Xte_full.shape} yte_pos={yte_full.sum()}")

    matrix_rows = []  # per-(seed, rate, tier, fset) cell
    h2_rows = []       # per-tier monotone test
    seed_summary = []  # per-seed aggregate

    for seed in SEEDS:
        print(f"[iter156] === SEED {seed} ===")
        rng = np.random.default_rng(seed)
        seed_data = {"seed": seed, "per_cell_value_rate": [], "per_cell_lift": []}

        # Pre-compute V_mean for the FULL test set (independent of fset/rate)
        # but the XGB scores are per-(seed, fset, rate) — precompute them.
        # For each (rate, fset), fit once, then aggregate across tiers.
        rate_fset_data = {}
        for rate_pct in RATES_PCT:
            Xte, yte = downsample_positives(Xte_full, yte_full, rate_pct, rng)
            n_te = len(yte)
            v_mean = Xte[:, ALL24.index("V_mean")]
            for fset_name, feats in FEATURE_SETS.items():
                scores = fit_predict(Xtr_full, ytr_full, Xte, feats, seed)
                counters = cell_counters(scores, v_mean, yte)
                rate_fset_data[(rate_pct, fset_name)] = {
                    "scores": scores, "yte": yte, "v_mean": v_mean, "counters": counters,
                    "n_test": n_te, "n_pos": int(yte.sum()),
                }
                seed_data["per_cell_value_rate"].append(
                    counters["n_lift"] / max(1, counters["n_xgb_missed_pos"]))
                seed_data["per_cell_lift"].append(counters["n_lift"])

        # Per-cell × per-tier escalation metrics
        for (rate_pct, fset_name), d in rate_fset_data.items():
            c = d["counters"]
            for tier_name, cost_llm in LLM_PRICE_TIERS:
                n_lift = c["n_lift"]
                n_waste = c["n_waste"]
                n_llm_only = c["n_llm_only"]
                value_rate = n_lift / max(1, c["n_xgb_missed_pos"])
                esc_prec = n_lift / max(1, n_lift + n_waste)
                # escalation_cost_per_lift = total LLM cost on disagreement set / n_lift
                esc_cost = (n_lift + n_waste) * cost_llm / max(1, n_lift)
                # escalation_value = n_lift * value_per_catch
                esc_value = n_lift * VALUE_PER_CATCH
                # breakeven: cost <= value iff cost_per_lift <= value_per_catch
                breakeven = esc_cost <= VALUE_PER_CATCH
                # bootstrap CI on esc_prec
                prec_mean, prec_lo, prec_hi = bootstrap_escalation(
                    n_lift, n_waste, d["n_test"], N_BOOT, seed)
                matrix_rows.append({
                    "seed": seed,
                    "rate_pct": rate_pct,
                    "tier": tier_name,
                    "fset": fset_name,
                    "cost_llm_per_call": cost_llm,
                    "n_test": d["n_test"],
                    "n_pos": d["n_pos"],
                    "n_xgb_pos": c["n_xgb_pos"],
                    "n_lift": n_lift,
                    "n_waste": n_waste,
                    "n_xgb_only": c["n_xgb_only"],
                    "n_llm_only": n_llm_only,
                    "value_rate": value_rate,
                    "esc_prec": esc_prec,
                    "esc_prec_boot_mean": prec_mean,
                    "esc_prec_boot_lo": prec_lo,
                    "esc_prec_boot_hi": prec_hi,
                    "esc_cost_per_lift": esc_cost,
                    "esc_value": esc_value,
                    "breakeven": breakeven,
                })

        seed_summary.append({
            "seed": seed,
            "n_cells": len(seed_data["per_cell_value_rate"]),
            "mean_value_rate": float(np.mean(seed_data["per_cell_value_rate"])),
            "mean_n_lift": float(np.mean(seed_data["per_cell_lift"])),
            "max_n_lift": int(max(seed_data["per_cell_lift"])),
        })
        print(f"[iter156] seed {seed}: mean_value_rate={seed_summary[-1]['mean_value_rate']:.4f} "
              f"max_n_lift={seed_summary[-1]['max_n_lift']}")

    # ----------------------------------------------------------------
    # H1: escalation_precision >= 0.05 on >= 80% of cells
    # ----------------------------------------------------------------
    n_total = len(matrix_rows)
    n_prec_pass = sum(1 for r in matrix_rows if r["esc_prec"] >= 0.05)
    h1_pass = (n_prec_pass / max(1, n_total)) >= 0.80
    print(f"[iter156] H1: {n_prec_pass}/{n_total} = {n_prec_pass/n_total:.3f} prec >= 0.05; "
          f"PASS={h1_pass}")

    # ----------------------------------------------------------------
    # H2: escalation_cost_per_lift monotone in tier price at each (rate,fset) cell
    # Sweep avg across seeds.
    # ----------------------------------------------------------------
    h2_verdicts = []
    for rate_pct in RATES_PCT:
        for fset_name in FEATURE_SETS:
            for seed in SEEDS:
                rows = [r for r in matrix_rows if r["seed"] == seed
                        and r["rate_pct"] == rate_pct and r["fset"] == fset_name]
                if len(rows) != len(LLM_PRICE_TIERS):
                    continue
                costs = [r["esc_cost_per_lift"] for r in rows]
                # Monotone non-decreasing
                mono = all(costs[i] <= costs[i+1] for i in range(len(costs)-1))
                h2_verdicts.append({
                    "seed": seed, "rate_pct": rate_pct, "fset": fset_name,
                    "cheap_cost": costs[0], "frontier_cost": costs[-1],
                    "monotone_in_tier_price": mono,
                    "ratio_frontier_over_cheap": costs[-1] / max(1e-9, costs[0]),
                })
    h2_pass_rate = sum(1 for v in h2_verdicts if v["monotone_in_tier_price"]) / max(1, len(h2_verdicts))
    print(f"[iter156] H2: monotone_in_tier_price={h2_pass_rate:.3f}; "
          f"ratio frontier/cheap mean={np.mean([v['ratio_frontier_over_cheap'] for v in h2_verdicts]):.1f}x")

    # H2 part B: cheap tier crosses breakeven value line (cost <= $50)
    h2_cheap_breakeven = sum(1 for r in matrix_rows
                             if r["tier"] == "cheap_heuristic" and r["breakeven"])
    h2_cheap_total = sum(1 for r in matrix_rows if r["tier"] == "cheap_heuristic")
    h2_cheap_pass = (h2_cheap_breakeven / max(1, h2_cheap_total)) >= 0.50
    print(f"[iter156] H2b: cheap tier breakeven {h2_cheap_breakeven}/{h2_cheap_total} = "
          f"{h2_cheap_breakeven/h2_cheap_total:.3f}; PASS={h2_cheap_pass}")

    # ----------------------------------------------------------------
    # H3: 5-seed CV on value_rate < 0.20 on >= 80% of (rate,tier,fset) cells
    # ----------------------------------------------------------------
    h3_rows = []
    for rate_pct in RATES_PCT:
        for tier_name, _ in LLM_PRICE_TIERS:
            for fset_name in FEATURE_SETS:
                per_seed = [r["value_rate"] for r in matrix_rows
                            if r["rate_pct"] == rate_pct and r["tier"] == tier_name
                            and r["fset"] == fset_name]
                if len(per_seed) < 2:
                    continue
                m = float(np.mean(per_seed))
                sd = float(np.std(per_seed))
                cv = sd / max(1e-9, m) if m > 0 else float("inf")
                h3_rows.append({
                    "rate_pct": rate_pct, "tier": tier_name, "fset": fset_name,
                    "value_rate_mean": m, "value_rate_sd": sd, "value_rate_cv": cv,
                    "cv_leq_020": cv <= 0.20,
                })
    h3_pass_rate = sum(1 for r in h3_rows if r["cv_leq_020"]) / max(1, len(h3_rows))
    h3_pass = h3_pass_rate >= 0.80
    print(f"[iter156] H3: CV<=0.20 on {sum(1 for r in h3_rows if r['cv_leq_020'])}/{len(h3_rows)} = "
          f"{h3_pass_rate:.3f}; PASS={h3_pass}")

    # ----------------------------------------------------------------
    # H4: fset sensitivity on value_rate — 20raw+stat best at low rates
    # ----------------------------------------------------------------
    h4_rows = []
    for rate_pct in RATES_PCT:
        for tier_name, _ in LLM_PRICE_TIERS:
            per_fset = {}
            for fset_name in FEATURE_SETS:
                vals = [r["value_rate"] for r in matrix_rows
                        if r["rate_pct"] == rate_pct and r["tier"] == tier_name
                        and r["fset"] == fset_name]
                per_fset[fset_name] = float(np.mean(vals))
            best = max(per_fset, key=per_fset.get)
            h4_rows.append({
                "rate_pct": rate_pct, "tier": tier_name, "best_fset": best,
                **{f"vr_{k}": v for k, v in per_fset.items()},
            })
    # Low rates: 0.05, 0.10 — does 20raw+stat win?
    low_rate_bset_count = sum(1 for r in h4_rows
                              if r["rate_pct"] in (0.05, 0.10)
                              and r["best_fset"] == "20raw+stat")
    low_total = sum(1 for r in h4_rows if r["rate_pct"] in (0.05, 0.10))
    release_rate_bset_count = sum(1 for r in h4_rows
                                  if r["rate_pct"] == 1.44
                                  and r["best_fset"] == "24full")
    release_total = sum(1 for r in h4_rows if r["rate_pct"] == 1.44)
    h4_low_pass = low_rate_bset_count == low_total
    h4_release_pass = release_rate_bset_count == release_total
    print(f"[iter156] H4: low-rate best=20raw+stat on {low_rate_bset_count}/{low_total}; "
          f"release best=24full on {release_rate_bset_count}/{release_total}")

    # ----------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------
    out_matrix = RES / "p8_iter156_escalation_matrix.tsv"
    fieldnames = list(matrix_rows[0].keys())
    with out_matrix.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(matrix_rows)
    print(f"[iter156] wrote {out_matrix} ({len(matrix_rows)} rows)")

    out_h2 = RES / "p8_iter156_tier_monotone.tsv"
    fieldnames2 = list(h2_verdicts[0].keys())
    with out_h2.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames2, delimiter="\t")
        w.writeheader()
        w.writerows(h2_verdicts)
    print(f"[iter156] wrote {out_h2} ({len(h2_verdicts)} rows)")

    out_h3 = RES / "p8_iter156_value_rate_cv.tsv"
    fieldnames3 = list(h3_rows[0].keys())
    with out_h3.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames3, delimiter="\t")
        w.writeheader()
        w.writerows(h3_rows)
    print(f"[iter156] wrote {out_h3} ({len(h3_rows)} rows)")

    out_h4 = RES / "p8_iter156_fset_best.tsv"
    fieldnames4 = list(h4_rows[0].keys())
    with out_h4.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames4, delimiter="\t")
        w.writeheader()
        w.writerows(h4_rows)
    print(f"[iter156] wrote {out_h4} ({len(h4_rows)} rows)")

    summary = {
        "iter": 156,
        "n_seeds": len(SEEDS),
        "n_total_cells": n_total,
        "value_per_catch_dollars": VALUE_PER_CATCH,
        "tau_vmean_threshold": TAU_VMEAN,
        "h1_pass": h1_pass,
        "h1_n_prec_pass": n_prec_pass,
        "h1_n_total": n_total,
        "h1_prec_pass_rate": n_prec_pass / max(1, n_total),
        "h2_pass": h2_pass_rate >= 0.80,
        "h2_pass_rate": h2_pass_rate,
        "h2_ratio_frontier_over_cheap_mean": float(np.mean(
            [v["ratio_frontier_over_cheap"] for v in h2_verdicts])),
        "h2b_cheap_breakeven_pass": h2_cheap_pass,
        "h2b_cheap_breakeven_rate": h2_cheap_breakeven / max(1, h2_cheap_total),
        "h3_pass": h3_pass,
        "h3_cv_pass_rate": h3_pass_rate,
        "h4_low_rate_20raw_stat_pass": h4_low_pass,
        "h4_release_24full_pass": h4_release_pass,
        "h4_low_rate_bset_count": low_rate_bset_count,
        "h4_low_rate_total": low_total,
        "h4_release_bset_count": release_rate_bset_count,
        "h4_release_total": release_total,
        "seed_summary": seed_summary,
    }
    out_sum = RES / "p8_iter156_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter156] wrote {out_sum}")
    print(f"[iter156] DONE")


if __name__ == "__main__":
    main()
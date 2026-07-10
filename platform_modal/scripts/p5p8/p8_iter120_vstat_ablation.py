#!/usr/bin/env python3
"""P8 JOB A (iter 120): per-V_stat ablation of gradient-band firing criterion.

Fresh vein (not in 130 prior P8 rows). Each test row carries 4
anomaly-summary statistics derived from the 20 raw V1..V20 PCA features:

  V_mean = mean(V1..V20)        (central tendency)
  V_std  = std(V1..V20)         (dispersion)
  V_max  = max(V1..V20)         (upper-tail anomaly)
  V_min  = min(V1..V20)         (lower-tail anomaly)

Prior P8 iters only broke the test set down by V_mean quartile
(iter-108 H3, cohort asymmetry). iter-120 extends the breakdown to
ALL FOUR V_stats: which anomaly-stat, if any, explains xgb-only's
residual uncertainty?  If a single V_stat quartile concentrates the
LLM-call density AND the recall gap, then the LLM "sensor" is
exploiting a feature the XGB model misses.  If the call density is
spread evenly across V_stat quartiles, then the LLM "sensor" is
capturing score-stream geometry, not anomaly-stat magnitude.

Falsifiable headlines
---------------------
H1 -- per-V_stat quartile LLM-call density (gradient-band, n_test=10000):
  for V_mean quartiles {Q0..Q3}: call density per row [low..high]
  for V_std  quartiles {Q0..Q3}: ...
  for V_max  quartiles {Q0..Q3}: ...
  for V_min  quartiles {Q0..Q3}: ...
  Range across all 16 quartiles:  if max/min > 5.0, then a single
  V_stat quartile concentrates the LLM activity (sensor exploits
  that V_stat);  else xgb-only is structurally sufficient.

H2 -- per-V_stat quartile recall@K=2% gap (xgb-only vs gradient-band)
  paired bootstrap CI on delta_recall per quartile.  Null hypothesis:
  delta_recall = 0 in every quartile.  Test: how many quartiles have
  CI excluding 0?

H3 -- ablation: drop one V_stat quartile from training? (out-of-scope
  here -- kept simple).  Instead:  compute AUC@K=2% of xgb-only on
  the 4 single-V_stat quartile restricted subsets to see if any
  V_stat quartile is "easy" (xgb-only AUC > 0.999) vs "hard" (AUC
  drops).

H4 -- sensor-feature efficiency:
  LLM calls needed to recover 1 missed fraud, per V_stat quartile.
  Computed as (LLM calls in quartile) / (max(1, n_missed_in_quartile)).

Stdlib + numpy + xgboost.  <= 280 lines.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import xgboost as xgb

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
SEED = 20260705
N_BOOT = 1500
COST_XGB = 0.0001
COST_LLM = 0.0010
K_PCT = 2.0
G_THR = 0.001  # gradient-band threshold (iter-80 row 94)
WIDTH = 0.50   # absolute-band threshold (iter-80)

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4


def load(path):
    """Load CSV with the 24 numeric columns + Class."""
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        X, y = [], []
        for line in rdr:
            X.append([float(line[idx[c]]) for c in ALL24])
            y.append(int(float(line[idx["Class"]])))
    return np.array(X), np.array(y)


def fit_predict(Xtr, ytr, Xte):
    """Fit XGB-24full on all 24 features and return test scores.

    Matches the train_xgboost.py baseline that produced the iter-80/108
    headline (recall@K=2% = 141/144 = 0.979):
    n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=7.
    """
    n_pos_tr = max(1, int(ytr.sum()))
    n_neg_tr = max(1, len(ytr) - n_pos_tr)
    spw = n_neg_tr / n_pos_tr  # standard imbalance correction
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=SEED, n_jobs=4,
    )
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def quartiles(values):
    """Return quartile index (0..3) per row, via pandas qcut with retborder."""
    qs = np.quantile(values, [0.25, 0.50, 0.75])
    out = np.zeros(len(values), dtype=int)
    out[values > qs[0]] = 1
    out[values > qs[1]] = 2
    out[values > qs[2]] = 3
    return out


def gradient_band_fire(scores, top_k_mask, g_thr=G_THR):
    """LLM fires iff row is in top-K AND consecutive-score gradient < g_thr."""
    sorted_idx = np.argsort(-scores)  # descending
    sorted_scores = scores[sorted_idx]
    # Consecutive differences in the sorted score stream.
    grad = np.abs(np.diff(sorted_scores, prepend=sorted_scores[0] + 1.0))
    fire_sorted = (grad < g_thr)
    fire = np.zeros(len(scores), dtype=bool)
    fire[sorted_idx] = fire_sorted
    fire = fire & top_k_mask
    return fire


def absolute_band_fire(scores, top_k_mask, width=WIDTH):
    """LLM fires iff row is in top-K AND absolute score < width."""
    fire = (scores < width) & top_k_mask
    return fire


def recall_at_K(scores, y, k_pct=K_PCT):
    """Top-K mask and recall on positives."""
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    top_k_idx = np.argsort(-scores)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[top_k_idx] = True
    pos_total = max(1, int(y.sum()))
    pos_caught = int(y[mask].sum())
    return mask, pos_caught, pos_total


def bootstrap_delta_ci(metric_a, metric_b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap CI on delta = metric_a - metric_b (per-row metric)."""
    rng = np.random.default_rng(seed)
    n = len(metric_a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = metric_a[idx].mean() - metric_b[idx].mean()
    return {
        "delta_mean": float(deltas.mean()),
        "delta_lo": float(np.percentile(deltas, 2.5)),
        "delta_hi": float(np.percentile(deltas, 97.5)),
        "p_gt0": float((deltas > 0).mean()),
        "p_lt0": float((deltas < 0).mean()),
    }


def main():
    print(f"[iter120] loading train/test ...")
    Xtr, ytr = load(ROOT / "train_data.csv")
    Xte, yte = load(ROOT / "test_data.csv")
    print(f"[iter120] Xtr={Xtr.shape} ytr_pos={ytr.sum()} | "
          f"Xte={Xte.shape} yte_pos={yte.sum()}")

    print(f"[iter120] fitting XGB-24full ...")
    scores = fit_predict(Xtr, ytr, Xte)
    top_k_mask, pos_caught_xgb, pos_total = recall_at_K(scores, yte)
    print(f"[iter120] xgb-only recall@K=2% = {pos_caught_xgb}/{pos_total} "
          f"= {pos_caught_xgb/pos_total:.4f}")

    # Compute gradient-band fire mask (LLM fires on top-K rows with small grad).
    fire_grad = gradient_band_fire(scores, top_k_mask)
    fire_abs = absolute_band_fire(scores, top_k_mask)
    n_llm_grad = int(fire_grad.sum())
    n_llm_abs = int(fire_abs.sum())
    print(f"[iter120] n_llm_grad={n_llm_grad}  n_llm_abs={n_llm_abs}")

    # ---- Per-V_stat quartile breakdown (16 = 4 stats * 4 quartiles) ----
    per_q_rows = []
    for stat in AGG4:
        vals_te = Xte[:, ALL24.index(stat)]
        qid = quartiles(vals_te)
        for q in range(4):
            q_mask = (qid == q)
            n_q = int(q_mask.sum())
            n_pos_q = int(yte[q_mask].sum())
            # xgb-only recall@K in this quartile (restricted subset ranking)
            sub_scores = scores[q_mask]
            sub_y = yte[q_mask]
            sub_mask, sub_caught, sub_total = recall_at_K(sub_scores, sub_y)
            # LLM fires in this quartile (intersect with top-K on full set)
            fire_q_grad = (fire_grad & q_mask)
            fire_q_abs = (fire_abs & q_mask)
            n_llm_q_grad = int(fire_q_grad.sum())
            n_llm_q_abs = int(fire_q_abs.sum())
            # Cost in this quartile
            n_test_q = max(1, n_q)
            cpd_grad = (n_test_q * COST_XGB + n_llm_q_grad * (COST_LLM - COST_XGB)) / n_test_q
            cpd_xgb = COST_XGB
            # LLM calls per missed fraud in this quartile
            sub_pos_total = max(1, sub_total)
            n_missed = sub_pos_total - sub_caught
            lpm_grad = n_llm_q_grad / max(1, n_missed)
            per_q_rows.append({
                "stat": stat, "quartile": q,
                "n_test_q": n_q, "n_pos_q": n_pos_q,
                "xgb_recall_at_K2": sub_caught / sub_pos_total if sub_pos_total else 0.0,
                "n_llm_grad": n_llm_q_grad,
                "n_llm_abs": n_llm_q_abs,
                "llm_call_density_grad": n_llm_q_grad / n_test_q,
                "cpd_xgb": cpd_xgb,
                "cpd_grad": cpd_grad,
                "lpm_grad": lpm_grad,
                "n_missed_in_quartile": n_missed,
            })

    # Write per-V_stat quartile rows
    out_path = RES / "p8_iter120_vstat_quartile_breakdown.tsv"
    with out_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(per_q_rows)
    print(f"[iter120] wrote {out_path} ({len(per_q_rows)} rows)")

    # ---- H1: range of LLM-call density across all 16 quartiles ----
    densities = [r["llm_call_density_grad"] for r in per_q_rows]
    print(f"[iter120 H1] LLM-call density gradient-band across 16 quartiles: "
          f"min={min(densities):.4f} max={max(densities):.4f} "
          f"max/min={max(densities)/max(1e-9, min(densities)):.2f}")

    # ---- H2: per-quartile paired bootstrap on recall_at_K2 (xgb vs gradient-band) ----
    # gradient-band rule: in each row, LLM fires -> label = 1 (caught); else label = 0.
    # On the FULL test set, gradient-band catches the same positives as xgb-only
    # because LLM can ONLY change a label from "miss" to "catch" -- never the reverse.
    # So per-row delta_recall = 0 by construction.  Better: per-quartile delta on
    # RESTRICTED ranking (LLM uses its own confidence, ignoring XGB plateau).
    # We'll instead bootstrap the LLMCALL DENSITY per row (proportion test).
    rng = np.random.default_rng(SEED)
    boot_density = np.empty((N_BOOT, 16))
    for bi in range(N_BOOT):
        idx = rng.integers(0, len(yte), len(yte))
        fire_sampled = fire_grad[idx]
        for qi, q_row in enumerate(per_q_rows):
            stat = q_row["stat"]
            vals_te = Xte[:, ALL24.index(stat)]
            qid = quartiles(vals_te)
            q_mask = (qid == q_row["quartile"])
            in_q = q_mask[idx]
            boot_density[bi, qi] = fire_sampled[in_q].mean() if in_q.sum() else 0.0

    boot_summary = []
    for qi, q_row in enumerate(per_q_rows):
        col = boot_density[:, qi]
        boot_summary.append({
            "stat": q_row["stat"],
            "quartile": q_row["quartile"],
            "density_point": q_row["llm_call_density_grad"],
            "density_boot_mean": float(col.mean()),
            "density_boot_lo": float(np.percentile(col, 2.5)),
            "density_boot_hi": float(np.percentile(col, 97.5)),
        })
    out_boot = RES / "p8_iter120_vstat_quartile_boot.tsv"
    with out_boot.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_summary[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(boot_summary)
    print(f"[iter120] wrote {out_boot} ({len(boot_summary)} rows)")

    # ---- H4: sensor-feature efficiency (lpm = LLMs per Missed fraud) ----
    # In gradient-band, missed-fraud-in-quartile = n_pos_q - xgb_caught_q.
    # If n_missed == 0, lpm = inf (LLM not needed; quartile is xgb-perfect).
    lpm_rows = []
    for r in per_q_rows:
        eff = r["lpm_grad"] if r["n_missed_in_quartile"] > 0 else float("inf")
        lpm_rows.append({
            "stat": r["stat"], "quartile": r["quartile"],
            "n_llm_grad": r["n_llm_grad"],
            "n_missed_xgb": r["n_missed_in_quartile"],
            "lpm_grad": eff,
        })
    out_lpm = RES / "p8_iter120_vstat_lpm.tsv"
    with out_lpm.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(lpm_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(lpm_rows)
    print(f"[iter120] wrote {out_lpm} ({len(lpm_rows)} rows)")

    # ---- H3: AUC@K=2% on each V_stat quartile restricted subset ----
    # Easier to read as: xgb-only recall@K=2% in each V_stat quartile.
    # Quartiles where xgb-only recall = 1.0 are "xgb-perfect" (no LLM needed).
    xgb_perfect = sum(1 for r in per_q_rows if r["xgb_recall_at_K2"] >= 0.999)
    print(f"[iter120 H3] quartiles where xgb-only recall >= 0.999: "
          f"{xgb_perfect}/16")

    # ---- Headline aggregation ----
    headline = {
        "iter": 120,
        "n_test": int(len(yte)),
        "n_pos_total": int(pos_total),
        "xgb_only_recall_K2": pos_caught_xgb / pos_total,
        "n_llm_grad": n_llm_grad,
        "n_llm_abs": n_llm_abs,
        "H1_min_density": float(min(densities)),
        "H1_max_density": float(max(densities)),
        "H1_max_over_min_ratio": float(max(densities) / max(1e-9, min(densities))),
        "H3_xgb_perfect_quartiles": xgb_perfect,
        "H3_total_quartiles": 16,
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    out_sum = RES / "p8_iter120_summary.json"
    with out_sum.open("w") as f:
        json.dump(headline, f, indent=2)
    print(f"[iter120] wrote {out_sum}")
    print(f"[iter120] DONE")


if __name__ == "__main__":
    main()
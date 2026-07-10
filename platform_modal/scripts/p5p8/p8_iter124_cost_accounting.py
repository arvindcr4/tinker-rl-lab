#!/usr/bin/env python3
"""P8 JOB A (iter 124): cost-per-decision accounting on the iter-120 V_stat
quartile breakdown + LLM-as-sensor feature ablation.

Fresh vein, not in 137 prior P8 rows.

Cost accounting
---------------
iter-108, iter-112, iter-116, iter-120 reported a single LLM-call
price point (COST_LLM=$0.001/call, COST_XGB=$0.0001/call, ratio 10x).
Real LLM API prices vary by 500x:
  - GPT-4-class:    $0.03 / call
  - mid-tier LLM:   $0.005 / call
  - small open LLM: $0.0006 / call
  - heuristic rule: $0.0001 / call (xgb-only baseline)

iter-124 sweeps 5 LLM price tiers (10x down to 10x up) and reports
the per-V_stat quartile:
  - cpd (cost per DECISION):     total $ / n_test_q
  - cppr (cost per POSITIVE      total $ / n_pos_recalled
          RECALLED):
  - acd (average cost ratio):    cpd(grad-band) / cpd(xgb-only)
  - sweet-spot price:            max(LLM price) where grad-band is still
                                 cheaper than xgb-only at cppr

LLM-as-sensor feature ablation
------------------------------
iter-108/120 used 24 features = 20 raw V1..V20 + 4 aggregates
(V_mean, V_std, V_max, V_min). iter-124 retrains XGB on:
  - 20raw       : drop the 4 aggregates (cleaner PCA-only backbone)
  - 20raw+minmax: keep V_min and V_max only (extreme-tail backbone)
  - 20raw+stat  : keep V_mean and V_std only (central-tendency backbone)
  - 24full      : the iter-108/120 anchor

For each backbone, recompute (a) xgb-only recall@K=2%, (b) LLM
gradient-band fires, (c) lpm ratio, (d) cost-per-positive-recalled.

Falsifiable headlines
---------------------
H1 -- The gradient-band rule is CHEAPER than xgb-only (cppr < cpd_xgb)
  at every LLM price tier tested.  Null hypothesis:  at the
  realistic $0.005/call LLM tier, grad-band cost exceeds xgb-only.

H2 -- 5-tier cost sweep: per-V_stat quartile, the
  worst-cost cell is (stat, quartile) = ?.  If the worst-cost cell is
  the same at every price tier, then the iter-80 rule has a
  stable cost structure; if it shifts across tiers, the rule is
  price-sensitive.

H3 -- LLM-as-sensor feature ablation: dropping the 4 aggregate
  features from training (20raw) does NOT change the gradient-band
  firing pattern (Mann-Whitney U on n_llm_grad).

H4 -- Sweet-spot price: the maximum LLM price at which
  cppr(grad-band) <= cppr(xgb-only).  Reported per V_stat quartile.

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
SEED = 20260705
N_BOOT = 1500
COST_XGB = 0.0001   # per-decision XGB inference cost ($)
K_PCT = 2.0
G_THR = 0.001       # iter-80 gradient-band threshold
WIDTH = 0.50        # iter-80 absolute-band threshold

# 5 LLM price tiers, $ per call.  Real 2026 prices vary 500x.
LLM_PRICE_TIERS = [
    ("cheap_heuristic", 0.0001),
    ("small_open",      0.0006),
    ("iter120_default", 0.0010),
    ("mid_tier",        0.0050),
    ("frontier_gpt4",   0.0300),
]

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
FEATURE_SETS = {
    "24full":       ALL24,
    "20raw":        RAW20,
    "20raw+minmax": RAW20 + ["V_min", "V_max"],
    "20raw+stat":   RAW20 + ["V_mean", "V_std"],
}


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


def fit_predict(Xtr, ytr, Xte, feats):
    """Fit XGB on selected feature subset; return test scores."""
    cols = [ALL24.index(c) for c in feats]
    Xtr_s = Xtr[:, cols]
    Xte_s = Xte[:, cols]
    n_pos_tr = max(1, int(ytr.sum()))
    n_neg_tr = max(1, len(ytr) - n_pos_tr)
    spw = n_neg_tr / n_pos_tr
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=SEED, n_jobs=4,
    )
    m.fit(Xtr_s, ytr)
    return m.predict_proba(Xte_s)[:, 1]


def quartiles(values):
    qs = np.quantile(values, [0.25, 0.50, 0.75])
    out = np.zeros(len(values), dtype=int)
    out[values > qs[0]] = 1
    out[values > qs[1]] = 2
    out[values > qs[2]] = 3
    return out


def gradient_band_fire(scores, top_k_mask, g_thr=G_THR):
    sorted_idx = np.argsort(-scores)
    sorted_scores = scores[sorted_idx]
    grad = np.abs(np.diff(sorted_scores, prepend=sorted_scores[0] + 1.0))
    fire_sorted = (grad < g_thr)
    fire = np.zeros(len(scores), dtype=bool)
    fire[sorted_idx] = fire_sorted
    return fire & top_k_mask


def recall_at_K(scores, y, k_pct=K_PCT):
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    top_k_idx = np.argsort(-scores)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[top_k_idx] = True
    pos_total = max(1, int(y.sum()))
    pos_caught = int(y[mask].sum())
    return mask, pos_caught, pos_total


def mann_whitney_u(a, b):
    """One-sided MWU: probability that samples from `a` are stochastically
    greater than samples from `b`.  Approximation: normal w/ tie correction."""
    a = np.asarray(a); b = np.asarray(b)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    combined = np.concatenate([a, b])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=float)
    # average ranks for ties
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[order[j]] == combined[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0  # 1-based average rank
        ranks[order[i:j]] = avg
        i = j
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (U1 - mu) / sigma if sigma > 0 else 0.0
    return float(U1), float(z)


def bootstrap_acd(fire_grad, n_test_q, cost_xgb=COST_XGB, cost_llm=0.0010,
                  n_boot=N_BOOT, seed=SEED):
    """Bootstrap the average cost ratio cpd_grad/cpd_xgb at a price tier."""
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    cpd_xgb = cost_xgb
    for bi in range(n_boot):
        idx = rng.integers(0, len(fire_grad), len(fire_grad))
        n_llm_s = int(fire_grad[idx].sum())
        cpd_grad = (n_test_q * cost_xgb + n_llm_s * (cost_llm - cost_xgb)) / n_test_q
        ratios[bi] = cpd_grad / cpd_xgb
    return float(ratios.mean()), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def sweet_spot_price(fire_grad, n_test_q, n_pos_recalled_xgb,
                     cost_xgb=COST_XGB, n_boot=N_BOOT, seed=SEED):
    """Return the max LLM price at which cppr(grad-band) <= cppr(xgb-only).

    cppr_xgb  = n_test_q * cost_xgb / n_pos_recalled_xgb
    cppr_grad = (n_test_q * cost_xgb + n_llm * (cost_llm - cost_xgb))
              / n_pos_recalled_combined

    Closed-form: cppr_grad <= cppr_xgb  iff
       cost_llm <= cost_xgb * n_pos_recalled_combined / n_llm
    """
    n_llm = int(fire_grad.sum())
    if n_llm == 0:
        return float("inf")  # grad-band never fires: always sweet
    pos_caught = n_pos_recalled_xgb  # grad-band can only ADD recall
    return float(cost_xgb * pos_caught / n_llm)


def main():
    print(f"[iter124] loading train/test ...")
    Xtr, ytr = load(ROOT / "train_data.csv")
    Xte, yte = load(ROOT / "test_data.csv")
    print(f"[iter124] Xtr={Xtr.shape} ytr_pos={ytr.sum()} | "
          f"Xte={Xte.shape} yte_pos={yte.sum()}")

    # ----------------------------------------------------------------
    # Sweep feature sets (LLM-as-sensor feature ablation backbone)
    # ----------------------------------------------------------------
    feat_summary = {}
    fire_per_feat = {}
    scores_per_feat = {}
    for fset_name, feats in FEATURE_SETS.items():
        print(f"[iter124] fitting {fset_name} ({len(feats)} feats) ...")
        scores = fit_predict(Xtr, ytr, Xte, feats)
        scores_per_feat[fset_name] = scores
        top_k_mask, pos_caught_xgb, pos_total = recall_at_K(scores, yte)
        fire_grad = gradient_band_fire(scores, top_k_mask)
        fire_per_feat[fset_name] = fire_grad
        n_llm = int(fire_grad.sum())
        feat_summary[fset_name] = {
            "n_feats": len(feats),
            "xgb_only_recall_K2": pos_caught_xgb / pos_total,
            "n_llm_grad": n_llm,
            "n_pos_total": int(pos_total),
            "n_pos_caught_xgb": int(pos_caught_xgb),
        }
        print(f"[iter124]   xgb-only recall@K=2% = {pos_caught_xgb}/{pos_total} "
              f"= {pos_caught_xgb/pos_total:.4f} | n_llm_grad={n_llm}")

    # ----------------------------------------------------------------
    # H3 -- MWU test: does 20raw have same n_llm_grad distribution as 24full?
    # Bootstrap resample: re-fit on bootstrap sample of training rows.
    # Cheaper: compare the LLM-call pattern on TEST data (paired).
    # ----------------------------------------------------------------
    h3_rows = []
    f_anchor = "24full"
    anchor_fire = fire_per_feat[f_anchor]
    for fset_name in FEATURE_SETS:
        if fset_name == f_anchor:
            continue
        f_fire = fire_per_feat[fset_name]
        U, z = mann_whitney_u(anchor_fire.astype(float), f_fire.astype(float))
        # both fire masks are boolean; MWU reduces to sign-rank of differences
        n_both = int((anchor_fire & f_fire).sum())
        n_anchor_only = int((anchor_fire & ~f_fire).sum())
        n_test_only = int((~anchor_fire & f_fire).sum())
        n_neither = int((~anchor_fire & ~f_fire).sum())
        agreement = (n_both + n_neither) / len(yte)
        h3_rows.append({
            "fset": fset_name,
            "n_fires_anchor": int(anchor_fire.sum()),
            "n_fires_fset": int(f_fire.sum()),
            "n_both": n_both,
            "n_anchor_only": n_anchor_only,
            "n_test_only": n_test_only,
            "n_neither": n_neither,
            "agreement": agreement,
            "mwu_U": U,
            "mwu_z": z,
        })

    out_h3 = RES / "p8_iter124_feature_ablation.tsv"
    with out_h3.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(h3_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(h3_rows)
    print(f"[iter124] wrote {out_h3} ({len(h3_rows)} rows)")

    # ----------------------------------------------------------------
    # H1, H2 -- 5-tier cost sweep on the 24full backbone
    # (matches iter-120 anchor); per-V_stat quartile.
    # ----------------------------------------------------------------
    fset_name = "24full"
    scores = scores_per_feat[fset_name]
    fire_grad = fire_per_feat[fset_name]
    top_k_mask, pos_caught_xgb, pos_total = recall_at_K(scores, yte)
    n_test = len(yte)

    # Per-V_stat quartile, 5 price tiers.  For each (stat, quartile, tier),
    # record cpd, cppr, acd.  Then per tier find the worst-cost quartile.
    sweep_rows = []
    sweet_per_q = {}
    for stat in AGG4:
        vals_te = Xte[:, ALL24.index(stat)]
        qid = quartiles(vals_te)
        for q in range(4):
            q_mask = (qid == q)
            n_q = max(1, int(q_mask.sum()))
            n_pos_q = int(yte[q_mask].sum())
            sub_scores = scores[q_mask]
            sub_y = yte[q_mask]
            sub_mask, sub_caught, sub_total = recall_at_K(sub_scores, sub_y)
            n_missed_q = max(1, sub_total - sub_caught)
            fire_q = (fire_grad & q_mask)
            n_llm_q = int(fire_q.sum())
            # Sweet-spot price (closed-form)
            sweet = sweet_spot_price(fire_q, n_q, sub_caught, cost_xgb=COST_XGB)
            sweet_per_q[(stat, q)] = sweet
            for tier_name, cost_llm in LLM_PRICE_TIERS:
                cpd_xgb_q = COST_XGB
                cpd_grad_q = (n_q * COST_XGB + n_llm_q * (cost_llm - COST_XGB)) / n_q
                cppr_xgb_q = (n_q * COST_XGB) / max(1, sub_caught)
                cppr_grad_q = (n_q * COST_XGB + n_llm_q * (cost_llm - COST_XGB)) / max(1, sub_caught)
                acd_q = cpd_grad_q / cpd_xgb_q
                sweep_rows.append({
                    "stat": stat,
                    "quartile": q,
                    "tier": tier_name,
                    "cost_llm_per_call": cost_llm,
                    "n_test_q": n_q,
                    "n_pos_q": n_pos_q,
                    "xgb_caught_q": sub_caught,
                    "n_llm_q": n_llm_q,
                    "cpd_xgb": cpd_xgb_q,
                    "cpd_grad": cpd_grad_q,
                    "cppr_xgb": cppr_xgb_q,
                    "cppr_grad": cppr_grad_q,
                    "acd": acd_q,
                    "sweet_spot_price": sweet,
                })

    out_sweep = RES / "p8_iter124_cost_sweep.tsv"
    with out_sweep.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(sweep_rows)
    print(f"[iter124] wrote {out_sweep} ({len(sweep_rows)} rows)")

    # Per-tier worst-cost quartile.
    worst_per_tier = {}
    for tier_name, _ in LLM_PRICE_TIERS:
        rows_t = [r for r in sweep_rows if r["tier"] == tier_name]
        worst = max(rows_t, key=lambda r: r["cppr_grad"] / max(1e-9, r["cppr_xgb"]))
        worst_per_tier[tier_name] = worst
        print(f"[iter124 H2] worst-cost at tier {tier_name}: "
              f"{worst['stat']} Q{worst['quartile']} "
              f"cppr_grad/cppr_xgb={worst['cppr_grad']/max(1e-9,worst['cppr_xgb']):.3f}")

    # ----------------------------------------------------------------
    # H1 -- sweep: at every LLM tier tested, is grad-band CHEAPER
    # at cppr (cost per positive recalled) than xgb-only?
    # Note: in 24full, grad-band may NOT add recall on top of xgb-only
    # (because both select top-K AND xgb-only uses xgb score to rank);
    # grad-band just fires LLM on rows where xgb score plateaued.
    # Since grad-band fires on TOP-K rows already, it doesn't add new
    # positives -- cppr changes only via the LLM-call cost.
    # So H1 here is whether cppr(grad-band) <= cppr(xgb-only) for any
    # tier: equivalent to n_llm_q * (cost_llm - cost_xgb) <= 0,
    # i.e., only when cost_llm <= cost_xgb.  Let's report this honestly.
    # ----------------------------------------------------------------
    h1_verdict = {}
    for tier_name, cost_llm in LLM_PRICE_TIERS:
        rows_t = [r for r in sweep_rows if r["tier"] == tier_name]
        # global: total cost / total caught across all 16 cells
        total_cost_xgb = sum(r["n_test_q"] * COST_XGB for r in rows_t) / 16  # average
        total_cost_grad = sum(r["n_test_q"] * COST_XGB + r["n_llm_q"] * (cost_llm - COST_XGB) for r in rows_t) / 16
        total_caught = sum(r["xgb_caught_q"] for r in rows_t)
        cppr_xgb_g = total_cost_xgb * 16 / max(1, total_caught)
        cppr_grad_g = total_cost_grad * 16 / max(1, total_caught)
        h1_verdict[tier_name] = {
            "cost_llm_per_call": cost_llm,
            "cppr_xgb_global": cppr_xgb_g,
            "cppr_grad_global": cppr_grad_g,
            "grad_cheaper": cppr_grad_g <= cppr_xgb_g,
            "ratio_cppr_grad_over_xgb": cppr_grad_g / max(1e-9, cppr_xgb_g),
        }
        print(f"[iter124 H1] tier={tier_name} cost_llm=${cost_llm:.4f}: "
              f"cppr_xgb={cppr_xgb_g:.6f} cppr_grad={cppr_grad_g:.6f} "
              f"grad_cheaper={cppr_grad_g <= cppr_xgb_g}")

    # ----------------------------------------------------------------
    # H4 -- Sweet-spot price per (stat, quartile).
    # ----------------------------------------------------------------
    sweet_rows = []
    for (stat, q), price in sweet_per_q.items():
        sweet_rows.append({"stat": stat, "quartile": q, "sweet_spot_price_per_call": price})
    out_sweet = RES / "p8_iter124_sweet_spot.tsv"
    with out_sweet.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(sweet_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(sweet_rows)
    print(f"[iter124] wrote {out_sweet} ({len(sweet_rows)} rows)")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    summary = {
        "iter": 124,
        "n_test": int(n_test),
        "n_pos_total": int(pos_total),
        "feat_summary": feat_summary,
        "h3_feature_ablation": h3_rows,
        "h1_5tier_verdict": h1_verdict,
        "h2_worst_quartile_per_tier": {
            t: {"stat": w["stat"], "quartile": w["quartile"],
                "cppr_grad": w["cppr_grad"], "cppr_xgb": w["cppr_xgb"]}
            for t, w in worst_per_tier.items()
        },
        "llm_price_tiers": [{"tier": n, "cost_per_call": c} for n, c in LLM_PRICE_TIERS],
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    out_sum = RES / "p8_iter124_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter124] wrote {out_sum}")
    print(f"[iter124] DONE")


if __name__ == "__main__":
    main()
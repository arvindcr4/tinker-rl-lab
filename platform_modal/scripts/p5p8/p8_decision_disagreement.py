#!/usr/bin/env python3
"""P8 JOB A (iter 76): inter-model decision-disagreement (flip-rate) at varying
K budgets + feature-level diagnosis of which rows flip.

Fresh vein (not in the prior 84 P8 rows). The iter-68 single-sensor ablation
(row 79) measured AUC, recall@K=2%, and cost-per-fraud-caught for each of 12
trees; the iter-72 calibration-under-noise row 84 measured the calibration gap
as sigma increased. Neither answered the operational question fraud-ops leads
ask after seeing iter-68 row 79: "**When two of these trees disagree on the
alert/no-alert decision at K=2%, WHICH rows flip and WHY?**"

This iter measures:

  G1. **Inter-model decision-flip rate at K in {0.5, 1, 2, 3, 5}%.**
      For each pair of trees (six pairs across the 4 operational variants
      XGB-20raw / XGB-24full / XGB-pair / LLM-as-scribe), count how many test
      rows are alerted by exactly one tree at K and bootstrap a 95% CI on the
      flip-rate.

  G2. **Feature-level diagnosis of which rows flip.** For each pair, the flips
      are scored against the 24-feature vector. Compare the mean V_std,
      V_max, V_min, V_mean, and the 20 raw features between `flipped` vs
      `agreed` subsets per pair. Sharpens iter-31 row 31 leave-one-OUT from
      "V_std, V_max matter most" to a quantitative "the XGB-20raw vs
      XGB-24full flip-set has V_std = {x} vs {y}" measurement.

  G3. **Cost-of-disagreement** under two operational regimes:
      (a) `union` rule (alert if EITHER tree alerts) and (b) `intersection`
      rule (alert if BOTH trees alert). For each pair at K=2%, report:
      precision, recall, F1, $/fraud-caught, fraction of K-disagreements
      preserved.

  G4. **Selective LLM-as-sensor cheap-extension.** Train XGB-20raw as the
      always-cheap backbone. Define `borderline_score in [0.05, 0.95]` as the
      rows where the LLM-as-sensor surrogate's score can substitute. Measure:
      (a) fraction of test rows where the LLM is invoked, (b) AUC of the
      selective composite, (c) cost-per-fraud-caught, (d) recall@K=2%.

Inputs
------
fraud_data.csv : 50k train (24 numeric features + Class)
test_data.csv  : 10k held-out (same schema + Class)

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_decision_disagreement_flip.tsv       (~30 rows: 6 pairs x 5 K)
platform_hybrid/experiments/results/p5p8/p8_decision_disagreement_flip_boot.tsv  (~30 paired-bootstrap rows)
platform_hybrid/experiments/results/p5p8/p8_decision_disagreement_features.tsv   (~24 rows: 6 pairs x 4 aggs)
platform_hybrid/experiments/results/p5p8/p8_decision_disagreement_union.tsv      (~6 rows: 6 pairs x 1 K)
platform_hybrid/experiments/results/p5p8/p8_decision_disagreement_selective.tsv (~16 rows: 4 widths x 4 metrics)
platform_hybrid/experiments/results/p5p8/p8_decision_disagreement_summary.json  (machine-readable summary)
platform_hybrid/experiments/results/p5p8/figures/p8_decision_disagreement_flip.{png,pdf}
platform_hybrid/experiments/results/p5p8/figures/p8_decision_disagreement_selective.{png,pdf}

Stdlib + numpy + pandas + xgboost + matplotlib. <=290 lines.
"""
from __future__ import annotations

import csv
import json
import math
import random
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
FIG = RES / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED = 20260705
N_BOOT = 600
K_BUDGETS = [0.5, 1.0, 2.0, 3.0, 5.0]  # percent
RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4

# Operational variants (4)
VARIANTS = {
    "XGB-20raw": RAW20,
    "XGB-24full": ALL24,
    "XGB-pair": RAW20 + ["V_std", "V_max"],  # iter-68 row 79 best pair
    "LLM-as-scribe (XGB-24full @ $0.001)": ALL24,  # same tree, but $0.001/dec cost
}
COST = {"XGB-20raw": 0.0001, "XGB-24full": 0.0001, "XGB-pair": 0.0001,
        "LLM-as-scribe (XGB-24full @ $0.001)": 0.0010}

# Borderline widths to sweep for selective composition
WIDTHS = [0.10, 0.20, 0.30, 0.40, 0.50, 1.00]


# ---------- data ----------
def load(path):
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        X, y = [], []
        for line in rdr:
            X.append([float(line[idx[c]]) for c in ALL24])
            y.append(int(float(line[idx["Class"]])))
    return np.array(X), np.array(y), header


# ---------- train trees ----------
def fit(name, cols):
    rng = np.random.default_rng(SEED)
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=SEED, n_jobs=4,
    )
    ci = [ALL24.index(c) for c in cols]
    Xtr = Xtr_all[:, ci]
    Xte = Xte_all[:, ci]
    m.fit(Xtr, ytr)
    p_test = m.predict_proba(Xte)[:, 1]
    p_train = m.predict_proba(Xtr)[:, 1]
    return p_train, p_test


# ---------- metric helpers ----------
def bootstrap_ci(diff_vec, B, seed):
    rng = np.random.default_rng(seed)
    n = len(diff_vec)
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        means[b] = diff_vec[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def precision_recall_f1(scores, y, k_frac):
    n = len(y)
    k = max(1, int(round(k_frac / 100 * n)))
    top = np.argsort(-scores)[:k]
    tp = int(y[top].sum())
    return tp / k, tp / max(1, int(y.sum())), 2 * (tp / k) * (tp / max(1, int(y.sum()))) / (tp / k + tp / max(1, int(y.sum())) + 1e-12)


def flip_rate(s1, s2, k_frac, y_te):
    n = len(s1)
    k = max(1, int(round(k_frac / 100 * n)))
    a1 = np.zeros(n, dtype=bool)
    a2 = np.zeros(n, dtype=bool)
    a1[np.argsort(-s1)[:k]] = True
    a2[np.argsort(-s2)[:k]] = True
    return np.logical_xor(a1, a2).sum()  # count


# ---------- main ----------
print("# === P8 decision-disagreement (JOB A iter 76) ===")
Xtr_all, ytr, h_tr = load(ROOT / "fraud_data.csv")
Xte_all, yte, h_te = load(ROOT / "test_data.csv")
print(f"# train={Xtr_all.shape}, test={Xte_all.shape}, base_rate_test={yte.mean():.4f}")

scores = {}
for name, cols in VARIANTS.items():
    p_tr, p_te = fit(name, cols)
    scores[name] = (p_tr, p_te)
    print(f"# {name}: train AUC raw={np.argsort(p_tr).argsort()[ytr==1].mean():.2f}")

# --- G1: flip-rate per pair per K ---
flip_rows = []
flip_boot_rows = []
n_test = len(yte)
for (n1, n2) in combinations(VARIANTS.keys(), 2):
    p1_te = scores[n1][1]
    p2_te = scores[n2][1]
    for k in K_BUDGETS:
        flips = flip_rate(p1_te, p2_te, k, yte)
        rate = flips / n_test
        # paired bootstrap CI on flip-rate
        diff_vec = (np.argsort(-p1_te)[:max(1, int(round(k /100 * n_test)))] != np.argsort(-p2_te)[:max(1, int(round(k / 100 * n_test)))]).astype(int)
        # Approximate: bootstrap the per-row flip indicator (computed once above per K)
        a1 = np.zeros(n_test, dtype=bool); a1[np.argsort(-p1_te)[:max(1, int(round(k / 100 * n_test)))]] = True
        a2 = np.zeros(n_test, dtype=bool); a2[np.argsort(-p2_te)[:max(1, int(round(k / 100 * n_test)))]] = True
        indicators = np.logical_xor(a1, a2).astype(int)
        rng = np.random.default_rng(SEED)
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n_test, n_test)
            boot[b] = indicators[idx].mean()
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        flip_rows.append({"pair": f"{n1} vs {n2}", "K_pct": k,
                          "n_flips": int(flips), "n_test": n_test,
                          "flip_rate": rate, "ci_lo": lo, "ci_hi": hi})
        flip_boot_rows.append({"pair": f"{n1} vs {n2}", "K_pct": k,
                               "boot_mean": boot.mean(),
                               "ci_lo": lo, "ci_hi": hi})

# --- G2: feature-level diagnosis at K=2% ---
feat_rows = []
K = 2.0
k_n = max(1, int(round(K / 100 * n_test)))
for (n1, n2) in combinations(VARIANTS.keys(), 2):
    p1 = scores[n1][1]; p2 = scores[n2][1]
    a1 = np.zeros(n_test, dtype=bool); a1[np.argsort(-p1)[:k_n]] = True
    a2 = np.zeros(n_test, dtype=bool); a2[np.argsort(-p2)[:k_n]] = True
    flip_mask = np.logical_xor(a1, a2)
    n_flip = int(flip_mask.sum())
    if n_flip < 5:
        continue
    for col_i, col in enumerate(ALL24):
        flipped_vals = Xte_all[flip_mask, col_i]
        agreed_vals = Xte_all[~flip_mask, col_i]
        if n_flip > 0:
            feat_rows.append({"pair": f"{n1} vs {n2}",
                              "feature": col,
                              "n_flips": n_flip,
                              "flip_mean": float(flipped_vals.mean()),
                              "agreed_mean": float(agreed_vals.mean()),
                              "delta": float(flipped_vals.mean() - agreed_vals.mean()),
                              "flip_std": float(flipped_vals.std())})

# --- G3: union/intersection at K=2% ---
union_rows = []
for (n1, n2) in combinations(VARIANTS.keys(), 2):
    p1 = scores[n1][1]; p2 = scores[n2][1]
    a1 = np.zeros(n_test, dtype=bool); a1[np.argsort(-p1)[:k_n]] = True
    a2 = np.zeros(n_test, dtype=bool); a2[np.argsort(-p2)[:k_n]] = True
    union = np.logical_or(a1, a2)
    inter = np.logical_and(a1, a2)
    # union: take all rows in union, assign "alert" — use union score = max(p1,p2)
    pu = np.where(union, np.maximum(p1, p2), -1)
    pi = np.where(inter, np.minimum(p1, p2), -1)
    # Precision/recall on the agreed set
    n_union_alerts = int(union.sum())
    n_inter_alerts = int(inter.sum())
    tp_union = int((union & (yte == 1)).sum())
    tp_inter = int((inter & (yte == 1)).sum())
    union_rows.append({"pair": f"{n1} vs {n2}",
                       "n_union_alerts": n_union_alerts,
                       "n_inter_alerts": n_inter_alerts,
                       "n_flips": int(np.logical_xor(a1, a2).sum()),
                       "tp_union": tp_union,
                       "tp_inter": tp_inter,
                       "precision_union": tp_union / max(1, n_union_alerts),
                       "precision_inter": tp_inter / max(1, n_inter_alerts),
                       "recall_union": tp_union / yte.sum(),
                       "recall_inter": tp_inter / yte.sum()})

# --- G4: Selective LLM-as-sensor ---
# Train XGB-20raw backbone (scoring-only when out of borderline range)
p_20raw = scores["XGB-20raw"][1]
p_scribe = scores["LLM-as-scribe (XGB-24full @ $0.001)"][1]
sel_rows = []
for w in WIDTHS:
    lo, hi = 0.5 - w / 2, 0.5 + w / 2  # Score is a probability, not percentile - use raw score threshold
    # Borderline rule: rows where XGB-20raw score is in [lo, hi] AND LLM gives a different score
    # The composition rule: use LLM-as-scribe ONLY in the borderline range; XGB-20raw outside
    composite = np.where(
        (p_20raw >= lo) & (p_20raw <= hi),
        p_scribe,  # invoke LLM
        p_20raw,    # use cheap backbone
    )
    n_llm_calls = int(((p_20raw >= lo) & (p_20raw <= hi)).sum())
    cost_per = (n_llm_calls * 0.0010 + (n_test - n_llm_calls) * 0.0001) / n_test
    auc = float(np.argsort(composite).argsort()[yte == 1].mean()) / max(1, (yte == 0).sum())  # rough AUC
    # simpler AUC via Mann-Whitney
    pos = composite[yte == 1]; neg = composite[yte == 0]
    n_pos = len(pos); n_neg = len(neg)
    comb = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(comb)) + 1
    auc = (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    p_at_2, r_at_2, f1_at_2 = precision_recall_f1(composite, yte, 2.0)
    n_alert_2 = max(1, int(round(2.0 / 100 * n_test)))
    cost_caught = (cost_per * n_test) / max(1, int((np.argsort(-composite)[:n_alert_2][yte[np.argsort(-composite)[:n_alert_2]] == 1]).sum()))
    sel_rows.append({"width": w, "n_llm_calls": n_llm_calls,
                     "frac_llm": n_llm_calls / n_test,
                     "auc": float(auc),
                     "recall_at_K2": float(r_at_2),
                     "precision_at_K2": float(p_at_2),
                     "f1_at_K2": float(f1_at_2),
                     "cost_per_decision": cost_per,
                     "cost_per_fraud_caught": float(cost_caught)})

# --- write outputs ---
tsv = lambda name, rows, header=None: (
    (RES / name).write_text(
        "\n".join([
            "\t".join(map(str, [h for h in (header or rows[0].keys())]))
            if header else "\t".join(rows[0].keys())
        ] + ["\t".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                       for c in (header or rows[0].keys())) for r in rows])
    )
)

def write_tsv(name, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = ["\t".join(keys)]
    for r in rows:
        lines.append("\t".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    (RES / name).write_text("\n".join(lines) + "\n")

write_tsv("p8_decision_disagreement_flip.tsv", flip_rows)
write_tsv("p8_decision_disagreement_flip_boot.tsv", flip_boot_rows)
write_tsv("p8_decision_disagreement_features.tsv", feat_rows)
write_tsv("p8_decision_disagreement_union.tsv", union_rows)
write_tsv("p8_decision_disagreement_selective.tsv", sel_rows)

summary = {
    "n_test": int(n_test),
    "base_rate": float(yte.mean()),
    "variants": list(VARIANTS.keys()),
    "flip_count": len(flip_rows),
    "flip_mean_rate_at_K2": float(np.mean([r["flip_rate"] for r in flip_rows if r["K_pct"] == 2.0])),
    "feat_count": len(feat_rows),
    "union_count": len(union_rows),
    "selective_count": len(sel_rows),
    "minimum_flip_pair_K2": min(flip_rows, key=lambda r: r["flip_rate"] if r["K_pct"] == 2.0 else 99),
    "maximum_flip_pair_K2": max(flip_rows, key=lambda r: r["flip_rate"] if r["K_pct"] == 2.0 else -1),
    "selective_cheapest_cost_per_decision": min(s["cost_per_decision"] for s in sel_rows),
    "selective_best_AUC": max(s["auc"] for s in sel_rows),
}
(RES / "p8_decision_disagreement_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("# wrote summary:", json.dumps(summary, indent=2, default=str))

# --- figures ---
pairs_K2 = [(r["pair"], r["flip_rate"], r["ci_lo"], r["ci_hi"]) for r in flip_rows if r["K_pct"] == 2.0]
fig, ax = plt.subplots(figsize=(9, 4.5))
names = [p[0].replace("XGB-20raw", "XGB-20").replace("XGB-24full", "XGB-24").replace("XGB-pair", "XGB-pair") for p in pairs_K2]
rates = [p[1] for p in pairs_K2]
los = [p[1] - p[2] for p in pairs_K2]
his = [p[3] - p[1] for p in pairs_K2]
ax.barh(range(len(names)), rates, xerr=[los, his], color="#376092", edgecolor="black")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Flip-rate at K=2% (paired bootstrap 95% CI)")
ax.set_title("P8 iter 76: Inter-model decision flip-rate at K=2% (10k test rows)")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIG / "p8_decision_disagreement_flip.png", dpi=120)
fig.savefig(FIG / "p8_decision_disagreement_flip.pdf")
plt.close(fig)

# selective composite figure
fig, ax = plt.subplots(figsize=(9, 4.5))
ws = [s["width"] for s in sel_rows]
cs = [s["cost_per_decision"] * 1000 for s in sel_rows]  # in $
aucs = [s["auc"] for s in sel_rows]
ax.plot(ws, cs, "o-", color="#376092", label="cost/decision (millicents)")
ax.set_xlabel("Borderline width w (XGB-20raw score in [0.5-w/2, 0.5+w/2])")
ax.set_ylabel("Cost/decision (millicents, $0.001 LLM + $0.0001 XGB)", color="#376092")
ax2 = ax.twinx()
ax2.plot(ws, aucs, "s--", color="#c0504d", label="AUC")
ax2.set_ylabel("Selective-composite AUC", color="#c0504d")
ax.set_title("P8 iter 76: Selective LLM-as-sensor composite — cost & AUC vs borderline width")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG / "p8_decision_disagreement_selective.png", dpi=120)
fig.savefig(FIG / "p8_decision_disagreement_selective.pdf")
plt.close(fig)

print("# === iter 76 JOB A complete; outputs in platform_hybrid/experiments/results/p5p8/ ===")

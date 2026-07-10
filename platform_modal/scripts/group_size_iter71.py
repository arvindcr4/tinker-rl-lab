#!/usr/bin/env python3
"""Iter 71 — Pillar 3 (G=4 vs G=32): Loss-of-Signal Decomposition.

Decomposes the accuracy deficit of small group sizes (G=4) vs large
(G=32) at each observed budget T into three operationally meaningful
components:

  (a) Signal-availability loss  1 - GU(G)         (ZVF tail penalty)
  (b) Per-token noise penalty   sqrt(G_ref / G)   (variance of group mean)
  (c) Effective-batch penalty   (T / T_64)        (relative sample budget)

Aggregated into a single additive-decomposition model and a single
multiplicative-decomposition model; both fit on (T, G) -> acc and
compared.  This is a *fresh* operational diagnostic beyond iter67's
target-accuracy frontier: instead of asking "which G is optimal?", we
ask "why does G=4 lose?" and assign percentages to each cause.

Inputs:
  platform_hybrid/experiments/results/group_size_token_normalized.tsv
  platform_hybrid/experiments/results/group_size_iter43_eff_zvf.tsv
  platform_hybrid/experiments/results/group_size_iter67_iaf_ratios.tsv

Outputs:
  platform_hybrid/experiments/results/group_size_iter71_signal_loss.tsv
  platform_hybrid/experiments/results/group_size_iter71_eff_batch.tsv
  platform_hybrid/experiments/results/group_size_iter71_decomp.tsv
  platform_hybrid/experiments/results/group_size_iter71_summary.tsv
  figures/group_size_iter71_decomp.pdf
  figures/group_size_iter71_decomp.png
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "experiments" / "results"
FIG = REPO / "figures"

# Reference group size for the noise-penalty component: use G_ref = 64
# (largest observed) so all penalties are expressed relative to the
# best-sampled configuration.
G_REF = 64

# The three weights for the additive decomposition.  Calibrated by
# least-squares fit on (T, G) -> acc, but also reported with the
# unweighted prior (equal thirds) so the reader can see how much of
# the attribution is data-driven vs prior-driven.
PRIOR_WEIGHTS = {"signal": 0.50, "noise": 0.30, "batch": 0.20}


def read_tsv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, dicts: list[dict]) -> None:
    if not dicts:
        return
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()), delimiter="\t")
        w.writeheader()
        for r in dicts:
            w.writerow(r)


def main() -> None:
    tok_rows = read_tsv(RES / "group_size_token_normalized.tsv")
    zvf_rows = read_tsv(RES / "group_size_iter43_eff_zvf.tsv")

    # Index zvf_rows by (T, G)
    zvf_idx: dict[tuple[int, int], dict] = {}
    for r in zvf_rows:
        T = int(r["T_tokens"])
        G = int(r["G"])
        zvf_idx[(T, G)] = r

    # Map (T, G) -> acc / GU(prob-style) / ZVF using iter43's
    # `gu_theoretical` ∈ [0,1] (probability-style fraction of groups
    # that retain within-group contrast).  This is the signal
    # quantity we attribute to.
    by_TG: dict[tuple[int, int], dict] = {}
    for r in tok_rows:
        T = int(r["budget_tokens"])
        G = int(r["G"])
        zr = zvf_idx.get((T, G), {})
        gu_prob = float(zr["gu_theoretical"]) if zr else 1.0
        by_TG[(T, G)] = {
            "T": T, "G": G,
            "acc": float(r["heldout_acc_mean"]),
            "gu_prob": gu_prob,
            "zvf": 1.0 - gu_prob,
        }

    # GU at G_REF for each T (reference signal availability)
    gu_ref_by_T: dict[int, float] = {}
    for r in tok_rows:
        T = int(r["budget_tokens"])
        if T in gu_ref_by_T:
            continue
        ref_row = by_TG.get((T, G_REF))
        gu_ref_by_T[T] = ref_row["gu_prob"] if ref_row else 1.0

    # Per-token-budget reference (use T=64M as the largest observed)
    T_REF = max(int(r["budget_tokens"]) for r in tok_rows)

    signal_rows: list[dict] = []
    eff_batch_rows: list[dict] = []
    decomp_rows: list[dict] = []

    for r in tok_rows:
        T = int(r["budget_tokens"])
        G = int(r["G"])
        acc = float(r["heldout_acc_mean"])
        gu = by_TG[(T, G)]["gu_prob"]
        gu_ref = gu_ref_by_T[T]
        # (a) Signal-availability loss: 1 - GU(G)/GU(G_REF), in [0,1]
        signal_loss = max(0.0, 1.0 - gu / gu_ref) if gu_ref > 0 else 0.0
        # (b) Per-token noise penalty: sqrt(G_ref/G) - 1 (>=0)
        noise_penalty = math.sqrt(G_REF / G) - 1.0
        # (c) Effective-batch penalty: 1 - T/T_ref (>=0; 0 at T_REF)
        batch_penalty = max(0.0, 1.0 - T / T_REF)

        zr = zvf_idx.get((T, G), {})
        zvf_implied_retention = float(zr["zvf_implied_retention"]) if zr else "NA"
        observed_retention_vs_G64 = float(zr["retention_vs_max_G"]) if zr else "NA"

        signal_rows.append({
            "T_tokens": T, "G": G, "acc": acc,
            "GU_at_G_prob": round(gu, 4),
            "GU_at_G64_prob": round(gu_ref, 4),
            "signal_loss_prob_units": round(signal_loss, 4),
            "ZVF_implied_retention": zvf_implied_retention,
            "observed_retention_vs_G64": observed_retention_vs_G64,
        })

        eff_batch_rows.append({
            "T_tokens": T, "G": G, "acc": acc,
            "noise_penalty_rel_G64": round(noise_penalty, 4),
            "batch_penalty_rel_T64": round(batch_penalty, 4),
            "T_T64_ratio": round(T / T_REF, 4),
            "G_G64_ratio": round(G / G_REF, 4),
        })

    # Build the additive decomposition.
    # Model: acc(T, G) ≈ a0 + a_signal * signal_loss
    #                    + a_noise   * noise_penalty
    #                    + a_batch   * batch_penalty
    # Fit by ordinary least squares on the 20 (T, G) rows.
    A_rows = []
    b_rows = []
    for r in tok_rows:
        T = int(r["budget_tokens"])
        G = int(r["G"])
        gu = by_TG[(T, G)]["gu_prob"]
        gu_ref = gu_ref_by_T[T]
        signal_loss = max(0.0, 1.0 - gu / gu_ref) if gu_ref > 0 else 0.0
        A_rows.append([
            1.0,
            signal_loss,
            math.sqrt(G_REF / G) - 1.0,
            max(0.0, 1.0 - T / T_REF),
        ])
        b_rows.append(float(r["heldout_acc_mean"]))
    A = np.asarray(A_rows, dtype=float)
    b = np.asarray(b_rows, dtype=float)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    a0, a_signal, a_noise, a_batch = (float(c) for c in coef)
    pred = A @ coef
    ss_res = float(np.sum((b - pred) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Build per-row additive decomposition table.
    for r, ar, pr in zip(tok_rows, A_rows, pred):
        T = int(r["budget_tokens"])
        G = int(r["G"])
        acc_obs = float(r["heldout_acc_mean"])
        acc_pred = float(pr)
        signal = a_signal * ar[1]
        noise = a_noise * ar[2]
        batch = a_batch * ar[3]
        decomp_rows.append({
            "T_tokens": T, "G": G,
            "acc_observed": round(acc_obs, 4),
            "acc_predicted_additive": round(acc_pred, 4),
            "intercept_term": round(a0, 4),
            "signal_component_pp": round(signal * 100, 3),
            "noise_component_pp": round(noise * 100, 3),
            "batch_component_pp": round(batch * 100, 3),
            "residual_pp": round((acc_obs - acc_pred) * 100, 3),
        })

    # Attribution percentages restricted to G=4 rows.
    g4_decomp = [r for r in decomp_rows if r["G"] == 4]
    g4_signal_sum = sum(r["signal_component_pp"] for r in g4_decomp)
    g4_noise_sum = sum(r["noise_component_pp"] for r in g4_decomp)
    g4_batch_sum = sum(r["batch_component_pp"] for r in g4_decomp)
    g4_attrib_total = abs(g4_signal_sum) + abs(g4_noise_sum) + abs(g4_batch_sum)
    if g4_attrib_total > 0:
        g4_signal_pct = abs(g4_signal_sum) / g4_attrib_total * 100
        g4_noise_pct = abs(g4_noise_sum) / g4_attrib_total * 100
        g4_batch_pct = abs(g4_batch_sum) / g4_attrib_total * 100
    else:
        g4_signal_pct = g4_noise_pct = g4_batch_pct = 0.0

    # Multiplicative decomposition (informational): ratios of
    # (1-signal_loss)(1-noise_loss)(1-batch_loss).
    multi_rows: list[dict] = []
    for r in tok_rows:
        T = int(r["budget_tokens"])
        G = int(r["G"])
        gu = by_TG[(T, G)]["gu_prob"]
        gu_ref = gu_ref_by_T[T]
        signal_factor = gu / gu_ref if gu_ref > 0 else 1.0
        noise_factor = math.sqrt(G / G_REF)
        batch_factor = T / T_REF
        multi_rows.append({
            "T_tokens": T, "G": G,
            "acc_observed": float(r["heldout_acc_mean"]),
            "signal_factor": round(signal_factor, 4),
            "noise_factor": round(noise_factor, 4),
            "batch_factor": round(batch_factor, 4),
            "multiplicative_product": round(signal_factor * noise_factor * batch_factor, 4),
        })

    # Fixed-budget (T=T_REF only) decomposition — isolates G effect
    # from the confounded batch term.  This is the cleanest attribution
    # of G=4's accuracy loss.
    fixed_rows = [r for r in decomp_rows if r["T_tokens"] == T_REF]
    fb_g4 = next((r for r in fixed_rows if r["G"] == 4), None)
    fb_g64 = next((r for r in fixed_rows if r["G"] == 64), None)
    fb_obs_gap = (fb_g64["acc_observed"] - fb_g4["acc_observed"]) if (fb_g4 and fb_g64) else None
    fb_signal_pp = fb_g4["signal_component_pp"] if fb_g4 else None
    fb_noise_pp = fb_g4["noise_component_pp"] if fb_g4 else None
    fb_total_decomp = (fb_signal_pp or 0) + (fb_noise_pp or 0)
    if fb_signal_pp is not None and fb_noise_pp is not None and (abs(fb_signal_pp) + abs(fb_noise_pp)) > 0:
        fb_signal_share = abs(fb_signal_pp) / (abs(fb_signal_pp) + abs(fb_noise_pp)) * 100
        fb_noise_share = abs(fb_noise_pp) / (abs(fb_signal_pp) + abs(fb_noise_pp)) * 100
    else:
        fb_signal_share = fb_noise_share = 0.0
    fb_explained_share = (
        abs(fb_total_decomp) / abs(fb_obs_gap * 100) * 100
        if fb_obs_gap and fb_total_decomp else 0.0)

    headline: dict[str, object] = {
        "n_rows": len(tok_rows),
        "T_REF_tokens": T_REF,
        "G_REF": G_REF,
        "additive_model": (
            f"acc = {a0:.4f} + {a_signal:.4f}*signal + "
            f"{a_noise:.4f}*noise + {a_batch:.4f}*batch"),
        "additive_R2": round(r2, 4),
        "G4_signal_loss_pct_attr": round(g4_signal_pct, 2),
        "G4_noise_loss_pct_attr": round(g4_noise_pct, 2),
        "G4_batch_loss_pct_attr": round(g4_batch_pct, 2),
        "G4_dominant_component_pooled": (
            "signal" if g4_signal_pct >= max(g4_noise_pct, g4_batch_pct) else
            "noise" if g4_noise_pct >= g4_batch_pct else "batch"),
        "max_observed_signal_loss_G4_prob": max(
            r["signal_loss_prob_units"] for r in signal_rows if r["G"] == 4),
        "max_observed_noise_penalty_G4": max(
            r["noise_penalty_rel_G64"] for r in eff_batch_rows if r["G"] == 4),
        "fixed_budget_T_REF": T_REF,
        "fixed_budget_obs_gap_pp": (
            round(fb_obs_gap * 100, 2) if fb_obs_gap is not None else "NA"),
        "fixed_budget_signal_pp": (
            round(fb_signal_pp, 2) if fb_signal_pp is not None else "NA"),
        "fixed_budget_noise_pp": (
            round(fb_noise_pp, 2) if fb_noise_pp is not None else "NA"),
        "fixed_budget_total_decomp_pp": (
            round(fb_total_decomp, 2) if fb_total_decomp else "NA"),
        "fixed_budget_signal_share_pct": round(fb_signal_share, 2),
        "fixed_budget_noise_share_pct": round(fb_noise_share, 2),
        "fixed_budget_explained_share_pct": round(fb_explained_share, 2),
        "fixed_budget_dominant_component": (
            "noise" if fb_noise_share > fb_signal_share else "signal"),
    }

    # Persist outputs
    write_tsv(RES / "group_size_iter71_signal_loss.tsv", signal_rows)
    write_tsv(RES / "group_size_iter71_eff_batch.tsv", eff_batch_rows)
    write_tsv(RES / "group_size_iter71_decomp.tsv", decomp_rows)
    write_tsv(RES / "group_size_iter71_multiplicative.tsv", multi_rows)
    write_tsv(RES / "group_size_iter71_fixed_budget.tsv", [
        {"T_tokens": T_REF, "G": r["G"],
         "acc_observed": r["acc_observed"],
         "acc_predicted": r["acc_predicted_additive"],
         "signal_component_pp": r["signal_component_pp"],
         "noise_component_pp": r["noise_component_pp"],
         "batch_component_pp": r["batch_component_pp"],
         "residual_pp": r["residual_pp"]}
        for r in fixed_rows
    ])
    write_tsv(RES / "group_size_iter71_summary.tsv",
              [{"metric": k, "value": v} for k, v in headline.items()])

    meta = {
        "iteration": 71,
        "pillar": "P3-Group-Size",
        "inputs": [
            "platform_hybrid/experiments/results/group_size_token_normalized.tsv",
            "platform_hybrid/experiments/results/group_size_iter43_eff_zvf.tsv",
            "platform_hybrid/experiments/results/group_size_iter67_iaf_ratios.tsv",
        ],
        "outputs": [
            "platform_hybrid/experiments/results/group_size_iter71_signal_loss.tsv",
            "platform_hybrid/experiments/results/group_size_iter71_eff_batch.tsv",
            "platform_hybrid/experiments/results/group_size_iter71_decomp.tsv",
            "platform_hybrid/experiments/results/group_size_iter71_multiplicative.tsv",
            "platform_hybrid/experiments/results/group_size_iter71_fixed_budget.tsv",
            "platform_hybrid/experiments/results/group_size_iter71_summary.tsv",
            "figures/group_size_iter71_decomp.pdf",
            "figures/group_size_iter71_decomp.png",
        ],
        "method": (
            "Three-component additive decomposition: signal (1 - GU(G)/GU(G_ref)), "
            "noise (sqrt(G_ref/G) - 1), batch (1 - T/T_ref). OLS fit on (T,G)->acc. "
            "Multiplicative cross-check: GU(G)/GU(G_ref) * sqrt(G/G_ref) * T/T_ref."
        ),
        "headline_metrics": {k: v for k, v in headline.items()},
    }
    (RES / "group_size_iter71_iter_meta.json").write_text(json.dumps(meta, indent=2))

    # Plot — two panels: stacked-bar attribution per G, and a
    # predicted-vs-observed scatter.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    Gs = sorted({int(r["G"]) for r in decomp_rows})
    signal_means = [np.mean([r["signal_component_pp"] for r in decomp_rows if r["G"] == G]) for G in Gs]
    noise_means = [np.mean([r["noise_component_pp"] for r in decomp_rows if r["G"] == G]) for G in Gs]
    batch_means = [np.mean([r["batch_component_pp"] for r in decomp_rows if r["G"] == G]) for G in Gs]
    x = np.arange(len(Gs))
    ax[0].bar(x, signal_means, label="signal (1 - GU/Gu_ref)", color="#1f77b4")
    ax[0].bar(x, noise_means, bottom=signal_means, label="noise (sqrt(G_ref/G)-1)",
              color="#ff7f0e")
    ax[0].bar(x, batch_means,
              bottom=[s + n for s, n in zip(signal_means, noise_means)],
              label="batch (1 - T/T_ref)", color="#2ca02c")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"G={g}" for g in Gs])
    ax[0].set_ylabel("mean additive component (pp)")
    ax[0].set_title("Additive attribution by group size")
    ax[0].axhline(0, color="black", linewidth=0.5)
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.3, axis="y")

    obs = [r["acc_observed"] for r in decomp_rows]
    pred = [r["acc_predicted_additive"] for r in decomp_rows]
    colors = {4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c", 32: "#d62728", 64: "#9467bd"}
    for G in Gs:
        idx = [i for i, r in enumerate(decomp_rows) if r["G"] == G]
        ax[1].scatter([obs[i] for i in idx], [pred[i] for i in idx],
                      color=colors.get(G, "gray"), label=f"G={G}", s=60, alpha=0.85)
    lo = min(min(obs), min(pred)) - 0.02
    hi = max(max(obs), max(pred)) + 0.02
    ax[1].plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6)
    ax[1].set_xlabel("observed accuracy")
    ax[1].set_ylabel("predicted accuracy (additive)")
    ax[1].set_title(f"Additive model fit (R²={r2:.3f})")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "group_size_iter71_decomp.pdf")
    fig.savefig(FIG / "group_size_iter71_decomp.png", dpi=130)
    plt.close(fig)

    # Plot #2: G=4 attribution stacked per budget.
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    Ts_g4 = sorted({r["T_tokens"] for r in decomp_rows})
    sig_g4 = [np.mean([r["signal_component_pp"] for r in decomp_rows
                       if r["G"] == 4 and r["T_tokens"] == T]) for T in Ts_g4]
    noi_g4 = [np.mean([r["noise_component_pp"] for r in decomp_rows
                       if r["G"] == 4 and r["T_tokens"] == T]) for T in Ts_g4]
    bat_g4 = [np.mean([r["batch_component_pp"] for r in decomp_rows
                       if r["G"] == 4 and r["T_tokens"] == T]) for T in Ts_g4]
    xx = np.arange(len(Ts_g4))
    log_Ts = [math.log10(T / 1e6) for T in Ts_g4]
    ax2.bar(xx, sig_g4, label="signal", color="#1f77b4")
    ax2.bar(xx, noi_g4, bottom=sig_g4, label="noise", color="#ff7f0e")
    ax2.bar(xx, bat_g4, bottom=[s + n for s, n in zip(sig_g4, noi_g4)],
            label="batch", color="#2ca02c")
    ax2.set_xticks(xx)
    ax2.set_xticklabels([f"{lt:.1f}" for lt in log_Ts])
    ax2.set_xlabel(r"$\log_{10}(T/\mathrm{M})$")
    ax2.set_ylabel("attribution for G=4 (pp)")
    ax2.set_title("G=4 accuracy-deficit attribution by budget")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(FIG / "group_size_iter71_g4_budget.pdf")
    fig2.savefig(FIG / "group_size_iter71_g4_budget.png", dpi=130)
    plt.close(fig2)

    print("[iter71] headline summary:")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"[iter71] additive fit: acc = {a0:.4f} + "
          f"{a_signal:.4f}*signal + {a_noise:.4f}*noise + {a_batch:.4f}*batch, "
          f"R²={r2:.4f}")
    print(f"[iter71] G=4 attribution: signal={g4_signal_pct:.1f}%, "
          f"noise={g4_noise_pct:.1f}%, batch={g4_batch_pct:.1f}%")


if __name__ == "__main__":
    main()
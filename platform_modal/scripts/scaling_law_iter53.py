"""Pillar 1 iter53 -- Rank preservation + temporal-peak coupling + Critic Degeneracy.

iter49 already fit R_max = a*log10(P) + b*log10(C) + c on 12 anchors with LOO
residual; iter45 fixed one axis; iter41/iter37/iter33/iter25 parsed the
trajectories. What has NOT been done is to test whether the LOO prediction
preserves the *ranking* of models, and to test whether the LOO residual is
coupled to the deterministic temporal-phase taxonomy from iter33.

Concretely, three exploratory questions:

  Q1. Rank preservation.
      Sort the 12 anchors by LOO-predicted R_max and compare to the actual
      R_max ordering. Metric: Kendall tau_b (no ties broken unless forced).
      Exploratory: tau > 0.50 (above chance) — indicates the joint fit is
      rank-useful, even if the LOO RMSE is loose.

  Q2. Temporal-peak coupling.
      Does a late-peaking trajectory correspond to a *negative* LOO residual?
      The saturation fit R_max*(1 - e^{-lambda t}) tends to read the peak
      value as R_max, so a trace peaking late AND high should look larger
      than its asymptotic plateau. Exploratory: spearman(peak_frac, LOO
      residual) < -0.30 (negative).

  Q3. Critic Degeneracy (frontier synthesis Pillar-1 critique).
      FrontChatGPT/Gemini argued that matched-stack PPO/GRPO are
      indistinguishable because the critic collapses into a static
      prompt-difficulty regressor. The same logic predicts that, after
      dropping the *collapse* anchors (which violate saturation), the
      remaining trace R_max must correlate more strongly with the
      DATA COMPLEXITY axis (log10_P or peak_var) and LESS with raw compute.
      Exploratory: Spearman(log10_P, R_max) on non-collapse anchors has
      sign opposite to / smaller absolute value than the same correlation on
      the full sample (the critic is doing the work, not compute).

This script does not refit; it imports iter49 outputs and adds three artefacts.

Outputs (5 artefacts):
  platform_hybrid/experiments/results/scaling_law_iter53_rank.tsv
  platform_hybrid/experiments/results/scaling_law_iter53_rank_summary.tsv
  platform_hybrid/experiments/results/scaling_law_iter53_peak_residual_coupling.tsv
  platform_hybrid/experiments/results/scaling_law_iter53_critic_degeneracy.tsv
  platform_hybrid/experiments/results/scaling_law_iter53_predictions.tsv
  paper/sections/scaling_law_iter53.tex
  figures/scaling_law_iter53_rank.{pdf,png}
"""

from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2] / "platform_hybrid"
CANDIDATE = REPO / "experiments" / "results" / "review27_v3"
RES = CANDIDATE
FIG_DIR = CANDIDATE / "figures"
PAPER_SEC = CANDIDATE / "sections"

PHASE_SRC = REPO / "experiments" / "results" / "scaling_law_iter33_phase_score.tsv"
LOO_SRC = RES / "scaling_law_iter49_loo_residuals.tsv"
COMPUTE_SRC = RES / "scaling_law_iter45_compute_proxy_v2.tsv"


# ---------- helpers ----------

def rank_with_ties(x):
    """Average-rank with ties (1..n)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def kendall_tau_b(x, y):
    try:
        from ._review27_rank import kendall
    except ImportError:
        from _review27_rank import kendall
    return kendall(x, y)



def spearman_with_perm(x, y, b=2000, seed=2026_0702):
    try:
        from ._review27_rank import spearman
    except ImportError:
        from _review27_rank import spearman
    return spearman(x, y, b=b, seed=seed)



def ols_3var(x1, x2, y):
    """OLS y = a*x1 + b*x2 + c."""
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    y = np.asarray(y, float)
    X = np.column_stack([x1, x2, np.ones_like(x1)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return float(coef[0]), float(coef[1]), float(coef[2]), yhat


# ---------- loaders ----------

def load_loo():
    with open(LOO_SRC) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k in ("R_max_actual", "R_max_predicted", "residual", "abs_residual",
                  "params_B", "log10_P", "log10_C"):
            if k in r:
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
    return rows


def load_phase():
    with open(PHASE_SRC) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for r in rows:
        if "model" in r:
            out[r["model"]] = {
                "phase_classifier": r.get("phase_classifier", "unknown"),
                "peak_step": float(r.get("peak_step", "nan")) if r.get("peak_step") not in ("", "nan") else float("nan"),
                "peak_frac": float(r.get("peak_frac", "nan")) if r.get("peak_frac") not in ("", "nan") else float("nan"),
                "peak_val": float(r.get("peak_val", "nan")) if r.get("peak_val") not in ("", "nan") else float("nan"),
                "n_steps": float(r.get("n_steps", "nan")) if r.get("n_steps") not in ("", "nan") else float("nan"),
                "mean_reward": float(r.get("mean_reward", "nan")) if r.get("mean_reward") not in ("", "nan") else float("nan"),
                "var_reward": float(r.get("var_reward", "nan")) if r.get("var_reward") not in ("", "nan") else float("nan"),
                "R_max": float(r.get("R_max", "nan")) if r.get("R_max") not in ("", "nan") else float("nan"),
            }
    return out


def load_compute():
    with open(COMPUTE_SRC) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


# ---------- main ----------

def main(output_dir=None, source_root=None):
    global REPO, CANDIDATE, FIG_DIR, PAPER_SEC
    global RES, PHASE_SRC, LOO_SRC, COMPUTE_SRC
    if source_root is not None:
        REPO = Path(source_root)
    if output_dir is not None:
        CANDIDATE = Path(output_dir)
    RES = CANDIDATE
    PHASE_SRC = REPO / "experiments/results/scaling_law_iter33_phase_score.tsv"
    LOO_SRC = RES / "scaling_law_iter49_loo_residuals.tsv"
    COMPUTE_SRC = RES / "scaling_law_iter45_compute_proxy_v2.tsv"
    FIG_DIR, PAPER_SEC = CANDIDATE / "figures", CANDIDATE / "sections"
    for directory in (CANDIDATE, FIG_DIR, PAPER_SEC):
        directory.mkdir(parents=True, exist_ok=True)

    loo = load_loo()
    if not loo or any(not all(math.isfinite(float(row[key])) for key in
                              ("R_max_actual", "R_max_predicted", "residual", "log10_P", "log10_C")) for row in loo):
        raise ValueError("finite identified LOO cohort required")
    if len({row["model"] for row in loo}) != len(loo):
        raise ValueError("duplicate model identities in LOO cohort")
    ph = load_phase()
    co = load_compute()
    co_by_model = {r["model"]: r for r in co}

    # Index by model
    loo_models = [r["model"] for r in loo]
    actual = np.array([r["R_max_actual"] for r in loo], float)
    predicted = np.array([r["R_max_predicted"] for r in loo], float)
    residuals = np.array([r["residual"] for r in loo], float)

    # Attach peak metrics
    peak_frac_list = []
    peak_step_list = []
    for i, m in enumerate(loo_models):
        info = ph.get(m, {})
        pf = info.get("peak_frac", float("nan"))
        ps = info.get("peak_step", float("nan"))
        peak_frac_list.append(pf)
        peak_step_list.append(ps)
    peak_frac = np.array(peak_frac_list, float)
    peak_step = np.array(peak_step_list, float)

    # ---- Q1: rank preservation ----
    tau = kendall_tau_b(predicted, actual)
    # Spearman too as cross-check
    rho_pa, p_pa = spearman_with_perm(predicted, actual, b=2000)

    # Build per-anchor rank rows
    rank_rows = []
    rank_pred_map = dict(zip(loo_models, rank_with_ties(-predicted)))
    rank_act_map = dict(zip(loo_models, rank_with_ties(-actual)))
    for i, m in enumerate(loo_models):
        rank_rows.append({
            "model": m,
            "params_B": next((r["params_B"] for r in co if r["model"] == m), float("nan")),
            "R_max_actual": round(float(actual[i]), 4),
            "R_max_predicted": round(float(predicted[i]), 4),
            "rank_actual": rank_act_map[m],
            "rank_predicted": rank_pred_map[m],
            "rank_diff": rank_act_map[m] - rank_pred_map[m],
            "abs_rank_diff": abs(rank_act_map[m] - rank_pred_map[m]),
            "phase": ph.get(m, {}).get("phase_classifier", "unknown"),
        })
    rank_rows.sort(key=lambda r: r["rank_actual"])
    with open(RES / "scaling_law_iter53_rank.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rank_rows)

    # Summary statistics
    abs_rank_diffs = np.array([r["abs_rank_diff"] for r in rank_rows], float)
    n = len(rank_rows)
    summary_rows = [
        {"metric": "n_anchors", "value": int(n), "note": "Total ranked"},
        {"metric": "kendall_tau_b_predicted_vs_actual", "value": round(tau, 4),
         "note": "SciPy tau-b; double ties excluded"},
        {"metric": "spearman_predicted_vs_actual", "value": round(rho_pa, 4),
         "note": f"permutation p={p_pa:.4f}"},
        {"metric": "mean_abs_rank_diff", "value": round(float(np.mean(abs_rank_diffs)), 4),
         "note": "Average |actual rank - predicted rank|"},
        {"metric": "median_abs_rank_diff", "value": round(float(np.median(abs_rank_diffs)), 4),
         "note": "Median |actual rank - predicted rank|"},
        {"metric": "max_abs_rank_diff", "value": round(float(np.max(abs_rank_diffs)), 4),
         "note": "Worst swap distance"},
        {"metric": "correct_top3", "value": int(sum(1 for r in rank_rows
                                                   if r["rank_actual"] <= 3 and r["rank_predicted"] <= 3)),
         "note": "Anchors in actual top-3 that LOO also places in top-3"},
        {"metric": "spearman_abs_residual_rank", "value": round(spearman_with_perm(np.abs(residuals), actual, b=2000)[0], 4),
         "note": "Does |residual| track the magnitude of R_max?"},
    ]
    with open(RES / "scaling_law_iter53_rank_summary.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "note"], delimiter="\t")
        w.writeheader()
        w.writerows(summary_rows)

    # ---- Q2: temporal-peak coupling ----
    # Drop NaN peak anchors
    valid = ~np.isnan(peak_frac) & ~np.isnan(residuals)
    if valid.sum() >= 4:
        rho_pr, p_pr = spearman_with_perm(peak_frac[valid], residuals[valid], b=2000)
    else:
        rho_pr, p_pr = float("nan"), float("nan")
    # Also do peak_step (absolute) vs residuals
    valid2 = ~np.isnan(peak_step) & ~np.isnan(residuals)
    if valid2.sum() >= 4:
        rho_ps, p_ps = spearman_with_perm(peak_step[valid2], residuals[valid2], b=2000)
    else:
        rho_ps, p_ps = float("nan"), float("nan")
    coupling_rows = [
        {"coupling": "peak_frac_vs_loo_residual",
         "n_anchors": int(valid.sum()),
         "spearman_rho": round(rho_pr, 4),
         "permutation_p": round(p_pr, 4),
         "interpretation": "Negative association: later peaks coincide with greater over-prediction", },
        {"coupling": "peak_step_vs_loo_residual",
         "n_anchors": int(valid2.sum()),
         "spearman_rho": round(rho_ps, 4),
         "permutation_p": round(p_ps, 4),
         "interpretation": "Negative association: later absolute peaks coincide with greater over-prediction"},
    ]
    with open(RES / "scaling_law_iter53_peak_residual_coupling.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coupling_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(coupling_rows)

    # ---- Q3: critic-degeneracy test ----
    # Compare Spearman(log10_P, R_max) on full sample vs dropping 'collapse' anchors.
    log_p_full = np.array([r["log10_P"] for r in loo if r["log10_P"] == r["log10_P"]], float)
    rmax_full = actual[~np.isnan(np.array([r["log10_P"] for r in loo], float))]
    rho_logP_full, p_logP_full = spearman_with_perm(log_p_full, rmax_full, b=2000)

    # Filter out collapse anchors
    mask_keep = np.array([
        ph.get(m, {}).get("phase_classifier", "unknown") != "collapse"
        for m in loo_models
    ])
    rho_logP_nocollapse, p_logP_nocollapse = spearman_with_perm(
        np.array([r["log10_P"] for r in loo], float)[mask_keep],
        actual[mask_keep], b=2000)
    # Also: log10_C vs R_max both
    log_c_full = np.array([r["log10_C"] for r in loo], float)
    rho_logC_full, p_logC_full = spearman_with_perm(log_c_full, actual, b=2000)
    rho_logC_nocollapse, p_logC_nocollapse = spearman_with_perm(
        log_c_full[mask_keep], actual[mask_keep], b=2000)

    # Bonus: Cross-check that R_max-via-phase_classifier identity is *stronger*
    # than R_max-via-compute. Compare var_reward vs log10_C.
    var_list = np.array([ph.get(m, {}).get("var_reward", float("nan")) for m in loo_models], float)
    valid3 = ~np.isnan(var_list)
    rho_var_full, p_var_full = spearman_with_perm(
        var_list[valid3], actual[valid3], b=2000)

    degeneracy_rows = [
        {"axis": "Spearman(log10_P, R_max)",
         "full_rho": round(rho_logP_full, 4), "full_p": round(p_logP_full, 4),
         "n_drop_collapse_rho": round(rho_logP_nocollapse, 4),
         "n_drop_collapse_p": round(p_logP_nocollapse, 4),
         "note": "Exploratory: drop-collapse |rho| smaller than full |rho|"},
        {"axis": "Spearman(log10_C, R_max)",
         "full_rho": round(rho_logC_full, 4), "full_p": round(p_logC_full, 4),
         "n_drop_collapse_rho": round(rho_logC_nocollapse, 4),
         "n_drop_collapse_p": round(p_logC_nocollapse, 4),
         "note": "Compute proxy should also shrink without collapse anchors"},
        {"axis": "Spearman(var_reward, R_max)",
         "full_rho": round(rho_var_full, 4), "full_p": round(p_var_full, 4),
         "n_drop_collapse_rho": "n/a", "n_drop_collapse_p": "n/a",
         "note": "Cross-check: does trajectory variance explain R_max at all?"},
    ]
    with open(RES / "scaling_law_iter53_critic_degeneracy.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(degeneracy_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(degeneracy_rows)

    # ---- exploratory predictions ----
    p1_pass = bool(tau > 0.50) if math.isfinite(tau) else None
    p2_pass = bool(rho_pr < -0.30) if math.isfinite(rho_pr) else None
    # Critic-degeneracy pass: |drop-collapse ρ of log10_P| < |full ρ of log10_P|
    # when full ρ > 0 (positive case). If full ρ was negative and shrink
    # would be away from zero; compare absolute values.
    full_abs = abs(rho_logP_full)
    drop_abs = abs(rho_logP_nocollapse)
    p3_pass = bool(drop_abs < full_abs - 0.05) if math.isfinite(drop_abs) and math.isfinite(full_abs) else None  # at least 0.05 smaller

    pred_rows = [
        {"prediction": "P1_Kendall_tau_b_gt_0p50",
         "value": round(tau, 4),
         "pass": p1_pass,
         "note": "rank-preservation under LOO two-param OLS"},
        {"prediction": "P2_spearman_peak_frac_resid_lt_-0p30",
         "value": round(rho_pr, 4),
         "pass": p2_pass,
         "note": f"late peak ⇒ negative LOO residual; p={p_pr:.4f}"},
        {"prediction": "P3_drop_collapse_log10_P_abs_rho_smaller",
         "value": f"full={rho_logP_full:.4f}, drop_coll={rho_logP_nocollapse:.4f}",
         "pass": p3_pass,
         "note": "Descriptive sensitivity to excluding collapse-labeled anchors; not a causal explanation"},
        {"prediction": "P4_spearman_peak_step_resid_lt_-0p20",
         "value": round(rho_ps, 4),
         "pass": bool(rho_ps < -0.20),
         "note": f"absolute peak_step vs residual; p={p_ps:.4f}"},
    ]
    with open(RES / "scaling_law_iter53_predictions.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(pred_rows)

    # ---- figure: rank parity plot ----
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=140)
    ax = axes[0]
    sc = ax.scatter(actual, predicted, c=[rank_act_map[m] for m in loo_models],
                    cmap="viridis_r", s=64, edgecolor="k", linewidths=0.6)
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", ls="--", lw=0.8)
    for r in rank_rows:
        ax.annotate(r["model"].replace("-", "\n"), (r["R_max_actual"], r["R_max_predicted"]),
                    fontsize=6, alpha=0.85, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Actual R_max")
    ax.set_ylabel("LOO predicted R_max")
    ax.set_title(f"Kendall tau = {tau:.3f}")
    plt.colorbar(sc, ax=ax, label="Actual rank (1=highest)")
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    ax2.scatter(peak_frac, residuals,
                c=[1 if ph.get(m, {}).get("phase_classifier") == "collapse" else 0
                    for m in loo_models],
                cmap="bwr", s=64, edgecolor="k", linewidths=0.6)
    for i, m in enumerate(loo_models):
        ax2.annotate(m.replace("-", "\n"), (peak_frac[i], residuals[i]),
                     fontsize=6, alpha=0.85, xytext=(2, 2), textcoords="offset points")
    ax2.axhline(0.0, color="grey", ls="--", lw=0.8)
    ax2.set_xlabel("peak_frac (step of peak / n_steps)")
    ax2.set_ylabel("LOO residual (actual − predicted)")
    ax2.set_title(f"Spearman(peak_frac, residual) = {rho_pr:.3f}")
    ax2.grid(alpha=0.25)
    fig.suptitle("iter53 — rank preservation + temporal-peak coupling")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scaling_law_iter53_rank.pdf")
    fig.savefig(FIG_DIR / "scaling_law_iter53_rank.png")
    plt.close(fig)

    # ---- paper section ----
    write_paper_section(tau, rho_pa, p_pa, rho_pr, p_pr,
                        rho_ps, p_ps,
                        rho_logP_full, p_logP_full,
                        rho_logP_nocollapse, p_logP_nocollapse,
                        rho_logC_full, rho_logC_nocollapse,
                        rho_var_full, summary_rows, rank_rows,
                        p1_pass, p2_pass, p3_pass,
                        abs_rank_diffs)

    try:
        from ._review27_rank import write_manifest
    except ImportError:
        from _review27_rank import write_manifest
    write_manifest(__file__, CANDIDATE, [PHASE_SRC, LOO_SRC, COMPUTE_SRC])


def write_paper_section(tau, rho_pa, p_pa, rho_pr, p_pr,
                        rho_ps, p_ps,
                        rho_logP_full, p_logP_full,
                        rho_logP_nocollapse, p_logP_nocollapse,
                        rho_logC_full, rho_logC_nocollapse,
                        rho_var_full,
                        summary_rows, rank_rows,
                        p1_pass, p2_pass, p3_pass,
                        abs_rank_diffs):
    # Find anchors with rank_diff >= 4 (significant swaps)
    """Render the current descriptive outcomes without frozen narrative verdicts."""
    labels=("P1 rank preservation","P2 temporal coupling","P3 collapse sensitivity")
    flags=(p1_pass,p2_pass,p3_pass)
    body=r"\subsection{Exploratory rank and temporal diagnostics}"+"\n"
    body+="These checks describe the supplied cohort; registration and causal interpretation are not established.\n"
    for label,flag in zip(labels,flags):
        body+=f"{label}: {'UNAVAILABLE' if flag is None else 'PASS' if flag else 'FAIL'}.\n"
    body+=f"Kendall tau-b: {tau:.6g}; observed rank displacement mean: {float(np.mean(abs_rank_diffs)):.6g}.\n"
    out=PAPER_SEC/"scaling_law_iter53_review27_v3.tex"
    out.write_text(body)
    return out




if __name__ == "__main__":
    main()

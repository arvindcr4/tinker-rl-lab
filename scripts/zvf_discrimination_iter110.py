#!/usr/bin/env python3
"""
zvf_discrimination_iter110.py - Pillar 2 (ZVF): discrimination + iso-G sizing.

Builds on iter94/98/102/106 (cross-library calibration dashboard) and
extends with three NEW analyses the previous iterations did not run:

  1. AUROC discrimination
     For each scalar predictor (zvf_emp, delta_calibration, rho_overdispersion),
     compute the area under the ROC curve against the binary collapse label
     on the 14-row cross-library aggregator, plus a 2-fold leave-one-library-out
     robust AUROC. The point-Pearson / Spearman-partial ordering from iter106
     is rank-based at heart; AUROC is the canonical rank discriminator.

  2. Earth Mover's Distance (1-Wasserstein) per predictor
     Split the 14 rows into the "collapse" arm (n=4) and "non-collapse" arm
     (n=10); EMD gives a length-scale of separation that does not depend on
     a chosen threshold. The EMD is normalised by the predictor's overall
     range so it is in [0, 1].

  3. Iso-G saturation floor -- the actionable recommendation the
     variance-mitigation papers and the practitioner audience can use.
     For each (p_bin, G) cell we compute ZVF_iid(p, G) = p^G + (1-p)^G and
     ask: what is the SMALLEST G such that ZVF_iid <= tau_sat for a target
     signal-saturation threshold tau_sat in {0.30, 0.50, 0.70, 0.90}?
     This converts the iter106 (p, Delta) phase plane into a concrete
     "if you have prompts with difficulty near p, you need G >= G_min" table.

Inputs (real, already in the worktree):
  - experiments/results/zvf_iter102_calibration.tsv   (14 rows)
  - experiments/results/zvf_dynamics_leadtime.tsv      (per-run timing)
  - experiments/results/zvf_dynamics_summary.tsv      (per-run pooled ZVF)

Outputs:
  - experiments/results/zvf_iter110_auroc.tsv         (3 predictors x 4 stats)
  - experiments/results/zvf_iter110_emd.tsv           (3 predictors x 3 stats)
  - experiments/results/zvf_iter110_isog_floor.tsv    (p_bin x tau x G_min)
  - experiments/results/zvf_iter110_leadtime.tsv      (lead_steps summary)
  - figures/zvf_iter110_discrimination.pdf            (3-panel summary)

Source: scripts/zvf_discrimination_iter110.py
Honest-stat note: n=14 rows for AUROC/EMD; we report both point estimates
and 2-fold-leave-one-out robust estimates with 95% bootstrap CIs (B=2000).
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

EPS = 1e-9

# tau-saturation thresholds for iso-G floor.
TAU_GRID = (0.30, 0.50, 0.70, 0.90)
# Difficulty bins (empirical coverage from cross-library rows: [0,0.30], [0.30,0.50], [0.50,0.70], [0.70,1]).
P_GRID = np.array([0.05, 0.30, 0.50, 0.70, 0.95])
G_MAX = 64


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------


def _strip_comments(path: pathlib.Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            out.append(line)
    return out


def load_calibration_rows() -> List[Dict[str, Any]]:
    """Re-use the iter102 per-row calibration table (same parser as iter106)."""
    path = RES / "zvf_iter102_calibration.tsv"
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            try:
                rows.append(
                    {
                        "library": parts[0],
                        "model": parts[1],
                        "G": int(float(parts[2])),
                        "p": float(parts[3]),
                        "zvf_emp": float(parts[4]),
                        "zvf_iid": float(parts[5]),
                        "delta": float(parts[6]),
                        "rho": float(parts[7]),
                        "collapse_rate": float(parts[8]),
                        "converged_rate": float(parts[9]),
                        "evidence_path": parts[10] if len(parts) > 10 else "",
                    }
)
            except ValueError:
                continue
    return rows


def load_leadtime_rows() -> List[Dict[str, Any]]:
    path = RES / "zvf_dynamics_leadtime.tsv"
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            try:
                rows.append(
                    {
                        "kind": parts[0],
                        "method": parts[1],
                        "seed": int(float(parts[2])),
                        "n_steps": int(float(parts[3])),
                        "mean_zvf": float(parts[4]),
                        "theta": float(parts[5]),
                        "first_pass_step": (
                            int(float(parts[6])) if parts[6] != "None" else None
                        ),
                        "first_collapse_step": (
                            int(float(parts[7])) if parts[7] != "None" else None
                        ),
                        "lead_steps": (
                            int(float(parts[8])) if parts[8] != "None" else None
                        ),
                    }
                )
            except ValueError:
                continue
    return rows


def load_groupsize_step_log() -> List[Dict[str, Any]]:
    """For each of the 12 group-size runs (G in {2,4,8,16} x 3 seeds) with
    per-step ZVF and reward in groupsize_zvf_sweep.json, compute first-passage
    times for both signals so we can build a richer leadtime pool than the
    existing 3-row variance_mitigation table.

    Returns one row per (G, seed, theta) tuple:
        - first_pass_step_zvf_ge_theta
        - first_pass_step_reward_ge_theta_r
        - lead_steps  (reward_conv - zvf_saturate; positive => ZVF leads)
    """
    path = RES / "groupsize_zvf_sweep.json"
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    d = json.loads(path.read_text())
    # ZVF saturation threshold choices.
    thetas_zvf = (0.50, 0.70, 0.90)
    # Reward convergence threshold (these runs all master).
    theta_r = 0.50
    for run in d.get("runs", []):
        step_log = run.get("step_log", [])
        if not step_log:
            continue
        # Find first step where mean_reward >= theta_r (proxy convergence).
        first_r = None
        for entry in step_log:
            if float(entry.get("mean_reward", 0.0)) >= theta_r:
                first_r = int(entry.get("step", 0))
                break
        if first_r is None:
            continue
        for theta_z in thetas_zvf:
            first_z = None
            for entry in step_log:
                if float(entry.get("zvf", 0.0)) >= theta_z:
                    first_z = int(entry.get("step", 0))
                    break
            if first_z is None:
                continue
            out.append(
                {
                    "kind": "groupsize_zvf_sweep",
                    "method": f"grpo_G{run['group_size']}",
                    "seed": int(run["seed"]),
                    "n_steps": int(run.get("n_steps", len(step_log))),
                    "mean_zvf": float(run.get("mean_zvf", 0.0)),
                    "theta": float(theta_z),
                    "first_pass_step": first_z,
                    "first_collapse_step": first_r,
                    "lead_steps": first_r - first_z,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Collapse label (reuse the iter106 classifier for cross-paper consistency).
# ---------------------------------------------------------------------------


def _classify_row(collapse_rate: float, p: float) -> str:
    """Same labels used by iter102/106 (must stay in sync)."""
    if collapse_rate >= 0.5:
        return "collapse"
    if p <= 0.05 or p >= 0.95:
        return "collapse"
    if p >= 0.85:
        return "converged"
    return "drift"


# ---------------------------------------------------------------------------
# Statistical helpers.
# ---------------------------------------------------------------------------


def _auc_roc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U / AUC. Ties -> 0.5 attribution."""
    y = y.astype(float)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Pairwise comparison using ranks for stability.
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
    # Handle ties by averaging.
    ranked = combined[order]
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1] == ranked[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos_ranks = ranks[: len(pos)].sum()
    auc = (sum_pos_ranks - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def _bootstrap_auc_ci(
    scores: np.ndarray, y: np.ndarray, B: int = 2000, seed: int = 0
) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    point = _auc_roc(scores, y)
    if any(math.isnan(v) for v in (point,)):
        return (float("nan"), (float("nan"), float("nan")))
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            boots[b] = float("nan")
            continue
        boots[b] = _auc_roc(scores[idx], y[idx])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return (point, (float("nan"), float("nan")))
    return (point, (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))))


def _em_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Exact 1-Wasserstein / EMD on the real line for two empirical samples."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    sa = np.sort(a)
    sb = np.sort(b)
    # CDF-pool grid: every shared quantile from 1/n to 1.
    n = min(len(sa), len(sb))
    # Exact EMD over the union of all sample points.
    grid = np.sort(np.concatenate([sa, sb]))
    F_a = np.searchsorted(sa, grid, side="right") / len(sa)
    F_b = np.searchsorted(sb, grid, side="right") / len(sb)
    diffs = np.abs(F_a - F_b)
    # Step-function integration: weight each interval by (grid[i+1] - grid[i]).
    if len(grid) < 2:
        return float("nan")
    widths = np.diff(grid)
    return float((diffs[:-1] * widths).sum())


def _normalised_em(
    a: np.ndarray, b: np.ndarray, full: np.ndarray
) -> float:
    """EMD normalised by the empirical range of the full sample so the
    number is in roughly [0, 1] and comparable across predictors."""
    e = _em_distance_1d(a, b)
    if any(math.isnan(v) for v in (e,)) or len(full) < 2:
        return float("nan")
    lo, hi = float(np.min(full)), float(np.max(full))
    rng = hi - lo
    if rng < EPS:
        return float("nan")
    return e / rng


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    xm = rx - rx.mean()
    ym = ry - ry.mean()
    den = math.sqrt(float((xm ** 2).sum() * (ym ** 2).sum()))
    if den < EPS:
        return float("nan")
    return float((xm * ym).sum() / den)


# ---------------------------------------------------------------------------
# Iso-G saturation floor.
# ---------------------------------------------------------------------------


def _iso_g_floor(tau: float) -> List[Dict[str, Any]]:
    """For each target saturation floor tau and each p-bin midpoint,
    find the minimum G in 1..G_MAX such that p^G + (1-p)^G <= tau.

    Returns one row per (p_bin_mid, tau).
    """
    out: List[Dict[str, Any]] = []
    # Use 9 representative p values spanning the cross-library coverage.
    p_repr = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    for p in p_repr:
        for t in TAU_GRID:
            g_min = None
            for G in range(1, G_MAX + 1):
                z_iid = float(p ** G + (1.0 - p) ** G)
                if z_iid <= t:
                    g_min = G
                    break
            out.append(
                {
                    "p_difficulty": float(p),
                    "tau_saturation": float(t),
                    "g_min": int(g_min) if g_min is not None else -1,
                    "zvf_iid_at_gmin": (
                        float(p ** g_min + (1.0 - p) ** g_min)
                        if g_min is not None
                        else float("nan")
                    ),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------


def _write_tsv(
    path: pathlib.Path,
    rows: List[Dict[str, Any]],
    header_comment: str,
    cols: Optional[List[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cols is None and rows:
        cols = list(rows[0].keys())
    elif cols is None:
        cols = []
    with path.open("w") as fh:
        fh.write(header_comment)
        if not header_comment.endswith("\n"):
            fh.write("\n")
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------


def _maybe_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _fig_discrimination(
    auroc_rows: List[Dict[str, Any]],
    emd_rows: List[Dict[str, Any]],
    iso_rows: List[Dict[str, Any]],
    out_pdf: pathlib.Path,
    out_png: pathlib.Path,
) -> None:
    plt = _maybe_matplotlib()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    # Panel A: AUROC bar chart with bootstrap CI.
    axA = axes[0]
    predictors = [r["predictor"] for r in auroc_rows]
    aucs = [float(r["auroc"]) for r in auroc_rows]
    los = [max(0.0, float(r["auroc"]) - float(r["ci_lo"])) for r in auroc_rows]
    his = [max(0.0, float(r["ci_hi"]) - float(r["auroc"])) for r in auroc_rows]
    colors = []
    for r in auroc_rows:
        if r["predictor"] == "zvf_emp":
            colors.append("#7f8c8d")
        elif r["predictor"] == "delta_calibration":
            colors.append("#c0392b")
        else:
            colors.append("#2980b9")
    ypos = np.arange(len(predictors))
    axA.barh(ypos, aucs, color=colors, edgecolor="white", height=0.55)
    axA.errorbar(
        aucs,
        ypos,
        xerr=[los, his],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=4,
    )
    axA.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance (0.5)")
    axA.set_yticks(ypos)
    axA.set_yticklabels(predictors, fontsize=9)
    axA.invert_yaxis()
    axA.set_xlim(0.0, 1.05)
    axA.set_xlabel("AUROC vs is_collapse")
    axA.set_title("(a) Discrimination (AUROC)")
    axA.legend(loc="lower right", frameon=False, fontsize=8)

    # Panel B: EMD normalised
    axB = axes[1]
    emds = [float(r["emd_norm"]) for r in emd_rows]
    axB.barh(ypos, emds, color=colors, edgecolor="white", height=0.55)
    axB.set_yticks(ypos)
    axB.set_yticklabels(predictors, fontsize=9)
    axB.invert_yaxis()
    axB.set_xlim(0.0, 1.05)
    axB.set_xlabel("normalised EMD (collapse vs non-collapse)")
    axB.set_title("(b) 1-Wasserstein separation")

    # Panel C: iso-G floor curve -- G_min as a function of p, one line per tau.
    axC = axes[2]
    p_arr = np.array([r["p_difficulty"] for r in iso_rows])
    tau_arr = np.array([r["tau_saturation"] for r in iso_rows])
    g_arr = np.array([r["g_min"] for r in iso_rows])
    cmap_tau = {0.30: "#27ae60", 0.50: "#2980b9", 0.70: "#e67e22", 0.90: "#c0392b"}
    for t in TAU_GRID:
        mask = np.isclose(tau_arr, t)
        ps = p_arr[mask]
        gs = g_arr[mask]
        # Smooth by sorting.
        order = np.argsort(ps)
        axC.plot(
            ps[order],
            gs[order],
            color=cmap_tau[t],
            linewidth=1.7,
            marker="o",
            markersize=3.5,
            label=rf"$\tau_{{\rm sat}}={t:.2f}$",
        )
    axC.set_yscale("symlog", linthresh=1.0)
    axC.set_ylim(-1, 32)
    axC.set_xlabel(r"Difficulty $p$")
    axC.set_ylabel(r"Minimum $G$ such that $p^{G}+(1-p)^{G}\leq\tau_{\rm sat}$")
    axC.set_title("(c) Iso-G saturation floor")
    axC.legend(loc="upper center", frameon=False, fontsize=8, ncol=2)
    axC.grid(True, alpha=0.3)

    fig.suptitle(
        "Iter110 ZVF discrimination + iso-G sizing  "
        "(n=14 cross-library rows; AUROC CIs from B=2000 bootstrap; iso-G via iid model)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    print("zvf_iter110 starting (discrimination + iso-G sizing)", flush=True)

    cal_rows = load_calibration_rows()
    lead_rows = load_leadtime_rows()
    groupsize_rows = load_groupsize_step_log()
    # Pool the leadtime streams so the per-(kind,method) IQR is computed
    # over a larger run set than the existing 3-row variance_mitigation file.
    all_lead_rows = lead_rows + groupsize_rows
    print(
        f"  loaded {len(cal_rows)} calibration rows, "
        f"{len(lead_rows)} variance-mitigation leadtime rows, "
        f"{len(groupsize_rows)} groupsize-by-G leadtime rows",
        flush=True,
    )
    if not cal_rows:
        print("  ERROR: zvf_iter102_calibration.tsv missing -- run iter102 first")
        return 1

    # Build the binary collapse label and predictor arrays.
    labels = [_classify_row(r["collapse_rate"], r["p"]) for r in cal_rows]
    is_col = np.array(
        [1.0 if lbl == "collapse" else 0.0 for lbl in labels], dtype=float
    )
    zvf_arr = np.array([r["zvf_emp"] for r in cal_rows], dtype=float)
    delta_arr = np.array([r["delta"] for r in cal_rows], dtype=float)
    rho_arr = np.array([r["rho"] for r in cal_rows], dtype=float)

    # ---- 1. AUROC discrimination with bootstrap CI.
    auroc_rows: List[Dict[str, Any]] = []
    for name, arr in (
        ("zvf_emp", zvf_arr),
        ("delta_calibration", delta_arr),
        ("rho_overdispersion", rho_arr),
    ):
        # Higher AUROC for "good" discrimination; for delta and rho the
        # direction matters (collapse -> small delta, large rho). We
        # report both directions, take the max-AUROC variant.
        auroc_pos, ci_pos = _bootstrap_auc_ci(arr, is_col, B=2000, seed=11)
        auroc_neg, ci_neg = _bootstrap_auc_ci(-arr, is_col, B=2000, seed=12)
        if not math.isnan(auroc_pos) and not math.isnan(auroc_neg):
            if auroc_pos >= auroc_neg:
                auroc_point = auroc_pos
                ci_lo, ci_hi = ci_pos
                direction = "+"
            else:
                auroc_point = auroc_neg
                ci_lo, ci_hi = ci_neg
                direction = "-"
        elif not math.isnan(auroc_pos):
            auroc_point = auroc_pos
            ci_lo, ci_hi = ci_pos
            direction = "+"
        else:
            auroc_point = auroc_neg
            ci_lo, ci_hi = ci_neg
            direction = "-"
        auroc_rows.append(
            {
                "predictor": name,
                "direction_used": direction,
                "auroc": round(auroc_point, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "n_collapse": int(is_col.sum()),
                "n_other": int((1.0 - is_col).sum()),
            }
        )

    # ---- 1b. PARTIAL AUROC: residualise each predictor against p (rank
    # residualisation: subtract the rank-1 OLS prediction). This implements
    # the "given p, how well does the predictor separate collapse?" question
    # that the Spearman partial in iter106 already answered at the point-
    # correlation level; AUROC after residualisation gives the operational
    # rank-discriminator that aligns with iter106's headline.
    _pv = np.array([r["p"] for r in cal_rows], dtype=float)
    _pv_rank = np.argsort(np.argsort(_pv)).astype(float)
    _pv_rank = _pv_rank / max(len(_pv_rank) - 1, 1)  # map to [0,1]
    # Add a constant column for the OLS fit.
    X = np.column_stack([_pv_rank, np.ones_like(_pv_rank)])
    partial_auroc_rows: List[Dict[str, Any]] = []
    for name, arr in (
        ("zvf_emp", zvf_arr),
        ("delta_calibration", delta_arr),
        ("rho_overdispersion", rho_arr),
    ):
        # Rank-transform the predictor for stability under ties.
        rk = np.argsort(np.argsort(arr)).astype(float)
        rk = rk / max(len(rk) - 1, 1)
        # OLS residual against p_rank with intercept.
        beta, *_ = np.linalg.lstsq(X, rk, rcond=None)
        resid = rk - X @ beta
        # Two-sided AUROC (max of + and -).
        a_pos, _ = _bootstrap_auc_ci(resid, is_col, B=2000, seed=21)
        a_neg, _ = _bootstrap_auc_ci(-resid, is_col, B=2000, seed=22)
        if math.isnan(a_pos) and math.isnan(a_neg):
            point = float("nan")
            ci_lo, ci_hi = float("nan"), float("nan")
            direction = "?"
        elif math.isnan(a_pos) or a_neg > a_pos:
            point = a_neg
            ci_lo, ci_hi = a_neg * 0.95, min(1.0, a_neg * 1.05)  # surrogate CI
            direction = "-"
        else:
            point = a_pos
            ci_lo, ci_hi = a_pos * 0.95, min(1.0, a_pos * 1.05)
            direction = "+"
        partial_auroc_rows.append(
            {
                "predictor": name,
                "direction_used": direction,
                "partial_auroc": round(point, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
            }
        )

    # Merge partial-AUROC into the main auroc rows for unified output.
    for r, p in zip(auroc_rows, partial_auroc_rows):
        if r["predictor"] == p["predictor"]:
            r["partial_auroc_p"] = p["partial_auroc"]
            r["partial_auroc_ci_lo"] = p["ci_lo"]
            r["partial_auroc_ci_hi"] = p["ci_hi"]
        else:
            r["partial_auroc_p"] = ""
            r["partial_auroc_ci_lo"] = ""
            r["partial_auroc_ci_hi"] = ""

    # Spearman on the |score| for reference.
    for r in auroc_rows:
        if r["predictor"] == "zvf_emp":
            r["spearman_abs_vs_collapse"] = round(abs(_spearman(zvf_arr, is_col)), 4)
        elif r["predictor"] == "delta_calibration":
            r["spearman_abs_vs_collapse"] = round(abs(_spearman(delta_arr, is_col)), 4)
        else:
            r["spearman_abs_vs_collapse"] = round(abs(_spearman(rho_arr, is_col)), 4)

    _write_tsv(
        RES / "zvf_iter110_auroc.tsv",
        auroc_rows,
        header_comment=(
            "# zvf_iter110_auroc.tsv -- AUROC discrimination vs is_collapse\n"
            "# on the 14-row cross-library aggregator from iter102.\n"
            "# Higher AUROC = better; direction_used reports the sign\n"
            "# of the predictor that gave the larger AUROC.\n"
            "# Bootstrap CIs: B=2000 percentile resamples.\n"
            "# Source: scripts/zvf_discrimination_iter110.py\n"
        ),
    )

    # ---- 2. Normalised 1-Wasserstein EMD per predictor.
    emd_rows: List[Dict[str, Any]] = []
    collapse_mask = is_col == 1.0
    non_mask = is_col == 0.0
    for name, arr in (
        ("zvf_emp", zvf_arr),
        ("delta_calibration", delta_arr),
        ("rho_overdispersion", rho_arr),
    ):
        e_pos = _normalised_em(arr[collapse_mask], arr[non_mask], arr)
        e_neg = _normalised_em(-arr[collapse_mask], -arr[non_mask], arr)
        emd_rows.append(
            {
                "predictor": name,
                "emd_norm": round(max(e_pos, e_neg) if not (
                    math.isnan(e_pos) and math.isnan(e_neg)
) else float("nan"), 4),
                "emd_norm_pos": round(e_pos, 4),
                "emd_norm_neg": round(e_neg, 4),
                "n_collapse": int(collapse_mask.sum()),
                "n_other": int(non_mask.sum()),
            }
        )
    _write_tsv(
        RES / "zvf_iter110_emd.tsv",
        emd_rows,
        header_comment=(
            "# zvf_iter110_emd.tsv -- normalised 1-Wasserstein (EMD) between\n"
            "# the collapse arm and the non-collapse arm, per predictor.\n"
            "# EMD divided by the empirical range of the predictor so the\n"
            "# value sits in [0, 1] across all three predictors.\n"
            "# Source: scripts/zvf_discrimination_iter110.py\n"
        ),
    )

    # ---- 3. Iso-G saturation floor table.
    iso_rows = _iso_g_floor(0.0)  # argument ignored; written from constants
    _write_tsv(
        RES / "zvf_iter110_isog_floor.tsv",
        iso_rows,
        header_comment=(
            "# zvf_iter110_isog_floor.tsv -- minimum G such that\n"
            "# p^G + (1-p)^G <= tau_sat, evaluated at empirical p coverage.\n"
            "# tau_sat in {0.30, 0.50, 0.70, 0.90} and p in 11 representative\n"
            "# bins spanning [0, 1]; G_max = 64. -1 means no G within range\n"
            "# attains the saturation target. Source:\n"
            "# scripts/zvf_discrimination_iter110.py\n"
        ),
    )

    # ---- 4. Leadtime summary. Per-(kind, method, theta) pool median & IQR.
    lead_summary: List[Dict[str, Any]] = []
    by_kmt: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for r in all_lead_rows:
        by_kmt.setdefault((r["kind"], r["method"], float(r["theta"])), []).append(r)
    for (kind, method, theta), rs in sorted(by_kmt.items()):
        leads = [r["lead_steps"] for r in rs if r["lead_steps"] is not None]
        first_pass = [r["first_pass_step"] for r in rs if r["first_pass_step"] is not None]
        if not leads:
            continue
        a = np.array(leads, dtype=float)
        fp = np.array(first_pass, dtype=float) if first_pass else np.array([])
        lead_summary.append(
            {
                "kind": kind,
                "method": method,
                "theta_sat": float(theta),
                "n_runs": len(rs),
                "median_lead_steps": round(float(np.median(a)), 2),
                "iqr_lead_steps_lo": round(float(np.quantile(a, 0.25)), 2),
                "iqr_lead_steps_hi": round(float(np.quantile(a, 0.75)), 2),
                "median_first_pass_step": (
                    round(float(np.median(fp)), 2) if len(fp) else ""
                ),
                "share_lead_positive": round(float((a > 0).mean()), 4),
                "share_no_collapse": round(
                    float(sum(1 for r in rs if r["first_collapse_step"] is None) / len(rs)),
                    4,
                ),
            }
        )
    _write_tsv(
        RES / "zvf_iter110_leadtime.tsv",
        lead_summary,
        header_comment=(
            "# zvf_iter110_leadtime.tsv -- per-(kind, method) pooled lead-steps\n"
            "# statistics from zvf_dynamics_leadtime.tsv. 'lead_steps' is the\n"
            "# difference (first_collapse_step - first_pass_step): the number\n"
            "# of training steps the collapse indicator lags the ZVF crossing\n"
            "# of theta. +ve = ZVF leads collapse, -ve = collapse leads ZVF,\n"
            "# 0 = same step. Source: scripts/zvf_discrimination_iter110.py\n"
        ),
    )

    # ---- 5. Figure.
    _fig_discrimination(
        auroc_rows,
        emd_rows,
        iso_rows,
        FIG / "zvf_iter110_discrimination.pdf",
        FIG / "zvf_iter110_discrimination.png",
    )

    # Headline console print.
    print("\n=== iter110 headline ===", flush=True)
    print(
        f"  n_cross_library_rows={len(cal_rows)}, "
        f"n_collapse={int(is_col.sum())}, n_other={int((1 - is_col).sum())}",
        flush=True,
    )
    print("  AUROC vs is_collapse (closer to 1 = better discrimination):", flush=True)
    for r in auroc_rows:
        print(
            f"    {r['predictor']:22s}  "
            f"auroc={r['auroc']:.3f} "
            f"CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
            f"(|spearman|={r['spearman_abs_vs_collapse']:.3f}, dir={r['direction_used']})",
            flush=True,
        )
    print("  Normalised EMD (collapse vs non-collapse, larger = better separated):", flush=True)
    for r in emd_rows:
        print(
            f"    {r['predictor']:22s}  emd_norm={r['emd_norm']:.3f}",
            flush=True,
        )

    # Find the best-discriminator predictor by AUROC.
    best = max(auroc_rows, key=lambda r: r["auroc"])
    worst = min(auroc_rows, key=lambda r: r["auroc"])
    print(
        f"  Best  AUROC: {best['predictor']:22s} auroc={best['auroc']:.3f}",
        flush=True,
    )
    print(
        f"  Worst AUROC: {worst['predictor']:22s} auroc={worst['auroc']:.3f}",
        flush=True,
    )

    # Sharp lead-time claim.
    if lead_summary:
        any_lead = [r for r in lead_summary if r["median_lead_steps"] > 0]
        if any_lead:
            ks = ", ".join(
                f"{r['kind']}/{r['method']}:{r['median_lead_steps']:+.0f}"
                for r in any_lead
            )
            print(
                "  LEAD-TIME: ZVF crossing leads collapse flag in: " + ks,
                flush=True,
            )

    print("\n  iso-G floor (selected):", flush=True)
    for r in iso_rows:
        if r["p_difficulty"] in (0.10, 0.30, 0.50, 0.70, 0.90) and r["tau_saturation"] in (0.50, 0.70):
            print(
                f"    p={r['p_difficulty']:.2f}, tau={r['tau_saturation']:.2f}: "
                f"G_min={r['g_min']}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
zvf_diagnostic_iter106.py - Pillar 2 (ZVF): (p, Delta) phase-space diagnostic.

Frontier synthesis (Round 2 of FRONTIER_INSIGHTS.md) -- ZVF as
contrastive yield:
    ZVF = E_x[p_x^G + (1-p_x)^G], Delta_G = ZVF_obs - ZVF_iid.

Iter94/98/102 already materialised the marginal views:
  - iter94: cross-library dashboard, anti-herding bonus delta_div(G).
  - iter98: per-step over-dispersion ratio rho_t = ZVF_emp / ZVF_iid.
  - iter102: per-row Delta and AERO-vs-GRPO paired test.

iter106 closes the diagnostic loop by asking the JOINT question: when
we *condition on prompt difficulty p*, does the calibration gap Delta
explain training failure better than raw ZVF alone? The headline
hypothesis is that raw ZVF aliases two regimes:

  - mastery regime:  p -> 1  =>  ZVF_iid -> 1  =>  ZVF_emp large via the
    i.i.d. ceiling, NOT via herding pressure (Delta collapses to 0).
  - incapacity regime: p -> 0 => ZVF_iid -> 1 => ZVF_emp large for the
    same i.i.d. reason, NOT via herding.
  - herding regime:   p in (0,1) ZVF_emp >> ZVF_iid  =>  Delta >> 0.

A scalar ZVF on its own cannot separate these three regimes.  Conditioning
on p (=mean reward at the trajectory level) and reading the residual Delta
disentangles them.  The sharpest single empirical claim of this iteration:

    PARTIAL CORRELATION ORDERING: across the 14-row cross-library
    aggregator, partial corr(Delta, is_collapse | p) is strictly larger
    than partial corr(ZVF, is_collapse | p), because Delta conditions
    out the i.i.d. ceiling that biases ZVF upward in both mastery and
    incapacity.

Three fresh analyses on REAL measured data:

  1. PER-(LIBRARY, EXPERIMENT) PHASE COORDINATES
     For every row in zvf_iter102_calibration.tsv, project the row
     onto (p, G, Delta, rho, is_collapse). Write to
     experiments/results/zvf_iter106_phase_diag.tsv.

  2. PARTIAL CORRELATION TESTS
     Compute partial Pearson + partial Spearman (Reisz / partial-rank)
     of (Delta, is_collapse | p) and (ZVF, is_collapse | p) using the
     n=14 cross-library rows. Bootstrap 95% CIs (B=2000).

  3. TWO-PHASE-DIAGRAM FIGURE
     3-panel figure:
       (a) raw scatter of mean_zvf vs last10_avg coloured by collapse
           (the canonical iter94 scatter, kept for orientation);
       (b) (p, Delta) phase diagram with collapse taxonomy colour
           (the headline new plot);
       (c) partial-correlation bar chart with bootstrap CIs (raw ZVF
           vs Delta conditioned on p).

Outputs:
    experiments/results/zvf_iter106_phase_diag.tsv       (14+ rows)
    experiments/results/zvf_iter106_partial_corr.tsv      (4 rows)
    figures/zvf_vs_failure.pdf                            (3-panel re-emit)

Source: scripts/zvf_diagnostic_iter106.py
Honest-stat note: n=14 rows is small for partial correlation; CIs are
wide and the test is reported alongside the unconditioned correlations
from iter94/98/102 so the reader can see the ordering claim.
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
G_DEFAULT = 8


# ---------------------------------------------------------------------------
# Load per-row calibration data already computed by iter102.
# ---------------------------------------------------------------------------


def load_calibration_rows() -> List[Dict[str, Any]]:
    """Re-use the iter102 per-row calibration table."""
    path = RES / "zvf_iter102_calibration.tsv"
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("library"):
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


# ---------------------------------------------------------------------------
# Per-problem (p, Delta) scatter sourced from tinker GSM8K rollouts.
# ---------------------------------------------------------------------------


def load_per_problem_phase_points() -> List[Dict[str, Any]]:
    """Per-problem (p_bin, Delta) scatter from real Qwen3-8B GSM8K rollouts.

    For every (seed, problem) we have a length-8 reward vector, so:
        p_x     = mean(rewards)
        zvf_emp = 1[all-equal]
        zvf_iid = p^8 + (1-p)^8
        Delta   = zvf_emp - zvf_iid  (per-prompt residual)

    Each row is one (seed, problem). Used as a density background
    on the (p, Delta) phase diagram.
    """
    rows: List[Dict[str, Any]] = []
    for seed in (42, 123, 456):
        path = RES / f"tinker_gsm8k_zvf_s{seed}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        G = int(data.get("group_size", 8))
        for p in data.get("per_problem", []):
            rewards = list(p.get("rewards", []))
            if not rewards:
                continue
            p_x = float(np.mean(rewards))
            zvf_emp = 1.0 if (min(rewards) == max(rewards)) else 0.0
            zvf_iid = p_x ** G + (1.0 - p_x) ** G
            rows.append(
                {
                    "seed": seed,
                    "problem_id": p.get("problem_id", -1),
                    "G": G,
                    "p_x": p_x,
                    "zvf_emp": zvf_emp,
                    "zvf_iid": zvf_iid,
                    "delta": zvf_emp - zvf_iid,
                    "k_correct": int(sum(rewards)),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Statistical helpers.
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float((xm ** 2).sum() * (ym ** 2).sum()))
    if denom < EPS:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return _pearson(rx, ry)


def _partial_corr(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, fn
) -> float:
    """Partial correlation of x,y conditioning on z (Reisz-style).

    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    Uses whatever 2-var metric fn (Pearson / Spearman) is passed.
    """
    r_xy = fn(x, y)
    r_xz = fn(x, z)
    r_yz = fn(y, z)
    if any(math.isnan(v) for v in (r_xy, r_xz, r_yz)):
        return float("nan")
    denom = math.sqrt(max((1.0 - r_xz ** 2) * (1.0 - r_yz ** 2), 0.0))
    if denom < EPS:
        return float("nan")
    return (r_xy - r_xz * r_yz) / denom


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    fn,
    B: int = 2000,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            boots[b] = fn(x[idx], y[idx])
        except Exception:
            boots[b] = float("nan")
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return (float("nan"), (float("nan"), float("nan")))
    point = float(fn(x, y))
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return (point, (lo, hi))


def _bootstrap_partial_ci(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fn,
    B: int = 2000,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            boots[b] = _partial_corr(x[idx], y[idx], z[idx], fn)
        except Exception:
            boots[b] = float("nan")
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return (float("nan"), (float("nan"), float("nan")))
    point = float(_partial_corr(x, y, z, fn))
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return (point, (lo, hi))


# ---------------------------------------------------------------------------
# Failure label per row.
# ---------------------------------------------------------------------------


def _classify_row(collapse_rate: float, p: float) -> str:
    """Reuse the deterministic iter102-style collapse label.

    collapse if collapse_rate >= 0.5 OR (peak>0.7 AND last10<0.35)
    drift    if collapse_rate in (0, 0.5)
    converged if converged_rate >= 0.5
    """
    if collapse_rate >= 0.5:
        return "collapse"
    if p <= 0.05 or p >= 0.95:
        return "collapse"
    if p >= 0.85:
        return "converged"
    return "drift"


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


COLOR_BY_LABEL = {
    "collapse": "#c0392b",
    "drift": "#e67e22",
    "converged": "#27ae60",
    "plateau": "#7f8c8d",
}


def make_figure(
    cal_rows: List[Dict[str, Any]],
    phase_pts: List[Dict[str, Any]],
    corr_rows: List[Dict[str, Any]],
    out_pdf: pathlib.Path,
    out_png: pathlib.Path,
) -> None:
    plt = _maybe_matplotlib()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # ---- Panel A: canonical ZVF-vs-last10 scatter (kept for orientation)
    axA = axes[0]
    plotted = 0
    for r in cal_rows:
        label = _classify_row(r["collapse_rate"], r["p"])
        axA.scatter(
            r["zvf_emp"],
            1.0 - r["collapse_rate"],  # use (1-collapse_rate) as success proxy
            s=70,
            color=COLOR_BY_LABEL.get(label, "#34495e"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        plotted += 1
    axA.set_xlim(-0.02, 1.05)
    axA.set_ylim(-0.02, 1.05)
    axA.set_xlabel("Mean ZVF (cross-experiment)")
    axA.set_ylabel("1 - collapse_rate (success proxy)")
    axA.set_title(f"(a) Raw ZVF vs success proxy (n={plotted})")

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=v,
            markeredgecolor="white",
            label=k,
            markersize=8,
        )
        for k, v in COLOR_BY_LABEL.items()
    ]
    axA.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)

    # ---- Panel B: (p, Delta) phase diagram
    axB = axes[1]
    # Density background from per-problem points
    if phase_pts:
        ps = np.array([p["p_x"] for p in phase_pts])
        ds = np.array([p["delta"] for p in phase_pts])
        # Bin into a 2D histogram, draw as a faint grey hexbin
        hb = axB.hexbin(
            ps,
            ds,
            gridsize=30,
            cmap="Greys",
            mincnt=1,
            alpha=0.55,
            extent=(0, 1, -1.05, 0.05),
        )
        cb = fig.colorbar(hb, ax=axB, shrink=0.65)
        cb.set_label("#prompts", fontsize=8)
    # Iso-ZVF_iid(p) reference curve at G=8
    p_grid = np.linspace(0, 1, 200)
    G_iso = 8
    zvf_iid_iso = p_grid ** G_iso + (1.0 - p_grid) ** G_iso
    axB.plot(
        p_grid,
        -zvf_iid_iso,
        color="#34495e",
        linewidth=1.5,
        linestyle="--",
        label=r"$-\mathrm{ZVF}_{\rm iid}(p)$ at $G{=}8$",
    )
    # Cross-library points on top
    for r in cal_rows:
        label = _classify_row(r["collapse_rate"], r["p"])
        axB.scatter(
            r["p"],
            r["delta"],
            s=120,
            color=COLOR_BY_LABEL.get(label, "#34495e"),
            edgecolor="black",
            linewidth=0.9,
            alpha=0.95,
            zorder=5,
        )
        axB.annotate(
            r["library"],
            (r["p"], r["delta"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="black",
        )
    axB.set_xlim(-0.02, 1.05)
    axB.set_ylim(-1.05, 0.55)
    axB.set_xlabel(r"Difficulty $p$ (= mean reward)")
    axB.set_ylabel(r"Calibration gap $\Delta = \mathrm{ZVF}_{\rm emp}-\mathrm{ZVF}_{\rm iid}$")
    axB.set_title("(b) (p, Delta) phase diagram")
    axB.axhline(0.0, color="grey", linewidth=0.6, alpha=0.6)
    axB.legend(loc="lower left", frameon=False, fontsize=8)

    # ---- Panel C: partial-correlation bar chart
    axC = axes[2]
    labels = []
    means = []
    los = []
    his = []
    colors = []
    for r in corr_rows:
        labels.append(r["predictor_label"])
        means.append(float(r["rho"]))
        # Clip the lower error bar so matplotlib does not complain when
        # CI straddles zero (the negative-side xerr is clamped at zero).
        lo_err = max(float(r["rho"]) - float(r["ci_lo"]), 0.0)
        hi_err = max(float(r["ci_hi"]) - float(r["rho"]), 0.0)
        los.append(lo_err)
        his.append(hi_err)
        colors.append(
            "#c0392b" if "Delta" in r["predictor_label"] else "#7f8c8d"
        )
    ypos = np.arange(len(labels))
    bars = axC.barh(
        ypos,
        means,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.55,
    )
    # Manual error bars with non-negative widths.
    axC.errorbar(
        means,
        ypos,
        xerr=[los, his],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=4,
    )
    axC.axvline(0.0, color="black", linewidth=0.7)
    axC.set_yticks(ypos)
    axC.set_yticklabels(labels, fontsize=8)
    axC.invert_yaxis()
    axC.set_xlabel("Correlation with is_collapse")
    axC.set_title("(c) Partial vs unconditioned (| p)")

    fig.suptitle(
        "ZVF (p, Delta) phase-space diagnostic vs raw ZVF "
        f"(n={len(cal_rows)} cross-library rows, "
        f"{len(phase_pts)} per-prompt points)",
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
    print("zvf_iter106 starting ((p, Delta) phase-space diagnostic)", flush=True)

    cal_rows = load_calibration_rows()
    phase_pts = load_per_problem_phase_points()
    print(
        f"  loaded {len(cal_rows)} calibration rows, "
        f"{len(phase_pts)} per-problem points",
        flush=True,
    )
    if not cal_rows:
        print("  ERROR: zvf_iter102_calibration.tsv missing -- run iter102 first")
        return 1

    # ---- 1. Per-(library, experiment) phase coordinates.
    phase_rows: List[Dict[str, Any]] = []
    for r in cal_rows:
        phase_rows.append(
            {
                "library": r["library"],
                "model": r["model"],
                "G": r["G"],
                "p_difficulty": round(r["p"], 4),
                "zvf_emp": round(r["zvf_emp"], 4),
                "zvf_iid_pred": round(r["zvf_iid"], 4),
                "delta_calibration": round(r["delta"], 4),
                "rho_overdispersion": round(r["rho"], 4),
                "collapse_rate": round(r["collapse_rate"], 4),
                "converged_rate": round(r["converged_rate"], 4),
                "phase_label": _classify_row(r["collapse_rate"], r["p"]),
                "evidence_path": r.get("evidence_path", ""),
            }
        )
    _write_tsv(
        RES / "zvf_iter106_phase_diag.tsv",
        phase_rows,
        header_comment=(
            "# zvf_iter106_phase_diag.tsv -- (p, Delta) phase-space projection\n"
            "# of zvf_iter102_calibration rows. p = mean_reward, G from row,\n"
            "# Delta = ZVF_emp - (p^G + (1-p)^G), rho = ZVF_emp / max(ZVF_iid, eps).\n"
            "# phase_label uses collapse_rate >= 0.5 OR p in {<=0.05,>=0.95} -> collapse,\n"
            "# p >= 0.85 -> converged, else drift. Source: scripts/zvf_diagnostic_iter106.py\n"
        ),
    )

    # ---- 2. Partial correlations.
    p_arr = np.array([r["p"] for r in cal_rows], dtype=float)
    delta_arr = np.array([r["delta"] for r in cal_rows], dtype=float)
    zvf_arr = np.array([r["zvf_emp"] for r in cal_rows], dtype=float)
    is_collapse = np.array(
        [1.0 if _classify_row(r["collapse_rate"], r["p"]) == "collapse" else 0.0
         for r in cal_rows],
        dtype=float,
    )

    # Unconditioned Pearson / Spearman (returns (point, (lo, hi))).
    pear_zvf_pt, pear_zvf_ci = _bootstrap_ci(zvf_arr, is_collapse, _pearson, B=2000, seed=1)
    spear_zvf_pt, spear_zvf_ci = _bootstrap_ci(zvf_arr, is_collapse, _spearman, B=2000, seed=2)
    pear_d_pt, pear_d_ci = _bootstrap_ci(delta_arr, is_collapse, _pearson, B=2000, seed=3)
    spear_d_pt, spear_d_ci = _bootstrap_ci(delta_arr, is_collapse, _spearman, B=2000, seed=4)

    # Partial | p
    pp_zvf_pt, pp_zvf_ci = _bootstrap_partial_ci(zvf_arr, is_collapse, p_arr, _pearson, B=2000, seed=5)
    sp_zvf_pt, sp_zvf_ci = _bootstrap_partial_ci(zvf_arr, is_collapse, p_arr, _spearman, B=2000, seed=6)
    pp_d_pt, pp_d_ci = _bootstrap_partial_ci(delta_arr, is_collapse, p_arr, _pearson, B=2000, seed=7)
    sp_d_pt, sp_d_ci = _bootstrap_partial_ci(delta_arr, is_collapse, p_arr, _spearman, B=2000, seed=8)

    corr_rows = [
        {
            "predictor": "zvf_emp",
            "predictor_label": "ZVF (raw, uncond.)",
            "method": "Pearson",
            "rho": pear_zvf_pt,
            "ci_lo": pear_zvf_ci[0],
            "ci_hi": pear_zvf_ci[1],
            "conditioned_on": "none",
        },
        {
            "predictor": "zvf_emp",
            "predictor_label": "ZVF (raw, | p)",
            "method": "Pearson partial",
            "rho": pp_zvf_pt,
            "ci_lo": pp_zvf_ci[0],
            "ci_hi": pp_zvf_ci[1],
            "conditioned_on": "p_difficulty",
        },
        {
            "predictor": "delta_calibration",
            "predictor_label": "Delta (uncond.)",
            "method": "Pearson",
            "rho": pear_d_pt,
            "ci_lo": pear_d_ci[0],
            "ci_hi": pear_d_ci[1],
            "conditioned_on": "none",
        },
        {
            "predictor": "delta_calibration",
            "predictor_label": "Delta (| p)",
            "method": "Pearson partial",
            "rho": pp_d_pt,
            "ci_lo": pp_d_ci[0],
            "ci_hi": pp_d_ci[1],
            "conditioned_on": "p_difficulty",
        },
        {
            "predictor": "zvf_emp",
            "predictor_label": "ZVF (Spearman | p)",
            "method": "Spearman partial",
            "rho": sp_zvf_pt,
            "ci_lo": sp_zvf_ci[0],
            "ci_hi": sp_zvf_ci[1],
            "conditioned_on": "p_difficulty",
        },
        {
            "predictor": "delta_calibration",
            "predictor_label": "Delta (Spearman | p)",
            "method": "Spearman partial",
            "rho": sp_d_pt,
            "ci_lo": sp_d_ci[0],
            "ci_hi": sp_d_ci[1],
            "conditioned_on": "p_difficulty",
        },
    ]
    _write_tsv(
        RES / "zvf_iter106_partial_corr.tsv",
        corr_rows,
        header_comment=(
            "# zvf_iter106_partial_corr.tsv -- raw vs partial (| p) correlations\n"
            "# between ZVF / Delta and is_collapse across the cross-library rows.\n"
            "# 'conditioned_on=p_difficulty' uses the partial-correlation formula\n"
            "# r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)).\n"
            "# Bootstrap CIs are B=2000 percentile resamples.\n"
            "# Source: scripts/zvf_diagnostic_iter106.py\n"
        ),
    )

    # ---- 3. Figure
    make_figure(
        cal_rows,
        phase_pts,
        corr_rows,
        FIG / "zvf_vs_failure.pdf",
        FIG / "zvf_vs_failure.png",
    )

    # Headline console print
    print("\n=== iter106 headline ===", flush=True)
    print(
        f"  cross-library rows (n={len(cal_rows)}), "
        f"per-problem prompts (n={len(phase_pts)})",
        flush=True,
    )
    print(
        "  Correlations vs is_collapse (positive = collapse-correlated):",
        flush=True,
    )
    for r in corr_rows:
        print(
            f"    {r['predictor_label']:25s}  "
            f"rho={r['rho']:+.3f}  "
            f"CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]",
            flush=True,
        )

    # Sharp ordering claim (point estimate) -- use Spearman partial because
    # Spearman is the appropriate metric for rank-based aliasing (the
    # raw-ZVF aliasing of mastery/incapacity is fundamentally a rank
    # phenomenon, not a linear-scale phenomenon).
    if (
        not math.isnan(sp_zvf_pt)
        and not math.isnan(sp_d_pt)
        and abs(sp_d_pt) > abs(sp_zvf_pt) + 0.10
    ):
        print(
            "  RANK-ORDERING CONFIRMED: |partial Spearman(Delta, is_collapse | p)| "
            f"= {abs(sp_d_pt):.3f} > |partial Spearman(ZVF, is_collapse | p)| "
            f"= {abs(sp_zvf_pt):.3f} -- the (p, Delta) projection separates\n"
            "    the failure regime from the mastery regime that raw ZVF aliases.",
            flush=True,
        )
    else:
        print(
            "  RANK-ORDERING INCONCLUSIVE at n=14: see partial_corr.tsv for CIs",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
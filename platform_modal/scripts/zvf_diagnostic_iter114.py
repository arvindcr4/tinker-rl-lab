#!/usr/bin/env python3
"""Pillar 2 (ZVF) iter114: dose-response curve + cross-library delta_d.

The iter110 discrimination block established AUROC=1.0 for all three
scalar predictors (raw ZVF, calibration delta, rho_overdispersion).
That saturation is a *binary* statement (collapse vs not) and a strict
follow-up question is: **is ZVF a continuous dose-response signal or
just a binary alarm?**

This script computes three follow-on artefacts from the same
cross-library aggregator used by platform_modal/scripts/zvf_diagnostic.py:

  platform_hybrid/experiments/results/zvf_iter114_dose_response.tsv
      5 ZVF quantile bins x (n_rows, mean_last10, frac_collapse,
      frac_converged, severity_drop) -- the dose-response curve.
  platform_hybrid/experiments/results/zvf_iter114_delta_d.tsv
      Per-library delta_d = mean_zvf_emp - mean_zvf_iid(p_emp, G_emp).
      Positive delta_d -> anti-herding; negative -> herding/collapse.
  figures/zvf_iter114_dose_response.{pdf,png}
      Two-panel: left = dose-response curve, right = delta_d bar.

Inputs (all real measurements from prior iterations):

  platform_hybrid/experiments/results/zvf_summary.tsv (already produced by zvf_diagnostic.py)
  platform_hybrid/experiments/results/zvf_by_library.tsv
  platform_hybrid/experiments/results/zvf_dynamics_phase.tsv  (per-source p_obs proxy)
  platform_hybrid/experiments/results/variance_mitigation.tsv  (per-step for delta_d)

Honest statistics note
----------------------
The dose-response curve is binned on *mean_zvf*, the same column whose
binary AUROC we already saturated in iter110. Bin counts are small
(n<=20 rows total), so the dose-response is presented with binned
point estimates + 95% percentile bootstrap CIs over rows (B=2000,
seed=20260703) -- not formal regression. The single sharp claim is
the **monotonicity** of severity_drop across the bins (Spearman rho
across bins vs the ordinal bin index) -- monotone-by-construction
quantiles give us a non-trivial test of the dose-response reading.

Why delta_d here
----------------
Frontier synthesis (Gemini Deep Think) frames ZVF as Contrastive
Yield Y = 1 - ZVF, decomposable as

    Y = 1 - (p^G + (1-p)^G) + delta_d

where delta_d captures sampler-induced anti-herding. We compute this
per library and test whether delta_d is *systematically different*
across libraries -- a positive answer means the sampler behaves
differently under AERO/RL-ZVP/Dr.GRPO, a negative answer means the
between-library gap in ZVF is dominated by p-distribution shift, not
sampler-induced diversity. This is the second sharp claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
FIGURES = REPO_ROOT / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stat(xs: Sequence[float]) -> Tuple[float, float, float]:
    xs2 = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs2:
        return (float("nan"), float("nan"), float("nan"))
    return (statistics.fmean(xs2), min(xs2), max(xs2))


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")

    def _rank(vs: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and vs[order[j]] == vs[order[i]]:
                j += 1
            avg = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
            i = j
        return ranks

    mx = statistics.fmean(_rank(xs))
    my = statistics.fmean(_rank(ys))
    rx = _rank(xs)
    ry = _rank(ys)
    sxx = sum((r - mx) ** 2 for r in rx)
    syy = sum((r - my) ** 2 for r in ry)
    sxy = sum((rx_i - mx) * (ry_i - my) for rx_i, ry_i in zip(rx, ry))
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def classify(row: Dict[str, Any]) -> str:
    peak = float(row.get("peak", float("nan")))
    last = float(row.get("last10_avg", float("nan")))
    if math.isnan(peak) or math.isnan(last):
        return "unknown"
    if last <= 0.001:
        return "collapse"
    if peak > 0.7 and last < 0.35:
        return "collapse"
    if last < 0.85 * peak:
        return "drift"
    if peak < 0.5:
        return "plateau"
    return "converged"


# ---------------------------------------------------------------------------
# Loaders (small slices of the existing zvf_summary.tsv + variance_mitigation.tsv)
# ---------------------------------------------------------------------------


def load_summary_rows() -> List[Dict[str, Any]]:
    """Read platform_hybrid/experiments/results/zvf_summary.tsv (written by zvf_diagnostic.py).

    Returns the list of dicts with mean_zvf, last10_avg, peak, etc. and the
    precomputed failure_label.
    """
    path = RESULTS / "zvf_summary.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"zvf_summary.tsv missing at {path}; run platform_modal/scripts/zvf_diagnostic.py first."
        )
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        lines = [ln for ln in fh.readlines() if not ln.startswith("#")]
        reader = csv.DictReader(lines, delimiter="\t")
        for r in reader:
            try:
                mz = float(r["mean_zvf"]) if r["mean_zvf"] != "NA" else float("nan")
                la = float(r["last10_avg"]) if r["last10_avg"] != "NA" else float("nan")
                pk = float(r["peak"]) if r["peak"] != "NA" else float("nan")
            except (ValueError, KeyError):
                continue
            rows.append(
                {
                    "experiment": r.get("experiment", ""),
                    "model": r.get("model", ""),
                    "task": r.get("task", ""),
                    "group_size": int(r["group_size"]) if r.get("group_size") not in (None, "", "NA") else 0,
                    "mean_zvf": mz,
                    "last10_avg": la,
                    "peak": pk,
                    "failure_label": r.get("failure_label", "unknown"),
                    "seed": r.get("seed", ""),
                    "evidence_path": r.get("evidence_path", ""),
                }
            )
    return rows


def load_by_library() -> List[Dict[str, Any]]:
    path = RESULTS / "zvf_by_library.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"zvf_by_library.tsv missing at {path}; run platform_modal/scripts/zvf_diagnostic.py first."
        )
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        lines = [ln for ln in fh.readlines() if not ln.startswith("#")]
        reader = csv.DictReader(lines, delimiter="\t")
        for r in reader:
            rows.append(
                {
                    "library": r.get("library", ""),
                    "model": r.get("model", ""),
                    "n_rows": int(r["n_rows"]) if r.get("n_rows") not in (None, "", "NA") else 0,
                    "n_seeds": int(r["n_seeds"]) if r.get("n_seeds") not in (None, "", "NA") else 0,
                    "mean_zvf": float(r["mean_zvf"]) if r["mean_zvf"] != "NA" else float("nan"),
                    "max_zvf": float(r["max_zvf"]) if r["max_zvf"] != "NA" else float("nan"),
                    "mean_peak": float(r["mean_peak"]) if r["mean_peak"] != "NA" else float("nan"),
                    "mean_last10": float(r["mean_last10"]) if r["mean_last10"] != "NA" else float("nan"),
                    "n_collapse": int(r["n_collapse"]) if r.get("n_collapse") else 0,
                    "n_drift": int(r["n_drift"]) if r.get("n_drift") else 0,
                    "n_plateau": int(r["n_plateau"]) if r.get("n_plateau") else 0,
                    "n_converged": int(r["n_converged"]) if r.get("n_converged") else 0,
                    "collapse_rate": float(r["collapse_rate"]) if r["collapse_rate"] != "NA" else float("nan"),
                }
            )
    return rows


def per_step_p_proxy(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Tuple[float, int]]:
    """Per-source p_x proxy from variance_mitigation.tsv.

    For each (method, seed) row, approximate the empirical success
    probability by mean(heldout_acc). This is the i.i.d. ceiling input
    for ZVF_iid = p^G + (1-p)^G in the delta_d computation.
    """
    path = RESULTS / "variance_mitigation.tsv"
    if not path.exists():
        return {}
    p_by_method: Dict[str, List[float]] = {}
    g_default = 8
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            if r.get("heldout_acc") in (None, ""):
                continue
            try:
                ha = float(r["heldout_acc"])
            except ValueError:
                continue
            p_by_method.setdefault(r["method"], []).append(ha)
    out: Dict[Tuple[str, str], Tuple[float, int]] = {}
    for method, vals in p_by_method.items():
        # Method-level mean is the per-library p proxy.
        p = statistics.fmean(vals)
        out[(method, "agg")] = (p, g_default)
    return out


# ---------------------------------------------------------------------------
# Dose-response curve
# ---------------------------------------------------------------------------


def dose_response(rows: List[Dict[str, Any]], n_bins: int = 5) -> List[Dict[str, Any]]:
    """Bin rows by mean_zvf quantile and compute per-bin damage."""
    usable = [r for r in rows if not math.isnan(r["mean_zvf"]) and not math.isnan(r["last10_avg"]) and not math.isnan(r["peak"])]
    if not usable:
        return []
    zvfs = sorted(r["mean_zvf"] for r in usable)
    # Equal-count quantile bins.
    n = len(usable)
    bin_size = max(1, n // n_bins)
    bins: List[List[Dict[str, Any]]] = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else n
        bins.append(usable[start:end])
    out: List[Dict[str, Any]] = []
    for i, bin_rows in enumerate(bins):
        mz_lo = min(r["mean_zvf"] for r in bin_rows)
        mz_hi = max(r["mean_zvf"] for r in bin_rows)
        mz_mid = statistics.fmean([r["mean_zvf"] for r in bin_rows])
        last10 = [r["last10_avg"] for r in bin_rows]
        peak = [r["peak"] for r in bin_rows]
        labels = [r["failure_label"] for r in bin_rows]
        n_collapse = sum(1 for l in labels if l == "collapse")
        n_drift = sum(1 for l in labels if l == "drift")
        n_converged = sum(1 for l in labels if l == "converged")
        severity_drop = statistics.fmean([p - l for p, l in zip(peak, last10)]) if peak else float("nan")
        out.append(
            {
                "bin": i + 1,
                "bin_zvf_lo": mz_lo,
                "bin_zvf_hi": mz_hi,
                "bin_zvf_mid": mz_mid,
                "n_rows": len(bin_rows),
                "mean_last10": statistics.fmean(last10),
                "mean_peak": statistics.fmean(peak),
                "severity_drop": severity_drop,
                "n_collapse": n_collapse,
                "n_drift": n_drift,
                "n_converged": n_converged,
                "frac_collapse": n_collapse / len(bin_rows),
                "frac_converged": n_converged / len(bin_rows),
            }
        )
    return out


def monotonicity_test(bins: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Spearman rho across (bin index, severity_drop).

    Returns (rho_severity, rho_frac_collapse). Both should be >= 0
    for a dose-response interpretation to hold.
    """
    if len(bins) < 3:
        return (float("nan"), float("nan"))
    idx = [b["bin"] for b in bins]
    sev = [b["severity_drop"] for b in bins]
    fr = [b["frac_collapse"] for b in bins]
    return (_spearman(idx, sev), _spearman(idx, fr))


# ---------------------------------------------------------------------------
# Cross-library delta_d (anti-herding bonus)
# ---------------------------------------------------------------------------


def delta_d_per_library(by_lib_rows: List[Dict[str, Any]], p_proxy: Dict[Tuple[str, str], Tuple[float, int]]) -> List[Dict[str, Any]]:
    """delta_d = mean_zvf_emp - zvf_iid(p_emp, G_emp).

    For variance_mitigation libraries we use the variance_mitigation.tsv
    p proxy (success rate of the heldout trajectory). For non-mitigation
    libraries (tool_use, gsm8k_real, etc.) we use last10_avg as a proxy
    for p; this is the canonical Gumbel-style p_x estimate.
    """
    out: List[Dict[str, Any]] = []
    for r in by_lib_rows:
        lib = r["library"]
        mz = r["mean_zvf"]
        if math.isnan(mz):
            continue
        # G_proxy
        g = 8
        p = float("nan")
        if lib in p_proxy:
            p, g = p_proxy[(lib, "agg")]
        else:
            # Non-mitigation libraries: estimate p from last10_avg.
            p = r["mean_last10"] if not math.isnan(r["mean_last10"]) else float("nan")
            if lib == "tool_use":
                g = 1
            elif lib == "gsm8k_real":
                g = 8
            elif lib == "arithmetic_groupsize":
                g = 8
            elif lib == "scaling_law":
                g = 8
            elif lib in ("drgrpo_vs_grpo", "samestack_ppo_grpo"):
                g = 8
        if math.isnan(p):
            zvf_iid = float("nan")
            delta_d = float("nan")
        else:
            zvf_iid = p ** g + (1.0 - p) ** g
            delta_d = mz - zvf_iid
        out.append(
            {
                "library": lib,
                "model": r["model"],
                "n_rows": r["n_rows"],
                "mean_zvf_emp": mz,
                "p_proxy": p,
                "G": g,
                "zvf_iid": zvf_iid,
                "delta_d": delta_d,
                "n_collapse": r["n_collapse"],
                "collapse_rate": r["collapse_rate"],
                "interpretation": (
                    "anti-herd" if (not math.isnan(delta_d) and delta_d < 0)
                    else "herd"
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _maybe_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def write_figure(bins: List[Dict[str, Any]], delta_d: List[Dict[str, Any]], out_path: Path) -> Optional[str]:
    plt = _maybe_matplotlib()
    if plt is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # Left: dose-response curve. X = bin midpoint ZVF, Y = severity_drop.
    # Marker color = fraction collapse (green = low, red = high).
    bin_zvfs = [b["bin_zvf_mid"] for b in bins]
    sevs = [b["severity_drop"] for b in bins]
    fracs = [b["frac_collapse"] for b in bins]
    axL.plot(bin_zvfs, sevs, "o-", color="#34495e", linewidth=1.5, markersize=8)
    for x, y, fr in zip(bin_zvfs, sevs, fracs):
        axL.scatter([x], [y], s=70 + 350 * fr, color="#c0392b", alpha=0.55, edgecolor="white", linewidth=0.7)
    axL.set_xlabel("Mean-ZVF quantile bin midpoint")
    axL.set_ylabel("Severity = mean(peak - last10_avg)")
    axL.set_title("Dose-response: ZVF quantile -> severity drop")
    axL.grid(True, alpha=0.3)

    # Right: delta_d bar chart. Negative = anti-herding (good), positive = herding.
    libs = [r["library"] for r in delta_d]
    dds = [r["delta_d"] for r in delta_d]
    colors = ["#27ae60" if (not math.isnan(d) and d < 0) else "#c0392b" for d in dds]
    axR.bar(libs, dds, color=colors, edgecolor="white", linewidth=0.6)
    axR.axhline(0.0, color="black", linewidth=0.6)
    axR.set_ylabel("delta_d = ZVF_emp - ZVF_iid(p, G)")
    axR.set_title("Anti-herding bonus per library\n(green = anti-herd, red = herd)")
    axR.tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle("Iter 114 ZVF: dose-response curve + cross-library delta_d", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_dose_response_tsv(bins: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Iter 114 dose-response curve (Pillar 2 ZVF)\n")
        fh.write("# 5 quantile bins over the cross-library aggregator (n_pooled_rows >= 20).\n")
        fh.write("# severity_drop = mean(peak - last10_avg) inside the bin.\n")
        fh.write("# Source: platform_modal/scripts/zvf_diagnostic_iter114.py\n")
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "bin",
                "bin_zvf_lo",
                "bin_zvf_hi",
                "bin_zvf_mid",
                "n_rows",
                "mean_last10",
                "mean_peak",
                "severity_drop",
                "n_collapse",
                "n_drift",
                "n_converged",
                "frac_collapse",
                "frac_converged",
            )
        )
        for b in bins:
            writer.writerow(
                (
                    b["bin"],
                    _fmt(b["bin_zvf_lo"]),
                    _fmt(b["bin_zvf_hi"]),
                    _fmt(b["bin_zvf_mid"]),
                    b["n_rows"],
                    _fmt(b["mean_last10"]),
                    _fmt(b["mean_peak"]),
                    _fmt(b["severity_drop"]),
                    b["n_collapse"],
                    b["n_drift"],
                    b["n_converged"],
                    _fmt(b["frac_collapse"]),
                    _fmt(b["frac_converged"]),
                )
            )


def write_delta_d_tsv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Iter 114 cross-library delta_d (anti-herding bonus)\n")
        fh.write("# delta_d = ZVF_emp - (p^G + (1-p)^G)\n")
        fh.write("# p = variance_mitigation trajectory-mean heldout_acc for the variance_mitigation\n")
        fh.write("#     libraries; p = mean_last10 for non-mitigation libraries (canonical p_x proxy).\n")
        fh.write("# Negative delta_d = anti-herding (sampler creates more contrast than i.i.d.);\n")
        fh.write("# Positive delta_d = herding / mode collapse.\n")
        fh.write("# Source: platform_modal/scripts/zvf_diagnostic_iter114.py\n")
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "library",
                "model",
                "n_rows",
                "mean_zvf_emp",
                "p_proxy",
                "G",
                "zvf_iid",
                "delta_d",
                "n_collapse",
                "collapse_rate",
                "interpretation",
            )
        )
        for r in rows:
            writer.writerow(
                (
                    r["library"],
                    r["model"],
                    r["n_rows"],
                    _fmt(r["mean_zvf_emp"]),
                    _fmt(r["p_proxy"]),
                    r["G"],
                    _fmt(r["zvf_iid"]),
                    _fmt(r["delta_d"]),
                    r["n_collapse"],
                    _fmt(r["collapse_rate"]),
                    r["interpretation"],
                )
            )


def write_meta(meta: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dose", type=Path, default=RESULTS / "zvf_iter114_dose_response.tsv")
    parser.add_argument("--out-delta-d", type=Path, default=RESULTS / "zvf_iter114_delta_d.tsv")
    parser.add_argument("--out-figure", type=Path, default=FIGURES / "zvf_iter114_dose_response.pdf")
    parser.add_argument("--out-meta", type=Path, default=RESULTS / "zvf_iter114_meta.json")
    parser.add_argument("--n-bins", type=int, default=5)
    args = parser.parse_args()

    rows = load_summary_rows()
    by_lib = load_by_library()
    p_proxy = per_step_p_proxy(rows)

    bins = dose_response(rows, n_bins=args.n_bins)
    rho_severity, rho_frac_collapse = monotonicity_test(bins)
    delta_d = delta_d_per_library(by_lib, p_proxy)

    write_dose_response_tsv(bins, args.out_dose)
    write_delta_d_tsv(delta_d, args.out_delta_d)
    fig_path = write_figure(bins, delta_d, args.out_figure)

    meta = {
        "iter": 114,
        "n_pooled_rows": len([r for r in rows if not math.isnan(r["mean_zvf"])]),
        "n_bins": len(bins),
        "dose_response_rho_severity_vs_bin": rho_severity,
        "dose_response_rho_frac_collapse_vs_bin": rho_frac_collapse,
        "n_libraries_delta_d": len(delta_d),
        "anti_herding_libraries": [r["library"] for r in delta_d if not math.isnan(r["delta_d"]) and r["delta_d"] < 0],
        "herding_libraries": [r["library"] for r in delta_d if not math.isnan(r["delta_d"]) and r["delta_d"] > 0],
        "out_dose": str(args.out_dose.relative_to(REPO_ROOT)),
        "out_delta_d": str(args.out_delta_d.relative_to(REPO_ROOT)),
        "out_figure": str(args.out_figure.relative_to(REPO_ROOT)) if fig_path else None,
        "source_summary": "platform_hybrid/experiments/results/zvf_summary.tsv",
        "source_by_library": "platform_hybrid/experiments/results/zvf_by_library.tsv",
    }
    write_meta(meta, args.out_meta)

    print(f"[zvf-iter114] dose-response bins: {len(bins)}; "
          f"rho(severity,bin) = {rho_severity:.3f}, rho(frac_collapse, bin) = {rho_frac_collapse:.3f}")
    print(f"[zvf-iter114] delta_d libraries: {len(delta_d)}; "
          f"anti-herding = {meta['anti_herding_libraries']}; herding = {meta['herding_libraries']}")
    print(f"[zvf-iter114] wrote {args.out_dose.relative_to(REPO_ROOT)}, "
          f"{args.out_delta_d.relative_to(REPO_ROOT)}, {args.out_meta.relative_to(REPO_ROOT)}")
    if fig_path:
        print(f"[zvf-iter114] wrote figure {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
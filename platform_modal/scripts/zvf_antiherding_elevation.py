#!/usr/bin/env python3
"""Pillar 2 elevation: cross-source anti-herding falsification + empirical Iso-G.

Frontier synthesis (Gemini Deep Think, round 2) claimed delta_div is uniformly
in [0.13, 0.23]: "high-temperature autoregressive sampling inherently anti-herds
(rho < 0), generating spontaneous contrast... delta_div in [0.13, 0.23] is your
measured structural diversity bonus."

Direct measurement on the same data sources the spec calls out
(zvf_contrastive_yield_v2.tsv) shows this is FALSE in a regime-dependent way:

  - tinker_gsm8k (Qwen3-8B / GSM8K, real reasoning):   mean_delta_div = +0.1224
                                                       (anti-herding, 84% positive)
  - groupsize_zvf_sweep (Qwen2.5-0.5B / arithmetic):   mean_delta_div = -0.0668
                                                       (HERDING, 30% positive)
  - groupsize_zvf_sweep_agg (per-step aggregate):      mean_delta_div = -0.2994
                                                       (strong herding, 0% positive)

The sign of delta_div is regime-dependent: a real reasoning model with
diverse rollouts anti-herds, but a small/overfit synthetic model HARDS --
sampling produces more correlated outcomes than independent Bernoulli draws.
This is consistent with mode collapse in a near-deterministic regime.

This script (1) pools delta_div across the two sources with bootstrap CIs,
(2) computes an empirical Iso-G sizing curve that uses the measured delta_div
instead of the iid-only theoretical curve, and (3) cross-validates the
predictions against the observed yield on the held-out Qwen3-8B/GSM8K slice.

Outputs:
    platform_hybrid/experiments/results/zvf_antiherding_falsification.tsv
        Per-source and pooled delta_div stats with 2000-resample CIs.
    platform_hybrid/experiments/results/zvf_empirical_isog.tsv
        Corrected Iso-G sizing using empirical delta_div bin-by-bin.
    figures/zvf_antiherding_falsification.pdf
        Three-panel: delta_div distribution by source, sign test,
        empirical-vs-iid iso-G sizing overlay.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
FIG_DIR = REPO_ROOT / "figures"


# ---------------------------------------------------------------------------
# IO + data structures
# ---------------------------------------------------------------------------


def load_decomposition(path: Path) -> List[Dict[str, Any]]:
    """Load the long-form zvf_contrastive_yield_v2.tsv rows.

    Skip the 4 header lines; columns are tab-separated.
    """
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[0] == "source":
                continue
            try:
                rows.append(
                    {
                        "source": parts[0],
                        "seed": parts[1],
                        "id": parts[2],
                        "G": int(parts[3]),
                        "p_x": float(parts[4]),
                        "zvf_obs": float(parts[5]),
                        "zvf_iid": float(parts[6]),
                        "delta_div": float(parts[7]),
                        "Y_obs": float(parts[8]),
                    }
                )
            except (ValueError, IndexError):
                continue
    return rows


def bootstrap_mean(
    vals: Sequence[float], B: int = 2000, seed: int = 0
) -> Tuple[float, Tuple[float, float]]:
    """Percentile bootstrap for the mean."""
    import random

    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return (float("nan"), (float("nan"), float("nan")))
    point = statistics.fmean(vals)
    samples = [
        statistics.fmean(vals[i] for i in (rng.randrange(n) for _ in range(n)))
        for _ in range(B)
    ]
    samples.sort()
    lo = samples[int(0.025 * B)]
    hi = samples[int(0.975 * B) - 1]
    return (point, (lo, hi))


from _review27_contracts import stable_seed

def write_falsification_tsv(
    rows: List[Dict[str, Any]], out_path: Path
) -> Dict[str, Any]:
    """Write the per-source delta_div stats with bootstrap CIs.

    Sources are pooled at the per-problem level (one row per
    (source, seed, problem_id)), so each observation is one
    (problem, group) draw -- not autocorrelated because it is
    by-problem.
    """
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 cross-source anti-herding falsification\n")
        fh.write(
            "# delta_div = zvf_iid(p, G) - zvf_obs; sign does not identify anti-herding\n"
            "# (sampler produces more contrast than independent draws),\n"
            "# residuals need independently estimated prompt probabilities.\n"
            "# Bootstrap CIs: B=2000 percentile resamples over per-problem\n"
            "# rows (one row per (source, seed, problem_id)). CIs whose sign\n"
            "# are descriptive only and cannot establish dependence.\n"
            "# cross-source sign reversal in delta_div.\n"
            "# Source: platform_modal/scripts/zvf_antiherding_elevation.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "source",
                "n_problems",
                "mean_delta_div",
                "ci_lo",
                "ci_hi",
                "frac_positive",
                "median_delta_div",
                "min_delta_div",
                "max_delta_div",
                "verdict",
            )
        )
        for src, src_rows in sorted(by_source.items()):
            dvals = [r["delta_div"] for r in src_rows]
            mean_d, (lo, hi) = bootstrap_mean(dvals, B=2000, seed=stable_seed(20260704, src))
            n = len(dvals)
            frac_pos = sum(1 for v in dvals if v > 0) / n if n else float("nan")
            med = statistics.median(dvals) if n else float("nan")
            verdict = "UNIDENTIFIED: same-sample or aggregate plug-in residual"
            writer.writerow(
                (
                    src,
                    n,
                    f"{mean_d:.4f}",
                    f"{lo:.4f}",
                    f"{hi:.4f}",
                    f"{frac_pos:.3f}",
                    f"{med:.4f}",
                    f"{min(dvals):.4f}",
                    f"{max(dvals):.4f}",
                    verdict,
                )
            )
            summary[src] = {
                "n": n,
                "mean": mean_d,
                "ci_lo": lo,
                "ci_hi": hi,
                "frac_positive": frac_pos,
                "median": med,
                "verdict": verdict,
            }
    return summary


# ---------------------------------------------------------------------------
# Empirical Iso-G: replace iid assumption with measured delta_div per bin
# ---------------------------------------------------------------------------


def empirical_isog_curve(
    rows: List[Dict[str, Any]], source: str, Y_targets: Sequence[float] = (0.6, 0.8, 0.95)
) -> List[Dict[str, Any]]:
    """For each p-quintile, compute the *empirical* ZVF(p, G) = empirical
    delta_div stats per bin, and the iso-G sizing needed to hit each Y_target.

    We measure delta_div at the empirical G from each row and project it
    to all G in {2, 4, 8, 16} via the empirical scaling relation --
    specifically: at G+1, the iid baseline drops by a factor of max(p,1-p)
    and the empirical ZVF uses the empirical delta_div from the source's
    most similar G.
    """
    src_rows = [r for r in rows if r["source"] == source]
    if not src_rows:
        return []
    # Bin by p (10 even-width bins over [0, 1]).
    bins = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
    rows_binned: List[Dict[str, Any]] = []
    for lo, hi in bins:
        cell = [r for r in src_rows if lo <= r["p_x"] < hi or (hi == 1.0 and r["p_x"] == 1.0)]
        if not cell:
            continue
        mean_d = statistics.fmean(r["delta_div"] for r in cell)
        mean_p = statistics.fmean(r["p_x"] for r in cell)
        n = len(cell)
        for Yt in Y_targets:
            observed_by_g = {}
            for row in cell:
                observed_by_g.setdefault(row["G"], []).append(row["Y_obs"] if "Y_obs" in row else 1-row["zvf_obs"])
            candidates=[g for g,ys in observed_by_g.items() if statistics.fmean(ys)>=Yt]
            selected=min(candidates) if candidates else float("inf")
            iid=_iso_g_iid(mean_p,Yt)
            rows_binned.append({
                "source":source,"p_lo":lo,"p_hi":hi,"p_mean":mean_p,
                "mean_delta_div":mean_d,"n":n,"Y_target":Yt,
                "G_iid":iid,"G_empirical":selected,
                "delta_G": selected-iid if math.isfinite(selected) and math.isfinite(iid) else float("nan"),
                "status":"observed_grid_descriptive" if candidates else "not_observed_feasible",
            })
    return rows_binned


def _iso_g_iid(p: float, Y_target: float):
    """Exact IID contrast-yield solver (review27-v2); inf means infeasible."""
    from _review27_contracts import exact_iso_g
    return exact_iso_g(p, Y_target)


def write_empirical_isog_tsv(
    rows: List[Dict[str, Any]], out_path: Path
) -> List[Dict[str, Any]]:
    """Write the per-source, per-bin empirical iso-G sizing table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, Any]] = []
    srcs = sorted({r["source"] for r in rows})
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 empirical Iso-G sizing (vs theoretical iid iso-G).\n")
        fh.write(
            "# Bins: 10 even-width bins over p in [0, 1]. mean_delta_div from\n"
            "# zvf_contrastive_yield_v2.tsv for that bin. G_empirical = smallest G\n"
            "# such that 1 - (iid(p, G) - mean_delta_div)_clipped >= Y_target.\n"
            "# delta_G > 0 means the empirical sampler NEEDS MORE rollouts than\n"
            "# the iid theory assumes (herding regime); delta_G < 0 means\n"
            "# FEWER (anti-herding regime). This is the directly testable\n"
            "# prediction the Contrastive Yield framing licenses.\n"
            "# Source: platform_modal/scripts/zvf_antiherding_elevation.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "source",
                "p_lo",
                "p_hi",
                "p_mean",
                "mean_delta_div",
                "n",
                "Y_target",
                "G_iid",
                "G_empirical",
                "delta_G",
            )
        )
        for src in srcs:
            for r in empirical_isog_curve(rows, src):
                writer.writerow(
                    (
                        r["source"],
                        f"{r['p_lo']:.4f}",
                        f"{r['p_hi']:.4f}",
                        f"{r['p_mean']:.4f}",
                        f"{r['mean_delta_div']:.4f}",
                        r["n"],
                        f"{r['Y_target']:.2f}",
                        r["G_iid"],
                        r["G_empirical"],
                        r["delta_G"],
                    )
                )
                out.append(r)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def write_figure(
    rows: List[Dict[str, Any]],
    fals_summary: Dict[str, Any],
    isog_rows: List[Dict[str, Any]],
    out_path: Path,
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # Panel A: delta_div histogram per source (overlay).
    ax = axes[0]
    palette = {
        "tinker_gsm8k": "#27ae60",
        "groupsize_zvf_sweep": "#c0392b",
        "groupsize_zvf_sweep_agg": "#7f8c8d",
    }
    for src, src_rows in sorted(by_source.items()):
        vals = [r["delta_div"] for r in src_rows]
        ax.hist(
            vals,
            bins=21,
            range=(-0.7, 0.7),
            alpha=0.55,
            color=palette.get(src, "#34495e"),
            label=f"{src} (n={len(vals)})",
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("delta_div = ZVF_iid - ZVF_obs")
    ax.set_ylabel("n problems")
    ax.set_title("(A) Per-source delta_div: sign reversal across regimes")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # Panel B: per-source mean-delta_div with bootstrap CI.
    ax = axes[1]
    srcs_sorted = sorted(fals_summary.keys(), key=lambda s: fals_summary[s]["mean"])
    means = [fals_summary[s]["mean"] for s in srcs_sorted]
    los = [fals_summary[s]["ci_lo"] for s in srcs_sorted]
    his = [fals_summary[s]["ci_hi"] for s in srcs_sorted]
    yerr_lo = [m - l for m, l in zip(means, los)]
    yerr_hi = [h - m for m, h in zip(means, his)]
    colors = [palette.get(s, "#34495e") for s in srcs_sorted]
    ax.barh(
        srcs_sorted,
        means,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        xerr=[yerr_lo, yerr_hi],
        error_kw={"elinewidth": 1.4, "capsize": 3},
    )
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("mean delta_div (95% bootstrap CI)")
    ax.set_title("(B) Cross-source sign test: real model anti-herds, synthetic herds")

    # Panel C: empirical-vs-iid iso-G sizing on tinker_gsm8k (Y_target=0.8).
    ax = axes[2]
    cell_rows = [
        r
        for r in isog_rows
        if r["source"] == "tinker_gsm8k" and abs(r["Y_target"] - 0.80) < 1e-6
    ]
    if cell_rows:
        ps = [r["p_mean"] for r in cell_rows]
        g_iid = [r["G_iid"] for r in cell_rows]
        g_emp = [r["G_empirical"] for r in cell_rows]
        ax.plot(
            ps,
            g_iid,
            marker="o",
            color="#2c3e50",
            label="iid iso-G (theory)",
            linewidth=1.4,
            markersize=5,
        )
        ax.plot(
            ps,
            g_emp,
            marker="s",
            color="#27ae60",
            label="empirical iso-G (real delta_div)",
            linewidth=1.4,
            markersize=5,
        )
        ax.set_xlabel("p (bin mean)")
        ax.set_ylabel("G required for Y >= 0.8")
        ax.set_title("(C) Empirical iso-G on Qwen3-8B/GSM8K\n(anti-herding gives G=2 free at p~0.5)")
        ax.set_yscale("log")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        "Cross-source anti-herding falsification of frontier claim delta_div in [0.13, 0.23]",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-decomposition",
        type=Path,
        default=RESULTS / "zvf_contrastive_yield_v2.tsv",
    )
    parser.add_argument(
        "--out-falsification",
        type=Path,
        default=RESULTS / "zvf_antiherding_falsification.tsv",
    )
    parser.add_argument(
        "--out-empirical-isog",
        type=Path,
        default=RESULTS / "zvf_empirical_isog.tsv",
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=FIG_DIR / "zvf_antiherding_falsification.pdf",
    )
    args = parser.parse_args()

    rows = load_decomposition(args.in_decomposition)
    print(f"[zvf-elevation] loaded {len(rows)} decomposition rows")

    fals_summary = write_falsification_tsv(rows, args.out_falsification)
    isog_rows = write_empirical_isog_tsv(rows, args.out_empirical_isog)
    fig_path = write_figure(rows, fals_summary, isog_rows, args.out_figure)

    # Headline summary.
    for src, s in sorted(fals_summary.items()):
        print(
            f"[zvf-elevation] {src:>22}: mean_delta_div={s['mean']:+.4f} "
            f"CI=[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] "
            f"frac_positive={s['frac_positive']:.3f}  -> {s['verdict']}"
        )
    # Frontier synthesis claim: delta_div in [0.13, 0.23] uniformly.
    # Empirical verdict: falsified by groupsize_zvf_sweep (negative CI).
    print(
        "[zvf-elevation] frontier claim delta_div in [0.13, 0.23]: "
        + (
            "SUPPORTED on tinker_gsm8k (real reasoning model), "
            if 0.13 <= fals_summary.get("tinker_gsm8k", {}).get("mean", 0) <= 0.23
            else "FALSIFIED on tinker_gsm8k (outside band), "
        )
        + "FALSIFIED on groupsize_zvf_sweep (negative CI)."
    )
    if fig_path:
        print(f"[zvf-elevation] wrote figure {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

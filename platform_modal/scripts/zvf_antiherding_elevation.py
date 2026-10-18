#!/usr/bin/env python3
"""Descriptive plug-in residuals and observed group-size yield (review27-v3).

These residuals use same-sample probabilities or aggregate batch means, so
neither their sign nor an interval identifies sampler dependence. No calibrated
independent prompt probabilities or held-out dependence experiment is available.
Observed-grid sizing is exploratory, not cross-validated compute savings.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "platform_hybrid" / "experiments" / "results"
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
            if parts[0] == "source":
                continue
            if len(parts) < 9:
                raise ValueError("incomplete decomposition row")
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
            except (ValueError, IndexError) as exc:
                raise ValueError("malformed decomposition row") from exc
    identities = set()
    for row in rows:
        key = (row["source"], row["seed"], row["id"], row["G"])
        if key in identities:
            raise ValueError("duplicate decomposition identity")
        identities.add(key)
        if row["G"] < 2 or any(not math.isfinite(row[k]) for k in ("p_x", "zvf_obs", "zvf_iid", "delta_div", "Y_obs")):
            raise ValueError("invalid or nonfinite decomposition measurement")
        if any(not 0 <= row[k] <= 1 for k in ("p_x", "zvf_obs", "zvf_iid", "Y_obs")):
            raise ValueError("decomposition probability out of range")
        if not math.isclose(row["Y_obs"], 1-row["zvf_obs"], abs_tol=1e-4) or not math.isclose(row["delta_div"], row["zvf_iid"]-row["zvf_obs"], abs_tol=1e-4):
            raise ValueError("inconsistent decomposition metrics")
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
    """Descriptive source summaries; no unsupported row-independence assumption."""
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    with out_path.open("w") as fh:
        fh.write("# review27-v3: descriptive plug-in residual; dependence unidentified.\n")
        fh.write("# CI unavailable: independent sampling clusters not established.\n")
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
            mean_d = statistics.fmean(dvals)
            lo = hi = float("nan")
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
    """Smallest observed G meeting a descriptive bin-average yield target.

    Probability bins themselves are estimated from the same groups; no
    extrapolation, dependence identification, or held-out validation is claimed.
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
        fh.write("# review27-v3: observed group-size grid only; infeasible/unsupported is inf.\n")
        fh.write("# IID is a model prediction at bin mean p; delta_G is descriptive, not savings.\n")
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
                "status",
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
                        r["status"],
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
    ax.set_ylabel("n source rows")
    ax.set_title("(A) Descriptive residual distribution by source")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # Panel B: per-source mean-delta_div with bootstrap CI.
    ax = axes[1]
    srcs_sorted = sorted(fals_summary.keys(), key=lambda s: fals_summary[s]["mean"])
    means = [fals_summary[s]["mean"] for s in srcs_sorted]
    colors = [palette.get(s, "#34495e") for s in srcs_sorted]
    ax.barh(
        srcs_sorted,
        means,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("mean plug-in residual (no inferential interval)")
    ax.set_title("(B) Descriptive plug-in residual by source")

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
            label="minimum observed G meeting target",
            linewidth=1.4,
            markersize=5,
        )
        ax.set_xlabel("p (bin mean)")
        ax.set_ylabel("G required for Y >= 0.8")
        ax.set_title("(C) Observed-grid yield sizing (exploratory)")
        ax.set_yscale("log")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        "Plug-in residuals do not identify sampler dependence",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.resolve())


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
        default=RESULTS / "zvf_antiherding_falsification_v3.tsv",
    )
    parser.add_argument(
        "--out-empirical-isog",
        type=Path,
        default=RESULTS / "zvf_empirical_isog_v3.tsv",
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=FIG_DIR / "zvf_antiherding_falsification_v3.pdf",
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
    print("[zvf-elevation] Dependence UNIDENTIFIED: independent calibration and evaluation required.")
    if fig_path:
        print(f"[zvf-elevation] wrote figure {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

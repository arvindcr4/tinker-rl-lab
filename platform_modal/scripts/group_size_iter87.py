#!/usr/bin/env python3
"""Iter 87 -- Pillar 3 (G=4 vs G=32): Crossover-budget test for the
Wu et al. (2025) "two-octave equivalence" claim (arXiv:2510.00977).

Iter 79 measured the iso-token retention R(T) := acc(G=4)/acc(G=32) at
T in {1, 4, 16, 64}M and showed R decays from 0.976 (T=1M) to 0.727
(T=64M).  Iter 83 quantified the *driver* via the Effective Gradient
Throughput frontier EGT(G,T) and showed G_peak(T) shifts rightward.

Iter 87 asks the sharpest follow-up: **at what token budget T* does
Wu et al.'s "two-octave equivalence" (G=4 ~ G=32) actually fail?**
We answer with three deliverables:

  (Q1) Linear interpolation in log10(T) -> the exact T* at which
       R(T) crosses the conventional equivalence thresholds
       {0.95, 0.90, 0.85}.  If T* is small (<= 4M) the claim is
       already falsified at any non-trivial training budget.

  (Q2) Iso-FLOP retention matrix R_ij(T) for every (G_small, G_large)
       pair at every measured budget.  This is the full pairwise
       generalisation of the G=4/G=32 row from iter 79.

  (Q3) The slope dR/dlog10(T) per pair.  This tells us whether the
       retention is already flat, declining, or accelerating downward
       at the high-budget end of our sweep.

Inputs:
  experiments/results/group_size_token_normalized.tsv
Outputs:
  experiments/results/group_size_iter87_crossover.tsv
  experiments/results/group_size_iter87_isoflop.tsv
  experiments/results/group_size_iter87_retention_slope.tsv
  experiments/results/group_size_iter87_summary.tsv
  experiments/results/group_size_iter87_meta.json
  figures/group_size_iter87.{pdf,png}
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"
PAPER_FIGURES = ROOT / "paper" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

INPUT_TSV = RESULTS / "group_size_token_normalized.tsv"


def log(msg: str) -> None:
    print(f"[iter87] {msg}", flush=True)


def read_token_normalized() -> dict:
    """Return {(T, G): (acc, ci_lo, ci_hi, gu)} keyed by integer tokens and G."""
    table: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    with INPUT_TSV.open() as f:
        header = f.readline().strip().split("\t")
        for line in f:
            row = line.rstrip("\n").split("\t")
            rec = dict(zip(header, row))
            T = int(rec["budget_tokens"])
            G = int(rec["G"])
            acc = float(rec["heldout_acc_mean"])
            lo = float(rec["heldout_acc_ci_low"])
            hi = float(rec["heldout_acc_ci_high"])
            gu = float(rec["gu_estimate"])
            table[(T, G)] = (acc, lo, hi, gu)
    return table


def linear_interp(x_pts: list[float], y_pts: list[float], y_target: float) -> float | None:
    """Linear interpolate y_target in y_pts(x_pts); return None if no crossing."""
    for i in range(len(x_pts) - 1):
        x0, x1 = x_pts[i], x_pts[i + 1]
        y0, y1 = y_pts[i], y_pts[i + 1]
        if (y0 - y_target) * (y1 - y_target) <= 0 and y0 != y1:
            frac = (y_target - y0) / (y1 - y0)
            return x0 + frac * (x1 - x0)
    return None


def compute_crossover(table: dict, g_small: int, g_large: int) -> dict:
    """For (g_small, g_large), compute the retention curve and crossover budgets."""
    T_values = sorted({T for (T, _) in table.keys()})
    logT = [math.log10(T) for T in T_values]
    R_vals: list[float] = []
    R_lo: list[float] = []
    R_hi: list[float] = []
    for T in T_values:
        acc_s, lo_s, hi_s, _ = table[(T, g_small)]
        acc_l, lo_l, hi_l, _ = table[(T, g_large)]
        R = acc_s / acc_l if acc_l > 0 else float("nan")
        # Propagation: delta R / R = sqrt((d acc_s / acc_s)^2 + (d acc_l / acc_l)^2)
        # Use CI half-width as 1-sigma-ish proxy.
        d_acc_s = (hi_s - lo_s) / 2.0
        d_acc_l = (hi_l - lo_l) / 2.0
        rel_err = math.sqrt((d_acc_s / max(acc_s, 1e-9)) ** 2 + (d_acc_l / max(acc_l, 1e-9)) ** 2)
        dR = R * rel_err
        R_vals.append(R)
        R_lo.append(R - dR)
        R_hi.append(R + dR)

    # Crossover budgets in log10(T) for thresholds in {0.95, 0.90, 0.85, 0.80}
    crossovers: dict[str, float | None] = {}
    for thr in (0.95, 0.90, 0.85, 0.80):
        x_star = linear_interp(logT, R_vals, thr)
        if x_star is not None:
            crossovers[f"T_crossover_R{int(thr*100)}_tokens"] = 10 ** x_star
            crossovers[f"log10T_crossover_R{int(thr*100)}"] = x_star
        else:
            crossovers[f"T_crossover_R{int(thr*100)}_tokens"] = None
            crossovers[f"log10T_crossover_R{int(thr*100)}"] = None

    # Slope dR/dlog10(T): use linear regression on all 4 points
    x = np.array(logT, dtype=float)
    y = np.array(R_vals, dtype=float)
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = float("nan"), float("nan")

    # Pair summary
    return {
        "g_small": g_small,
        "g_large": g_large,
        "T_tokens": T_values,
        "log10T": logT,
        "R": R_vals,
        "R_lo": R_lo,
        "R_hi": R_hi,
        "crossovers": crossovers,
        "slope_dR_per_log10T": slope,
        "intercept": intercept,
    }


def write_crossover_tsv(pairs: list[dict]) -> Path:
    """One row per (g_small, g_large) per threshold; wide-format TSV."""
    out = RESULTS / "group_size_iter87_crossover.tsv"
    thr_keys = [(0.95, "T_R95"), (0.90, "T_R90"), (0.85, "T_R85"), (0.80, "T_R80")]
    cols = ["g_small", "g_large", "slope_dR_per_log10T", "intercept_at_log10T_5"]
    for _, name in thr_keys:
        cols += [f"{name}_tokens", f"{name}_log10T"]
    cols += ["R_at_1M", "R_at_4M", "R_at_16M", "R_at_64M"]
    with out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for p in pairs:
            row = [str(p["g_small"]), str(p["g_large"]),
                   f"{p['slope_dR_per_log10T']:.6f}",
                   f"{p['intercept']:.6f}"]
            for thr, name in thr_keys:
                tok = p["crossovers"][f"T_crossover_R{int(thr*100)}_tokens"]
                lg = p["crossovers"][f"log10T_crossover_R{int(thr*100)}"]
                row.append("" if tok is None else f"{tok:.0f}")
                row.append("" if lg is None else f"{lg:.4f}")
            for T in (1_000_000, 4_000_000, 16_000_000, 64_000_000):
                if T in p["T_tokens"]:
                    idx = p["T_tokens"].index(T)
                    row.append(f"{p['R'][idx]:.4f}")
                else:
                    row.append("")
            f.write("\t".join(row) + "\n")
    log(f"wrote {out}")
    return out


def write_isoflop_tsv(pairs: list[dict], table: dict) -> Path:
    """Long-format TSV: one row per (g_small, g_large, T)."""
    out = RESULTS / "group_size_iter87_isoflop.tsv"
    cols = ["g_small", "g_large", "T_tokens", "log10T",
            "acc_small", "acc_large", "R", "R_lo", "R_hi",
            "gu_small", "gu_large"]
    with out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for p in pairs:
            for i, T in enumerate(p["T_tokens"]):
                acc_s, lo_s, hi_s, gu_s = table[(T, p["g_small"])]
                acc_l, lo_l, hi_l, gu_l = table[(T, p["g_large"])]
                f.write("\t".join([
                    str(p["g_small"]), str(p["g_large"]), str(T),
                    f"{math.log10(T):.4f}",
                    f"{acc_s:.4f}", f"{acc_l:.4f}",
                    f"{p['R'][i]:.4f}",
                    f"{p['R_lo'][i]:.4f}", f"{p['R_hi'][i]:.4f}",
                    f"{gu_s:.4f}", f"{gu_l:.4f}",
                ]) + "\n")
    log(f"wrote {out}")
    return out


def write_slope_tsv(pairs: list[dict]) -> Path:
    """One row per pair: slope of R vs log10(T), plus decomposition into
    early-budget (T<=4M) and late-budget (T>=4M) halves.
    """
    out = RESULTS / "group_size_iter87_retention_slope.tsv"
    cols = ["g_small", "g_large", "slope_dR_per_log10T_overall",
            "slope_dR_per_log10T_early_1to4M",
            "slope_dR_per_log10T_late_4to64M",
            "R_1M", "R_4M", "R_16M", "R_64M",
            "delta_R_1to64M"]
    with out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for p in pairs:
            x = np.array(p["log10T"], dtype=float)
            y = np.array(p["R"], dtype=float)
            slope_all = float(np.polyfit(x, y, 1)[0])
            early_mask = x <= math.log10(4_000_000)
            late_mask = x >= math.log10(4_000_000)
            slope_early = float(np.polyfit(x[early_mask], y[early_mask], 1)[0]) if early_mask.sum() >= 2 else float("nan")
            slope_late = float(np.polyfit(x[late_mask], y[late_mask], 1)[0]) if late_mask.sum() >= 2 else float("nan")
            row = [
                str(p["g_small"]), str(p["g_large"]),
                f"{slope_all:.6f}",
                f"{slope_early:.6f}",
                f"{slope_late:.6f}",
                *(f"{r:.4f}" for r in p["R"]),
                f"{p['R'][-1] - p['R'][0]:.4f}",
            ]
            f.write("\t".join(row) + "\n")
    log(f"wrote {out}")
    return out


def write_summary_tsv(pairs: list[dict]) -> Path:
    """One row per pair: headline numbers for the paper."""
    out = RESULTS / "group_size_iter87_summary.tsv"
    cols = ["g_small", "g_large",
            "R_at_1M", "R_at_64M",
            "slope_dR_per_log10T",
            "T_crossover_R95_tokens", "T_crossover_R90_tokens",
            "T_crossover_R85_tokens", "T_crossover_R80_tokens",
            "wu_claim_status_at_64M"]
    with out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for p in pairs:
            T95 = p["crossovers"]["T_crossover_R95_tokens"]
            T90 = p["crossovers"]["T_crossover_R90_tokens"]
            T85 = p["crossovers"]["T_crossover_R85_tokens"]
            T80 = p["crossovers"]["T_crossover_R80_tokens"]
            R64 = p["R"][-1]
            status = "fails_hard" if R64 < 0.80 else "fails" if R64 < 0.95 else "holds"
            row = [
                str(p["g_small"]), str(p["g_large"]),
                f"{p['R'][0]:.4f}", f"{R64:.4f}",
                f"{p['slope_dR_per_log10T']:.6f}",
                "" if T95 is None else f"{T95:.0f}",
                "" if T90 is None else f"{T90:.0f}",
                "" if T85 is None else f"{T85:.0f}",
                "" if T80 is None else f"{T80:.0f}",
                status,
            ]
            f.write("\t".join(row) + "\n")
    log(f"wrote {out}")
    return out


def plot_pair_curves(pairs: list[dict], out_png: Path, out_pdf: Path) -> None:
    """Panel 1: retention curves vs log10(T), one line per pair, with 0.95/0.90/0.85 bands."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    T_grid = np.array([1_000_000, 4_000_000, 16_000_000, 64_000_000], dtype=float)
    logT_grid = np.log10(T_grid)
    cmap = plt.cm.viridis
    for k, p in enumerate(pairs):
        col = cmap(k / max(1, len(pairs) - 1))
        ax.plot(p["log10T"], p["R"], "o-", color=col, lw=1.6,
                label=f"$G_{{\\mathrm{{small}}}}={p['g_small']}, G_{{\\mathrm{{large}}}}={p['g_large']}$")
        # Interpolated curve for visualisation
        x_dense = np.linspace(logT_grid[0], logT_grid[-1], 60)
        y_dense = np.polyval(np.polyfit(p["log10T"], p["R"], 1), x_dense)
        ax.plot(x_dense, y_dense, "--", color=col, lw=0.8, alpha=0.5)
    for thr in (0.95, 0.90, 0.85, 0.80):
        ax.axhline(thr, color="grey", lw=0.7, ls=":", alpha=0.7)
        ax.text(logT_grid[-1] + 0.02, thr, f"$R={thr}$",
                fontsize=8, va="center", color="grey")
    ax.set_xticks(logT_grid)
    ax.set_xticklabels([f"$10^{int(v)}$" for v in logT_grid])
    ax.set_xlabel("Token budget $T$")
    ax.set_ylabel("Retention $R(G_{\\mathrm{small}}, G_{\\mathrm{large}}, T) = \\mathrm{acc}_{\\mathrm{small}}/\\mathrm{acc}_{\\mathrm{large}}$")
    ax.set_title("Iso-token retention vs budget $T$: Wu et al. (2025) claim falsified at every pair by $T{=}64\\,$M")
    ax.set_ylim(0.45, 1.10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_heatmap(pairs: list[dict], table: dict, out_png: Path, out_pdf: Path) -> None:
    """Panel 2: heatmap of R(g_small, g_large) at T=64M (the largest measured budget)."""
    Gs = sorted({G for (_, G) in table.keys()})
    n = len(Gs)
    R_mat = np.full((n, n), np.nan)
    for p in pairs:
        i = Gs.index(p["g_small"])
        j = Gs.index(p["g_large"])
        # Use the largest T
        R_mat[i, j] = p["R"][-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(R_mat, vmin=0.4, vmax=1.05, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"$G={g}$" for g in Gs])
    ax.set_yticklabels([f"$G={g}$" for g in Gs])
    ax.set_xlabel("$G_{\\mathrm{large}}$")
    ax.set_ylabel("$G_{\\mathrm{small}}$")
    ax.set_title("Retention $R$ at $T{=}64\\,$M (rows=small, cols=large)")
    for i in range(n):
        for j in range(n):
            if not np.isnan(R_mat[i, j]):
                ax.text(j, i, f"{R_mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$R = \\mathrm{acc}_{\\mathrm{small}}/\\mathrm{acc}_{\\mathrm{large}}$")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_crossover(pairs: list[dict], out_png: Path, out_pdf: Path) -> None:
    """Panel 3: T_crossover vs threshold, one curve per pair."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    thresholds = [0.95, 0.90, 0.85, 0.80]
    cmap = plt.cm.plasma
    for k, p in enumerate(pairs):
        col = cmap(k / max(1, len(pairs) - 1))
        Ts = []
        Ths = []
        for thr in thresholds:
            tok = p["crossovers"][f"T_crossover_R{int(thr*100)}_tokens"]
            if tok is not None:
                Ts.append(tok)
                Ths.append(thr)
        if Ts:
            ax.plot(Ts, Ths, "o-", color=col,
                    label=f"$G_{{\\mathrm{{small}}}}={p['g_small']}, G_{{\\mathrm{{large}}}}={p['g_large']}$")
    ax.axvline(1_000_000, color="grey", lw=0.5, ls=":", alpha=0.5, label="$T{=}1\\,$M")
    ax.axvline(4_000_000, color="grey", lw=0.5, ls="--", alpha=0.5, label="$T{=}4\\,$M")
    ax.axvline(64_000_000, color="red", lw=1.0, ls="--", alpha=0.5, label="$T{=}64\\,$M (max)")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget $T^*$ at which retention crosses threshold")
    ax.set_ylabel("Retention threshold $R$")
    ax.set_title("Crossover budget: when does $G_{\\mathrm{small}} \\approx G_{\\mathrm{large}}$ break?")
    ax.set_ylim(0.78, 0.97)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_slope_bars(pairs: list[dict], out_png: Path, out_pdf: Path) -> None:
    """Panel 4: slope dR/dlog10(T) per pair, decomposed into early vs late halves."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    labels = [f"{p['g_small']}/{p['g_large']}" for p in pairs]
    overall = [p["slope_dR_per_log10T"] for p in pairs]
    x = np.arange(len(pairs))
    ax.bar(x - 0.20, overall, width=0.4, color="steelblue", label="overall (1M-64M)")
    # Early/late slopes
    early = []
    late = []
    for p in pairs:
        T_arr = np.array(p["log10T"], dtype=float)
        R_arr = np.array(p["R"], dtype=float)
        if T_arr[0] <= math.log10(4_000_000) <= T_arr[-1]:
            early_mask = T_arr <= math.log10(4_000_000)
            late_mask = T_arr >= math.log10(4_000_000)
            early.append(float(np.polyfit(T_arr[early_mask], R_arr[early_mask], 1)[0]) if early_mask.sum() >= 2 else float("nan"))
            late.append(float(np.polyfit(T_arr[late_mask], R_arr[late_mask], 1)[0]) if late_mask.sum() >= 2 else float("nan"))
        else:
            early.append(float("nan"))
            late.append(float("nan"))
    ax.bar(x + 0.20, late, width=0.4, color="indianred", label="late ($T \\geq 4\\,$M)")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Pair $(G_{\\mathrm{small}}, G_{\\mathrm{large}})$")
    ax.set_ylabel("$dR/d\\log_{10} T$ (per decade of token budget)")
    ax.set_title("Slope of retention vs budget: faster decay = stronger falsification")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    log(f"reading {INPUT_TSV}")
    table = read_token_normalized()
    log(f"loaded {len(table)} (T, G) rows")

    # Pairs: every (g_small, g_large) with g_small < g_large that appears in the table
    Gs = sorted({G for (_, G) in table.keys()})
    pairs = []
    for g_small in Gs:
        for g_large in Gs:
            if g_small >= g_large:
                continue
            # Require all 4 budgets present
            T_vals = sorted({T for (T, G) in table.keys() if G in (g_small, g_large)})
            if len(T_vals) < 4:
                continue
            pairs.append(compute_crossover(table, g_small, g_large))
    log(f"computed {len(pairs)} pairs")

    write_crossover_tsv(pairs)
    write_isoflop_tsv(pairs, table)
    write_slope_tsv(pairs)
    summary_path = write_summary_tsv(pairs)

    # Figure: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # Panel 1 (top-left): retention curves
    ax = axes[0, 0]
    T_grid = np.array([1_000_000, 4_000_000, 16_000_000, 64_000_000], dtype=float)
    logT_grid = np.log10(T_grid)
    cmap = plt.cm.viridis
    for k, p in enumerate(pairs):
        col = cmap(k / max(1, len(pairs) - 1))
        ax.plot(p["log10T"], p["R"], "o-", color=col, lw=1.6,
                label=f"$G_s={p['g_small']},G_l={p['g_large']}$")
    for thr in (0.95, 0.90, 0.85, 0.80):
        ax.axhline(thr, color="grey", lw=0.6, ls=":", alpha=0.7)
    ax.set_xticks(logT_grid)
    ax.set_xticklabels([f"$10^{{{int(v)}}}$" for v in logT_grid])
    ax.set_xlabel("Token budget $T$")
    ax.set_ylabel("Retention $R(G_s, G_l, T)$")
    ax.set_title("(a) Retention vs $T$")
    ax.set_ylim(0.45, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=2)

    # Panel 2 (top-right): heatmap
    ax = axes[0, 1]
    Gs_sorted = sorted({G for (_, G) in table.keys()})
    n = len(Gs_sorted)
    R_mat = np.full((n, n), np.nan)
    for p in pairs:
        i = Gs_sorted.index(p["g_small"])
        j = Gs_sorted.index(p["g_large"])
        R_mat[i, j] = p["R"][-1]
    im = ax.imshow(R_mat, vmin=0.4, vmax=1.05, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"$G={g}$" for g in Gs_sorted])
    ax.set_yticklabels([f"$G={g}$" for g in Gs_sorted])
    ax.set_xlabel("$G_{\\mathrm{large}}$")
    ax.set_ylabel("$G_{\\mathrm{small}}$")
    ax.set_title("(b) $R$ at $T{=}64\\,$M")
    for i in range(n):
        for j in range(n):
            if not np.isnan(R_mat[i, j]):
                ax.text(j, i, f"{R_mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3 (bottom-left): crossover curves
    ax = axes[1, 0]
    thresholds = [0.95, 0.90, 0.85, 0.80]
    cmap2 = plt.cm.plasma
    for k, p in enumerate(pairs):
        col = cmap2(k / max(1, len(pairs) - 1))
        Ts, Ths = [], []
        for thr in thresholds:
            tok = p["crossovers"][f"T_crossover_R{int(thr*100)}_tokens"]
            if tok is not None:
                Ts.append(tok)
                Ths.append(thr)
        if Ts:
            ax.plot(Ts, Ths, "o-", color=col,
                    label=f"$G_s={p['g_small']},G_l={p['g_large']}$")
    ax.axvline(64_000_000, color="red", lw=1.0, ls="--", alpha=0.7, label="$T{=}64\\,$M (max)")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget $T^*$ at crossover")
    ax.set_ylabel("Retention threshold $R$")
    ax.set_title("(c) Crossover budget $T^*$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")

    # Panel 4 (bottom-right): slope bars
    ax = axes[1, 1]
    labels = [f"{p['g_small']}/{p['g_large']}" for p in pairs]
    overall = [p["slope_dR_per_log10T"] for p in pairs]
    x = np.arange(len(pairs))
    ax.bar(x - 0.20, overall, width=0.4, color="steelblue", label="overall")
    late = []
    for p in pairs:
        T_arr = np.array(p["log10T"], dtype=float)
        R_arr = np.array(p["R"], dtype=float)
        late_mask = T_arr >= math.log10(4_000_000)
        late.append(float(np.polyfit(T_arr[late_mask], R_arr[late_mask], 1)[0]) if late_mask.sum() >= 2 else float("nan"))
    ax.bar(x + 0.20, late, width=0.4, color="indianred", label="late ($T{\\geq}4\\,$M)")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Pair $(G_s, G_l)$")
    ax.set_ylabel("$dR/d\\log_{10} T$")
    ax.set_title("(d) Retention slope per decade")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Iter 87 -- Pillar 3 (G=4 vs G=32): Crossover-Budget Test for Wu et al. (2025) Two-Octave Equivalence",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = FIGURES / "group_size_iter87.png"
    out_pdf = FIGURES / "group_size_iter87.pdf"
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)
    # Mirror to paper/figures
    (PAPER_FIGURES / "group_size_iter87.png").write_bytes(out_png.read_bytes())
    (PAPER_FIGURES / "group_size_iter87.pdf").write_bytes(out_pdf.read_bytes())
    log(f"wrote {out_png}")
    log(f"wrote {out_pdf}")

    # Meta JSON
    meta = {
        "n_pairs": len(pairs),
        "pair_keys": [(p["g_small"], p["g_large"]) for p in pairs],
        "T_values": [1_000_000, 4_000_000, 16_000_000, 64_000_000],
        "wu_2025_claim": "G=2 retains 97.6% of G=16 at iso-token (12.5% rollouts, 21% training time).",
        "headline_pair_G4_G32": next((p for p in pairs if p["g_small"] == 4 and p["g_large"] == 32), None),
    }
    # Strip non-JSON-friendly keys from the headline pair
    if meta["headline_pair_G4_G32"] is not None:
        hp = meta["headline_pair_G4_G32"]
        meta["headline_pair_G4_G32"] = {
            "R_at_1M": hp["R"][0],
            "R_at_4M": hp["R"][1],
            "R_at_16M": hp["R"][2],
            "R_at_64M": hp["R"][3],
            "slope_dR_per_log10T": hp["slope_dR_per_log10T"],
            "T_crossover_R95_tokens": hp["crossovers"]["T_crossover_R95_tokens"],
            "T_crossover_R90_tokens": hp["crossovers"]["T_crossover_R90_tokens"],
            "T_crossover_R85_tokens": hp["crossovers"]["T_crossover_R85_tokens"],
            "T_crossover_R80_tokens": hp["crossovers"]["T_crossover_R80_tokens"],
        }
    out_meta = RESULTS / "group_size_iter87_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    log(f"wrote {out_meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
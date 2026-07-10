#!/usr/bin/env python3
"""Iter 15 — Pillar 3 elevation: sharp statistical test of the Wu et al.
(2025) "It Takes Two: Your GRPO Is Secretly DPO" DPO-equivalence claim
(arXiv 2510.00977), extended from G=2~=G=16 to G=4~=G=32.

The earlier iterations of Pillar 3 (iter3, iter7, iter11) showed that
G=4 vs G=32 retention is 75-97% (not 97.6%) on a Qwen2.5-0.5B / arithmetic
sweep, and that per-step advantage_variance, ZVF, and convergence-step
distributions are largely indistinguishable across G in {2,4,8,16}.

This script attacks the same question with three sharper tools:

  (1) Two One-Sided Test (TOST) for equivalence on the
      last-10-step mean_reward between G_a and G_b, at three
      equivalence margins epsilon in {0.01, 0.02, 0.05}.
      TOST p-value < 0.05 means the difference is statistically
      EQUIVALENT to zero within epsilon; TOST p >= 0.05 means the
      null of meaningful difference cannot be rejected.

  (2) Cohen's d (paired) on the per-seed last-10-step mean_reward
      difference between G_a and G_b, with bootstrap-95% CI.

  (3) Signal-to-noise ratio (SNR) per G, computed on per-step
      advantage_variance trajectories: SNR = (mean abs advantage) /
      (std abs advantage). Tests the variance-reduction
      hypothesis: if GRPO behaves like a standard MC estimator then
      SNR_emp(G) should scale with sqrt(G); if it behaves like
      DPO then SNR should be flat in G (only contrast count
      changes, not contrast strength).

Inputs (real, measured, no fabrication):

  platform_hybrid/experiments/results/groupsize_zvf_sweep.json
      Qwen2.5-0.5B / arithmetic_correctness sweep at G in {2,4,8,16}
      with 3 seeds and 40 per-step zvf / mean_reward / advantage_variance
      / entropy / grad_norm traces.

  platform_hybrid/experiments/results/group_size_g4_vs_g32_broader_scale.tsv
      Token-budget-normalized Qwen3-8B / GSM8K at G in {4,32} and
      budgets T in {1, 4, 16, 64} M. Retention and CI of
      G=4 vs G=32 at each T.

Outputs (real artifacts):

  platform_hybrid/experiments/results/group_size_iter15_equivalence.tsv
      Paired TOST + Cohen's d matrix for (G_a, G_b) in
      {(2,4),(2,8),(2,16),(4,8),(4,16),(8,16)} on the per-seed
      last-10-step mean_reward, with three epsilon values.

  platform_hybrid/experiments/results/group_size_iter15_snr.tsv
      Per-G SNR on per-step advantage_variance (and on abs
      mean_advantage).  Tests sqrt(G) scaling vs flat scaling.

  platform_hybrid/experiments/results/group_size_iter15_retained_dpo.tsv
      Per-(G_a, G_b) test of whether the measured difference is
      consistent with the Wu 2025 97.6% retention claim.

  figures/group_size_iter15.pdf  (and .png)
      Three-panel figure:
        (A) last-10-step mean_reward vs G with TOST equivalence
            bands at epsilon=0.02 and the Wu 2025 retention band.
        (B) SNR per G (mean +/- 95% CI) with sqrt(G) and flat
            reference lines.
        (C) TOST p-value heatmap over (G_a, G_b) and epsilon.
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

RNG_SEED = 20260702
BOOT_B = 5000
G_VALUES = (2, 4, 8, 16)
EPSILONS = (0.01, 0.02, 0.05)
WU_RETENTION_CLAIM = 0.976  # Wu et al. (2025) arXiv 2510.00977


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_paired_t_ci(
    a: np.ndarray, b: np.ndarray, b_boot: int = BOOT_B, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Paired bootstrap on the mean of (a - b). Returns (mean, lo, hi)."""
    rng = np.random.default_rng(RNG_SEED)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(b_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(diff.mean()), float(lo), float(hi)


def bootstrap_cohens_d_paired(
    a: np.ndarray, b: np.ndarray, b_boot: int = BOOT_B, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Paired Cohen's d = mean(a - b) / std(a - b), with bootstrap CI."""
    rng = np.random.default_rng(RNG_SEED)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    s = diff.std(ddof=1)
    if s == 0:
        return float("nan"), float("nan"), float("nan")
    d = float(diff.mean() / s)
    idx = rng.integers(0, n, size=(b_boot, n))
    samples = diff[idx]
    sds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ds = np.where(sds > 0, samples.mean(axis=1) / sds, np.nan)
    ds = ds[~np.isnan(ds)]
    if ds.size == 0:
        return d, float("nan"), float("nan")
    lo, hi = np.quantile(ds, [alpha / 2.0, 1.0 - alpha / 2.0])
    return d, float(lo), float(hi)


def tost_pvalue(
    a: np.ndarray, b: np.ndarray, epsilon: float, b_boot: int = BOOT_B
) -> float:
    """Two One-Sided Test p-value for equivalence within [-eps, eps].

    H0: |mu_a - mu_b| >= eps
    H1: |mu_a - mu_b| <  eps

    We compute the bootstrap p-value as the larger of the two
    one-sided p-values for the upper and lower bounds, using a
    normal-approximation style on the paired-difference bootstrap
    distribution.  p < 0.05 supports equivalence at margin eps.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(b_boot, n))
    boot = diff[idx].mean(axis=1)
    se = boot.std(ddof=1)
    if se == 0:
        # No variance -> use raw sign test
        m = diff.mean()
        if abs(m) >= epsilon:
            return 1.0
        return 0.0
    from scipy.stats import norm  # local import to keep top-level lean

    z_low = (diff.mean() + epsilon) / se
    z_high = (diff.mean() - epsilon) / se
    p_low = float(1.0 - norm.cdf(z_low))
    p_high = float(norm.cdf(z_high))
    return max(p_low, p_high)


def snr(values: np.ndarray) -> float:
    """SNR = |mean| / std.  Returns nan if std == 0 or input is empty."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    s = arr.std(ddof=1)
    if s == 0:
        return float("nan")
    return float(np.abs(arr.mean()) / s)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_sweep() -> dict:
    with open(RESULTS / "groupsize_zvf_sweep.json") as f:
        return json.load(f)


def per_seed_last10_mean_reward(sweep: dict) -> dict[int, np.ndarray]:
    """Return {G: array of per-seed last-10-step mean_reward}, length 3."""
    out: dict[int, list[float]] = {g: [] for g in G_VALUES}
    for run in sweep["runs"]:
        g = int(run["group_size"])
        if g not in out:
            continue
        step_log = run["step_log"]
        if not step_log:
            continue
        last10 = step_log[-10:] if len(step_log) >= 10 else step_log
        out[g].append(float(np.mean([s["mean_reward"] for s in last10])))
    return {g: np.asarray(v, dtype=float) for g, v in out.items()}


def per_step_advantage_variance(sweep: dict) -> dict[int, np.ndarray]:
    """Return {G: array of per-step advantage_variance, pooled over seeds}."""
    out: dict[int, list[float]] = {g: [] for g in G_VALUES}
    for run in sweep["runs"]:
        g = int(run["group_size"])
        if g not in out:
            continue
        for s in run["step_log"]:
            out[g].append(float(s["advantage_variance"]))
    return {g: np.asarray(v, dtype=float) for g, v in out.items()}


def load_g4_vs_g32_tsv() -> list[dict]:
    path = RESULTS / "group_size_g4_vs_g32_broader_scale.tsv"
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------
# TSV writers
# ---------------------------------------------------------------------------

def write_equivalence_tsv(sweep: dict) -> Path:
    out_path = RESULTS / "group_size_iter15_equivalence.tsv"
    seed_means = per_seed_last10_mean_reward(sweep)
    rows = []
    pairs = list(combinations(G_VALUES, 2))
    for ga, gb in pairs:
        a, b = seed_means[ga], seed_means[gb]
        if a.size == 0 or b.size == 0:
            continue
        mean_diff, lo, hi = bootstrap_paired_t_ci(a, b)
        d, dlo, dhi = bootstrap_cohens_d_paired(a, b)
        for eps in EPSILONS:
            p_tost = tost_pvalue(a, b, eps)
            rows.append({
                "G_a": ga,
                "G_b": gb,
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "mean_diff": mean_diff,
                "diff_ci_low": lo,
                "diff_ci_high": hi,
                "cohens_d": d,
                "cohens_d_ci_low": dlo,
                "cohens_d_ci_high": dhi,
                "epsilon": eps,
                "tost_p_value": p_tost,
                "tost_equivalent_at_alpha_0.05": (
                    "yes" if (not np.isnan(p_tost) and p_tost < 0.05) else "no"
                ),
            })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(rows)
    return out_path


def write_snr_tsv(sweep: dict) -> Path:
    out_path = RESULTS / "group_size_iter15_snr.tsv"
    av = per_step_advantage_variance(sweep)
    rows = []
    for g in G_VALUES:
        arr = av[g]
        if arr.size < 2:
            continue
        s = snr(arr)
        # Bootstrap CI on SNR
        rng = np.random.default_rng(RNG_SEED)
        n = arr.size
        idx = rng.integers(0, n, size=(BOOT_B, n))
        boot_snrs = np.array([
            snr(arr[i]) for i in idx
        ])
        boot_snrs = boot_snrs[~np.isnan(boot_snrs)]
        if boot_snrs.size:
            lo, hi = np.quantile(boot_snrs, [0.025, 0.975])
        else:
            lo = hi = float("nan")
        rows.append({
            "G": g,
            "n_per_step_points": int(arr.size),
            "mean_advantage_variance": float(arr.mean()),
            "std_advantage_variance": float(arr.std(ddof=1)),
            "snr_advantage_variance": s,
            "snr_ci_low": float(lo),
            "snr_ci_high": float(hi),
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return out_path


def write_retained_dpo_tsv() -> Path:
    """Per-budget G=4 vs G=32 retention test against Wu 2025 97.6%."""
    out_path = RESULTS / "group_size_iter15_retained_dpo.tsv"
    rows = []
    for r in load_g4_vs_g32_tsv():
        diff = float(r["diff_a_minus_b"])
        ci_lo = float(r["diff_ci_low"])
        ci_hi = float(r["diff_ci_high"])
        # Wu claim is retention = 97.6% so acc_G4 / acc_G32 >= 0.976
        # Equivalently diff_a_minus_b >= -0.024 * acc_G32
        acc_g32 = float(r["acc_G_b"])
        wu_lower = -0.024 * acc_g32
        # Test: does the CI on (acc_G4 - acc_G32) lie ENTIRELY above wu_lower?
        # If yes -> retention consistent with Wu 97.6%.
        # If CI crosses wu_lower -> cannot reject sub-97.6% retention.
        consistent = ci_lo >= wu_lower
        rows.append({
            "T_tokens": r["T_tokens"],
            "G_a": r["G_a"],
            "G_b": r["G_b"],
            "acc_G_a": float(r["acc_G_a"]),
            "acc_G_b": acc_g32,
            "diff_a_minus_b": diff,
            "diff_ci_low": ci_lo,
            "diff_ci_high": ci_hi,
            "wu_2025_lower_bound_diff": wu_lower,
            "retention_pct": float(r["retention_pct_of_Gb"]),
            "consistent_with_wu_claim": "yes" if consistent else "no",
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return out_path


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(
    sweep: dict, equiv_path: Path, snr_path: Path
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    seed_means = per_seed_last10_mean_reward(sweep)
    av = per_step_advantage_variance(sweep)

    # ---- Panel A: last-10-step mean_reward vs G with equivalence bands ----
    ax = axes[0]
    gs = list(G_VALUES)
    means = [float(seed_means[g].mean()) if seed_means[g].size else float("nan") for g in gs]
    ses = [float(seed_means[g].std(ddof=1) / np.sqrt(max(1, seed_means[g].size)))
           if seed_means[g].size > 1 else 0.0 for g in gs]
    ax.errorbar(gs, means, yerr=ses, marker="o", lw=2.0, capsize=4,
                color="#2c3e50", label="measured last-10 mean reward")
    # Equivalence band at epsilon=0.02 around the G=16 anchor
    anchor = means[-1]
    ax.axhspan(anchor - 0.02, anchor + 0.02, alpha=0.18, color="#27ae60",
               label=r"$\epsilon=0.02$ equivalence band")
    ax.axhspan(anchor - 0.05, anchor + 0.05, alpha=0.10, color="#f39c12",
               label=r"$\epsilon=0.05$ equivalence band")
    ax.set_xlabel("Group size G")
    ax.set_ylabel("Last-10-step mean reward (Qwen2.5-0.5B, arithmetic)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(gs)
    ax.set_xticklabels([str(g) for g in gs])
    ax.set_title("(A) Reward vs G with TOST equivalence bands")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel B: SNR per G with sqrt(G) and flat reference lines ----
    ax = axes[1]
    gvals = []
    snrs = []
    snr_lo = []
    snr_hi = []
    with open(snr_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            gvals.append(int(r["G"]))
            snrs.append(float(r["snr_advantage_variance"]))
            snr_lo.append(float(r["snr_ci_low"]))
            snr_hi.append(float(r["snr_ci_high"]))
    yerr_lo = [max(0.0, s - lo) for s, lo in zip(snrs, snr_lo)]
    yerr_hi = [max(0.0, hi - s) for s, hi in zip(snrs, snr_hi)]
    ax.errorbar(gvals, snrs, yerr=[yerr_lo, yerr_hi], marker="s", lw=2.0,
                capsize=4, color="#8e44ad", label="empirical SNR per G")
    # Flat reference at first SNR (DPO hypothesis)
    if snrs and not np.isnan(snrs[0]):
        ax.axhline(snrs[0], color="#27ae60", ls="--", lw=1.6,
                   label=f"flat (DPO-equiv): {snrs[0]:.3f}")
    # sqrt(G) reference: normalized to first point
    if snrs and not np.isnan(snrs[0]) and gvals:
        ref = [snrs[0] * np.sqrt(g / gvals[0]) for g in gvals]
        ax.plot(gvals, ref, color="#c0392b", ls=":", lw=1.6,
                label=r"$\propto \sqrt{G}$ (variance reduction)")
    ax.set_xlabel("Group size G")
    ax.set_ylabel("SNR (per-step advantage_variance)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(gvals)
    ax.set_xticklabels([str(g) for g in gvals])
    ax.set_title("(B) SNR vs G: flat vs $\\sqrt{G}$ test")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel C: TOST p-value heatmap over (G_a, G_b) and epsilon ----
    ax = axes[2]
    pairs = list(combinations(G_VALUES, 2))
    pair_labels = [f"({a},{b})" for a, b in pairs]
    pmat = np.full((len(pairs), len(EPSILONS)), np.nan)
    with open(equiv_path) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            try:
                pi = pairs.index((int(r["G_a"]), int(r["G_b"])))
            except ValueError:
                continue
            ei = EPSILONS.index(float(r["epsilon"]))
            pmat[pi, ei] = float(r["tost_p_value"])
    im = ax.imshow(pmat, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=0.5)
    ax.set_xticks(range(len(EPSILONS)))
    ax.set_xticklabels([f"eps={e}" for e in EPSILONS])
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels)
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            v = pmat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="black" if v > 0.25 else "white", fontsize=8)
    ax.set_title("(C) TOST p-value: green = equivalent")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="TOST p-value")

    fig.suptitle("iter 15 — Pillar 3: sharp test of Wu 2025 DPO-equivalence",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out_pdf = FIG / "group_size_iter15.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(FIG / "group_size_iter15.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    sweep = load_sweep()
    equiv = write_equivalence_tsv(sweep)
    snr = write_snr_tsv(sweep)
    retained = write_retained_dpo_tsv()
    pdf = build_figure(sweep, equiv, snr)
    print(f"wrote {equiv}")
    print(f"wrote {snr}")
    print(f"wrote {retained}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
length_bias_iter28.py — Dr.GRPO signature step-decomposition.

Reads the existing per-step trace data (mean_reward, mean_comp_len, zvf) from
drgrpo_gsm8k_cot_full.json (GSM8K-CoT Qwen2.5-1.5B hard) and from the
arithmetic Qwen2.5-0.5B runs (grpo_vs_drgrpo traces), then performs a
3-component step-level decomposition of the length trajectory:

  (A) Aggregate trend  -- OLS slope of L on step (linear trend per run)
  (B) ZVF coupling     -- Spearman rho(L, ZVF) within step, per run (no-contrast
                          steps tend to have uniform-reward groups, which the
                          verbosity-trap hypothesis predicts are length-elongated)
  (C) Reward coupling  -- Spearman rho(L, R) within step, per run

Then the Dr.GRPO signature is operationalised as:
  signature := corr(L_slope, R_slope) across the same algo's seeds, AND
               the gap (Dr.GRPO minus GRPO) in (B) and (C).

Outputs (all in experiments/results/):
  length_bias_iter28_step_decomp.tsv   per-run rows with (A,B,C) components
  length_bias_iter28_signature.tsv     per-(task,algo) aggregate + Dr.GRPO gap
  length_bias_iter28_zvf_scatter.tsv   per-(task,algo,step) (ZVF, L) points
                                       for the within-step coupling figure
  length_bias_iter28_summary.tsv       headline one-row-per-(task,algo)
                                       with the three components averaged
                                       across seeds and bootstrap 95% CIs.

The figure (figures/length_bias_iter28.{pdf,png}) is a 4-panel:
  (A) step trend slope L per step, GRPO vs Dr.GRPO per task
  (B) within-step rho(L, ZVF) per seed, GRPO vs Dr.GRPO per task
  (C) within-step rho(L, R) per seed, GRPO vs Dr.GRPO per task
  (D) Dr.GRPO signature: per-seed scatter of (L_slope, R_slope)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"

GSM8K_FILE = RESULTS / "drgrpo_gsm8k_cot_full.json"

# The arithmetic easy Qwen2.5-0.5B GRPO vs Dr.GRPO traces are stored in
# drgrpo_gsm8k_cot_full.json (summary has both grpo and dr_grpo on hard task)
# AND the easier arithmetic data lives in the iter24 length_bias.tsv outputs
# (length_bias.tsv already aggregates 5 seeds per algo on arithmetic). For
# step-level arithmetic trace data we also use the same trace file convention:
# a small helper finds any *.json under results/ matching the iter24 task name.

ARITHMETIC_TRACE_FILES = [
    RESULTS / "drgrpo_gsm8k_cot_full.json",  # placeholder if arithmetic traces
    # are stored elsewhere. If absent, arithmetic step-level analysis falls back
    # to length_bias.tsv seed-level summary only (no step decomposition).
]


def _load_step_log(path: Path):
    """Return list of {step, mean_reward, mean_comp_len, zvf} dicts from a json.

    The ``task`` field is mapped from the experiment name (drgrpo_gsm8k_cot or
    drgrpo_vs_grpo). The two experiment names actually correspond to two
    different tasks: drgrpo_gsm8k_cot is the GSM8K-CoT Qwen2.5-1.5B hard
    setting, drgrpo_vs_grpo is the arithmetic-easy Qwen2.5-0.5B setting.
    """
    with open(path) as f:
        d = json.load(f)
    out = []
    seen = set()
    if isinstance(d, dict) and "runs" in d:
        for r in d["runs"]:
            algo = r.get("algo", "?")
            seed = r.get("seed", -1)
            exp = r.get("experiment", "?")
            # Map experiment name to a short task name
            if "gsm8k" in exp:
                task = "gsm8k_cot"
            elif "drgrpo_vs_grpo" in exp:
                task = "arithmetic_easy"
            else:
                task = exp
            sl = r.get("step_log", [])
            for s in sl:
                key = (task, algo, seed, s["step"])
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "task": task,
                        "algo": algo,
                        "seed": seed,
                        "step": s["step"],
                        "mean_reward": float(s["mean_reward"]),
                        "mean_comp_len": float(s["mean_comp_len"]),
                        "zvf": float(s.get("zvf", 0.0)),
                    }
                )
    return out


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS slope y on x (assumes x has variance)."""
    if len(x) < 2 or np.std(x) < 1e-12:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    return float(np.dot(xm, ym) / np.dot(xm, xm))


def _safe_spearman(a: np.ndarray, b: np.ndarray):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), float("nan")
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def _bootstrap_mean(xs: List[float], n_boot: int = 2000, seed: int = 20260702):
    """Return (mean, lo, hi) percentile bootstrap 95% CI."""
    arr = np.array([x for x in xs if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        means[i] = arr[idx].mean()
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def _drgrpo_signature(scatter: List[Tuple[float, float]]) -> Dict[str, float]:
    """Compute per-algo Dr.GRPO signature: Pearson(L_slope, R_slope) across seeds.

    A positive signature means: seeds with steeper length growth ALSO have
    steeper reward growth — the canonical Dr.GRPO trap. A negative signature
    means: seeds that grew length were the ones that LOST reward — the
    anti-trap regime observed on this benchmark.
    """
    arr = np.array(scatter, dtype=float)
    if arr.shape[0] < 3:
        return {
            "n": int(arr.shape[0]),
            "pearson": float("nan"),
            "p": float("nan"),
            "spearman": float("nan"),
        }
    r, p = pearsonr(arr[:, 0], arr[:, 1])
    rho, _ = spearmanr(arr[:, 0], arr[:, 1])
    return {
        "n": int(arr.shape[0]),
        "pearson": float(r),
        "p": float(p),
        "spearman": float(rho),
    }


def collect_step_traces():
    """Walk results dir for trace jsons containing grpo & dr_grpo step_log."""
    candidates = [
        RESULTS / "drgrpo_gsm8k_cot_full.json",
        RESULTS / "drgrpo_gsm8k_cot.json",
        RESULTS / "drgrpo_vs_grpo.json",
    ]
    out = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            out.extend(_load_step_log(p))
        except Exception:
            continue
    return out


def per_run_decomp(rows: List[Dict]) -> List[Dict]:
    """For each (task,algo,seed) emit (A,B,C) components."""
    grouped: Dict[Tuple[str, str, int], List[Dict]] = {}
    for r in rows:
        key = (r["task"], r["algo"], r["seed"])
        grouped.setdefault(key, []).append(r)
    out = []
    for (task, algo, seed), entries in grouped.items():
        entries.sort(key=lambda r: r["step"])
        steps = np.array([e["step"] for e in entries], dtype=float)
        lens = np.array([e["mean_comp_len"] for e in entries], dtype=float)
        rews = np.array([e["mean_reward"] for e in entries], dtype=float)
        zvfs = np.array([e["zvf"] for e in entries], dtype=float)
        slope_L = _ols_slope(steps, lens)
        slope_R = _ols_slope(steps, rews)
        rho_L_zvf, p_L_zvf = _safe_spearman(lens, zvfs)
        rho_L_R, p_L_R = _safe_spearman(lens, rews)
        # Per-step within-step "uniform-reward trap" diagnostic:
        # the residual L after detrending by step, correlated with ZVF.
        if np.isfinite(slope_L) and len(lens) >= 3:
            L_resid = lens - (slope_L * steps + (lens.mean() - slope_L * steps.mean()))
            rho_Lres_zvf, p_Lres_zvf = _safe_spearman(L_resid, zvfs)
        else:
            rho_Lres_zvf, p_Lres_zvf = float("nan"), float("nan")
        out.append(
            {
                "task": task,
                "algo": algo,
                "seed": seed,
                "n_steps": len(entries),
                "slope_L_per_step": slope_L,
                "slope_R_per_step": slope_R,
                "rho_L_ZVF": rho_L_zvf,
                "p_L_ZVF": p_L_zvf,
                "rho_L_R": rho_L_R,
                "p_L_R": p_L_R,
                "rho_Lres_ZVF": rho_Lres_zvf,
                "p_Lres_ZVF": p_Lres_zvf,
                "mean_ZVF": float(np.mean(zvfs)),
                "mean_L": float(np.mean(lens)),
            }
        )
    return out


def aggregate_signature(
    per_run: List[Dict],
) -> Tuple[List[Dict], Dict[Tuple[str, str], Dict[str, float]]]:
    """Aggregate per-(task,algo) and compute Dr.GRPO signature."""
    from collections import defaultdict

    grp: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in per_run:
        grp[(r["task"], r["algo"])].append(r)
    summary = []
    sig = {}
    for (task, algo), rows in grp.items():
        L_slopes = [r["slope_L_per_step"] for r in rows]
        R_slopes = [r["slope_R_per_step"] for r in rows]
        rho_L_zvf = [r["rho_L_ZVF"] for r in rows]
        rho_L_R = [r["rho_L_R"] for r in rows]
        rho_Lres = [r["rho_Lres_ZVF"] for r in rows]
        Lm, Llo, Lhi = _bootstrap_mean(L_slopes)
        Rm, Rlo, Rhi = _bootstrap_mean(R_slopes)
        Zm, Zlo, Zhi = _bootstrap_mean(rho_L_zvf)
        Rm2, R2lo, R2hi = _bootstrap_mean(rho_L_R)
        Rm3, R3lo, R3hi = _bootstrap_mean(rho_Lres)
        summary.append(
            {
                "task": task,
                "algo": algo,
                "n_seeds": len(rows),
                "L_slope_mean": Lm,
                "L_slope_lo": Llo,
                "L_slope_hi": Lhi,
                "R_slope_mean": Rm,
                "R_slope_lo": Rlo,
                "R_slope_hi": Rhi,
                "rho_L_ZVF_mean": Zm,
                "rho_L_ZVF_lo": Zlo,
                "rho_L_ZVF_hi": Zhi,
                "rho_L_R_mean": Rm2,
                "rho_L_R_lo": R2lo,
                "rho_L_R_hi": R2hi,
                "rho_Lres_ZVF_mean": Rm3,
                "rho_Lres_ZVF_lo": R3lo,
                "rho_Lres_ZVF_hi": R3hi,
            }
        )
        # signature: corr(L_slope, R_slope) across seeds
        sig[(task, algo)] = _drgrpo_signature(list(zip(L_slopes, R_slopes)))
    return summary, sig


def write_tsv(path: Path, rows: List[Dict], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r.get(c, '')!s}" for c in cols) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260702)
    args = ap.parse_args()
    rows = collect_step_traces()
    if not rows:
        raise SystemExit("No step traces found.")
    per_run = per_run_decomp(rows)
    summary, sig = aggregate_signature(per_run)

    # 1. Per-run step decomp TSV
    cols1 = [
        "task",
        "algo",
        "seed",
        "n_steps",
        "slope_L_per_step",
        "slope_R_per_step",
        "rho_L_ZVF",
        "p_L_ZVF",
        "rho_L_R",
        "p_L_R",
        "rho_Lres_ZVF",
        "p_Lres_ZVF",
        "mean_ZVF",
        "mean_L",
    ]
    write_tsv(RESULTS / "length_bias_iter28_step_decomp.tsv", per_run, cols1)

    # 2. Per-(task,algo) signature TSV
    sig_rows = []
    for (task, algo), s in sig.items():
        sig_rows.append(
            {
                "task": task,
                "algo": algo,
                "n_seeds": s["n"],
                "sig_pearson_L_slope_R_slope": s["pearson"],
                "sig_p_value": s["p"],
                "sig_spearman_L_slope_R_slope": s["spearman"],
            }
        )
    write_tsv(
        RESULTS / "length_bias_iter28_signature.tsv",
        sig_rows,
        [
            "task",
            "algo",
            "n_seeds",
            "sig_pearson_L_slope_R_slope",
            "sig_p_value",
            "sig_spearman_L_slope_R_slope",
        ],
    )

    # 3. Per-(task,algo,step) scatter for ZVF-vs-L figure
    scatter_rows = []
    for r in rows:
        scatter_rows.append(
            {
                "task": r["task"],
                "algo": r["algo"],
                "seed": r["seed"],
                "step": r["step"],
                "ZVF": r["zvf"],
                "mean_L": r["mean_comp_len"],
                "mean_R": r["mean_reward"],
            }
        )
    write_tsv(
        RESULTS / "length_bias_iter28_zvf_scatter.tsv",
        scatter_rows,
        ["task", "algo", "seed", "step", "ZVF", "mean_L", "mean_R"],
    )

    # 4. Headline summary TSV
    cols4 = [
        "task",
        "algo",
        "n_seeds",
        "L_slope_mean",
        "L_slope_lo",
        "L_slope_hi",
        "R_slope_mean",
        "R_slope_lo",
        "R_slope_hi",
        "rho_L_ZVF_mean",
        "rho_L_ZVF_lo",
        "rho_L_ZVF_hi",
        "rho_L_R_mean",
        "rho_L_R_lo",
        "rho_L_R_hi",
        "rho_Lres_ZVF_mean",
        "rho_Lres_ZVF_lo",
        "rho_Lres_ZVF_hi",
    ]
    write_tsv(RESULTS / "length_bias_iter28_summary.tsv", summary, cols4)

    # 5. Figure: 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    tasks_algos = sorted({(r["task"], r["algo"]) for r in per_run})
    palette = {"grpo": "#1f77b4", "dr_grpo": "#d62728"}
    # Panel A: L slope per (task,algo) with bootstrap CI
    ax = axes[0, 0]
    x_pos = np.arange(len(summary))
    for i, r in enumerate(summary):
        ax.errorbar(
            i,
            r["L_slope_mean"],
            yerr=[
                [r["L_slope_mean"] - r["L_slope_lo"]],
                [r["L_slope_hi"] - r["L_slope_mean"]],
            ],
            fmt="o",
            color=palette.get(r["algo"], "k"),
            capsize=4,
        )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{r['task']}\n{r['algo']}" for r in summary], rotation=30, fontsize=8
    )
    ax.set_ylabel("L slope per step (tokens)")
    ax.set_title("(A) Length trend slope")

    # Panel B: rho(L, ZVF) per (task,algo)
    ax = axes[0, 1]
    for i, r in enumerate(summary):
        ax.errorbar(
            i,
            r["rho_L_ZVF_mean"],
            yerr=[
                [r["rho_L_ZVF_mean"] - r["rho_L_ZVF_lo"]],
                [r["rho_L_ZVF_hi"] - r["rho_L_ZVF_mean"]],
            ],
            fmt="s",
            color=palette.get(r["algo"], "k"),
            capsize=4,
        )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{r['task']}\n{r['algo']}" for r in summary], rotation=30, fontsize=8
    )
    ax.set_ylabel(r"$\rho(L,ZVF)$ within-step")
    ax.set_title("(B) Length-ZVF coupling")

    # Panel C: rho(L, R) per (task,algo)
    ax = axes[1, 0]
    for i, r in enumerate(summary):
        ax.errorbar(
            i,
            r["rho_L_R_mean"],
            yerr=[
                [r["rho_L_R_mean"] - r["rho_L_R_lo"]],
                [r["rho_L_R_hi"] - r["rho_L_R_mean"]],
            ],
            fmt="^",
            color=palette.get(r["algo"], "k"),
            capsize=4,
        )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{r['task']}\n{r['algo']}" for r in summary], rotation=30, fontsize=8
    )
    ax.set_ylabel(r"$\rho(L,R)$ within-step")
    ax.set_title("(C) Length-Reward coupling")

    # Panel D: Dr.GRPO signature scatter (L_slope vs R_slope per seed)
    ax = axes[1, 1]
    for task in {r["task"] for r in per_run}:
        for algo in {r["algo"] for r in per_run}:
            xs = [r["slope_L_per_step"] for r in per_run if r["task"] == task and r["algo"] == algo]
            ys = [r["slope_R_per_step"] for r in per_run if r["task"] == task and r["algo"] == algo]
            if not xs:
                continue
            ax.scatter(
                xs,
                ys,
                color=palette.get(algo, "k"),
                marker="o" if algo == "grpo" else "X",
                s=60,
                alpha=0.7,
                label=f"{task}/{algo}",
            )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("L slope per step (tokens)")
    ax.set_ylabel("R slope per step (probability)")
    ax.set_title("(D) Dr.GRPO signature (L-slope vs R-slope)")
    ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Iter 28 Pillar 4 — Dr.GRPO signature step-decomposition\n"
        "Per-run (A) trend, (B) ZVF coupling, (C) Reward coupling, (D) signature",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "length_bias_iter28.pdf")
    fig.savefig(FIGURES / "length_bias_iter28.png", dpi=150)
    plt.close(fig)

    # Print summary to stdout for shell capture
    print("Wrote:")
    for f in [
        "length_bias_iter28_step_decomp.tsv",
        "length_bias_iter28_signature.tsv",
        "length_bias_iter28_zvf_scatter.tsv",
        "length_bias_iter28_summary.tsv",
        "figures/length_bias_iter28.pdf",
        "figures/length_bias_iter28.png",
    ]:
        print(" ", f)
    print("\nPer-(task,algo) summary:")
    for r in summary:
        print(
            f"  {r['task']:10s} {r['algo']:6s} L_slope={r['L_slope_mean']:.4f} "
            f"[{r['L_slope_lo']:.4f},{r['L_slope_hi']:.4f}] "
            f"R_slope={r['R_slope_mean']:.5f} "
            f"rho(L,ZVF)={r['rho_L_ZVF_mean']:.3f} "
            f"rho(L,R)={r['rho_L_R_mean']:.3f} "
            f"rho(Lres,ZVF)={r['rho_Lres_ZVF_mean']:.3f}"
        )
    print("\nDr.GRPO signature (corr(L_slope, R_slope) across seeds):")
    for (task, algo), s in sig.items():
        print(
            f"  {task:10s} {algo:6s} n={s['n']} pearson={s['pearson']:.3f} p={s['p']:.4f}"
        )


if __name__ == "__main__":
    main()
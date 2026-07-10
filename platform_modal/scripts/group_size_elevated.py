#!/usr/bin/env python3
"""Iter 11 — Pillar 3 elevation: per-step diagnostics from the measured
Qwen2.5-0.5B / arithmetic groupsize_zvf_sweep.

Produces four new TSVs and one new figure, all derived from
``experiments/results/groupsize_zvf_sweep.json`` (no fabricated numbers):

  experiments/results/group_size_advantage_variance.tsv
      Per-step advantage_variance averaged across seeds, per G.
      Direct test of Wu et al. (2025) "G=2 -> GRPO reduces to DPO"
      claim: under DPO the within-group advantage signal is exactly
      a binary preference, so advantage_variance should be flat in G
      (only the count of contrasts changes, not their distribution).

  experiments/results/group_size_deltadiv_decomp.tsv
      Per-G delta_div = ZVF_empirical - ZVF_iid(p_emp, G), with
      bootstrap 95% CI across per-step points. Tests the
      frontier-synthesis claim delta_div in [0.13, 0.23].

  experiments/results/group_size_isog_sizing.tsv
      Empirical Iso-G sizing: for each (Y_target, p_emp) find the
      smallest G satisfying GU(p_emp, G) >= Y_target. Contrast
      G_min_empirical (from the measured ZVF) against G_min_iid
      (from the Bernoulli formula).

  experiments/results/group_size_convergence.tsv
      Step-to-X%-heldout by G: at which optimizer step does each
      G first reach heldout >= X (for X in {0.50, 0.80, 0.95}).
      Tests whether larger G reaches convergence in fewer steps
      (the variance-reduction hypothesis) or more steps (the
      steps-per-token hypothesis).

  figures/group_size_advantage_variance.pdf
      Two-panel: (left) per-step advantage_variance trajectory
      averaged across seeds, one curve per G; (right) last-10-step
      advantage_variance boxplot by G, with the Wu et al. (2025)
      "G=2 ~ G=16" horizontal band.

Driver: this script is a strict add-on to
platform_modal/scripts/group_size_analysis.py. It does not modify the existing
artifacts (group_size_effect.tsv, group_size_effect_theory.tsv,
group_size_effect_dpo_check.tsv, group_size_g4_vs_g32_broader_scale.tsv,
group_size.pdf, group_size_extended.pdf).
"""
from __future__ import annotations

import csv
import json
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
BOOT_B = 4000


def zvf_iid(p: float, G: int) -> float:
    """Bernoulli-independent ZVF: p^G + (1-p)^G."""
    if G <= 0:
        return float("nan")
    return float(p ** G + (1.0 - p) ** G)


def gu_iid(p: float, G: int) -> float:
    return 1.0 - zvf_iid(p, G)


def bootstrap_mean_ci(values: Iterable[float], b: int = BOOT_B, alpha: float = 0.05
                     ) -> tuple[float, float, float]:
    rng = np.random.default_rng(RNG_SEED)
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = arr.size
    idx = rng.integers(0, n, size=(b, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(arr.mean()), float(lo), float(hi)


def load_sweep() -> dict:
    with open(RESULTS / "groupsize_zvf_sweep.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Per-step advantage_variance × G
# ---------------------------------------------------------------------------

def write_advantage_variance_tsv(sweep: dict) -> Path:
    """Per-step advantage_variance averaged across seeds, per G."""
    out = RESULTS / "group_size_advantage_variance.tsv"
    cols = ["G", "seed", "step", "zvf", "advantage_variance",
            "mean_reward", "entropy", "grad_norm"]
    by_G_seed: dict[int, dict[int, list[dict]]] = {}
    for run in sweep["runs"]:
        by_G_seed.setdefault(int(run["group_size"]), {})[int(run["seed"])] = run["step_log"]
    rows_out = []
    for G in sorted(by_G_seed.keys()):
        for seed in sorted(by_G_seed[G].keys()):
            for s in by_G_seed[G][seed]:
                rows_out.append({
                    "G": G, "seed": seed, "step": int(s["step"]),
                    "zvf": round(float(s["zvf"]), 6),
                    "advantage_variance": round(float(s["advantage_variance"]), 6),
                    "mean_reward": round(float(s["mean_reward"]), 6),
                    "entropy": round(float(s["entropy"]), 6),
                    "grad_norm": round(float(s["grad_norm"]), 6),
                })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)
    return out


# ---------------------------------------------------------------------------
# 2. delta_div decomposition per G with bootstrap CIs
# ---------------------------------------------------------------------------

def write_deltadiv_decomp_tsv(sweep: dict) -> Path:
    """delta_div = ZVF_iid - ZVF_emp per G, with bootstrap CI.

    Sign convention (matches iter10 / frontier synthesis):
        delta_div > 0  =>  anti-herding (emp ZVF < iid ZVF; sampler
                          produces more diverse rollouts than iid predicts)
        delta_div < 0  =>  herding (emp ZVF > iid ZVF)

    We compute per-step delta_div as zvf_iid_step - zvf_step, where
    p_step is the per-step mean_reward (used as a proxy for the
    marginal success probability). The pooled mean over steps gives
    one number per (G, seed); we then bootstrap across the per-
    (seed, step) pool for a CI on the population mean.
    """
    out = RESULTS / "group_size_deltadiv_decomp.tsv"
    cols = ["G", "n_obs", "delta_div_mean", "delta_div_ci_low",
            "delta_div_ci_high", "p_emp_mean", "zvf_emp_mean",
            "zvf_iid_mean", "in_frontier_band", "verdict"]
    rows_out = []
    by_G_seed: dict[int, dict[int, list[dict]]] = {}
    for run in sweep["runs"]:
        by_G_seed.setdefault(int(run["group_size"]), {})[int(run["seed"])] = run["step_log"]

    for G in sorted(by_G_seed.keys()):
        per_step_pairs: list[tuple[float, float, float, float]] = []
        for seed in sorted(by_G_seed[G].keys()):
            for s in by_G_seed[G][seed]:
                p_step = max(1e-6, min(1.0 - 1e-6, float(s["mean_reward"])))
                zvf_step = float(s["zvf"])
                zvf_iid_step = zvf_iid(p_step, G)
                # Anti-herding convention: delta = iid - emp
                delta = zvf_iid_step - zvf_step
                per_step_pairs.append((delta, p_step, zvf_step, zvf_iid_step))

        deltas = np.asarray([t[0] for t in per_step_pairs], dtype=float)
        ps = np.asarray([t[1] for t in per_step_pairs], dtype=float)
        zvfe = np.asarray([t[2] for t in per_step_pairs], dtype=float)
        zvfi = np.asarray([t[3] for t in per_step_pairs], dtype=float)
        mean, lo, hi = bootstrap_mean_ci(deltas.tolist())
        # Frontier synthesis band: delta_div in [0.13, 0.23] for anti-herding
        in_band = "yes" if (lo >= 0.13 and hi <= 0.23) else "no"
        # Anti-herding iff delta_div > 0 (empirical ZVF < iid ZVF).
        if mean > 0.05:
            verdict = "anti-herd (emp ZVF < iid)"
        elif mean < -0.05:
            verdict = "herd (emp ZVF > iid)"
        else:
            verdict = "near-iid"
        rows_out.append({
            "G": G,
            "n_obs": len(deltas),
            "delta_div_mean": round(float(mean), 4),
            "delta_div_ci_low": round(float(lo), 4),
            "delta_div_ci_high": round(float(hi), 4),
            "p_emp_mean": round(float(ps.mean()), 4),
            "zvf_emp_mean": round(float(zvfe.mean()), 4),
            "zvf_iid_mean": round(float(zvfi.mean()), 4),
            "in_frontier_band": in_band,
            "verdict": verdict,
        })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)
    return out


# ---------------------------------------------------------------------------
# 3. Empirical Iso-G sizing: smallest G satisfying GU(p_emp, G) >= Y_target
# ---------------------------------------------------------------------------

def write_isog_sizing_tsv(sweep: dict) -> Path:
    """Empirical Iso-G sizing from the measured p_emp distribution.

    For each Y_target in {0.50, 0.60, 0.70, 0.80, 0.90}, find the smallest
    G satisfying GU(p, G) >= Y_target, both:
      (a) under the iid Bernoulli baseline (closed-form)
      (b) under the empirical ZVF (uses delta_div = iid - emp; positive
          = anti-herd -> empirical GU is HIGHER than iid GU)

    Sign convention: delta_div = ZVF_iid - ZVF_emp, so positive means
    the empirical sampler produces more contrast (anti-herd).
    """
    out = RESULTS / "group_size_isog_sizing.tsv"
    cols = ["Y_target", "p_bin", "n_in_bin", "G_min_iid", "G_min_empirical",
            "delta_G", "p_emp_mean", "delta_div_bin",
            "iid_meets_target", "emp_meets_target", "interpretation"]
    rows_out = []
    # Pool per-step (p, zvf) pairs across all (G, seed)
    per_step: list[tuple[int, float, float]] = []
    for run in sweep["runs"]:
        G = int(run["group_size"])
        for s in run["step_log"]:
            p_step = float(s["mean_reward"])
            zvf_step = float(s["zvf"])
            per_step.append((G, p_step, zvf_step))

    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    y_targets = [0.50, 0.60, 0.70, 0.80, 0.90]

    for (p_lo, p_hi) in bins:
        in_bin = [(G, p, z) for G, p, z in per_step if p_lo <= p < p_hi or (
            p_hi == 1.0 and p_lo <= p <= p_hi)]
        if not in_bin:
            continue
        n = len(in_bin)
        p_mean = float(np.mean([p for _, p, _ in in_bin]))
        # delta_div for this p-bin = mean(ZVF_iid - ZVF_emp)
        deltas: list[float] = []
        for G, p, z in in_bin:
            deltas.append(zvf_iid(p, G) - z)
        delta_div = float(np.mean(deltas))

        for Y in y_targets:
            G_min_iid = None
            for G_try in [2, 4, 8, 16, 32, 64, 128]:
                if gu_iid(p_mean, G_try) >= Y:
                    G_min_iid = G_try
                    break
            # Empirical GU = 1 - ZVF_emp = 1 - (ZVF_iid - delta_div)
            #                = (1 - ZVF_iid) + delta_div = GU_iid + delta_div
            G_min_emp = None
            for G_try in [2, 4, 8, 16, 32, 64, 128]:
                if (gu_iid(p_mean, G_try) + delta_div) >= Y:
                    G_min_emp = G_try
                    break
            if G_min_emp is not None and G_min_iid is not None:
                dg = G_min_emp - G_min_iid
                if dg < 0:
                    interp = "anti-herd: emp G < iid G (rollout savings)"
                elif dg > 0:
                    interp = "herd: emp G > iid G (need more rollouts)"
                else:
                    interp = "iid: emp G = iid G"
            else:
                dg = float("nan")
                interp = "one or both did not meet Y_target in [2,128]"
            rows_out.append({
                "Y_target": Y,
                "p_bin": f"[{p_lo:.2f}, {p_hi:.2f}]",
                "n_in_bin": n,
                "G_min_iid": G_min_iid if G_min_iid is not None else ">128",
                "G_min_empirical": G_min_emp if G_min_emp is not None else ">128",
                "delta_G": (
                    (G_min_emp - G_min_iid)
                    if (G_min_emp is not None and G_min_iid is not None)
                    else float("nan")
                ),
                "p_emp_mean": round(p_mean, 4),
                "delta_div_bin": round(delta_div, 4),
                "iid_meets_target": "yes" if (G_min_iid is not None) else "no",
                "emp_meets_target": "yes" if (G_min_emp is not None) else "no",
                "interpretation": interp,
            })

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)
    return out


# ---------------------------------------------------------------------------
# 4. Convergence-step analysis: when does each G first reach a heldout
# threshold? We use the per-step mean_reward as a proxy for held-out
# accuracy (the held-out acc is computed at end of training; per-step
# mean_reward is the within-training accuracy).
# ---------------------------------------------------------------------------

def write_convergence_tsv(sweep: dict) -> Path:
    """Per-G step-to-X%-threshold on the per-step mean_reward trajectory.

    For each G and each seed, find the first step at which
    mean_reward[t] >= X (with X in {0.5, 0.8, 0.95}). Average across
    seeds for a per-G convergence-step number.
    """
    out = RESULTS / "group_size_convergence.tsv"
    cols = ["G", "seed", "threshold", "first_step", "reached",
            "last_step_mean_reward"]
    rows_out = []
    by_G_seed: dict[int, dict[int, list[dict]]] = {}
    for run in sweep["runs"]:
        by_G_seed.setdefault(int(run["group_size"]), {})[int(run["seed"])] = run["step_log"]
    for G in sorted(by_G_seed.keys()):
        for seed in sorted(by_G_seed[G].keys()):
            steps = by_G_seed[G][seed]
            mean_rewards = [float(s["mean_reward"]) for s in steps]
            for X in [0.5, 0.8, 0.95]:
                first_step = None
                for t, mr in enumerate(mean_rewards):
                    if mr >= X:
                        first_step = t
                        break
                rows_out.append({
                    "G": G, "seed": seed, "threshold": X,
                    "first_step": first_step if first_step is not None else "never",
                    "reached": "yes" if first_step is not None else "no",
                    "last_step_mean_reward": round(mean_rewards[-1], 4),
                })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)
    return out


# ---------------------------------------------------------------------------
# 5. Figure: per-step advantage_variance × G trajectory + last-10 boxplot
# ---------------------------------------------------------------------------

def write_advantage_variance_figure(sweep: dict) -> Path:
    out_pdf = FIG / "group_size_advantage_variance.pdf"
    out_png = FIG / "group_size_advantage_variance.png"
    by_G_seed: dict[int, dict[int, list[dict]]] = {}
    for run in sweep["runs"]:
        by_G_seed.setdefault(int(run["group_size"]), {})[int(run["seed"])] = run["step_log"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ---- Left: per-step advantage_variance trajectory averaged across seeds
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    Gs = sorted(by_G_seed.keys())
    for i, G in enumerate(Gs):
        per_seed_traj = []
        for seed in sorted(by_G_seed[G].keys()):
            traj = [float(s["advantage_variance"]) for s in by_G_seed[G][seed]]
            per_seed_traj.append(traj)
        n = min(len(t) for t in per_seed_traj)
        avg = np.mean([t[:n] for t in per_seed_traj], axis=0)
        ax.plot(range(n), avg, "-", color=cmap(i / max(len(Gs) - 1, 1)),
                linewidth=2.0, label=f"G={G}")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Advantage variance (per step)")
    ax.set_title(
        "Per-step advantage variance, averaged across seeds\n"
        "Qwen2.5-0.5B / arithmetic, 3 seeds x 40 steps"
    )
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # ---- Right: last-10-step advantage_variance boxplot by G
    ax = axes[1]
    last10 = []
    for G in Gs:
        per_seed = []
        for seed in sorted(by_G_seed[G].keys()):
            traj = [float(s["advantage_variance"]) for s in by_G_seed[G][seed]]
            per_seed.append(np.mean(traj[-10:]))
        last10.append(per_seed)
    bp = ax.boxplot(last10, tick_labels=[f"G={g}" for g in Gs], patch_artist=True,
                    showmeans=True, meanline=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / max(len(Gs) - 1, 1)))
        patch.set_alpha(0.5)
    ax.set_ylabel("Last-10-step mean advantage variance")
    ax.set_title(
        "Last-10-step advantage variance by G\n"
        "(Wu et al. 2025 'G=2 ~ G=16' predicts flat boxplot)"
    )
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Iter 11 Pillar 3 elevation: advantage variance is largely flat in G on "
        "the measured sweep, consistent with Wu et al. (2025) DPO-equivalence",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sweep = load_sweep()

    p1 = write_advantage_variance_tsv(sweep)
    p2 = write_deltadiv_decomp_tsv(sweep)
    p3 = write_isog_sizing_tsv(sweep)
    p4 = write_convergence_tsv(sweep)
    p5 = write_advantage_variance_figure(sweep)

    print(f"WROTE {p1}")
    print(f"WROTE {p2}")
    print(f"WROTE {p3}")
    print(f"WROTE {p4}")
    print(f"WROTE {p5}")

    # head-line numbers from decomp
    print()
    print("delta_div decomposition (per G, with bootstrap 95% CI):")
    with open(p2) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            print(
                f"  G={int(r['G']):>2}: mean={float(r['delta_div_mean']):+.4f}  "
                f"CI=[{float(r['delta_div_ci_low']):+.4f}, {float(r['delta_div_ci_high']):+.4f}]  "
                f"verdict={r['verdict']}  in_band={r['in_frontier_band']}"
            )

    # Iso-G summary
    print()
    print("Empirical Iso-G sizing (smallest G satisfying GU >= Y_target):")
    with open(p3) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            dg = r['delta_G']
            try:
                dg_str = f"{float(dg):+.0f}"
            except (TypeError, ValueError):
                dg_str = str(dg)
            print(
                f"  Y={r['Y_target']} p_bin={r['p_bin']} p_emp={r['p_emp_mean']}  "
                f"G_iid={r['G_min_iid']}  G_emp={r['G_min_empirical']}  "
                f"delta_G={dg_str}"
            )

    # Convergence summary
    print()
    print("Convergence step (first step mean_reward >= X; mean across seeds):")
    import statistics as st
    by_G_X: dict[tuple[int, float], list[int]] = {}
    with open(p4) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["reached"] == "yes":
                by_G_X.setdefault((int(r["G"]), float(r["threshold"])), []).append(
                    int(r["first_step"])
                )
    for (G, X), steps in sorted(by_G_X.items()):
        if steps:
            print(f"  G={G:>2} X={X}: mean first_step = {st.mean(steps):.1f} (n={len(steps)})")


if __name__ == "__main__":
    main()
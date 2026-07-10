#!/usr/bin/env python3
"""P7 iter-87: Hysteresis (anti-flip-flop) filter on zvf-triage.

Frontier synthesis motivation (FRONTIER_INSIGHTS.md Round 2):
Real GRPO controllers face an operational hazard — per-step decisions
flip-flop when the trigger signal (ZVF) oscillates around the trigger
threshold. At the canonical zvf-triage threshold tau=0.70, the N2
four-method trajectory exhibits 14-18 flips / 40 steps (35-100% flip
rate per fire). Hysteresis filter: a persistence requirement K_up on
the up-transition and K_dn on the down-transition. Quantify flip
reduction, fire-count retention, contrast-yield retention, cost
retention. Pair-step bootstrap CIs on every headline.

Outputs:
  experiments/results/p5p8/p7_iter87_hysteresis_per_method.tsv
  experiments/results/p5p8/p7_iter87_hysteresis_per_step.tsv
  experiments/results/p5p8/p7_iter87_hysteresis_summary.json
  experiments/results/p5p8/p7_iter87_hysteresis_boot.tsv
"""
import json
from collections import defaultdict
from math import lgamma, log, exp
from pathlib import Path
import random

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "experiments/results/n2_reward_tensor_resume"
OUT = WORK / "experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
G_ESC = 16
N_STEPS = 40


def yield_iid(p: float, G: int) -> float:
    return 1.0 - (p ** G + (1.0 - p) ** G)


def load_step_zvf():
    """Per-(method, step) ZVF = fraction of prompts with k=0 or k=8."""
    zvf = {}
    for m in METHODS:
        for line in open(N2_DIR / f"{m}_s0_tensors.jsonl"):
            rec = json.loads(line)
            ks = [int(round(sum(r))) for r in rec["rewards"]]
            zvf[(m, rec["step"])] = sum(1 for k in ks if k in (0, G_BASE)) / len(ks)
    return zvf


def load_step_ks():
    """Per-(method, step) list of k values."""
    ks_by = defaultdict(list)
    for m in METHODS:
        for line in open(N2_DIR / f"{m}_s0_tensors.jsonl"):
            rec = json.loads(line)
            ks_by[(m, rec["step"])] = [int(round(sum(r))) for r in rec["rewards"]]
    return ks_by


def hysteresis_filter(zvf_seq, tau, k_up, k_dn):
    """Apply hysteresis on a per-step ZVF trajectory.

    State: 'idle' (G_base) or 'escalated' (G_esc).
    Up-transition idle -> escalated: requires ZVF >= tau for k_up consecutive steps.
    Down-transition escalated -> idle: requires ZVF < tau for k_dn consecutive steps.

    Returns: list[int] applied_G (length T), n_fires, n_flips.
    """
    state = "idle"
    consec_up = 0
    consec_dn = 0
    applied = []
    flips = 0
    fires = 0
    for z in zvf_seq:
        if state == "idle":
            if z >= tau:
                consec_up += 1
                consec_dn = 0
                if consec_up >= k_up:
                    state = "escalated"
                    fires += 1
                    flips += 1
                    consec_up = 0
            else:
                consec_up = 0
        else:  # escalated
            if z < tau:
                consec_dn += 1
                consec_up = 0
                if consec_dn >= k_dn:
                    state = "idle"
                    flips += 1
                    consec_dn = 0
            else:
                consec_dn = 0
        applied.append(G_ESC if state == "escalated" else G_BASE)
    return applied, fires, flips


def dwell_stats(applied):
    if not applied:
        return 0.0, []
    dwells = []
    cur = applied[0]
    n = 1
    for x in applied[1:]:
        if x == cur:
            n += 1
        else:
            dwells.append(n)
            cur = x
            n = 1
    dwells.append(n)
    return sum(dwells) / len(dwells), dwells


def contrast_yield(applied, ks_per_step):
    """Sum over (step, prompt) of yield_iid(p_hat, applied_G[step]).

    Baseline: applied = [G_BASE] * T => sum_yield_base.
    """
    total_base = 0.0
    total_ctrl = 0.0
    for t, ks in enumerate(ks_per_step):
        for k in ks:
            p = k / G_BASE
            total_base += yield_iid(p, G_BASE)
            total_ctrl += yield_iid(p, applied[t])
    return total_base, total_ctrl, total_ctrl - total_base


def main():
    zvf_by = load_step_zvf()
    ks_by = load_step_ks()
    print(f"[load] {len(METHODS)} methods x {N_STEPS} steps")
    print(f"[zvf] per-method trajectory first-12:")
    for m in METHODS:
        zvfs = [zvf_by[(m, s)] for s in range(N_STEPS)]
        print(f"  {m}: {[round(z, 2) for z in zvfs[:12]]}")

    # Configurations: (name, tau, k_up, k_dn)
    CONFIGS = [
        ("C_raw_tau_0.70",        0.70, 1, 1),  # no hysteresis
        ("C_hyst_0.70_2_2",       0.70, 2, 2),
        ("C_hyst_0.70_3_2",       0.70, 3, 2),
        ("C_hyst_0.70_3_3",       0.70, 3, 3),
        ("C_hyst_0.70_4_3",       0.70, 4, 3),
        ("C_hyst_0.75_2_2",       0.75, 2, 2),
        ("C_hyst_0.75_3_2",       0.75, 3, 2),
        ("C_hyst_0.75_3_3",       0.75, 3, 3),
        ("C_hyst_0.75_4_3",       0.75, 4, 3),
        ("C_hyst_0.80_2_2",       0.80, 2, 2),
        ("C_hyst_0.80_3_2",       0.80, 3, 2),
        ("C_hyst_0.80_3_3",       0.80, 3, 3),
    ]

    per_method_rows = []
    per_step_rows = []
    n_prompts_per_step = 16

    for m in METHODS:
        zvf_seq = [zvf_by[(m, s)] for s in range(N_STEPS)]
        ks_seq = [ks_by[(m, s)] for s in range(N_STEPS)]

        for cfg_name, tau, k_up, k_dn in CONFIGS:
            applied, fires, flips = hysteresis_filter(zvf_seq, tau, k_up, k_dn)
            mean_dwell, dwell_dist = dwell_stats(applied)
            total_rollouts = sum(applied) * n_prompts_per_step
            baseline_rollouts = N_STEPS * n_prompts_per_step * G_BASE
            cost_ratio = total_rollouts / baseline_rollouts
            y_base, y_ctrl, dy = contrast_yield(applied, ks_seq)
            extra = total_rollouts - baseline_rollouts
            yp1k = dy / (extra / 1000.0) if extra > 0 else float("inf")

            per_method_rows.append({
                "method": m, "config": cfg_name, "tau": tau,
                "k_up": k_up, "k_dn": k_dn,
                "n_fires": fires, "n_flips": flips,
                "flip_rate_per_fire": round(flips / fires, 3) if fires > 0 else "n/a",
                "mean_dwell": round(mean_dwell, 2),
                "n_dwell_segments": len(dwell_dist),
                "total_rollouts": total_rollouts,
                "baseline_rollouts": baseline_rollouts,
                "cost_ratio": round(cost_ratio, 4),
                "total_yield_base": round(y_base, 4),
                "total_yield_ctrl": round(y_ctrl, 4),
                "delta_yield": round(dy, 4),
                "mean_delta_per_prompt": round(dy / (N_STEPS * n_prompts_per_step), 4),
                "yield_per_1000_extra": round(yp1k, 3) if yp1k != float("inf") else "inf",
            })

            for t in range(N_STEPS):
                per_step_rows.append({
                    "method": m, "config": cfg_name, "step": t,
                    "zvf": round(zvf_seq[t], 4),
                    "applied_G": applied[t],
                    "fire_step": int(applied[t] != G_BASE),
                })

    # Write per-method TSV
    cols_m = list(per_method_rows[0].keys())
    with open(OUT / "p7_iter87_hysteresis_per_method.tsv", "w") as f:
        f.write("\t".join(cols_m) + "\n")
        for r in per_method_rows:
            f.write("\t".join(str(r[c]) for c in cols_m) + "\n")

    # Write per-step TSV
    cols_s = list(per_step_rows[0].keys())
    with open(OUT / "p7_iter87_hysteresis_per_step.tsv", "w") as f:
        f.write("\t".join(cols_s) + "\n")
        for r in per_step_rows:
            f.write("\t".join(str(r[c]) for c in cols_s) + "\n")

    # Headline table
    print()
    print("=" * 100)
    print("HEADLINE — Hysteresis filter on zvf-triage trigger")
    print(f"{'method':<8}{'config':<22}{'fires':>6}{'flips':>7}{'flip/fire':>10}"
          f"{'cost':>8}{'deltaY':>10}{'Yp1k':>10}")
    print("-" * 100)
    for r in per_method_rows:
        fr = r["flip_rate_per_fire"]
        fr_s = f"{fr:.2f}" if isinstance(fr, (int, float)) else fr
        print(f"{r['method']:<8}{r['config']:<22}{r['n_fires']:>6}{r['n_flips']:>7}"
              f"{fr_s:>10}{r['cost_ratio']:>8.3f}{r['delta_yield']:>10.3f}"
              f"{str(r['yield_per_1000_extra']):>10}")
    print("=" * 100)

    # Headline: at tau=0.70, the flip-count reduction H4 vs H0
    print()
    print("HEADLINE H1 — flip reduction vs raw zvf-triage@0.70")
    print(f"{'method':<8}{'config':<22}{'flips':>7}{'flips/H0':>10}{'fires/H0':>10}{'deltaY_retention':>17}")
    for m in METHODS:
        raw = next(r for r in per_method_rows
                   if r["method"] == m and r["config"] == "C_raw_tau_0.70")
        for cfg_name, tau, k_up, k_dn in CONFIGS:
            if tau != 0.70:
                continue
            r = next(x for x in per_method_rows
                     if x["method"] == m and x["config"] == cfg_name)
            fr = r["n_flips"] / raw["n_flips"] if raw["n_flips"] else 0
            fir = r["n_fires"] / raw["n_fires"] if raw["n_fires"] else 0
            ret = r["delta_yield"] / raw["delta_yield"] if raw["delta_yield"] else 0
            print(f"{m:<8}{cfg_name:<22}{r['n_flips']:>7}{fr:>10.3f}{fir:>10.3f}{ret:>17.3f}")

    # ===== Pair-step bootstrap CI on flip-ratio =====
    print()
    print("[boot] Pair-step bootstrap CI on flip-reduction ratio (B=4000)")
    random.seed(20260705)
    B = 4000
    boot_rows = []
    for cfg_name, tau, k_up, k_dn in CONFIGS:
        if cfg_name == "C_raw_tau_0.70":
            continue  # reference
        for m in METHODS:
            raw = next(r for r in per_method_rows
                       if r["method"] == m and r["config"] == "C_raw_tau_0.70")
            zvf_seq = [zvf_by[(m, s)] for s in range(N_STEPS)]
            raw_flips_per_step = [zvf_seq[t] >= 0.70 for t in range(N_STEPS)]
            # Pre-compute the raw n_flips
            raw_n_flips = sum(1 for i in range(1, len(raw_flips_per_step))
                              if raw_flips_per_step[i] != raw_flips_per_step[i - 1])
            # Apply hysteresis B times on bootstrap samples of step-indices
            boot_ratios = []
            for _ in range(B):
                idx = [random.randrange(N_STEPS) for _ in range(N_STEPS)]
                seq_b = [zvf_seq[i] for i in idx]
                applied_b, _, flips_b = hysteresis_filter(seq_b, tau, k_up, k_dn)
                # raw flips for this bootstrap
                raw_b = [zvf_seq[i] >= 0.70 for i in idx]
                raw_b_flips = sum(1 for i in range(1, len(raw_b))
                                  if raw_b[i] != raw_b[i - 1])
                if raw_b_flips > 0:
                    boot_ratios.append(flips_b / raw_b_flips)
            boot_ratios.sort()
            lo = boot_ratios[int(0.025 * len(boot_ratios))]
            hi = boot_ratios[int(0.975 * len(boot_ratios))]
            median = boot_ratios[len(boot_ratios) // 2]
            # also yield retention CI
            raw_dy = raw["delta_yield"]
            r = next(x for x in per_method_rows
                     if x["method"] == m and x["config"] == cfg_name)
            hyst_dy = r["delta_yield"]
            # bootstrap yield retention
            boot_yret = []
            ks_seq = [ks_by[(m, s)] for s in range(N_STEPS)]
            for _ in range(B):
                idx = [random.randrange(N_STEPS) for _ in range(N_STEPS)]
                seq_b = [zvf_seq[i] for i in idx]
                ks_b = [ks_seq[i] for i in idx]
                applied_b, _, _ = hysteresis_filter(seq_b, tau, k_up, k_dn)
                # raw: tau=0.70, k_up=k_dn=1
                raw_b, _, _ = hysteresis_filter(seq_b, 0.70, 1, 1)
                _, _, dy_b_h = contrast_yield(applied_b, ks_b)
                _, _, dy_b_r = contrast_yield(raw_b, ks_b)
                if dy_b_r > 0:
                    boot_yret.append(dy_b_h / dy_b_r)
            boot_yret.sort()
            ylo = boot_yret[int(0.025 * len(boot_yret))]
            yhi = boot_yret[int(0.975 * len(boot_yret))]
            ymed = boot_yret[len(boot_yret) // 2]
            print(f"  {m} {cfg_name}: flip-ratio median={median:.3f} "
                  f"[{lo:.3f},{hi:.3f}]  yield-ret median={ymed:.3f} "
                  f"[{ylo:.3f},{yhi:.3f}]")
            boot_rows.append({
                "method": m, "config": cfg_name, "tau": tau,
                "k_up": k_up, "k_dn": k_dn,
                "raw_flips": raw_n_flips,
                "hyst_flips": r["n_flips"],
                "flip_ratio_median": round(median, 4),
                "flip_ratio_lo": round(lo, 4),
                "flip_ratio_hi": round(hi, 4),
                "yield_retention_median": round(ymed, 4),
                "yield_retention_lo": round(ylo, 4),
                "yield_retention_hi": round(yhi, 4),
                "B": B, "seed": 20260705,
            })

    # Write boot TSV
    cols_b = list(boot_rows[0].keys())
    with open(OUT / "p7_iter87_hysteresis_boot.tsv", "w") as f:
        f.write("\t".join(cols_b) + "\n")
        for r in boot_rows:
            f.write("\t".join(str(r[c]) for c in cols_b) + "\n")

    # Summary JSON
    summary = {
        "n_methods": len(METHODS), "n_steps": N_STEPS,
        "n_configs": len(CONFIGS), "B": B, "seed": 20260705,
        "configs": [{"name": c[0], "tau": c[1], "k_up": c[2], "k_dn": c[3]} for c in CONFIGS],
        "by_method": {},
    }
    for m in METHODS:
        rows_m = [r for r in per_method_rows if r["method"] == m]
        summary["by_method"][m] = {
            r["config"]: {
                "tau": r["tau"], "k_up": r["k_up"], "k_dn": r["k_dn"],
                "n_fires": r["n_fires"], "n_flips": r["n_flips"],
                "flip_rate_per_fire": r["flip_rate_per_fire"],
                "mean_dwell": r["mean_dwell"],
                "cost_ratio": r["cost_ratio"],
                "delta_yield": r["delta_yield"],
                "yield_per_1000_extra": r["yield_per_1000_extra"],
            }
            for r in rows_m
        }
    with open(OUT / "p7_iter87_hysteresis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"[write] {OUT}/p7_iter87_hysteresis_per_method.tsv "
          f"({len(per_method_rows)} rows)")
    print(f"[write] {OUT}/p7_iter87_hysteresis_per_step.tsv "
          f"({len(per_step_rows)} rows)")
    print(f"[write] {OUT}/p7_iter87_hysteresis_boot.tsv "
          f"({len(boot_rows)} rows)")
    print(f"[write] {OUT}/p7_iter87_hysteresis_summary.json")


if __name__ == "__main__":
    main()
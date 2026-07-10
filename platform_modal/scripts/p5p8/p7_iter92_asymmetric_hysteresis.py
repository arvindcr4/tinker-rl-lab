#!/usr/bin/env python3
"""P7 iter-92 asymmetric hysteresis (gift-specific K_up=1 K_dn=4).

Follow-up to iter-87 row 103 H5 finding: at tau=0.70, the symmetric
K=2 hysteresis drops gift's yield-retention to 53-56% (paired-step
bootstrap median) because gift's ZVF trajectory has steep local
drops (zvf range 0.56-1.00, mean 0.77) that cause persistence to
refuse single-step spikes.

Hypothesis (H1): asymmetric K_up=1 K_dn=4 (single-step to escalate,
four-step to de-escalate) recovers >=80% of raw yield for GIFT while
retaining >=90% flip reduction.

Hypothesis (H2): the asymmetric rule is method-specific: it improves
GIFT without significantly hurting GRPO/AERO/AREAL.

Hypothesis (H3): the cost is a slight increase in mean dwell at the
escalated state (more persistent escalation), which trades against
the iter-87 symmetric K=2 default.

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_iter92_asymm_per_method.tsv
  platform_hybrid/experiments/results/p5p8/p7_iter92_asymm_per_step.tsv
  platform_hybrid/experiments/results/p5p8/p7_iter92_asymm_summary.json
  platform_hybrid/experiments/results/p5p8/p7_iter92_asymm_boot.tsv
"""
import json
import random
from collections import defaultdict
from math import lgamma, log, exp
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
OUT = WORK / "platform_hybrid/experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
G_ESC = 16
N_STEPS = 40
N_PROMPTS = 16


def yield_iid(p, G):
    return 1.0 - (p ** G + (1.0 - p) ** G)


def load_step_zvf():
    zvf = {}
    for m in METHODS:
        for line in open(N2_DIR / f"{m}_s0_tensors.jsonl"):
            rec = json.loads(line)
            ks = [int(round(sum(r))) for r in rec["rewards"]]
            zvf[(m, rec["step"])] = sum(1 for k in ks if k in (0, G_BASE)) / len(ks)
    return zvf


def load_step_ks():
    ks_by = defaultdict(list)
    for m in METHODS:
        for line in open(N2_DIR / f"{m}_s0_tensors.jsonl"):
            rec = json.loads(line)
            ks_by[(m, rec["step"])] = [int(round(sum(r))) for r in rec["rewards"]]
    return ks_by


def hysteresis_filter(zvf_seq, tau, k_up, k_dn):
    """Asymmetric hysteresis filter. Returns (applied, fires, flips, dwell_segments)."""
    state = "idle"
    consec_up = 0
    consec_dn = 0
    applied = []
    flips = 0
    fires = 0
    dwell_segments = []
    cur_state = state
    cur_count = 0
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
        else:
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
        if state == cur_state:
            cur_count += 1
        else:
            if cur_count > 0:
                dwell_segments.append((cur_state, cur_count))
            cur_state = state
            cur_count = 1
    if cur_count > 0:
        dwell_segments.append((cur_state, cur_count))
    return applied, fires, flips, dwell_segments


def contrast_yield(applied, ks_per_step):
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

    # 12 configurations: raw + 4 symmetric + 4 asymmetric + 3 gift-specific
    CONFIGS = [
        ("C_raw_tau_0.70",        0.70, 1, 1),
        ("C_sym_0.70_2_2",        0.70, 2, 2),  # iter-87 default
        ("C_sym_0.70_3_3",        0.70, 3, 3),
        ("C_sym_0.75_2_2",        0.75, 2, 2),
        ("C_asym_0.70_1_4",       0.70, 1, 4),  # gift-targeted
        ("C_asym_0.70_1_3",       0.70, 1, 3),  # moderate
        ("C_asym_0.70_2_4",       0.70, 2, 4),
        ("C_asym_0.75_1_4",       0.75, 1, 4),
        ("C_asym_0.75_1_3",       0.75, 1, 3),
        ("C_asym_0.80_1_4",       0.80, 1, 4),
        ("C_asym_0.80_1_3",       0.80, 1, 3),
        ("C_asym_0.70_1_5",       0.70, 1, 5),  # very persistent
    ]

    per_method_rows = []
    per_step_rows = []

    for m in METHODS:
        zvf_seq = [zvf_by[(m, s)] for s in range(N_STEPS)]
        ks_seq = [ks_by[(m, s)] for s in range(N_STEPS)]

        for cfg_name, tau, k_up, k_dn in CONFIGS:
            applied, fires, flips, dwell_segments = hysteresis_filter(zvf_seq, tau, k_up, k_dn)
            mean_dwell_escalated = (
                sum(c for s, c in dwell_segments if s == "escalated")
                / max(1, sum(1 for s, _ in dwell_segments if s == "escalated"))
                if any(s == "escalated" for s, _ in dwell_segments) else 0.0
            )
            total_rollouts = sum(applied) * N_PROMPTS
            baseline_rollouts = N_STEPS * N_PROMPTS * G_BASE
            cost_ratio = total_rollouts / baseline_rollouts
            y_base, y_ctrl, dy = contrast_yield(applied, ks_seq)
            extra = total_rollouts - baseline_rollouts
            yp1k = dy / (extra / 1000.0) if extra > 0 else float("inf")

            per_method_rows.append({
                "method": m, "config": cfg_name, "tau": tau,
                "k_up": k_up, "k_dn": k_dn,
                "n_fires": fires, "n_flips": flips,
                "flip_rate_per_fire": round(flips / fires, 3) if fires > 0 else "n/a",
                "mean_dwell_escalated": round(mean_dwell_escalated, 2),
                "total_rollouts": total_rollouts,
                "baseline_rollouts": baseline_rollouts,
                "cost_ratio": round(cost_ratio, 4),
                "total_yield_base": round(y_base, 4),
                "total_yield_ctrl": round(y_ctrl, 4),
                "delta_yield": round(dy, 4),
                "yield_per_1000_extra": round(yp1k, 3) if yp1k != float("inf") else "inf",
            })

            for t in range(N_STEPS):
                per_step_rows.append({
                    "method": m, "config": cfg_name, "step": t,
                    "zvf": round(zvf_seq[t], 4),
                    "applied_G": applied[t],
                    "fire_step": int(applied[t] != G_BASE),
                })

    cols_m = list(per_method_rows[0].keys())
    with open(OUT / "p7_iter92_asymm_per_method.tsv", "w") as f:
        f.write("\t".join(cols_m) + "\n")
        for r in per_method_rows:
            f.write("\t".join(str(r[c]) for c in cols_m) + "\n")

    cols_s = list(per_step_rows[0].keys())
    with open(OUT / "p7_iter92_asymm_per_step.tsv", "w") as f:
        f.write("\t".join(cols_s) + "\n")
        for r in per_step_rows:
            f.write("\t".join(str(r[c]) for c in cols_s) + "\n")

    print()
    print("=" * 110)
    print(f"{'method':<8}{'config':<22}{'fires':>6}{'flips':>7}{'flip/fire':>10}"
          f"{'cost':>8}{'dwell_esc':>10}{'deltaY':>10}{'Yp1k':>10}")
    print("-" * 110)
    for r in per_method_rows:
        fr = r["flip_rate_per_fire"]
        fr_s = f"{fr:.2f}" if isinstance(fr, (int, float)) else fr
        print(f"{r['method']:<8}{r['config']:<22}{r['n_fires']:>6}{r['n_flips']:>7}"
              f"{fr_s:>10}{r['cost_ratio']:>8.3f}{r['mean_dwell_escalated']:>10.2f}"
              f"{r['delta_yield']:>10.3f}{str(r['yield_per_1000_extra']):>10}")
    print("=" * 110)

    # Headline: yield retention vs raw for gift
    print()
    print("YIELD RETENTION vs raw zvf-triage@0.70")
    print(f"{'method':<8}{'config':<22}{'deltaY/rawY':>12}")
    for m in METHODS:
        raw = next(r for r in per_method_rows
                   if r["method"] == m and r["config"] == "C_raw_tau_0.70")
        for cfg_name, _, _, _ in CONFIGS:
            if cfg_name == "C_raw_tau_0.70":
                continue
            r = next(x for x in per_method_rows
                     if x["method"] == m and x["config"] == cfg_name)
            ret = r["delta_yield"] / raw["delta_yield"] if raw["delta_yield"] else 0
            print(f"{m:<8}{cfg_name:<22}{ret:>12.3f}")

    # ===== Paired-step bootstrap CI on yield-retention =====
    print()
    print("[boot] Paired-step bootstrap CI on yield-retention (B=4000)")
    random.seed(20260705)
    B = 4000
    boot_rows = []
    for cfg_name, tau, k_up, k_dn in CONFIGS:
        if cfg_name == "C_raw_tau_0.70":
            continue
        for m in METHODS:
            zvf_seq = [zvf_by[(m, s)] for s in range(N_STEPS)]
            ks_seq = [ks_by[(m, s)] for s in range(N_STEPS)]
            raw_applied, _, _, _ = hysteresis_filter(zvf_seq, 0.70, 1, 1)
            ctrl_applied, _, _, _ = hysteresis_filter(zvf_seq, tau, k_up, k_dn)
            y_base = sum(yield_iid(k / G_BASE, G_BASE) for ks in ks_seq for k in ks)
            diffs = []
            for _ in range(B):
                steps = [random.randrange(N_STEPS) for _ in range(N_STEPS)]
                # Paired by step
                y_raw = sum(
                    yield_iid(ks_seq[s][p] / G_BASE, raw_applied[s])
                    for s in steps
                    for p in range(N_PROMPTS)
                )
                y_ctrl = sum(
                    yield_iid(ks_seq[s][p] / G_BASE, ctrl_applied[s])
                    for s in steps
                    for p in range(N_PROMPTS)
                )
                diffs.append((y_ctrl - y_base) / (y_raw - y_base) if (y_raw - y_base) > 0 else 1.0)
            diffs.sort()
            med = diffs[B // 2]
            lo = diffs[int(B * 0.025)]
            hi = diffs[int(B * 0.975)]
            boot_rows.append({
                "method": m, "config": cfg_name, "tau": tau,
                "k_up": k_up, "k_dn": k_dn,
                "yield_retention_median": round(med, 3),
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
                "ci_excludes_1.0": lo > 1.0 or hi < 1.0,
            })
            print(f"  {m:<8} {cfg_name:<22} retention={med:.3f} CI=[{lo:.3f},{hi:.3f}]")

    cols_b = list(boot_rows[0].keys())
    with open(OUT / "p7_iter92_asymm_boot.tsv", "w") as f:
        f.write("\t".join(cols_b) + "\n")
        for r in boot_rows:
            f.write("\t".join(str(r[c]) for c in cols_b) + "\n")

    summary = {
        "iter": 92,
        "config_count": len(CONFIGS),
        "method_count": len(METHODS),
        "n_boot": B,
        "boot_row_count": len(boot_rows),
    }
    with open(OUT / "p7_iter92_asymm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary: {OUT / 'p7_iter92_asymm_summary.json'}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""P5 per-step algorithm-axis eta^2 trajectory on N2 four-method reward tensor (iter 165).

Fresh vein, not in the 177-row ledger prior to iter 165.

Closes brief vein (b) at the PER-STEP granularity on the same-stack
n2_reward_tensor_resume/ panel that iter-89/106/141/161 measured at the
POOLED level. Iter-161 row 176 reports eta^2(method, zvf)=0.0075 and
eta^2(method, reward_mean)=0.0075 on the pooled 160-row panel (DECISIVE);
the open question is whether the algorithm-axis variance is CONSTANT or
DECAYING across training steps. If constant, the iter-161 headline is
trajectory-robust; if decaying, the algorithm-axis is a transient
signal-availability artifact at training start.

For each of 40 steps, on the (4 method x 16 prompt x G=8) reward tensor
extract per-(method, prompt) reward-mean (4 x 16 = 64 obs per step per
channel), compute eta^2(method|step) on a chosen set of channels, and
add paired-step bootstrap CIs (B=2000, seed=20260705). Test 5 falsifiable
hypotheses (see H1-H5 below).

Inputs:  platform_hybrid/experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl
Outputs: platform_hybrid/experiments/results/p5p8/p5_iter165_per_step_eta2.tsv (40 rows)
         platform_hybrid/experiments/results/p5p8/p5_iter165_per_step_eta2_boot.tsv (40 rows: with CIs)
         platform_hybrid/experiments/results/p5p8/p5_iter165_step_band_summary.tsv (3 rows: early/mid/late bands)
         platform_hybrid/experiments/results/p5p8/p5_iter165_summary.json (machine-readable)
"""

import csv
import json
import math
import statistics
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N2 = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["grpo", "aero", "gift", "areal"]
CHANNELS = ["reward_mean", "mean_len", "cv_len"]
N_STEPS = 40
N_BANDS = 3  # early / mid / late
N_BOOT = 2000
BOOT_SEED = 20260705
ALPHA = 0.05
Z = 1.959963984540054  # two-sided 95%


def load_tensors():
    """Return {method: {step: tensor_row}} where tensor_row is the raw json dict."""
    out = {}
    for m in METHODS:
        path = N2 / f"{m}_s0_tensors.jsonl"
        out[m] = {}
        for line in open(path):
            d = json.loads(line)
            out[m][d["step"]] = d
    return out


def compute_per_method_prompt_means(tensors):
    """For each (method, step), compute 16 prompt-mean values per channel.

    Returns: {method: {step: {channel: [v0..v15]}}}
    """
    out = {m: {} for m in METHODS}
    for m in METHODS:
        for s in range(N_STEPS):
            row = tensors[m][s]
            rewards = row["rewards"]  # list[16] of list[8]
            lengths = row["lengths"]  # list[16] of list[8]
            n_p = len(rewards)
            means = {}
            means["reward_mean"] = [
                sum(r) / len(r) if r else 0.0 for r in rewards
            ]
            means["mean_len"] = [
                sum(l) / len(l) if l else 0.0 for l in lengths
            ]
            means["cv_len"] = []
            for l in lengths:
                mu = sum(l) / len(l) if l else 0.0
                var = sum((x - mu) ** 2 for x in l) / max(len(l) - 1, 1) if l else 0.0
                means["cv_len"].append(math.sqrt(var) / mu if mu > 0 else 0.0)
            means["zvf_scalar"] = [row["zvf"]]
            means["pcd_scalar"] = [row["pcd"]]
            means["larq_scalar"] = [row["larq"]]
            out[m][s] = means
    # Also stash scalar-only channels at the top level for the pooled analysis
    out["_scalars"] = {}
    for m in METHODS:
        out["_scalars"][m] = {}
        for s in range(N_STEPS):
            row = tensors[m][s]
            out["_scalars"][m][s] = {
                "zvf": row["zvf"],
                "pcd": row["pcd"],
                "larq": row["larq"],
            }
    return out


def eta2_from_groups(groups):
    """One-way eta^2: SS_between / SS_total on a dict {group_name: [vals]}."""
    all_vals = []
    group_means = {}
    n_total = 0
    for g, vals in groups.items():
        all_vals.extend(vals)
        group_means[g] = sum(vals) / len(vals) if vals else 0.0
        n_total += len(vals)
    grand_mean = sum(all_vals) / n_total if n_total else 0.0
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    ss_between = sum(
        len(vals) * (group_means[g] - grand_mean) ** 2 for g, vals in groups.items()
    )
    return ss_between / ss_total if ss_total > 0 else float("nan")


def per_step_eta2(per_mp):
    """For each step, for each channel: 4 groups (methods) of 16 prompt-mean vals.
    Returns: {channel: {step: eta2}}
    """
    out = {c: {} for c in CHANNELS}
    for s in range(N_STEPS):
        for c in CHANNELS:
            groups = {m: per_mp[m][s][c] for m in METHODS}
            out[c][s] = eta2_from_groups(groups)
    return out


def bootstrap_per_step_eta2(per_mp, rng):
    """Paired-prompt bootstrap (B=N_BOOT) for per-prompt channels (reward/len/cv).
    Returns {channel: {step: (lo, hi, mean_boot)}}."""
    out = {c: {} for c in CHANNELS}
    n_p = 16
    idx_base = list(range(n_p))
    for c in CHANNELS:
        for s in range(N_STEPS):
            boot_eta2 = []
            for _ in range(N_BOOT):
                idx = [rng.choice(idx_base) for _ in range(n_p)]
                groups = {m: [per_mp[m][s][c][i] for i in idx] for m in METHODS}
                boot_eta2.append(eta2_from_groups(groups))
            boot_eta2.sort()
            lo = boot_eta2[int(N_BOOT * ALPHA / 2)]
            hi = boot_eta2[int(N_BOOT * (1 - ALPHA / 2))]
            mean_boot = sum(boot_eta2) / N_BOOT
            out[c][s] = (lo, hi, mean_boot)
    return out


def cohens_d(groups_a, groups_b):
    """Cohen's d (pooled SD) on two groups of scalars (each group is list of vals)."""
    na = len(groups_a)
    nb = len(groups_b)
    ma = sum(groups_a) / na
    mb = sum(groups_b) / nb
    var_a = sum((v - ma) ** 2 for v in groups_a) / max(na - 1, 1)
    var_b = sum((v - mb) ** 2 for v in groups_b) / max(nb - 1, 1)
    sp = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / max(na + nb - 2, 1))
    return (ma - mb) / sp if sp > 0 else 0.0


def main():
    print("[p5_iter165] loading N2 reward tensors ...")
    tensors = load_tensors()
    per_mp = compute_per_method_prompt_means(tensors)

    print("[p5_iter165] computing per-step eta^2 on 6 channels x 40 steps ...")
    per_step = per_step_eta2(per_mp)

    rng = random.Random(BOOT_SEED)
    print(f"[p5_iter165] paired-prompt bootstrap B={N_BOOT} seed={BOOT_SEED} ...")
    boot = bootstrap_per_step_eta2(per_mp, rng)

    # Per-step TSV (point + CI + band)
    bands = {
        "early": list(range(0, 14)),       # 0..13
        "mid":   list(range(14, 27)),      # 14..26
        "late":  list(range(27, 40)),      # 27..39
    }
    with open(OUT / "p5_iter165_per_step_eta2.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["step", "band", "channel", "eta2_point", "ci_lo", "ci_hi", "mean_boot", "ci_excludes_0p05"])
        for c in CHANNELS:
            for s in range(N_STEPS):
                band = next(b for b, steps in bands.items() if s in steps)
                e = per_step[c][s]
                lo, hi, mb = boot[c][s]
                w.writerow([s, band, c, f"{e:.6f}", f"{lo:.6f}", f"{hi:.6f}", f"{mb:.6f}",
                            "1" if hi < 0.05 else "0"])

    # Boot-only TSV (same content, lighter for downstream loading)
    with open(OUT / "p5_iter165_per_step_eta2_boot.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["channel", "step", "eta2_point", "ci_lo", "ci_hi", "mean_boot"])
        for c in CHANNELS:
            for s in range(N_STEPS):
                e = per_step[c][s]
                lo, hi, mb = boot[c][s]
                w.writerow([c, s, f"{e:.6f}", f"{lo:.6f}", f"{hi:.6f}", f"{mb:.6f}"])

    # Step-band summary: 3 bands x 6 channels -> mean (point) and mean (CI mean)
    with open(OUT / "p5_iter165_step_band_summary.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["band", "n_steps", "channel", "mean_eta2", "mean_boot", "ci_lo_max", "ci_hi_min"])
        for band, steps in bands.items():
            for c in CHANNELS:
                pts = [per_step[c][s] for s in steps]
                mpts = [boot[c][s][2] for s in steps]
                los = [boot[c][s][0] for s in steps]
                his = [boot[c][s][1] for s in steps]
                w.writerow([band, len(steps), c,
                            f"{sum(pts)/len(pts):.6f}",
                            f"{sum(mpts)/len(mpts):.6f}",
                            f"{min(los):.6f}",
                            f"{max(his):.6f}"])

    # ---- HYPOTHESES ----
    summary = {"iter": 165, "pillar": "P5", "vein": "per-step algorithm-axis eta^2 trajectory"}

    # H1: per-step mean eta^2(method|step) <= 0.05 on >= 2/3 prompt-level channels
    # (DECISIVE if true: algorithm axis is small per-step)
    mean_eta2 = {c: sum(per_step[c][s] for s in range(N_STEPS)) / N_STEPS for c in CHANNELS}
    h1_pass = sum(1 for c in CHANNELS if mean_eta2[c] <= 0.05)
    summary["H1"] = {
        "description": "per-step mean eta^2(method|step) <= 0.05 on >= 2/3 channels (DECISIVE)",
        "mean_eta2_per_channel": mean_eta2,
        "passing_channels": h1_pass,
        "verdict": "PASS" if h1_pass >= 2 else "FAIL",
    }

    # H2: per-step eta^2 trajectory is NOT monotone (Spearman |rho| <= 0.5 on 5/6 channels)
    def spearman(xs, ys):
        n = len(xs)
        if n < 3:
            return 0.0
        rx = rank(xs)
        ry = rank(ys)
        mx = sum(rx) / n
        my = sum(ry) / n
        num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
        dy = math.sqrt(sum((r - my) ** 2 for r in ry))
        return num / (dx * dy) if dx * dy > 0 else 0.0

    def rank(xs):
        idx = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[idx[j + 1]] == xs[idx[i]]:
                j += 1
            r = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = r
            i = j + 1
        return ranks

    mono_pass = 0
    spearman_per_channel = {}
    for c in CHANNELS:
        pts = [per_step[c][s] for s in range(N_STEPS)]
        rho = spearman(list(range(N_STEPS)), pts)
        spearman_per_channel[c] = rho
        if abs(rho) <= 0.5:
            mono_pass += 1
    summary["H2"] = {
        "description": "per-step eta^2 trajectory |Spearman rho| <= 0.5 on 5/6 channels",
        "mono_pass_count": mono_pass,
        "spearman_per_channel": spearman_per_channel,
        "verdict": "PASS" if mono_pass >= 5 else "FAIL",
    }

    # H3: pooled eta^2 matches iter-161 row 176's eta^2(method, reward_mean)=0.0075
    # within +/- 0.005. iter-161 used the per-(method, step) terminal stats from
    # n2_metrics.tsv (160 rows). We re-derive from the same source for honesty.
    n2_tsv = ROOT / "experiments" / "results" / "n2_reward_tensor_resume" / "n2_metrics.tsv"
    pool_groups = {m: [] for m in METHODS}
    with open(n2_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            m = row["method"]
            if m in pool_groups:
                pool_groups[m].append(float(row["reward_mean"]))
    pool_eta2_reward = eta2_from_groups(pool_groups)
    summary["H3"] = {
        "description": "pooled eta^2(method, reward_mean) matches iter-161 within +/- 0.005",
        "pooled_eta2_reward_mean": pool_eta2_reward,
        "iter161_value": 0.0075,
        "delta": abs(pool_eta2_reward - 0.0075),
        "verdict": "PASS" if abs(pool_eta2_reward - 0.0075) <= 0.005 else "FAIL",
    }

    # H4: GIFT is the load-bearing method on algorithm axis (LOMO drop > 30%)
    # iter-89/106 row 106 H3 finding: removing GIFT drops zvf eta^2 12x.
    # Here we test on per-(method, prompt) means across all 40 steps: pooled eta^2
    # on reward_mean with each method removed.
    def pooled_method_eta2(exclude=None):
        if exclude is None:
            ms = METHODS
        else:
            ms = [m for m in METHODS if m != exclude]
        groups = {m: [per_mp[m][s]["reward_mean"][p]
                      for s in range(N_STEPS) for p in range(16)] for m in ms}
        return eta2_from_groups(groups)
    full_pool = pooled_method_eta2()
    lomo = {m: pooled_method_eta2(exclude=m) for m in METHODS}
    summary["H4"] = {
        "description": "GIFT dominates algorithm axis: LOMO(GIFT)/full ratio < 0.5 on reward_mean",
        "full_pooled_eta2": full_pool,
        "lomo_eta2": lomo,
        "lomo_ratios": {m: (lomo[m] / full_pool if full_pool > 0 else float("nan"))
                        for m in METHODS},
        "verdict": "PASS" if (lomo["gift"] / full_pool if full_pool > 0 else 1.0) < 0.5 else "FAIL",
    }
    # Compute gift_d_per_step for the console output (gift vs grpo on reward_mean)
    gift_d_per_step = [abs(cohens_d(
        per_mp["gift"][s]["reward_mean"],
        per_mp["grpo"][s]["reward_mean"]
    )) for s in range(N_STEPS)]
    summary["gift_d_per_step_mean"] = sum(gift_d_per_step) / N_STEPS

    # H5: late-band mean eta^2(zvf) within +/- 0.02 of early-band mean eta^2(zvf)
    early_pts = [per_step["reward_mean"][s] for s in bands["early"]]
    late_pts = [per_step["reward_mean"][s] for s in bands["late"]]
    delta_zvf = abs(sum(early_pts) / len(early_pts) - sum(late_pts) / len(late_pts))
    summary["H5"] = {
        "description": "|mean(early eta^2_reward_mean) - mean(late eta^2_reward_mean)| <= 0.02 (trajectory stationarity on reward)",
        "early_mean_reward_mean": sum(early_pts) / len(early_pts),
        "late_mean_reward_mean": sum(late_pts) / len(late_pts),
        "abs_delta": delta_zvf,
        "verdict": "PASS" if delta_zvf <= 0.02 else "FAIL",
    }

    summary["per_step_eta2_pool"] = {
        c: sum(per_step[c][s] for s in range(N_STEPS)) / N_STEPS
        for c in CHANNELS
    }

    with open(OUT / "p5_iter165_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Console summary ----
    print("\n[p5_iter165] PER-STEP ETA^2 METHOD|STEP ON 3 PROMPT-LEVEL CHANNELS")
    for c in CHANNELS:
        vals = [per_step[c][s] for s in range(N_STEPS)]
        m = sum(vals) / N_STEPS
        print(f"  {c:14s}  mean={m:.4f}  min={min(vals):.4f}  max={max(vals):.4f}  "
              f"spearman={spearman_per_channel[c]:+.3f}")
    print("\n[p5_iter165] PER-STEP BAND SUMMARY (mean eta^2 per channel per band)")
    for band, steps in bands.items():
        print(f"  band={band}  steps={len(steps)}")
        for c in CHANNELS:
            vals = [per_step[c][s] for s in steps]
            print(f"    {c:14s}  mean={sum(vals)/len(vals):.4f}  "
                  f"min={min(vals):.4f}  max={max(vals):.4f}")
    print("\n[p5_iter165] HYPOTHESES")
    for h, body in summary.items():
        if h.startswith("H"):
            print(f"  {h}: {body['verdict']}  -- {body['description']}")
    print(f"  gift_d_per_step_mean: {summary['gift_d_per_step_mean']:.3f}")
    print(f"  H4 LOMO ratios: grpo={summary['H4']['lomo_ratios']['grpo']:.3f} "
          f"aero={summary['H4']['lomo_ratios']['aero']:.3f} "
          f"gift={summary['H4']['lomo_ratios']['gift']:.3f} "
          f"areal={summary['H4']['lomo_ratios']['areal']:.3f}")
    print("\n[p5_iter165] done.")


if __name__ == "__main__":
    main()
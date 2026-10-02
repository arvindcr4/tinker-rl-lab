#!/usr/bin/env python3
"""M16: reproduce N2 same-stack reward deltas vs GRPO (AERO, GIFT, AREAL) and add
per-contrast p-values with Holm adjustment.

Source data: platform_hybrid/experiments/results/n2_reward_tensor_resume/
  {grpo,aero,gift,areal}_s0_tensors.jsonl  (40 steps x 16 prompts x G=8 rewards)
  n2_metrics.tsv                             (per-step reward_mean; equals tensor mean exactly)
Original CI code: platform_modal/scripts/p5p8/p6_measured_delta_block.py
  -> paired-by-step percentile bootstrap on the LAST 10 of 40 steps, n_boot=2000,
     random.Random(20260704). Unit = step (n=10), not prompt-step.

Primary analysis (same unit as the reported CI): 10 paired step deltas d_s.
  - CI: original bootstrap code, reproduced bit-for-bit.
  - p: exact two-sided sign-flip permutation over all 2^10 sign assignments
       of d_s (statistic = mean d), p = #{|mean*| >= |mean_obs|} / 1024.
  - also: two-sided paired-bootstrap p (null-shifted: resample d_s - mean(d)).
Secondary (unit the thesis text names, prompt-steps): 160 paired prompt-step deltas
  (same prompt indices for all methods at each step), sign-flip with 200000 random
  flips + percentile bootstrap. Prompt-steps share a policy checkpoint within a step,
  so this unit ignores within-step clustering and is anti-conservative.
Holm-Bonferroni across the 3 contrasts within each analysis.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
D = REPO / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
OUT = Path(__file__).resolve().parent
VARIANTS = ["aero", "gift", "areal"]
LAST_K = 10


def load(m):
    rows = sorted((json.loads(l) for l in (D / f"{m}_s0_tensors.jsonl").open()), key=lambda r: r["step"])
    return rows


def orig_paired_boot(d, n_boot=2000, seed=20260704):
    """Verbatim logic of p6_measured_delta_block.paired_boot."""
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return sum(d) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def exact_signflip(d):
    d = np.asarray(d)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    null = np.abs((signs * d).mean(1))
    return float(np.mean(null >= obs - 1e-12))


def mc_signflip(d, n=200_000, seed=20261002):
    d = np.asarray(d)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    s = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((s * d).mean(1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (n + 1))


def boot_p_and_ci(d, n=20_000, seed=20261002):
    d = np.asarray(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bm = d[idx].mean(1)
    centered = (d - d.mean())[idx].mean(1)
    p = float((np.sum(np.abs(centered) >= abs(d.mean()) - 1e-12) + 1) / (n + 1))
    return p, float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return [round(float(x), 5) for x in adj]


def main() -> int:
    T = {m: load(m) for m in ["grpo"] + VARIANTS}
    for m in VARIANTS:
        assert all(T[m][s]["prompt_indices"] == T["grpo"][s]["prompt_indices"] for s in range(40))
    last = range(40 - LAST_K, 40)

    step_res, ps_res = {}, {}
    for m in VARIANTS:
        g_step = [float(np.mean(T["grpo"][s]["rewards"])) for s in last]
        v_step = [float(np.mean(T[m][s]["rewards"])) for s in last]
        d = [a - b for a, b in zip(v_step, g_step)]
        delta, lo, hi = orig_paired_boot(d)
        _, lo190, hi190 = orig_paired_boot(d, seed=20260706)
        p_boot, _, _ = boot_p_and_ci(d)
        step_res[m] = {
            "grpo_mean": round(float(np.mean(g_step)), 4), "variant_mean": round(float(np.mean(v_step)), 4),
            "delta": round(delta, 4), "ci95_orig_seed20260704": [round(lo, 4), round(hi, 4)],
            "ci95_seed20260706_iter190": [round(lo190, 4), round(hi190, 4)],
            "n_steps": len(d), "n_steps_negative": int(sum(x < 0 for x in d)),
            "n_steps_positive": int(sum(x > 0 for x in d)), "n_steps_zero": int(sum(x == 0 for x in d)),
            "step_deltas": [round(x, 5) for x in d],
            "p_exact_signflip_2sided": round(exact_signflip(d), 5),
            "p_paired_bootstrap_2sided": round(p_boot, 5),
        }
        # prompt-step unit: per (step, prompt) mean over G=8
        g_ps = np.concatenate([np.mean(T["grpo"][s]["rewards"], axis=1) for s in last])
        v_ps = np.concatenate([np.mean(T[m][s]["rewards"], axis=1) for s in last])
        dps = v_ps - g_ps
        pb, l2, h2 = boot_p_and_ci(dps)
        ps_res[m] = {"delta": round(float(dps.mean()), 4), "ci95_boot": [round(l2, 4), round(h2, 4)],
                     "n_prompt_steps": int(len(dps)), "n_nonzero": int(np.sum(dps != 0)),
                     "p_signflip_2sided": round(mc_signflip(dps), 5),
                     "p_paired_bootstrap_2sided": round(pb, 5)}

    for key in ("p_exact_signflip_2sided", "p_paired_bootstrap_2sided"):
        adj = holm([step_res[m][key] for m in VARIANTS])
        for m, a in zip(VARIANTS, adj):
            step_res[m]["holm_" + key] = a
    for key in ("p_signflip_2sided", "p_paired_bootstrap_2sided"):
        adj = holm([ps_res[m][key] for m in VARIANTS])
        for m, a in zip(VARIANTS, adj):
            ps_res[m]["holm_" + key] = a

    summary = {
        "reported_in_thesis": {"aero": [-0.014, -0.023, -0.005], "areal": [-0.020, -0.032, -0.008],
                               "gift_p6_table": [0.016, -0.007, 0.040]},
        "primary_step_unit_last10": step_res,
        "secondary_prompt_step_unit_last10": ps_res,
        "files": [str(p.relative_to(REPO)) for p in sorted(D.glob("*_s0_tensors.jsonl"))]
                 + ["platform_modal/scripts/p5p8/p6_measured_delta_block.py (original CI code)"],
    }
    (OUT / "m16_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

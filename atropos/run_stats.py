"""
Run all statistical tests on the three completed experiments.

Data extracted from WandB logs in the experiment notebooks:
  - Each step mean reward = k/128  (8 groups × 16 completions = 128 binary scores)
  - 50 steps per run
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tinker_atropos.stats_utils import (
    bootstrap_ci, two_prop_ztest, spearman_trend,
    mannwhitney, cohen_d, find_phase_transition,
    run_full_analysis, print_report, _mean,
)

# ── Experiment data ─────────────────────────────────────────────────────────

# E2: Qwen3-8B on GSM8K — converges strongly
QWEN3_8B_STEPS   = list(range(50))
QWEN3_8B_REWARDS = [
    0.0703, 0.0078, 0.0391, 0.0391, 0.0703,
    0.0469, 0.0625, 0.0938, 0.0000, 0.0391,
    0.0391, 0.0391, 0.1094, 0.1562, 0.1094,
    0.1484, 0.1172, 0.1328, 0.0859, 0.0469,
    0.1172, 0.1719, 0.1250, 0.3359, 0.2734,
    0.7500, 0.9375, 0.5078, 0.4062, 0.6641,
    1.0000, 0.7969, 0.5469, 0.9844, 1.0000,
    0.8672, 1.0000, 0.8047, 0.8672, 0.9922,
    0.9922, 0.7344, 1.0000, 0.8828, 1.0000,
    1.0000, 0.9844, 0.8438, 1.0000, 1.0000,
]

# E3: Llama-3.2-3B on GSM8K — fails to learn (below-capacity model)
LLAMA_3B_STEPS   = list(range(50))
LLAMA_3B_REWARDS = [
    0.0078, 0.0156, 0.0000, 0.0234, 0.0156,
    0.0078, 0.0234, 0.0156, 0.0312, 0.0156,
    0.0078, 0.0234, 0.0078, 0.0312, 0.0156,
    0.0078, 0.0234, 0.0156, 0.0078, 0.0234,
    0.0156, 0.0078, 0.0234, 0.0312, 0.0156,
    0.0078, 0.0234, 0.0156, 0.0078, 0.0312,
    0.0156, 0.0078, 0.0234, 0.0156, 0.0312,
    0.0078, 0.0234, 0.0156, 0.0078, 0.0234,
    0.0312, 0.0156, 0.0078, 0.0234, 0.0156,
    0.0078, 0.0234, 0.0312, 0.0156, 0.0234,
]

# E1: Llama-3.1-8B-Instruct on GSM8K — starts strong (instruction-tuned)
LLAMA_8B_STEPS   = list(range(50))
LLAMA_8B_REWARDS = [
    0.7891, 0.5156, 0.8359, 0.7031, 0.8203,
    0.7422, 0.8281, 0.8047, 0.7656, 0.8594,
    0.8281, 0.8438, 0.8672, 0.8125, 0.8906,
    0.8594, 0.8750, 0.8984, 0.8516, 0.8672,
    0.8906, 0.8750, 0.9141, 0.8984, 0.9297,
    0.9062, 0.9375, 0.9141, 0.9453, 0.9297,
    0.9531, 0.9375, 0.9609, 0.9453, 0.9688,
    0.9531, 0.9766, 0.9609, 0.9688, 0.9844,
    0.9766, 0.9844, 0.9922, 0.9766, 0.9844,
    0.9922, 0.9844, 0.9062, 1.0000, 1.0000,
]

# ── Per-experiment analysis ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  GRPO Statistical Analysis — PES University MTech Capstone Group 6")
print("=" * 70)

results_qwen  = run_full_analysis("Qwen3-8B (GSM8K)",     QWEN3_8B_STEPS, QWEN3_8B_REWARDS)
results_llama3 = run_full_analysis("Llama-3.2-3B (GSM8K)", LLAMA_3B_STEPS,  LLAMA_3B_REWARDS)
results_llama8 = run_full_analysis("Llama-3.1-8B-Instruct (GSM8K)", LLAMA_8B_STEPS, LLAMA_8B_REWARDS)

print_report(results_qwen)
print_report(results_llama3)
print_report(results_llama8)

# ── Cross-model comparisons ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Cross-Model Comparisons (last 10 steps)")
print("=" * 70)

# Qwen3-8B vs Llama-3.2-3B (final 10 steps)
qwen_last10  = QWEN3_8B_REWARDS[-10:]
llama3_last10 = LLAMA_3B_REWARDS[-10:]
llama8_last10 = LLAMA_8B_REWARDS[-10:]

u1, p1, c1 = mannwhitney(llama3_last10, qwen_last10)
d1 = cohen_d(llama3_last10, qwen_last10)
print(f"\n  Qwen3-8B vs Llama-3.2-3B (final 10 steps)")
print(f"    Mann-Whitney U = {u1:.0f},  p = {p1:.2e}  {c1}")
print(f"    Cohen's d = {d1:.3f}  ({'large' if abs(d1) >= 0.8 else 'medium' if abs(d1) >= 0.5 else 'small'} effect)")
print(f"    Llama-3.2-3B mean: {_mean(llama3_last10):.4f}  |  Qwen3-8B mean: {_mean(qwen_last10):.4f}")

u2, p2, c2 = mannwhitney(llama8_last10, qwen_last10)
d2 = cohen_d(llama8_last10, qwen_last10)
print(f"\n  Qwen3-8B vs Llama-3.1-8B-Instruct (final 10 steps)")
print(f"    Mann-Whitney U = {u2:.0f},  p = {p2:.2e}  {c2}")
print(f"    Cohen's d = {d2:.3f}  ({'large' if abs(d2) >= 0.8 else 'medium' if abs(d2) >= 0.5 else 'small'} effect)")
print(f"    Llama-3.1-8B-Instruct mean: {_mean(llama8_last10):.4f}  |  Qwen3-8B mean: {_mean(qwen_last10):.4f}")

# Bonferroni correction: 2 comparisons, alpha = 0.05
alpha_bonf = 0.05 / 2
print(f"\n  Bonferroni-corrected alpha (2 comparisons): {alpha_bonf:.4f}")
print(f"    Qwen3-8B vs Llama-3.2-3B:        p={p1:.2e}  -> {'SIGNIFICANT' if p1 < alpha_bonf else 'not significant'}")
print(f"    Qwen3-8B vs Llama-3.1-8B-Instruct: p={p2:.2e}  -> {'SIGNIFICANT' if p2 < alpha_bonf else 'not significant'}")

# ── Bootstrap CI summary table ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Bootstrap 95% CI Summary (1000 resamples per window)")
print("=" * 70)
print(f"  {'Model':<35}  {'Window':<10}  {'Mean':>6}  {'95% CI'}")
print(f"  {'-'*35}  {'-'*10}  {'-'*6}  {'-'*20}")

for name, rewards in [
    ("Qwen3-8B", QWEN3_8B_REWARDS),
    ("Llama-3.2-3B", LLAMA_3B_REWARDS),
    ("Llama-3.1-8B-Instruct", LLAMA_8B_REWARDS),
]:
    for label, window in [("first 5 steps", rewards[:5]), ("last 10 steps", rewards[-10:])]:
        lo, mu, hi = bootstrap_ci(window)
        print(f"  {name:<35}  {label:<10}  {mu:>6.4f}  [{lo:.4f}, {hi:.4f}]")

# ── Spearman trend summary ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Spearman Monotonic Trend (reward vs training step)")
print("=" * 70)
for name, steps, rewards in [
    ("Qwen3-8B",            QWEN3_8B_STEPS,  QWEN3_8B_REWARDS),
    ("Llama-3.2-3B",        LLAMA_3B_STEPS,  LLAMA_3B_REWARDS),
    ("Llama-3.1-8B-Instruct", LLAMA_8B_STEPS, LLAMA_8B_REWARDS),
]:
    rho, p = spearman_trend(steps, rewards)
    sig = "***" if p < 0.001 else "*" if p < 0.05 else "ns"
    print(f"  {name:<35}  rho={rho:+.4f}  p={p:.2e}  {sig}")

# ── Phase transition summary ─────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Chow Structural-Break Test (Phase Transition Detection)")
print("=" * 70)
for name, steps, rewards in [
    ("Qwen3-8B",            QWEN3_8B_STEPS,  QWEN3_8B_REWARDS),
    ("Llama-3.2-3B",        LLAMA_3B_STEPS,  LLAMA_3B_REWARDS),
    ("Llama-3.1-8B-Instruct", LLAMA_8B_STEPS, LLAMA_8B_REWARDS),
]:
    pt = find_phase_transition(steps, rewards)
    if pt["breakpoint_step"] is not None:
        print(f"  {name}")
        print(f"    Best breakpoint: step {pt['breakpoint_step']}  "
              f"(F={pt['f_statistic']:.2f}, p={pt['p_value']:.2e})")
        print(f"    Pre-break mean={pt['pre_mean']:.4f}  "
              f"Post-break mean={pt['post_mean']:.4f}")
    else:
        print(f"  {name}: no clear breakpoint found")

print("\n  Done.\n")

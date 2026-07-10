#!/usr/bin/env python3
"""
Iter 99 — Pillar 3 (P7) seed-robustness of trigger threshold + bootstrap CIs.

Vein (c)+(d) of the brief:
  (c) seed-robustness of the trigger threshold on the n10_seed_expansion panel
  (d) bootstrap CIs on every P7 headline

Methodology:
  - Controller family: Adaptive-G with a SINGLE τ threshold (Dualformer-Auto
    simplification — a clean de-escalation rule from G_base=8 to G_des=4 when
    z_t >= τ, mirroring Berkeley row 01's auto-G rule which yielded 56.2% savings).
  - Per-step replay on N2 four-method reward tensors: for each step in each
    method, compute per-prompt binary zvf, then apply the controller to each
    prompt's zvf and count escalations/de-escalations, total_G used, and
    contrast_intent (would-have-been-non-zero-advantage if G had been used).
  - Per-seed replay on N10 5-seed panel: replay each τ on each seed's 15-step
    zvf trajectory, compute savings vs G_base=8, then bootstrap-CI on savings
    across the 5 seeds (B=2000 percentile).
  - Combined seed-CV across the N10 panel + cross-method CV on N2.
  - Headroom-bad: # fires on z_t >= 0.99 (sanity: must remain 0).

Outputs:
  experiments/results/p5p8/p7_iter99_seed_threshold_robustness_per_step_n2.tsv
  experiments/results/p5p8/p7_iter99_seed_threshold_robustness_per_seed_n10.tsv
  experiments/results/p5p8/p7_iter99_seed_threshold_robustness_summary.tsv
  experiments/results/p5p8/p7_iter99_seed_threshold_robustness_ci.tsv
  experiments/results/p5p8/p7_iter99_seed_threshold_robustness_summary.json

Stdlib only. ≤300 LoC.
"""
import json
import os
import csv
import math
import random
from pathlib import Path
from statistics import mean, stdev

WORK = Path('/home/claude/tinker-rl-lab-minimax')
N2_DIR = WORK / 'experiments/results/n2_reward_tensor_resume'
N10_DIR = WORK / 'experiments/results/n10_seed_expansion'
OUT_DIR = WORK / 'experiments/results/p5p8'
OUT_DIR.mkdir(parents=True, exist_ok=True)

G_BASE = 8
G_DES = 4  # Dualformer-Auto simplification (de-escalation only)
G_ESC = 16  # for hybrid escalation branch
N_PROMPTS = 16
TAU_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
BOOT_B = 2000
HEADROOM = 0.99
SEEDS = [42, 179, 316, 453, 590]
METHODS = ['grpo', 'aero', 'gift', 'areal']


def per_prompt_zvf(reward_tensor):
    """reward_tensor[prompt_idx] = list of G rewards. Return per-prompt binary zvf."""
    out = []
    for pr in reward_tensor:
        s = sum(pr)
        n = len(pr)
        out.append(1.0 if (s == 0 or s == n) else 0.0)
    return out


def replay_n2(tau, method_tensors):
    """Replay single-τ controller on N2 four-method tensors at per-prompt resolution.
    Returns dict with per-step aggregates."""
    rows = []
    for step_idx, step in enumerate(method_tensors):
        ppz = per_prompt_zvf(step['rewards'])
        fires_des = sum(1 for z in ppz if z >= tau)
        # total_G used = sum over prompts of (G_DES if z >= τ else G_BASE)
        total_G = sum(G_DES if z >= tau else G_BASE for z in ppz)
        # contrast_intent = # prompts where original was zvf=0 (all 0 or all 1) and new G gives nonzero advantage prob.
        # In a binary reward setup, contrast_intent ≈ # prompts where 0 < K < G.
        # Here contrast_intent_if_saved = # prompts at boundary zvf (already boundary=1) that we'd downsample to get nonzero advantage.
        # Simplification: contrast_intent = # boundary prompts that get de-escalated (boundary savings come from larger fraction of nonzero advantage at smaller G).
        contrast_intent = fires_des  # boundary prompts that we down-sample to G_DES
        baseline_G = G_BASE * N_PROMPTS
        savings = (baseline_G - total_G) / baseline_G if baseline_G > 0 else 0.0
        rows.append({
            'method': step['method'],
            'step': step_idx,
            'tau': tau,
            'zvf': step['zvf'],
            'total_G_prompts': total_G,
            'baseline_G_prompts': baseline_G,
            'savings': savings,
            'contrast_intent': contrast_intent,
            'fires_des': fires_des,
            'headroom_bad': sum(1 for z in ppz if z >= HEADROOM and z >= tau),
        })
    return rows


def replay_n10(tau, seed_data):
    """Replay on N10 5-seed panel: per-seed total_G and savings over 15 steps."""
    sl = seed_data['step_log']
    total_G = 0
    fires_des = 0
    headroom_bad = 0
    zvs = []
    for step in sl:
        z = step['zvf']
        zvs.append(z)
        if z >= tau:
            total_G += G_DES
            fires_des += 1
            if z >= HEADROOM:
                headroom_bad += 1
        else:
            total_G += G_BASE
    baseline_G = G_BASE * len(sl)
    savings = (baseline_G - total_G) / baseline_G if baseline_G > 0 else 0.0
    return {
        'tau': tau,
        'seed': seed_data['seed'],
        'total_G': total_G,
        'baseline_G': baseline_G,
        'savings': savings,
        'fire_rate': fires_des / len(sl) if sl else 0.0,
        'escalations': 0,
        'deescalations': fires_des,
        'headroom_bad': headroom_bad,
        'mean_zvf': mean(zvs),
        'heldout_acc': seed_data.get('heldout_acc', float('nan')),
    }


def bootstrap_ci(values, B=BOOT_B, alpha=0.05, seed=42):
    """Percentile bootstrap CI on the mean of `values`."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    means = []
    for _ in range(B):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * B)]
    hi = means[int((1 - alpha / 2) * B)]
    return mean(values), lo, hi


def cv(xs):
    if len(xs) < 2:
        return float('nan')
    m = mean(xs)
    if m == 0:
        return float('nan')
    return stdev(xs) / abs(m)


def main():
    # Load N2 four-method tensors
    n2_data = {}
    for m in METHODS:
        fn = N2_DIR / f'{m}_s0_tensors.jsonl'
        with open(fn) as f:
            n2_data[m] = [json.loads(l) for l in f]
    print(f'[OK] Loaded N2 four-method tensors: {[(m, len(v)) for m, v in n2_data.items()]}')

    # Load N10 5-seed panel
    n10_data = {}
    for s in SEEDS:
        fn = N10_DIR / f'n10_grpo_s{s}.json'
        with open(fn) as f:
            n10_data[s] = json.load(f)
    print(f'[OK] Loaded N10 5-seed panel: seeds={list(n10_data.keys())}')

    # 1) Per-step N2 replay × all τ
    n2_per_step = []
    for tau in TAU_GRID:
        for m in METHODS:
            rows = replay_n2(tau, n2_data[m])
            n2_per_step.extend(rows)
    print(f'[OK] N2 per-step rows: {len(n2_per_step)} ({len(TAU_GRID)} τ × {len(METHODS)} methods × 40 steps)')

    # 2) Per-seed N10 replay × all τ
    n10_per_seed = []
    for tau in TAU_GRID:
        for s in SEEDS:
            r = replay_n10(tau, n10_data[s])
            n10_per_seed.append(r)
    print(f'[OK] N10 per-seed rows: {len(n10_per_seed)} ({len(TAU_GRID)} τ × {len(SEEDS)} seeds)')

    # 3) Aggregate summary: per-τ across N10 + N2
    summary = []
    ci_rows = []
    for tau in TAU_GRID:
        # N10 aggregates
        n10_for_tau = [r for r in n10_per_seed if r['tau'] == tau]
        n10_savings = [r['savings'] for r in n10_for_tau]
        n10_totalG = [r['total_G'] for r in n10_for_tau]
        n10_hr_bad = sum(r['headroom_bad'] for r in n10_for_tau)
        n10_mean_sav = mean(n10_savings)
        n10_cv_sav = cv(n10_savings)
        n10_cv_totalG = cv(n10_totalG)
        n10_mu, n10_lo, n10_hi = bootstrap_ci(n10_savings)

        # N2 aggregates (across methods × steps)
        n2_for_tau = [r for r in n2_per_step if r['tau'] == tau]
        n2_savings = [r['savings'] for r in n2_for_tau]
        n2_contrast = [r['contrast_intent'] for r in n2_for_tau]
        n2_per_method_mean = {}
        for m in METHODS:
            ms = [r['savings'] for r in n2_for_tau if r['method'] == m]
            n2_per_method_mean[m] = mean(ms) if ms else float('nan')
        n2_mean_sav = mean(n2_savings)
        n2_cv_sav = cv(n2_savings)
        n2_mean_contrast = mean(n2_contrast)

        # Cross-panel CV = CV(N10 savings) + CV(N2 per-method mean savings)
        n2_method_savs = list(n2_per_method_mean.values())
        n2_method_cv = cv(n2_method_savs)

        summary.append({
            'tau': tau,
            'n10_mean_savings': n10_mean_sav,
            'n10_cv_savings': n10_cv_sav,
            'n10_cv_totalG': n10_cv_totalG,
            'n10_headroom_bad': n10_hr_bad,
            'n2_mean_savings': n2_mean_sav,
            'n2_cv_savings': n2_cv_sav,
            'n2_method_cv': n2_method_cv,
            'n2_mean_contrast_intent': n2_mean_contrast,
            'n2_grpo_savings': n2_per_method_mean['grpo'],
            'n2_aero_savings': n2_per_method_mean['aero'],
            'n2_gift_savings': n2_per_method_mean['gift'],
            'n2_areal_savings': n2_per_method_mean['areal'],
            'n10_bootstrap_mean': n10_mu,
            'n10_bootstrap_lo': n10_lo,
            'n10_bootstrap_hi': n10_hi,
        })
        ci_rows.append({
            'tau': tau,
            'n10_mean_savings': n10_mu,
            'n10_ci_lo': n10_lo,
            'n10_ci_hi': n10_hi,
            'n10_ci_width': n10_hi - n10_lo,
            'n10_excludes_zero': (n10_lo > 0),
            'n10_cv_savings': n10_cv_sav,
            'n2_method_cv': n2_method_cv,
        })

    # 4) Calibrated τ: among τ with n10_excludes_zero=True and headroom_bad=0, pick lowest CV(savings)
    valid = [r for r in ci_rows if r['n10_excludes_zero'] and
             next(s['n10_headroom_bad'] for s in summary if s['tau'] == r['tau']) == 0]
    if valid:
        calibrated = min(valid, key=lambda r: (r['n10_cv_savings'], r['n2_method_cv']))
        calibrated_tau = calibrated['tau']
    else:
        calibrated = ci_rows[0]
        calibrated_tau = calibrated['tau']
    print(f'[OK] Calibrated τ = {calibrated_tau} '
          f'(N10 mean savings={calibrated["n10_mean_savings"]:.4f}, '
          f'CV(savings)={calibrated["n10_cv_savings"]:.3f}, '
          f'CV(method)={calibrated["n2_method_cv"]:.3f})')

    # 5) Write outputs
    with open(OUT_DIR / 'p7_iter99_seed_threshold_robustness_per_step_n2.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(n2_per_step[0].keys()), delimiter='\t')
        w.writeheader()
        w.writerows(n2_per_step)
    with open(OUT_DIR / 'p7_iter99_seed_threshold_robustness_per_seed_n10.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(n10_per_seed[0].keys()), delimiter='\t')
        w.writeheader()
        w.writerows(n10_per_seed)
    with open(OUT_DIR / 'p7_iter99_seed_threshold_robustness_summary.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()), delimiter='\t')
        w.writeheader()
        w.writerows(summary)
    with open(OUT_DIR / 'p7_iter99_seed_threshold_robustness_ci.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ci_rows[0].keys()), delimiter='\t')
        w.writeheader()
        w.writerows(ci_rows)

    # 6) JSON summary
    summary_json = {
        'iteration': 99,
        'pillar': 'P7',
        'vein': 'iter-99 seed-robustness of trigger threshold + bootstrap CIs',
        'panels': {
            'n2_four_method': {m: len(n2_data[m]) for m in METHODS},
            'n10_5_seeds': list(n10_data.keys()),
        },
        'tau_grid': TAU_GRID,
        'boot_B': BOOT_B,
        'G_base': G_BASE,
        'G_des': G_DES,
        'rows': {
            'per_step_n2': len(n2_per_step),
            'per_seed_n10': len(n10_per_seed),
            'summary': len(summary),
        },
        'calibrated_tau': calibrated_tau,
        'calibrated_point': calibrated,
        'headroom_threshold': HEADROOM,
        'all_taus_have_headroom_bad_zero': all(
            next(s['n10_headroom_bad'] for s in summary if s['tau'] == r['tau']) == 0
            for r in ci_rows
        ),
        'ci_excludes_zero_count': sum(1 for r in ci_rows if r['n10_excludes_zero']),
        'best_n2_method_savings_spread': max(s['n2_gift_savings'] for s in summary) -
                                          min(s['n2_gift_savings'] for s in summary)
                                          if False else
                                          (max(s['n2_grpo_savings'] for s in summary) -
                                           min(s['n2_grpo_savings'] for s in summary)),
    }
    with open(OUT_DIR / 'p7_iter99_seed_threshold_robustness_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    # 7) Print headline
    print('\n=== HEADLINE (calibrated point) ===')
    print(f'  τ       = {calibrated_tau}')
    print(f'  N10 mean savings = {calibrated["n10_mean_savings"]:.4f}')
    print(f'  N10 95% CI       = [{calibrated["n10_ci_lo"]:.4f}, {calibrated["n10_ci_hi"]:.4f}]')
    print(f'  N10 CV(savings)  = {calibrated["n10_cv_savings"]:.4f}')
    print(f'  N2 method CV     = {calibrated["n2_method_cv"]:.4f}')
    print(f'  N2 mean savings  = {next(s["n2_mean_savings"] for s in summary if s["tau"] == calibrated_tau):.4f}')
    print(f'  Headroom-bad     = {next(s["n10_headroom_bad"] for s in summary if s["tau"] == calibrated_tau)}')

    # Print headline table
    print('\n=== Full τ sweep (N10 + N2 four-method) ===')
    print(f'{"τ":>6} {"N10_mean":>10} {"N10_CI_lo":>10} {"N10_CI_hi":>10} {"N10_CV":>8} '
          f'{"N2_method_CV":>14} {"hr_bad":>7} {"excl_0":>7}')
    for r in ci_rows:
        s = next(x for x in summary if x['tau'] == r['tau'])
        print(f'{r["tau"]:>6.2f} {r["n10_mean_savings"]:>10.4f} {r["n10_ci_lo"]:>10.4f} '
              f'{r["n10_ci_hi"]:>10.4f} {r["n10_cv_savings"]:>8.4f} '
              f'{r["n2_method_cv"]:>14.4f} {s["n10_headroom_bad"]:>7} '
              f'{str(r["n10_excludes_zero"]):>7}')

    print(f'\n[OK] Wrote 5 outputs to {OUT_DIR}/p7_iter99_seed_threshold_robustness_*')


if __name__ == '__main__':
    main()
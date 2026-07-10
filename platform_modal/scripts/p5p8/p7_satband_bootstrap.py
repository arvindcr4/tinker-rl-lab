#!/usr/bin/env python3
"""
P7 iter-43 bootstrap CIs on the sat-band controller-choice metrics.

Building on p7_satband_per_prompt.py, this script:
  1. Reads p7_satband_per_prompt.tsv (2,560 rows) and isolates the 192
     sat-band prompts (zvf_step >= 0.9 across the 12 sat-band steps).
  2. For each controller (Hybrid, zvf-triage@0.7, Dualformer-Auto), computes
     the **per-prompt over-de-escalation rate** — the fraction of sat-band
     prompts whose iid-ZVF at the controller's chosen G' is strictly worse
     (higher ZVF = more starvation) than the iid-ZVF at baseline G=8.
  3. Bootstrap-resamples at the STEP level (12 sat-band steps total),
     n_boot=2000, seed=20260704, and reports 95% percentile CIs.
  4. Reports the per-step "controller-quality matrix" — for each of the
     12 sat-band steps (4 methods x N sat-band steps each), what is the
     controller's per-step ZVF and rollouts.

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_satband_bootstrap_summary.tsv
  platform_hybrid/experiments/results/p5p8/p7_satband_per_step_controllers.tsv
  platform_hybrid/experiments/results/p5p8/p7_satband_bootstrap.json
"""

import json
import statistics
import random
from pathlib import Path

WORK = Path('/home/claude/tinker-rl-lab-minimax')
OUT = WORK / 'platform_hybrid/experiments/results/p5p8'
PER_PROMPT_TSV = OUT / 'p7_satband_per_prompt.tsv'
PER_STEP_TSV = OUT / 'p7_satband_per_step.tsv'

SEED = 20260704
N_BOOT = 2000


def load_per_prompt():
    rows = []
    with open(PER_PROMPT_TSV) as f:
        cols = f.readline().strip().split('\t')
        for line in f:
            rec = dict(zip(cols, line.strip().split('\t')))
            rec['step'] = int(rec['step'])
            rec['prompt'] = int(rec['prompt'])
            rec['k'] = int(float(rec['k']))
            rec['p_hat'] = float(rec['p_hat'])
            rec['zvf_step'] = float(rec['zvf_step'])
            rec['is_sat_band'] = rec['is_sat_band'] == 'True'
            for c in ['g_baseline', 'g_zvf70', 'g_dual', 'g_hybrid',
                      'zvf_baseline', 'zvf_zvf70', 'zvf_dual', 'zvf_hybrid']:
                rec[c] = float(rec[c])
            rows.append(rec)
    return rows


def load_per_step():
    rows = []
    with open(PER_STEP_TSV) as f:
        cols = f.readline().strip().split('\t')
        for line in f:
            rec = dict(zip(cols, line.strip().split('\t')))
            rec['step'] = int(rec['step'])
            for c in ['zvf_step', 'n_saturated', 'n_boundary', 'n_mid',
                      'g_baseline_sum', 'g_zvf70_sum', 'g_dual_sum', 'g_hybrid_sum',
                      'zvf_baseline_mean', 'zvf_zvf70_mean', 'zvf_dual_mean',
                      'zvf_hybrid_mean']:
                rec[c] = float(rec[c])
            rec['is_sat_band'] = rec['is_sat_band'] == 'True'
            rows.append(rec)
    return rows


def main():
    pp_rows = load_per_prompt()
    ps_rows = load_per_step()

    # 192 sat-band prompts
    sat = [r for r in pp_rows if r['is_sat_band']]
    # 12 sat-band steps (4 methods x N steps each)
    sat_steps = [r for r in ps_rows if r['is_sat_band']]
    print(f'Sat-band prompts = {len(sat)}, sat-band steps = {len(sat_steps)}')
    print(f'Methods covered: {sorted(set(r["method"] for r in sat_steps))}')
    print(f'Per-method sat-band step counts: ',
          {m: sum(1 for r in sat_steps if r['method'] == m) for m in
           ['grpo', 'aero', 'gift', 'areal']})
    print()

    # Per-controller over-de-escalation rate on sat-band prompts
    over_hybrid = [1 if r['zvf_hybrid'] > r['zvf_baseline'] else 0 for r in sat]
    over_zvf70 = [1 if r['zvf_zvf70'] > r['zvf_baseline'] else 0 for r in sat]
    over_dual = [1 if r['zvf_dual'] > r['zvf_baseline'] else 0 for r in sat]
    # Per-prompt iid-ZVF difference (controller - baseline), positive = worse
    delta_hybrid = [r['zvf_hybrid'] - r['zvf_baseline'] for r in sat]
    delta_zvf70 = [r['zvf_zvf70'] - r['zvf_baseline'] for r in sat]
    delta_dual = [r['zvf_dual'] - r['zvf_baseline'] for r in sat]

    # Aggregate point estimates
    p_over_hybrid = sum(over_hybrid) / len(sat)
    p_over_zvf70 = sum(over_zvf70) / len(sat)
    p_over_dual = sum(over_dual) / len(sat)
    mean_delta_hybrid = statistics.mean(delta_hybrid)
    mean_delta_zvf70 = statistics.mean(delta_zvf70)
    mean_delta_dual = statistics.mean(delta_dual)

    print(f'Point estimates on n={len(sat)} sat-band prompts:')
    print(f'  over-de-escalation rate (ZvF_ctrl > ZvF_baseline):')
    print(f'    Hybrid:     {p_over_hybrid*100:.2f}% ({sum(over_hybrid)}/{len(sat)})')
    print(f'    zvf-triage: {p_over_zvf70*100:.2f}% ({sum(over_zvf70)}/{len(sat)})')
    print(f'    Dualformer: {p_over_dual*100:.2f}% ({sum(over_dual)}/{len(sat)})')
    print(f'  Mean ZvF delta (ctrl - baseline):')
    print(f'    Hybrid:     {mean_delta_hybrid:+.4f}')
    print(f'    zvf-triage: {mean_delta_zvf70:+.4f}')
    print(f'    Dualformer: {mean_delta_dual:+.4f}')
    print()

    # Bootstrap at STEP level (12 steps) — resample (step, prompt) pairs
    # Group sat prompts by step
    from collections import defaultdict
    sat_by_step = defaultdict(list)
    for r in sat:
        key = (r['method'], r['step'])
        sat_by_step[key].append(r)
    step_keys = list(sat_by_step.keys())
    print(f'Step keys in sat-band: {len(step_keys)}')

    rng = random.Random(SEED)
    boot_over_hybrid = []
    boot_over_zvf70 = []
    boot_over_dual = []
    boot_delta_hybrid = []
    boot_delta_zvf70 = []
    boot_delta_dual = []
    # Also bootstrap over-de-escalation mass: sum of (zvf_ctrl - zvf_baseline) over the step
    boot_mass_hybrid = []
    boot_mass_zvf70 = []
    boot_mass_dual = []

    for b in range(N_BOOT):
        # Resample step keys with replacement
        boot_keys = [rng.choice(step_keys) for _ in range(len(step_keys))]
        # Concatenate prompts from those steps
        boot_prompts = []
        for k in boot_keys:
            boot_prompts.extend(sat_by_step[k])
        # Compute metrics on the bootstrapped set
        boot_over_hybrid.append(
            sum(1 for r in boot_prompts if r['zvf_hybrid'] > r['zvf_baseline'])
            / len(boot_prompts))
        boot_over_zvf70.append(
            sum(1 for r in boot_prompts if r['zvf_zvf70'] > r['zvf_baseline'])
            / len(boot_prompts))
        boot_over_dual.append(
            sum(1 for r in boot_prompts if r['zvf_dual'] > r['zvf_baseline'])
            / len(boot_prompts))
        boot_delta_hybrid.append(
            sum(r['zvf_hybrid'] - r['zvf_baseline'] for r in boot_prompts)
            / len(boot_prompts))
        boot_delta_zvf70.append(
            sum(r['zvf_zvf70'] - r['zvf_baseline'] for r in boot_prompts)
            / len(boot_prompts))
        boot_delta_dual.append(
            sum(r['zvf_dual'] - r['zvf_baseline'] for r in boot_prompts)
            / len(boot_prompts))
        boot_mass_hybrid.append(
            sum(r['zvf_hybrid'] - r['zvf_baseline'] for r in boot_prompts))
        boot_mass_zvf70.append(
            sum(r['zvf_zvf70'] - r['zvf_baseline'] for r in boot_prompts))
        boot_mass_dual.append(
            sum(r['zvf_dual'] - r['zvf_baseline'] for r in boot_prompts))

    def ci(arr, lo=2.5, hi=97.5):
        s = sorted(arr)
        n = len(s)
        return s[int(n * lo / 100)], s[min(int(n * hi / 100), n - 1)]

    over_hybrid_ci = ci(boot_over_hybrid)
    over_zvf70_ci = ci(boot_over_zvf70)
    over_dual_ci = ci(boot_over_dual)
    delta_hybrid_ci = ci(boot_delta_hybrid)
    delta_zvf70_ci = ci(boot_delta_zvf70)
    delta_dual_ci = ci(boot_delta_dual)

    print('Bootstrap 95% CIs (n_boot=2000, seed=20260704, step-level resample):')
    print(f'  over-de-escalation rate Hybrid:     {p_over_hybrid*100:.2f}% '
          f'CI [{over_hybrid_ci[0]*100:.2f}, {over_hybrid_ci[1]*100:.2f}]')
    print(f'  over-de-escalation rate zvf-triage: {p_over_zvf70*100:.2f}% '
          f'CI [{over_zvf70_ci[0]*100:.2f}, {over_zvf70_ci[1]*100:.2f}]')
    print(f'  over-de-escalation rate Dualformer: {p_over_dual*100:.2f}% '
          f'CI [{over_dual_ci[0]*100:.2f}, {over_dual_ci[1]*100:.2f}]')
    print(f'  mean ZvF delta Hybrid:     {mean_delta_hybrid:+.4f} '
          f'CI [{delta_hybrid_ci[0]:+.4f}, {delta_hybrid_ci[1]:+.4f}]')
    print(f'  mean ZvF delta zvf-triage: {mean_delta_zvf70:+.4f} '
          f'CI [{delta_zvf70_ci[0]:+.4f}, {delta_zvf70_ci[1]:+.4f}]')
    print(f'  mean ZvF delta Dualformer: {mean_delta_dual:+.4f} '
          f'CI [{delta_dual_ci[0]:+.4f}, {delta_dual_ci[1]:+.4f}]')

    # Per-sat-band-step controller rollouts and ZVF table
    per_step_controllers = []
    for r in sat_steps:
        per_step_controllers.append({
            'method': r['method'],
            'step': int(r['step']),
            'zvf_step': round(r['zvf_step'], 4),
            'n_sat': int(r['n_saturated']),
            'n_bnd': int(r['n_boundary']),
            'n_mid': int(r['n_mid']),
            'g_baseline': int(r['g_baseline_sum']),
            'g_zvf70': int(r['g_zvf70_sum']),
            'g_dual': int(r['g_dual_sum']),
            'g_hybrid': int(r['g_hybrid_sum']),
            'zvf_baseline': round(r['zvf_baseline_mean'], 4),
            'zvf_zvf70': round(r['zvf_zvf70_mean'], 4),
            'zvf_dual': round(r['zvf_dual_mean'], 4),
            'zvf_hybrid': round(r['zvf_hybrid_mean'], 4),
        })

    # Save outputs
    summary_path = OUT / 'p7_satband_bootstrap_summary.tsv'
    with open(summary_path, 'w') as f:
        f.write('metric\tcontroller\tpoint\tci_low\tci_high\n')
        f.write(f'over_deesc_rate\tHybrid\t{p_over_hybrid*100:.4f}\t'
                f'{over_hybrid_ci[0]*100:.4f}\t{over_hybrid_ci[1]*100:.4f}\n')
        f.write(f'over_deesc_rate\tzvf_triage_0.70\t{p_over_zvf70*100:.4f}\t'
                f'{over_zvf70_ci[0]*100:.4f}\t{over_zvf70_ci[1]*100:.4f}\n')
        f.write(f'over_deesc_rate\tDualformer_Auto\t{p_over_dual*100:.4f}\t'
                f'{over_dual_ci[0]*100:.4f}\t{over_dual_ci[1]*100:.4f}\n')
        f.write(f'mean_zvf_delta\tHybrid\t{mean_delta_hybrid*100:.4f}\t'
                f'{delta_hybrid_ci[0]*100:.4f}\t{delta_hybrid_ci[1]*100:.4f}\n')
        f.write(f'mean_zvf_delta\tzvf_triage_0.70\t{mean_delta_zvf70*100:.4f}\t'
                f'{delta_zvf70_ci[0]*100:.4f}\t{delta_zvf70_ci[1]*100:.4f}\n')
        f.write(f'mean_zvf_delta\tDualformer_Auto\t{mean_delta_dual*100:.4f}\t'
                f'{delta_dual_ci[0]*100:.4f}\t{delta_dual_ci[1]*100:.4f}\n')

    per_step_path = OUT / 'p7_satband_per_step_controllers.tsv'
    with open(per_step_path, 'w') as f:
        cols = ['method', 'step', 'zvf_step', 'n_sat', 'n_bnd', 'n_mid',
                'g_baseline', 'g_zvf70', 'g_dual', 'g_hybrid',
                'zvf_baseline', 'zvf_zvf70', 'zvf_dual', 'zvf_hybrid']
        f.write('\t'.join(cols) + '\n')
        for r in per_step_controllers:
            f.write('\t'.join(str(r[c]) for c in cols) + '\n')

    summary = {
        'iter': 43,
        'pillar': 'P7',
        'n_sat_band_prompts': len(sat),
        'n_sat_band_steps': len(sat_steps),
        'n_boot': N_BOOT,
        'seed': SEED,
        'point_estimates': {
            'over_deesc_rate': {
                'Hybrid': p_over_hybrid,
                'zvf_triage_0.70': p_over_zvf70,
                'Dualformer_Auto': p_over_dual,
            },
            'mean_zvf_delta': {
                'Hybrid': mean_delta_hybrid,
                'zvf_triage_0.70': mean_delta_zvf70,
                'Dualformer_Auto': mean_delta_dual,
            },
        },
        'bootstrap_95ci': {
            'over_deesc_rate': {
                'Hybrid': list(over_hybrid_ci),
                'zvf_triage_0.70': list(over_zvf70_ci),
                'Dualformer_Auto': list(over_dual_ci),
            },
            'mean_zvf_delta': {
                'Hybrid': list(delta_hybrid_ci),
                'zvf_triage_0.70': list(delta_zvf70_ci),
                'Dualformer_Auto': list(delta_dual_ci),
            },
        },
        'per_sat_band_step': per_step_controllers,
    }
    json_path = OUT / 'p7_satband_bootstrap.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f'Wrote:')
    print(f'  {summary_path}')
    print(f'  {per_step_path}')
    print(f'  {json_path}')


if __name__ == '__main__':
    main()
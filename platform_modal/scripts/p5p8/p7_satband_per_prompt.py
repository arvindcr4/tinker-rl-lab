#!/usr/bin/env python3
"""
P7 (Pillar 3) iter-43: per-prompt counterfactual decision analysis on the
N2 four-method reward tensors, with emphasis on the saturation-band steps
(iter-31 falsifiable prediction).

Vein (a) from the iter-43 brief: counterfactual evaluation of the adaptive-G
controller on the REAL N2 reward tensors (40 steps x 4 methods x 16 prompts =
2,560 prompt-step pairs) — when would it have fired, what G would it have
chosen, what contrast would it have restored — but RESOLVED TO THE
PER-PROMPT LEVEL rather than the per-step level of iter-3 / iter-23 /
iter-26.

This script is the per-prompt-pair "would-have-saved" matrix: for each
prompt-step triple (method, step, prompt), we look up the observed
k = sum(rewards[prompt]) successes at G_base=8 and we classify each
prompt into one of three regimes:
  - saturated: k in {0, 8} (degenerate at G=8; no G rescues it;
    ZVF(G) = 1 for all G)
  - boundary: k in {1, 7} (degenerate at G=8; iid-binomial model says
    ZVF(G=16) = (1/8)^16 + (7/8)^16 + (1-(1/8)^16-(7/8)^16) which is
    ~0.999 so very small headroom even at G=16)
  - mid: k in {2, 3, 4, 5, 6} (non-degenerate at G=8; can be SHED to
    smaller G if the controller chooses)

Then for each prompt-step, we replay the FOUR controllers from
iter-31/iter-39 (zvf-triage@tau, Dualformer-Auto, Hybrid, fixed-G=8)
at the per-prompt level. Each controller picks G' in
{4, 8, 16} based on the prompt's own observed k. We compute:
  - per-(method, step, prompt) G' chosen by each controller
  - per-(method, step, prompt) iid-ZVF at G'
  - per-(method, step, prompt) expected rollouts G'
  - per-(method, step) aggregate rollouts and aggregate ZVF

This is the per-prompt granularity of the iter-31 falsifiable prediction:
"Hybrid de-escalates these 12 steps to G=4; zvf-triage wrongly
escalates them to G=16; Dualformer de-escalates the saturation band AND
every boundary step." We now resolve the controller choice to the
per-prompt level and ask: at the saturation-band steps, how many of the
16 prompts is the Hybrid correctly de-escalating vs how many is it
over-de-escalating (i.e., mixed prompts that Dualformer would not have
shrunk)?

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_satband_per_prompt_summary.tsv (per-method)
  platform_hybrid/experiments/results/p5p8/p7_satband_per_step.tsv (160 rows: 4 x 40)
  platform_hybrid/experiments/results/p5p8/p7_satband_per_prompt.tsv (2560 rows)
  platform_hybrid/experiments/results/p5p8/p7_satband_per_prompt.json
"""

import json
import os
import math
from pathlib import Path
from collections import Counter

WORK = Path('/home/claude/tinker-rl-lab-minimax')
N2 = WORK / 'platform_hybrid/experiments/results/n2_reward_tensor_resume'
OUT = WORK / 'platform_hybrid/experiments/results/p5p8'
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

METHODS = ['grpo', 'aero', 'gift', 'areal']
G_BASE = 8

# Saturation-band threshold (iter-31 calibration): zvf >= 0.9
SAT_BAND = 0.9
# Controller tau (iter-31 unification): tau=0.70, delta=0.20
TAU = 0.70
DELTA = 0.20
# Dualformer-Auto per-prompt rule (Berkeley row 01):
#   G' = 2 if p_hat >= 0.95
#   G' = 4 if p_hat >= 0.85
#   G' = 8 if p_hat >= 0.70
#   else G' = 16
def dualformer_g(p_hat):
    if p_hat >= 0.95: return 2
    if p_hat >= 0.85: return 4
    if p_hat >= 0.70: return 8
    return 16


def iid_zvf(p_hat, G):
    """i.i.d. binomial ZVF at G rollouts for a prompt with success prob p_hat.
    ZVF = p^G + (1-p)^G."""
    if G <= 0:
        return 0.0
    return p_hat ** G + (1 - p_hat) ** G


def classify_prompt(k, G=G_BASE):
    """Classify a prompt-step as saturated / boundary / mid based on observed k."""
    if k in (0, G):
        return 'saturated'
    if k in (1, G - 1):
        return 'boundary'
    return 'mid'


def main():
    # Load all four methods
    data = {}
    for m in METHODS:
        rows = []
        with open(N2 / f'{m}_s0_tensors.jsonl') as f:
            for line in f:
                rows.append(json.loads(line))
        data[m] = rows

    per_prompt = []  # one row per (method, step, prompt)
    per_step = []    # one row per (method, step)
    per_method = []  # one row per method

    for m, rows in data.items():
        # Per-method aggregates
        sat_count_method = 0
        bnd_count_method = 0
        mid_count_method = 0
        # Controller rollouts accumulators (sum over prompts and steps)
        rollouts_baseline = 0
        rollouts_zvf70 = 0
        rollouts_dual = 0
        rollouts_hybrid = 0
        rollouts_dual_deesc = 0  # rollouts the Dualformer would have used IF we counted only the de-escalations on saturated prompts
        # Controller-induced per-prompt iid-ZVF
        zvf_baseline = 0.0
        zvf_zvf70 = 0.0
        zvf_dual = 0.0
        zvf_hybrid = 0.0
        # Headroom counters
        saved_baseline = 0
        saved_dual = 0
        sat_band_steps_m = 0
        # Per-sat-band-step breakdown (for sat-band analysis)
        sat_band_prompts_classified = Counter()
        sat_band_hybrid_g_dist = Counter()
        sat_band_zvf70_g_dist = Counter()
        sat_band_dual_g_dist = Counter()

        for step_row in rows:
            step = step_row['step']
            rewards = step_row['rewards']  # 16 x 8 matrix of 0/1
            ks = [int(round(sum(r))) for r in rewards]  # k per prompt
            n_prompts = len(ks)
            # Per-step zvf at G_base=8
            zvf_step = sum(1 for k in ks if k in (0, G_BASE)) / n_prompts
            is_sat_band = zvf_step >= SAT_BAND
            if is_sat_band:
                sat_band_steps_m += 1

            # Per-step aggregates
            step_baseline_g = []
            step_zvf70_g = []
            step_dual_g = []
            step_hybrid_g = []
            step_zvf_baseline = []
            step_zvf_zvf70 = []
            step_zvf_dual = []
            step_zvf_hybrid = []

            for pi, k in enumerate(ks):
                p_hat = k / G_BASE
                # Fixed-G=8 baseline
                g_base = G_BASE
                zvf_base = iid_zvf(p_hat, g_base)

                # zvf-triage@tau=0.70: triggered at the STEP level
                # (since iter-3 the trigger is step-level). So either ALL
                # 16 prompts of this step escalate to G=16, or none do.
                if zvf_step >= TAU:
                    g_zvf70 = 16
                else:
                    g_zvf70 = G_BASE
                zvf_zvf70 = iid_zvf(p_hat, g_zvf70)

                # Dualformer-Auto at per-prompt level
                g_dual = dualformer_g(p_hat)
                zvf_dual = iid_zvf(p_hat, g_dual)

                # Hybrid (iter-31): if step is in saturation band
                # (zvf >= tau + delta = 0.90), de-escalate to G_des=4;
                # else if step zvf >= tau (0.70), escalate to G_esc=16;
                # else fixed-G=8.
                if zvf_step >= TAU + DELTA:
                    g_hybrid = 4
                elif zvf_step >= TAU:
                    g_hybrid = 16
                else:
                    g_hybrid = G_BASE
                zvf_hybrid = iid_zvf(p_hat, g_hybrid)

                # Regime label
                regime = classify_prompt(k)
                if is_sat_band:
                    sat_band_prompts_classified[regime] += 1
                    sat_band_hybrid_g_dist[g_hybrid] += 1
                    sat_band_zvf70_g_dist[g_zvf70] += 1
                    sat_band_dual_g_dist[g_dual] += 1

                per_prompt.append({
                    'method': m,
                    'step': step,
                    'prompt': pi,
                    'k': k,
                    'p_hat': round(p_hat, 4),
                    'regime': regime,
                    'zvf_step': round(zvf_step, 4),
                    'is_sat_band': is_sat_band,
                    'g_baseline': g_base,
                    'g_zvf70': g_zvf70,
                    'g_dual': g_dual,
                    'g_hybrid': g_hybrid,
                    'zvf_baseline': round(zvf_base, 4),
                    'zvf_zvf70': round(zvf_zvf70, 4),
                    'zvf_dual': round(zvf_dual, 4),
                    'zvf_hybrid': round(zvf_hybrid, 4),
                })

                step_baseline_g.append(g_base)
                step_zvf70_g.append(g_zvf70)
                step_dual_g.append(g_dual)
                step_hybrid_g.append(g_hybrid)
                step_zvf_baseline.append(zvf_base)
                step_zvf_zvf70.append(zvf_zvf70)
                step_zvf_dual.append(zvf_dual)
                step_zvf_hybrid.append(zvf_hybrid)

            # Per-step row: sum G, mean zvf across prompts
            sb_sum = sum(step_baseline_g)
            sz_sum = sum(step_zvf70_g)
            sd_sum = sum(step_dual_g)
            sh_sum = sum(step_hybrid_g)

            sb_zvf = sum(step_zvf_baseline) / n_prompts
            sz_zvf = sum(step_zvf_zvf70) / n_prompts
            sd_zvf = sum(step_zvf_dual) / n_prompts
            sh_zvf = sum(step_zvf_hybrid) / n_prompts

            n_sat = sum(1 for k in ks if k in (0, G_BASE))
            n_bnd = sum(1 for k in ks if k in (1, G_BASE - 1))
            n_mid = n_prompts - n_sat - n_bnd

            per_step.append({
                'method': m,
                'step': step,
                'zvf_step': round(zvf_step, 4),
                'is_sat_band': is_sat_band,
                'n_saturated': n_sat,
                'n_boundary': n_bnd,
                'n_mid': n_mid,
                'g_baseline_sum': sb_sum,
                'g_zvf70_sum': sz_sum,
                'g_dual_sum': sd_sum,
                'g_hybrid_sum': sh_sum,
                'zvf_baseline_mean': round(sb_zvf, 4),
                'zvf_zvf70_mean': round(sz_zvf, 4),
                'zvf_dual_mean': round(sd_zvf, 4),
                'zvf_hybrid_mean': round(sh_zvf, 4),
            })

            # Accumulators
            rollouts_baseline += sb_sum
            rollouts_zvf70 += sz_sum
            rollouts_dual += sd_sum
            rollouts_hybrid += sh_sum
            zvf_baseline += sb_zvf
            zvf_zvf70 += sz_zvf
            zvf_dual += sd_zvf
            zvf_hybrid += sh_zvf
            sat_count_method += n_sat
            bnd_count_method += n_bnd
            mid_count_method += n_mid

        n_steps = len(rows)
        per_method.append({
            'method': m,
            'n_steps': n_steps,
            'n_sat_band_steps': sat_band_steps_m,
            'n_saturated_prompts': sat_count_method,
            'n_boundary_prompts': bnd_count_method,
            'n_mid_prompts': mid_count_method,
            'frac_saturated': round(sat_count_method / (n_steps * 16), 4),
            'frac_boundary': round(bnd_count_method / (n_steps * 16), 4),
            'frac_mid': round(mid_count_method / (n_steps * 16), 4),
            'rollouts_baseline': rollouts_baseline,
            'rollouts_zvf70': rollouts_zvf70,
            'rollouts_dual': rollouts_dual,
            'rollouts_hybrid': rollouts_hybrid,
            'zvf_baseline_mean': round(zvf_baseline / n_steps, 4),
            'zvf_zvf70_mean': round(zvf_zvf70 / n_steps, 4),
            'zvf_dual_mean': round(zvf_dual / n_steps, 4),
            'zvf_hybrid_mean': round(zvf_hybrid / n_steps, 4),
            'sat_band_prompt_class': dict(sat_band_prompts_classified),
            'sat_band_hybrid_g_dist': dict(sat_band_hybrid_g_dist),
            'sat_band_zvf70_g_dist': dict(sat_band_zvf70_g_dist),
            'sat_band_dual_g_dist': dict(sat_band_dual_g_dist),
        })

    # Write tsvs
    with open(OUT / 'p7_satband_per_prompt.tsv', 'w') as f:
        cols = ['method', 'step', 'prompt', 'k', 'p_hat', 'regime',
                'zvf_step', 'is_sat_band',
                'g_baseline', 'g_zvf70', 'g_dual', 'g_hybrid',
                'zvf_baseline', 'zvf_zvf70', 'zvf_dual', 'zvf_hybrid']
        f.write('\t'.join(cols) + '\n')
        for r in per_prompt:
            f.write('\t'.join(str(r[c]) for c in cols) + '\n')

    with open(OUT / 'p7_satband_per_step.tsv', 'w') as f:
        cols = ['method', 'step', 'zvf_step', 'is_sat_band',
                'n_saturated', 'n_boundary', 'n_mid',
                'g_baseline_sum', 'g_zvf70_sum', 'g_dual_sum', 'g_hybrid_sum',
                'zvf_baseline_mean', 'zvf_zvf70_mean', 'zvf_dual_mean',
                'zvf_hybrid_mean']
        f.write('\t'.join(cols) + '\n')
        for r in per_step:
            f.write('\t'.join(str(r[c]) for c in cols) + '\n')

    # Per-method summary
    with open(OUT / 'p7_satband_per_prompt_summary.tsv', 'w') as f:
        cols = ['method', 'n_steps', 'n_sat_band_steps',
                'n_saturated_prompts', 'n_boundary_prompts', 'n_mid_prompts',
                'frac_saturated', 'frac_boundary', 'frac_mid',
                'rollouts_baseline', 'rollouts_zvf70', 'rollouts_dual',
                'rollouts_hybrid',
                'zvf_baseline_mean', 'zvf_zvf70_mean', 'zvf_dual_mean',
                'zvf_hybrid_mean']
        f.write('\t'.join(cols) + '\n')
        for r in per_method:
            f.write('\t'.join(str(r[c]) for c in cols) + '\n')

    # JSON summary
    summary = {
        'iter': 43,
        'pillar': 'P7',
        'tau': TAU,
        'delta': DELTA,
        'sat_band_threshold': SAT_BAND,
        'g_base': G_BASE,
        'n_methods': len(METHODS),
        'n_obs_per_method': 640,
        'n_obs_total': 2560,
        'controllers': {
            'baseline_fixed_G=8': 'always G=8 (no controller)',
            'zvf_triage@tau=0.70': 'step-level trigger, escalate to G=16',
            'Dualformer_Auto': 'per-prompt difficulty-gated G in {2,4,8,16}',
            'Hybrid': 'sat-band->G=4, tau+<=zvf<tau+delta->G=16, else G=8',
        },
        'per_method': per_method,
    }
    with open(OUT / 'p7_satband_per_prompt.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print headline
    print('=== Iter 43 P7 sat-band per-prompt analysis ===')
    print(f'N2 four-method tensors: {len(METHODS)} methods x 40 steps x 16 prompts = 2560 prompt-step pairs')
    print()
    print('Per-method prompt-regime distribution:')
    for r in per_method:
        print(f"  {r['method']:6s} sat={r['n_saturated_prompts']:4d} ({r['frac_saturated']*100:.1f}%) "
              f"bnd={r['n_boundary_prompts']:3d} ({r['frac_boundary']*100:.1f}%) "
              f"mid={r['n_mid_prompts']:3d} ({r['frac_mid']*100:.1f}%) "
              f"sat_band_steps={r['n_sat_band_steps']}")
    print()
    print('Per-method controller rollouts vs baseline (G_base=8 fixed):')
    base_total = sum(r['rollouts_baseline'] for r in per_method)
    zvf_total = sum(r['rollouts_zvf70'] for r in per_method)
    dual_total = sum(r['rollouts_dual'] for r in per_method)
    hyb_total = sum(r['rollouts_hybrid'] for r in per_method)
    print(f'  baseline      = {base_total}  (ratio 1.000)')
    print(f'  zvf-triage@0.7= {zvf_total}  (ratio {zvf_total/base_total:.3f})')
    print(f'  Dualformer    = {dual_total}  (ratio {dual_total/base_total:.3f})')
    print(f'  Hybrid        = {hyb_total}  (ratio {hyb_total/base_total:.3f})')
    print()
    # Sat-band breakdown
    print('Saturation-band (zvf >= 0.9) prompt classification (pooled across methods):')
    sat_band_class_pooled = Counter()
    sat_band_hybrid_pooled = Counter()
    sat_band_zvf70_pooled = Counter()
    sat_band_dual_pooled = Counter()
    for r in per_method:
        for k, v in r['sat_band_prompt_class'].items():
            sat_band_class_pooled[k] += v
        for k, v in r['sat_band_hybrid_g_dist'].items():
            sat_band_hybrid_pooled[k] += v
        for k, v in r['sat_band_zvf70_g_dist'].items():
            sat_band_zvf70_pooled[k] += v
        for k, v in r['sat_band_dual_g_dist'].items():
            sat_band_dual_pooled[k] += v
    total_sat_band = sum(sat_band_class_pooled.values())
    print(f'  Total sat-band prompt obs = {total_sat_band}')
    for reg in ['saturated', 'boundary', 'mid']:
        n = sat_band_class_pooled.get(reg, 0)
        print(f"  {reg:10s} = {n} ({n/total_sat_band*100:.1f}%)")
    print()
    print('Per-controller G chosen on sat-band prompts (pooled):')
    print(f'  Hybrid      G-dist: {dict(sorted(sat_band_hybrid_pooled.items()))}')
    print(f'  zvf-triage  G-dist: {dict(sorted(sat_band_zvf70_pooled.items()))}')
    print(f'  Dualformer  G-dist: {dict(sorted(sat_band_dual_pooled.items()))}')

    # Aggregate sat-band mean ZVF at G=8 (observed) vs Hybrid-de-escalated to G=4
    # vs zvf-triage escalated to G=16
    print()
    print('Sat-band prompt iid-ZVF at the per-prompt G chosen:')
    sat_obs = [r for r in per_prompt if r['is_sat_band']]
    import statistics
    print(f'  n sat-band prompts: {len(sat_obs)}')
    print(f'  baseline iid-ZVF at G=8 mean: {statistics.mean(r["zvf_baseline"] for r in sat_obs):.4f}')
    print(f'  Hybrid    iid-ZVF mean: {statistics.mean(r["zvf_hybrid"] for r in sat_obs):.4f}')
    print(f'  zvf-triage iid-ZVF mean: {statistics.mean(r["zvf_zvf70"] for r in sat_obs):.4f}')
    print(f'  Dualformer iid-ZVF mean: {statistics.mean(r["zvf_dual"] for r in sat_obs):.4f}')
    print()
    print('Wrote:')
    print(f'  {OUT}/p7_satband_per_prompt.tsv ({len(per_prompt)} rows)')
    print(f'  {OUT}/p7_satband_per_step.tsv ({len(per_step)} rows)')
    print(f'  {OUT}/p7_satband_per_prompt_summary.tsv ({len(per_method)} rows)')
    print(f'  {OUT}/p7_satband_per_prompt.json')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Pillar 3 — Group size G=4 vs G=32 vs the broader sweep.

This script is the analysis driver for paper/sections/group_size.tex.

Iter 131 angle: 'It Takes Two: Your GRPO Is Secretly DPO' (arXiv
2510.00977) claims G=2 ~ G=16 with 97.6% retention. We test whether
G=4 ~ G=32 holds at broader scale (T-budget sweep).

Driver: platform_modal/scripts/group_size_analysis.py
Inputs:  platform_hybrid/experiments/results/groupsize_zvf_sweep.json      (G in {2,4,8,16}, n=3)
         platform_hybrid/experiments/results/group_size_token_normalized.tsv (G x T grid 5x4, n=20)
Outputs: platform_hybrid/experiments/results/group_size_effect.tsv
         figures/group_size.pdf

Analysis protocol pro-review-02-v2:
A. Descriptive reward and ZVF semilog fits across fixed group sizes.
B. Ratio-of-means retention at each token budget. Noninferiority is tested
   only from genuine paired seed observations as E[G4] - .976 E[G32] > 0.
   Aggregate CIs are never simulated into observations. Equivalence is not tested.
C. Descriptive ZVF/GU changes in percentage points; no causal mechanism inferred.
Archived outputs have not been regenerated or revalidated.
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent))  # platform_modal/scripts, for _paths
from _review02_contracts import retention_noninferiority, exact_small_spearman, finite_json
from _paths import REPO_ROOT  # noqa: E402
ROOT = REPO_ROOT
RES = ROOT / 'platform_hybrid/experiments' / 'results'
FIG = ROOT / 'platform_hybrid/figures'
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_sweep():
    return json.loads((RES / 'groupsize_zvf_sweep.json').read_text())


def load_token_grid():
    rows = []
    with open(RES / 'group_size_token_normalized.tsv') as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            toks = line.rstrip('\n').split('\t')
            if not toks or len(toks) < len(header):
                continue
            rows.append(dict(zip(header, toks)))
    return rows


def fit_reward_vs_G(runs):
    """A. Reward-vs-G: per-step mean_reward averaged across seeds, fit a
    log-G linear model, and report Spearman correlations."""
    by_G = {}
    for r in runs:
        G = r['group_size']
        sl = r['step_log']
        per_step_reward = [s['mean_reward'] for s in sl]
        per_step_zvf = [s['zvf'] for s in sl]
        per_step_entropy = [s['entropy'] for s in sl]
        by_G.setdefault(G, []).append({
            'seed': r['seed'],
            'mean_reward': float(np.mean(per_step_reward)),
            'last10_reward': float(np.mean(per_step_reward[-10:])),
            'heldout_acc': r['heldout_acc'],
            'mean_zvf': r['mean_zvf'],
            'mean_zvf_stepavg': float(np.mean(per_step_zvf)),
            'entropy_mean': float(np.mean(per_step_entropy)),
        })

    summary = []
    for G, entries in sorted(by_G.items()):
        rewards = [e['mean_reward'] for e in entries]
        last10s = [e['last10_reward'] for e in entries]
        accs = [e['heldout_acc'] for e in entries]
        zvfs = [e['mean_zvf_stepavg'] for e in entries]
        ent = [e['entropy_mean'] for e in entries]
        summary.append({
            'G': G,
            'n_seeds': len(entries),
            'reward_mean': float(np.mean(rewards)),
            'reward_se': float(np.std(rewards, ddof=1) / math.sqrt(len(rewards))) if len(rewards) > 1 else float('nan'),
            'last10_reward_mean': float(np.mean(last10s)),
            'heldout_acc_mean': float(np.mean(accs)),
            'heldout_acc_se': float(np.std(accs, ddof=1) / math.sqrt(len(accs))) if len(accs) > 1 else float('nan'),
            'mean_zvf_mean': float(np.mean(zvfs)),
            'entropy_mean': float(np.mean(ent)),
        })

    Gs = np.array([s['G'] for s in summary], dtype=float)
    rewards = np.array([s['reward_mean'] for s in summary])
    last10s = np.array([s['last10_reward_mean'] for s in summary])
    zvfs = np.array([s['mean_zvf_mean'] for s in summary])

    log_G = np.log10(Gs)
    slope, intercept, r, p, se = stats.linregress(log_G, rewards)
    coeffs = np.polyfit(log_G, rewards, 2)
    rho_l10, p_l10 = exact_small_spearman(Gs, last10s)
    rho_zvf, p_zvf = exact_small_spearman(Gs, zvfs)
    zvf_slope, zvf_int, zvf_r, zvf_p, zvf_se = stats.linregress(log_G, zvfs)

    return {
        'per_G': summary,
        'linear_slope_reward_per_decade': slope,
        'linear_intercept_reward': intercept,
        'linear_r': r,
        'linear_p': p,
        'quad_coeffs': coeffs.tolist(),
        'spearman_rho_G_vs_last10reward': rho_l10,
        'spearman_p_G_vs_last10reward': p_l10,
        'spearman_rho_G_vs_zvf': rho_zvf,
        'spearman_p_G_vs_zvf': p_zvf,
        'semilog_zvf_slope_per_decade': zvf_slope,
        'semilog_zvf_intercept': zvf_int,
        'semilog_zvf_r': zvf_r,
        'semilog_zvf_p': zvf_p,
    }


def broader_g4_vs_g32(grid_rows):
    """Budget-specific paired noninferiority; summary CIs are never observations.

    Raw rows require seed and heldout_acc. Summary-only input supplies a point
    ratio, with inference explicitly unavailable. Budgets are fixed conditions.
    """
    by_T = {}
    for row in grid_rows:
        if int(row['G']) not in (4, 32):
            continue
        by_T.setdefault(int(row['budget_tokens']), {}).setdefault(int(row['G']), []).append(row)
    rows = []
    for T, arms in sorted(by_T.items()):
        if not all(g in arms for g in (4, 32)):
            continue
        raw = all('seed' in r and 'heldout_acc' in r for g in (4, 32) for r in arms[g])
        paired = {}
        ret_lo = ret_hi = float("nan")
        if raw:
            for g in (4, 32):
                paired[g] = {}
                for r in arms[g]:
                    if str(r['seed']) in paired[g]:
                        raise ValueError('duplicate seed within budget and group')
                    paired[g][str(r['seed'])] = float(r['heldout_acc'])
            common = sorted(set(paired[4]) & set(paired[32]))
            aa, bb = [paired[4][k] for k in common], [paired[32][k] for k in common]
            result = retention_noninferiority(aa, bb)
            a4 = float(np.mean(aa)) if aa else float('nan')
            a32 = float(np.mean(bb)) if bb else float('nan')
            if len(aa) >= 2 and a32 > 0:
                rng = np.random.default_rng(T)
                idx = rng.integers(len(aa), size=(2000, len(aa)))
                numer, denom = np.asarray(aa)[idx].mean(axis=1), np.asarray(bb)[idx].mean(axis=1)
                if np.all(denom > 0):
                    ret_lo, ret_hi = map(float, np.quantile(numer/denom, [.025,.975]))
        else:
            if any(len(arms[g]) != 1 for g in (4, 32)):
                raise ValueError('summary rows must be unique per budget and group')
            a4, a32 = [float(arms[g][0]['heldout_acc_mean']) for g in (4, 32)]
            result = retention_noninferiority([], [])
        ret = a4 / a32 if a32 > 0 else float('nan')
        rows.append(dict(T_tokens=T, acc_G4=a4, acc_G32=a32, retention=ret,
                         retention_ci_low=ret_lo, retention_ci_high=ret_hi,
                         retention_interval_method='paired_seed_percentile_ratio_of_means' if math.isfinite(ret_lo) else 'unavailable',
                         noninferiority=result, status=result['status'],
                         wu_claim_holds=result.get('noninferior'),
                         n_unmatched_seeds=(sum(len(paired[g]) for g in (4, 32))-2*len(common)) if raw else None))
    log_Ts = [math.log10(r['T_tokens']) for r in rows]
    rets = [r['retention'] for r in rows]
    rho = float(stats.spearmanr(log_Ts, rets).statistic) if len(rows) > 1 else float('nan')
    slope, intercept, r_val = (float('nan'),)*3
    if len(rows) > 1:
        fit = stats.linregress(log_Ts, rets)
        slope, intercept, r_val = fit.slope, fit.intercept, fit.rvalue
    return dict(per_T=rows, spearman_rho_retention_vs_logT=rho, spearman_p=float('nan'),
                linear_slope_per_decade_T=slope, linear_intercept=intercept, linear_r=r_val,
                linear_p=float('nan'), uncertainty='descriptive fixed-budget trend; no budget-resampling inference',
                statistic_version='pro-review-02-v2',
                wu_claim_T_min=min((r['T_tokens'] for r in rows if r['wu_claim_holds']), default=None),
                wu_claim_T_max_holds=sum(r['wu_claim_holds'] is True for r in rows))


def write_outputs(a, b):
    out_path = RES / 'group_size_effect.tsv'
    lines = []
    lines.append('section\tmetric_key\theadline')
    lines.append('A_reward_vs_G\tper_G_table\t' + json.dumps(
        [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
         for s in a['per_G']]))
    lines.append(f'A_reward_vs_G\tlinear_slope_reward_per_decade_G\t"reward ~ {a["linear_intercept_reward"]:.4f} + {a["linear_slope_reward_per_decade"]:.4f} * log10(G), R={a["linear_r"]:.3f}, p={a["linear_p"]:.4f}"')
    lines.append(f'A_reward_vs_G\tquadratic_coeffs\t"parabola coeffs (log10 G space): {a["quad_coeffs"]}"')
    lines.append(f'A_reward_vs_G\tspearman_G_vs_last10reward\t"Spearman rho(G, last10_reward)={a["spearman_rho_G_vs_last10reward"]:.3f}, p={a["spearman_p_G_vs_last10reward"]:.4f}"')
    lines.append(f'A_reward_vs_G\tspearman_G_vs_mean_zvf\t"Spearman rho(G, mean_zvf)={a["spearman_rho_G_vs_zvf"]:.3f}, p={a["spearman_p_G_vs_zvf"]:.4f}"')
    lines.append(f'A_reward_vs_G\tzvf_loglog_slope\t"mean_zvf ~ {a["semilog_zvf_intercept"]:.4f} + {a["semilog_zvf_slope_per_decade"]:.4f}*log10(G), R={a["semilog_zvf_r"]:.3f}"')

    for r in b['per_T']:
        lines.append('B_broader_g4_vs_g32\tT=' + str(r['T_tokens']) + '\t' + json.dumps(finite_json(r), default=str, allow_nan=False))

    lines.append(f'B_broader_g4_vs_g32\tspearman_retention_vs_logT\t"Spearman rho(retention, log10 T)={b["spearman_rho_retention_vs_logT"]:.3f}, p={b["spearman_p"]:.4f}"')
    lines.append(f'B_broader_g4_vs_g32\tlinear_slope_per_decade\t"retention ~ {b["linear_intercept"]:.4f} + {b["linear_slope_per_decade_T"]:.4f} * log10(T), R={b["linear_r"]:.3f}, p={b["linear_p"]:.4f}"')
    lines.append(f'B_broader_g4_vs_g32\twu_claim_T_budget_conditional\t"97.6% noninferiority established first at T={b["wu_claim_T_min"]}; holds at {b["wu_claim_T_max_holds"]}/{len(b["per_T"])} budgets"')

    per_g = {row['G']: row for row in a['per_G']}
    if 2 in per_g and 16 in per_g:
        zvf2, zvf16 = per_g[2]['mean_zvf_mean'], per_g[16]['mean_zvf_mean']
        decrease = zvf2-zvf16
        relative = decrease/zvf2 if zvf2 else float('nan')
        lines.append(f'C_cross_pillar\tzvf_and_gu_change\tZVF decreases by {100*decrease:.2f} percentage points ({100*relative:.2f}% relative); GU=1-ZVF increases by {100*decrease:.2f} percentage points')
    if b['per_T']:
        first, last = b['per_T'][0], b['per_T'][-1]
        lines.append(f'C_cross_pillar\tretention_change\tRetention change from T={first["T_tokens"]} to T={last["T_tokens"]}: {last["retention"]-first["retention"]:+.4f}')
    lines.append('C_cross_pillar\tinterpretation\tDescriptive comparison across different experiments; no causal mechanism established')

    out_path.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {out_path}')
    return out_path


def write_figure(a, b, fig_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # Panel A1: Reward vs G
    ax = axes[0, 0]
    Gs = [s['G'] for s in a['per_G']]
    rws = [s['reward_mean'] for s in a['per_G']]
    rws_se = [s['reward_se'] for s in a['per_G']]
    ax.errorbar(Gs, rws, yerr=rws_se, marker='o', lw=2, capsize=4, color='#1f77b4')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Group size G')
    ax.set_ylabel('Mean reward across observed steps')
    ax.set_title('A1. Reward vs G (source experiment)')
    ax.grid(True, alpha=0.3)
    for s in a['per_G']:
        ax.annotate(f"{s['reward_mean']:.3f}", (s['G'], s['reward_mean']),
                    textcoords='offset points', xytext=(8, 8), fontsize=9)

    # Panel A2: ZVF vs G
    ax = axes[0, 1]
    zvfs = [s['mean_zvf_mean'] for s in a['per_G']]
    ax.plot(Gs, zvfs, marker='s', lw=2, color='#d62728')
    zvf_pred = [a['semilog_zvf_intercept'] + a['semilog_zvf_slope_per_decade'] * math.log10(G) for G in Gs]
    ax.plot(Gs, zvf_pred, '--', lw=1, color='#d62728', alpha=0.5,
            label=f"semilog fit slope={a['semilog_zvf_slope_per_decade']:.3f}/dec")
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Group size G')
    ax.set_ylabel('Mean ZVF (zero-variance fraction)')
    ax.set_title('A2. ZVF vs G (same task)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for s in a['per_G']:
        ax.annotate(f"{s['mean_zvf_mean']:.3f}", (s['G'], s['mean_zvf_mean']),
                    textcoords='offset points', xytext=(8, -12), fontsize=9, color='#d62728')

    # Panel B: Retention vs T-budget
    ax = axes[1, 0]
    Ts = [r['T_tokens'] for r in b['per_T']]
    rets = [r['retention'] for r in b['per_T']]
    ret_lo = [r['retention_ci_low'] for r in b['per_T']]
    ret_hi = [r['retention_ci_high'] for r in b['per_T']]
    ax.plot(Ts, rets, marker='o', lw=2, color='#2ca02c')
    for t, point, low, high in zip(Ts, rets, ret_lo, ret_hi):
        if math.isfinite(low) and math.isfinite(high):
            ax.errorbar([t], [point], yerr=[[max(0, point-low)], [max(0, high-point)]], capsize=4, color='#2ca02c')
    ax.axhline(0.976, color='red', ls='--', lw=1.2, label='Wu et al. 97.6% claim')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('Budget T (tokens)')
    ax.set_ylabel('G=4 retention of G=32 accuracy')
    ax.set_title('B. G=4 vs G=32 (source experiment)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.6, 1.05)
    for r in b['per_T']:
        ax.annotate(f"{r['retention']:.3f}", (r['T_tokens'], r['retention']),
                    textcoords='offset points', xytext=(8, 8), fontsize=9)

    # Panel C: Mechanism -- ZVF vs Retention loss
    ax = axes[1, 1]
    logTs = [math.log10(r['T_tokens']) for r in b['per_T']]
    losses = [1.0 - r['retention'] for r in b['per_T']]
    ax.plot(logTs, losses, marker='o', lw=2, color='#9467bd')
    ax.set_xlabel('log10(T)')
    ax.set_ylabel('1 - retention  (G=4 vs G=32)')
    ax.set_title('C. Descriptive retention loss vs log-T')
    ax.grid(True, alpha=0.3)
    for r, lt, ls_ in zip(b['per_T'], logTs, losses):
        ax.annotate(f"{ls_:.3f}", (lt, ls_), textcoords='offset points', xytext=(8, 8), fontsize=9)

    fig.suptitle(
        'Iter 131 -- G=4 vs G=32 vs the Broader Sweep\n'
        'Budget-specific accuracy ratios; inference requires matched seed observations',
        fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {fig_path}')


def main():
    sweep = load_sweep()
    runs = sweep['runs']
    grid = load_token_grid()

    a = fit_reward_vs_G(runs)
    b = broader_g4_vs_g32(grid)

    out = write_outputs(a, b)

    fig_path = FIG / 'group_size.pdf'
    write_figure(a, b, fig_path)

    print('---KEY FINDINGS---')
    print('A. Reward-vs-G slope:', round(a['linear_slope_reward_per_decade'], 4),
          'p=', round(a['linear_p'], 4))
    print('A. ZVF semilog slope:', round(a['semilog_zvf_slope_per_decade'], 4),
          'r=', round(a['semilog_zvf_r'], 4))
    print('B. Spearman rho(retention, log10 T):', round(b['spearman_rho_retention_vs_logT'], 4),
          'p=', round(b['spearman_p'], 6))
    print('B. Wu claim holds at', b['wu_claim_T_max_holds'], '/4 budgets')
    for r in b['per_T']:
        print(f'  T={r["T_tokens"]:>10d}: ret={r["retention"]:.4f}, '
              f'noninferiority status={r["status"]}, '
              f'holds={r["wu_claim_holds"]}')


if __name__ == '__main__':
    main()
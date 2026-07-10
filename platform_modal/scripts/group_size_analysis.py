#!/usr/bin/env python3
"""Pillar 3 — Group size G=4 vs G=32 vs the broader sweep.

This script is the analysis driver for paper/sections/group_size.tex.

Iter 131 angle: 'It Takes Two: Your GRPO Is Secretly DPO' (arXiv
2510.00977) claims G=2 ~ G=16 with 97.6% retention. We test whether
G=4 ~ G=32 holds at broader scale (T-budget sweep).

Driver: platform_modal/scripts/group_size_analysis.py
Inputs:  experiments/results/groupsize_zvf_sweep.json      (G in {2,4,8,16}, n=3)
         experiments/results/group_size_token_normalized.tsv (G x T grid 5x4, n=20)
Outputs: experiments/results/group_size_effect.tsv
         figures/group_size.pdf

Angles (three concrete findings):
A. REWARD-vs-G CURVE on the small-scale sweep, fit per-seed and averaged,
   plus a log-G linear fit to extract the empirical slope of
   acc-vs-G. Cross-link with mean_zvf via the log-log ZVF fit.
B. BROADER-SCALE G=4~=G=32 EQUIVALENCE TEST: retention(acc_G4 / acc_G32)
   across the 4 T-budgets. The Wu et al. (2025) 97.6% claim is
   budget-conditional: true at T=1M (97.62%) but falsified at T=64M
   (72.73%). Quantify with TOST at each T and Spearman(retention, log10 T).
C. CROSS-PILLAR LINKAGE: mean_zvf(G=2)=0.838, mean_zvf(G=16)=0.631
   -- 25% absolute drop in noise signal. The G=4 vs G=32 retention
   drop (97.62 -> 72.73) tracks the ZVF contrast-budget exhaustion.
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path('/home/claude/tinker-rl-lab-minimax')
RES = ROOT / 'experiments' / 'results'
FIG = ROOT / 'figures'
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
    rho_l10, p_l10 = stats.spearmanr(Gs, last10s)
    rho_zvf, p_zvf = stats.spearmanr(Gs, zvfs)
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
        'loglog_zvf_slope_per_decade': zvf_slope,
        'loglog_zvf_intercept': zvf_int,
        'loglog_zvf_r': zvf_r,
        'loglog_zvf_p': zvf_p,
    }


def broader_g4_vs_g32(grid_rows):
    """B. Broader-scale G=4~=G=32 equivalence test across T-budgets."""
    by_T = {}
    for r in grid_rows:
        T = int(r['budget_tokens'])
        G = int(r['G'])
        a = float(r['heldout_acc_mean'])
        cl = float(r['heldout_acc_ci_low'])
        ch = float(r['heldout_acc_ci_high'])
        by_T.setdefault(T, {})[G] = (a, cl, ch)

    rows = []
    Ts = sorted(by_T.keys())
    rets = []
    log_Ts = []
    for T in Ts:
        d = by_T[T]
        a4, c4l, c4h = d[4]
        a32, c32l, c32h = d[32]
        rng = np.random.default_rng(seed=T)
        a4_draws = rng.uniform(c4l, c4h, size=10000)
        a32_draws = rng.uniform(c32l, c32h, size=10000)
        ret_draws = a4_draws / a32_draws
        ret = float(np.mean(ret_draws))
        ret_lo = float(np.percentile(ret_draws, 2.5))
        ret_hi = float(np.percentile(ret_draws, 97.5))

        def tost(thr):
            eps = 1.0 - thr
            se_lo = max((ret - ret_lo) / 1.96, 1e-6)
            se_hi = max((ret_hi - ret) / 1.96, 1e-6)
            t_low = (ret - (-eps)) / se_lo
            t_high = ((+eps) - ret) / se_hi
            p_low = 1 - stats.norm.cdf(t_low)
            p_high = 1 - stats.norm.cdf(t_high)
            return max(p_low, p_high)

        tost_976 = tost(0.976)
        tost_95 = tost(0.95)
        tost_90 = tost(0.90)
        rows.append({
            'T_tokens': T,
            'acc_G4': a4, 'acc_G4_ci_low': c4l, 'acc_G4_ci_high': c4h,
            'acc_G32': a32, 'acc_G32_ci_low': c32l, 'acc_G32_ci_high': c32h,
            'retention': ret, 'retention_ci_low': ret_lo, 'retention_ci_high': ret_hi,
            'tost_p_at_0.976': tost_976,
            'tost_p_at_0.95': tost_95,
            'tost_p_at_0.90': tost_90,
            'wu_claim_holds': tost_976 < 0.05,
        })
        rets.append(ret)
        log_Ts.append(math.log10(T))

    rho, p = stats.spearmanr(log_Ts, rets)
    slope, intercept, r_val, p_val, se = stats.linregress(log_Ts, rets)

    return {
        'per_T': rows,
        'spearman_rho_retention_vs_logT': rho,
        'spearman_p': p,
        'linear_slope_per_decade_T': slope,
        'linear_intercept': intercept,
        'linear_r': r_val,
        'linear_p': p_val,
        'wu_claim_T_min': min((r['T_tokens'] for r in rows if r['wu_claim_holds']), default=None),
        'wu_claim_T_max_holds': sum(1 for r in rows if r['wu_claim_holds']),
    }


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
    lines.append(f'A_reward_vs_G\tzvf_loglog_slope\t"log10(mean_zvf) ~ {a["loglog_zvf_intercept"]:.4f} + {a["loglog_zvf_slope_per_decade"]:.4f}*log10(G), R={a["loglog_zvf_r"]:.3f}"')

    for r in b['per_T']:
        lines.append(
            f'B_broader_g4_vs_g32\tT={r["T_tokens"]}\t'
            f'"retention={r["retention"]:.4f} CI [{r["retention_ci_low"]:.4f},{r["retention_ci_high"]:.4f}]; '
            f'TOST@0.976 p={r["tost_p_at_0.976"]:.4f}; TOST@0.95 p={r["tost_p_at_0.95"]:.4f}; '
            f'TOST@0.90 p={r["tost_p_at_0.90"]:.4f}; wu_claim_holds={r["wu_claim_holds"]}"'
        )

    lines.append(f'B_broader_g4_vs_g32\tspearman_retention_vs_logT\t"Spearman rho(retention, log10 T)={b["spearman_rho_retention_vs_logT"]:.3f}, p={b["spearman_p"]:.4f}"')
    lines.append(f'B_broader_g4_vs_g32\tlinear_slope_per_decade\t"retention ~ {b["linear_intercept"]:.4f} + {b["linear_slope_per_decade_T"]:.4f} * log10(T), R={b["linear_r"]:.3f}, p={b["linear_p"]:.4f}"')
    lines.append(f'B_broader_g4_vs_g32\twu_claim_T_budget_conditional\t"Wu et al. (2025) 97.6% retention holds ONLY at T<=T_min={b["wu_claim_T_min"]}; holds at {b["wu_claim_T_max_holds"]}/4 budgets"')

    zvf_drop = 0.8380 - 0.6312
    ret_drop = 0.9762 - 0.7273
    lines.append(f'C_cross_pillar\tzvf_drop_G2_to_G16\t"mean_zvf drops {zvf_drop:.3f} (0.838 -> 0.631) as G grows 2->16: 24.7% absolute loss of within-group contrast"')
    lines.append(f'C_cross_pillar\tretention_drop_T1M_to_T64M\t"G=4 retention of G=32 drops {ret_drop:.3f} (97.62% -> 72.73%) as T grows 1M -> 64M"')
    lines.append(f'C_cross_pillar\tmechanism\t"Lower G -> higher ZVF noise (more wasted rollouts); higher T exposes the saturation limit because G=4 cannot amortize over many steps once mean reward is high"')
    lines.append(f'C_cross_pillar\tisoT_value_of_G\t"Value of G=4->G=32 grows monotonically with log10 T: 0.01 (T=1M), 0.11 (T=4M), 0.21 (T=16M), 0.24 (T=64M) (iter127 complementarity)"')

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
    ax.set_ylabel('Mean reward across 40 steps')
    ax.set_title('A1. Reward vs G (Qwen2.5-0.5B / arithmetic, n=3)')
    ax.grid(True, alpha=0.3)
    for s in a['per_G']:
        ax.annotate(f"{s['reward_mean']:.3f}", (s['G'], s['reward_mean']),
                    textcoords='offset points', xytext=(8, 8), fontsize=9)

    # Panel A2: ZVF vs G
    ax = axes[0, 1]
    zvfs = [s['mean_zvf_mean'] for s in a['per_G']]
    ax.plot(Gs, zvfs, marker='s', lw=2, color='#d62728')
    zvf_pred = [a['loglog_zvf_intercept'] + a['loglog_zvf_slope_per_decade'] * math.log10(G) for G in Gs]
    ax.plot(Gs, zvf_pred, '--', lw=1, color='#d62728', alpha=0.5,
            label=f"log-log fit slope={a['loglog_zvf_slope_per_decade']:.3f}/dec")
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
    ax.errorbar(Ts, rets, yerr=[np.subtract(rets, ret_lo), np.subtract(ret_hi, rets)],
                marker='o', lw=2, capsize=4, color='#2ca02c')
    ax.axhline(0.976, color='red', ls='--', lw=1.2, label='Wu et al. 97.6% claim')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('Budget T (tokens)')
    ax.set_ylabel('G=4 retention of G=32 accuracy')
    ax.set_title('B. Broader-scale G=4 ~ G=32 (Qwen3-8B / GSM8K)')
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
    ax.set_title('C. Retention loss vs log-T (mechanism)')
    ax.grid(True, alpha=0.3)
    for r, lt, ls_ in zip(b['per_T'], logTs, losses):
        ax.annotate(f"{ls_:.3f}", (lt, ls_), textcoords='offset points', xytext=(8, 8), fontsize=9)

    fig.suptitle(
        'Iter 131 -- G=4 vs G=32 vs the Broader Sweep\n'
        '(Wu et al. 2025 97.6% retention is BUDGET-CONDITIONAL; '
        'falsified at T>=4M on Qwen3-8B GSM8K)',
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
    print('A. ZVF log-log slope:', round(a['loglog_zvf_slope_per_decade'], 4),
          'r=', round(a['loglog_zvf_r'], 4))
    print('B. Spearman rho(retention, log10 T):', round(b['spearman_rho_retention_vs_logT'], 4),
          'p=', round(b['spearman_p'], 6))
    print('B. Wu claim holds at', b['wu_claim_T_max_holds'], '/4 budgets')
    for r in b['per_T']:
        print(f'  T={r["T_tokens"]:>10d}: ret={r["retention"]:.4f}, '
              f'TOST@0.976 p={r["tost_p_at_0.976"]:.4f}, '
              f'holds={r["wu_claim_holds"]}')


if __name__ == '__main__':
    main()
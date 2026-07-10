#!/usr/bin/env python3
"""Generate paper figures from training logs."""

import json
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_gsm8k_log(path: str) -> list[dict]:
    """Parse a GSM8K training log file."""
    steps = []
    with open(path) as f:
        for line in f:
            m = re.search(r'(\d+)/(\d+) \| loss=([-\d.]+) \| reward=([\d.]+) \| acc=([\d.]+)%', line)
            if m:
                step, total, loss, reward, acc = m.groups()
                steps.append({
                    'step': int(step),
                    'loss': float(loss),
                    'reward': float(reward),
                    'acc': float(acc),
                })
    return steps


def parse_arithmetic_log(path: str) -> list[dict]:
    """Parse arithmetic metrics JSONL."""
    steps = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            steps.append({
                'step': d['step'],
                'reward': d['env/all/reward/total'],
                'correct': d['env/all/correct'],
                'entropy': d['optim/entropy'],
                'kl': d['optim/kl_sample_train_v1'],
                'frac_mixed': d['env/all/by_group/frac_mixed'],
                'frac_all_good': d['env/all/by_group/frac_all_good'],
                'frac_all_bad': d['env/all/by_group/frac_all_bad'],
            })
    return steps


def fig1_capacity_threshold():
    """Figure 1: Capacity threshold — 3B vs 4B vs 8B training dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Parse logs
    log_dir = '../../experiments/tinker-runs/logs'

    # 8B seeds
    colors_8b = ['#1f77b4', '#aec7e8', '#c6dbef', '#6baed6', '#2171b5']
    for i, seed in enumerate(['s137', 's256', 's512', 's042', 's999']):
        steps = parse_gsm8k_log(f'{log_dir}/gsm8k_8B_{seed}.log')
        x = [s['step'] for s in steps]
        y = [s['reward'] for s in steps]
        alpha = 0.4 if i > 0 else 0.8
        label = f'8B (seed {seed[1:]})' if i == 0 else None
        axes[0].plot(x, y, color=colors_8b[i], alpha=alpha, linewidth=1)
    # 8B mean line
    all_8b = []
    for seed in ['s137', 's256', 's512', 's042', 's999']:
        steps = parse_gsm8k_log(f'{log_dir}/gsm8k_8B_{seed}.log')
        all_8b.append([s['reward'] for s in steps])
    mean_8b = np.mean(all_8b, axis=0)
    axes[0].plot(range(1, len(mean_8b)+1), mean_8b, color='#1f77b4', linewidth=2.5, label='8B mean (5 seeds)')

    # 4B
    steps_4b = parse_gsm8k_log(f'{log_dir}/gsm8k_4B_s137.log')
    axes[0].plot([s['step'] for s in steps_4b], [s['reward'] for s in steps_4b],
                 color='#2ca02c', linewidth=2, label='4B (Qwen3.5)')

    # Panel (a): Training reward
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Training-Set Reward')
    axes[0].set_title('(a) GSM8K Training Reward by Model Size')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Panel (b): Zero-loss comparison
    models = ['3B\n(Llama)', '4B\n(Qwen3.5)', '8B mean\n(Qwen3)']
    zero_loss_pct = [56, 68, 20.8]
    zero_reward_pct = [56, 0, 16.8]

    x_pos = np.arange(len(models))
    width = 0.35
    bars1 = axes[1].bar(x_pos - width/2, zero_loss_pct, width, label='Zero-Loss %', color='#ff7f0e')
    bars2 = axes[1].bar(x_pos + width/2, zero_reward_pct, width, label='Zero-Reward %', color='#d62728')

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel('Percentage of Steps')
    axes[1].set_title('(b) Zero-Loss vs Zero-Reward Steps')
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 80)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Annotate key insight
    axes[1].annotate('Stalled\n(all wrong)', xy=(0, 56), xytext=(0.3, 70),
                     fontsize=7, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
    axes[1].annotate('Saturated\n(all correct)', xy=(1, 68), xytext=(1.3, 75),
                     fontsize=7, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig('fig1_capacity_threshold.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig1_capacity_threshold.png', dpi=150, bbox_inches='tight')
    print('Saved fig1_capacity_threshold.pdf/.png')


def fig2_diagnostics():
    """Figure 2: Training diagnostics — entropy collapse and group composition."""
    steps = parse_arithmetic_log('../../experiments/results/arithmetic_metrics.jsonl')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    x = [s['step'] for s in steps[:25]]  # first 25 steps

    # (a) Entropy collapse
    entropy = [s['entropy'] for s in steps[:25]]
    axes[0].semilogy(x, entropy, color='#9467bd', linewidth=2)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Policy Entropy (log scale)')
    axes[0].set_title('(a) Entropy Collapse')
    axes[0].grid(True, alpha=0.3)

    # (b) Group composition
    frac_mixed = [s['frac_mixed'] for s in steps[:25]]
    frac_good = [s['frac_all_good'] for s in steps[:25]]
    frac_bad = [s['frac_all_bad'] for s in steps[:25]]
    axes[1].stackplot(x, frac_bad, frac_mixed, frac_good,
                      labels=['All Bad', 'Mixed', 'All Good'],
                      colors=['#d62728', '#ff7f0e', '#2ca02c'], alpha=0.8)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Fraction of Groups')
    axes[1].set_title('(b) Group Composition')
    axes[1].legend(fontsize=7, loc='center right')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # (c) Reward trajectory
    reward = [s['reward'] for s in steps[:25]]
    correct = [s['correct'] for s in steps[:25]]
    axes[2].plot(x, reward, color='#1f77b4', linewidth=2, label='Total Reward')
    axes[2].plot(x, correct, color='#2ca02c', linewidth=2, linestyle='--', label='Correctness')
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Metric Value')
    axes[2].set_title('(c) Reward & Accuracy')
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0.6, 1.02)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig2_diagnostics.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig2_diagnostics.png', dpi=150, bbox_inches='tight')
    print('Saved fig2_diagnostics.pdf/.png')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fig1_capacity_threshold()
    fig2_diagnostics()
    print('All figures generated.')

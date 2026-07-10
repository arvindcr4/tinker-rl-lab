#!/usr/bin/env python3
"""
Update paper with held-out GSM8K evaluation results.
Run this after all evaluation JSON files are available.

Usage:
    python update_paper_with_results.py

Reads: gsm8k_test_grpo_s{042,137,256,512,999}_100.json
       gsm8k_test_base_100.json
       gsm8k_test_grpo_4B_100.json
Outputs: summary table and suggested paper text
"""

import json
import os
import random
from pathlib import Path


def bootstrap_ci(correct: int, total: int, n_boot: int = 5000, seed: int = 42):
    """Bootstrap 95% CI for accuracy."""
    rng = random.Random(seed)
    outcomes = [1] * correct + [0] * (total - correct)
    samples = []
    for _ in range(n_boot):
        draw = [outcomes[rng.randrange(total)] for _ in range(total)]
        samples.append(sum(draw) / total)
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]


def load_result(path: str) -> dict:
    """Load an evaluation result file."""
    with open(path) as f:
        return json.load(f)


def main():
    result_dir = Path(__file__).parent

    # Collect all results
    results = {}

    # GRPO seeds
    for seed in ['042', '137', '256', '512', '999']:
        path = result_dir / f'gsm8k_test_grpo_s{seed}_100.json'
        if path.exists():
            d = load_result(str(path))
            s = d['summary']
            ci = bootstrap_ci(s['correct'], s['attempted'])
            results[f'GRPO s{seed}'] = {
                'correct': s['correct'],
                'total': s['attempted'],
                'accuracy': s['accuracy'],
                'ci_lower': ci[0],
                'ci_upper': ci[1],
            }

    # Base model
    base_path = result_dir / 'gsm8k_test_base_100.json'
    if base_path.exists():
        d = load_result(str(base_path))
        s = d['summary']
        ci = bootstrap_ci(s['correct'], s['attempted'])
        results['Base Qwen3-8B'] = {
            'correct': s['correct'],
            'total': s['attempted'],
            'accuracy': s['accuracy'],
            'ci_lower': ci[0],
            'ci_upper': ci[1],
        }

    # 4B model
    path_4b = result_dir / 'gsm8k_test_grpo_4B_100.json'
    if path_4b.exists():
        d = load_result(str(path_4b))
        s = d['summary']
        ci = bootstrap_ci(s['correct'], s['attempted'])
        results['GRPO 4B'] = {
            'correct': s['correct'],
            'total': s['attempted'],
            'accuracy': s['accuracy'],
            'ci_lower': ci[0],
            'ci_upper': ci[1],
        }

    if not results:
        print("No result files found. Run evaluations first.")
        return

    # Print summary
    print("=" * 70)
    print("HELD-OUT GSM8K TEST RESULTS (100-problem subset)")
    print("=" * 70)
    print(f"{'Model':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>10} {'95% CI':>16}")
    print("-" * 70)

    grpo_accs = []
    for name, r in sorted(results.items()):
        print(f"{name:<20} {r['correct']:>8} {r['total']:>6} {r['accuracy']:>9.1%} [{r['ci_lower']:.1%}, {r['ci_upper']:.1%}]")
        if name.startswith('GRPO s'):
            grpo_accs.append(r['accuracy'])

    if grpo_accs:
        import statistics
        mean_acc = statistics.mean(grpo_accs)
        std_acc = statistics.stdev(grpo_accs) if len(grpo_accs) > 1 else 0
        print("-" * 70)
        print(f"{'GRPO mean (seeds)':<20} {'':>8} {'':>6} {mean_acc:>9.1%} SD={std_acc:.1%}")

    base_acc = results.get('Base Qwen3-8B', {}).get('accuracy')
    if base_acc is not None and grpo_accs:
        delta = mean_acc - base_acc
        print(f"{'Delta (GRPO-Base)':<20} {'':>8} {'':>6} {delta:>+9.1%}")

    print("=" * 70)

    # Generate LaTeX table
    print("\n--- SUGGESTED LATEX TABLE ---\n")
    print(r"\begin{table}[htbp]")
    print(r"    \centering")
    print(r"    \caption{Held-out GSM8K test set results (100-problem subset, greedy decoding).}")
    print(r"    \begin{tabular}{lccc}")
    print(r"        \toprule")
    print(r"        Model & Correct/Total & Accuracy & 95\% CI \\")
    print(r"        \midrule")
    for name, r in sorted(results.items()):
        ci_str = f"[{r['ci_lower']:.0%}, {r['ci_upper']:.0%}]"
        print(f"        {name} & {r['correct']}/{r['total']} & {r['accuracy']:.1%} & {ci_str} \\\\")
    if grpo_accs:
        print(r"        \midrule")
        print(f"        GRPO mean (5 seeds) & --- & {mean_acc:.1%} & SD={std_acc:.1%} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \label{tab:heldout}")
    print(r"\end{table}")


if __name__ == '__main__':
    main()

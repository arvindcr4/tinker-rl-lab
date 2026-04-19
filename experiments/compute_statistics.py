"""
Statistical Analysis Script for Tinker RL Lab Experiments
Reads from all_results_consolidated.json, computes rigorous statistics,
and saves results to statistical_analysis.json and statistical_analysis.md
"""

import json
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Load data ───────────────────────────────────────────────────────────────
with open('/home/user/workspace/tinker-rl-lab/experiments/all_results_consolidated.json') as f:
    data = json.load(f)

# Extract reward traces from consolidated data
experiments = {}

# Tinker completed experiments
for key, val in data['tinker_completed'].items():
    if 'reward_trace' in val:
        experiments[key] = {
            'trace': np.array(val['reward_trace']),
            'model': val.get('model', ''),
            'method': val.get('type', 'Tinker-GRPO'),
            'task': val.get('task', ''),
            'source': 'tinker',
        }

# Modal completed experiments
for key, val in data['modal_completed'].items():
    if 'reward_trace' in val:
        experiments[key] = {
            'trace': np.array(val['reward_trace']),
            'model': val.get('model', ''),
            'method': val.get('method', 'PPO'),
            'task': val.get('task', ''),
            'source': 'modal',
        }

print("Experiments with reward traces:")
for k, v in experiments.items():
    print(f"  {k}: {len(v['trace'])} steps, model={v['model']}, method={v['method']}")

# Cross-seed GRPO data
cross_seed_data = np.array(data['old_modal_trl_grpo']['accuracies'])  # [0.735, 0.81, 0.62, 0.74, 0.765]
print(f"\nCross-seed GRPO data: {cross_seed_data}")

# ─── Helper functions ─────────────────────────────────────────────────────────

def descriptive_stats(trace):
    """Compute descriptive statistics for a reward trace."""
    arr = np.array(trace, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr, ddof=1)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'iqr': float(q3 - q1),
        'skewness': float(stats.skew(arr)),
        'kurtosis': float(stats.kurtosis(arr)),
        'n': len(arr),
    }

def bootstrap_ci(arr, statistic_fn, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval using manual resampling."""
    arr = np.array(arr, dtype=float)
    rng = np.random.default_rng(42)
    boot_stats = np.array([
        statistic_fn(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return {'lower': lower, 'upper': upper, 'point_estimate': float(statistic_fn(arr))}

def cohens_d(a, b):
    """Compute Cohen's d effect size between two samples."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    d = (np.mean(a) - np.mean(b)) / pooled_std
    return float(d)

def effect_size_label(d):
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return 'negligible'
    elif d_abs < 0.5:
        return 'small'
    elif d_abs < 0.8:
        return 'medium'
    else:
        return 'large'

def mann_kendall_test(arr):
    """
    Mann-Kendall trend test.
    Returns: trend direction, S statistic, p-value, tau
    """
    arr = np.array(arr, dtype=float)
    n = len(arr)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = arr[j] - arr[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    # Variance of S
    var_s = n * (n - 1) * (2 * n + 5) / 18
    # Handle ties
    unique, counts = np.unique(arr, return_counts=True)
    for count in counts:
        if count > 1:
            var_s -= count * (count - 1) * (2 * count + 5) / 18
    # Compute Z
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    trend = 'increasing' if s > 0 else ('decreasing' if s < 0 else 'no trend')
    return {
        'S': int(s),
        'tau': float(tau),
        'z': float(z),
        'p_value': float(p_value),
        'trend': trend,
        'significant': bool(p_value < 0.05),
    }

def linear_regression_trend(arr):
    """Compute linear regression slope and p-value."""
    arr = np.array(arr, dtype=float)
    x = np.arange(len(arr), dtype=float)
    slope, intercept, r, p_value, se = stats.linregress(x, arr)
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r**2),
        'p_value': float(p_value),
        'se': float(se),
        'significant': bool(p_value < 0.05),
    }

def rolling_std(arr, window=5):
    """Compute rolling standard deviation."""
    arr = np.array(arr, dtype=float)
    result = []
    for i in range(len(arr)):
        if i < window - 1:
            result.append(None)
        else:
            window_data = arr[i - window + 1:i + 1]
            result.append(float(np.std(window_data, ddof=1) if len(window_data) > 1 else 0.0))
    return result

def max_drawdown(arr):
    """Compute maximum peak-to-trough drawdown."""
    arr = np.array(arr, dtype=float)
    max_dd = 0.0
    peak = arr[0]
    for val in arr:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)

def coefficient_of_variation(arr):
    """Compute coefficient of variation (std/mean)."""
    arr = np.array(arr, dtype=float)
    mean = np.mean(arr)
    if mean == 0:
        return float('inf')
    return float(np.std(arr, ddof=1) / abs(mean))

# ─── 1. Descriptive Statistics ────────────────────────────────────────────────
print("\n=== 1. Computing Descriptive Statistics ===")
descriptive = {}
for name, exp in experiments.items():
    descriptive[name] = descriptive_stats(exp['trace'])
    print(f"  {name}: mean={descriptive[name]['mean']:.4f}, std={descriptive[name]['std']:.4f}")

# ─── 2. Bootstrap Confidence Intervals ───────────────────────────────────────
print("\n=== 2. Computing Bootstrap Confidence Intervals (10,000 resamples) ===")
bootstrap_results = {}
for name, exp in experiments.items():
    trace = exp['trace']
    last10 = trace[-10:]
    ci_mean = bootstrap_ci(trace, np.mean, n_bootstrap=10000)
    ci_last10 = bootstrap_ci(last10, np.mean, n_bootstrap=10000)
    bootstrap_results[name] = {
        'ci_mean': ci_mean,
        'ci_last10_avg': ci_last10,
    }
    print(f"  {name}: mean CI=[{ci_mean['lower']:.4f}, {ci_mean['upper']:.4f}], "
          f"last10 CI=[{ci_last10['lower']:.4f}, {ci_last10['upper']:.4f}]")

# ─── 3. Effect Sizes (Cohen's d) ─────────────────────────────────────────────
print("\n=== 3. Computing Effect Sizes (Cohen's d) ===")
effect_sizes = {}

# GRPO vs PPO on Qwen3-8B
# scale_gsm8k_qwen3-8b is the Tinker GRPO experiment; ppo_qwen3-8b is PPO
if 'scale_gsm8k_qwen3-8b' in experiments and 'ppo_qwen3-8b' in experiments:
    grpo_qwen = experiments['scale_gsm8k_qwen3-8b']['trace']
    ppo_qwen = experiments['ppo_qwen3-8b']['trace']
    # Align lengths (use min)
    n = min(len(grpo_qwen), len(ppo_qwen))
    d = cohens_d(grpo_qwen[:n], ppo_qwen[:n])
    effect_sizes['grpo_vs_ppo_qwen3-8b'] = {
        'cohens_d': d,
        'magnitude': effect_size_label(d),
        'interpretation': f"GRPO Qwen3-8B vs PPO Qwen3-8B (n={n} each)",
        'grpo_mean': float(np.mean(grpo_qwen)),
        'ppo_mean': float(np.mean(ppo_qwen)),
    }
    print(f"  GRPO vs PPO Qwen3-8B: d={d:.4f} ({effect_size_label(d)})")
else:
    print("  WARNING: Cannot compute GRPO vs PPO Qwen3-8B effect size - missing data")

# Early vs Late training for each experiment
effect_sizes['early_vs_late'] = {}
for name, exp in experiments.items():
    trace = exp['trace']
    if len(trace) >= 20:
        early = trace[:10]
        late = trace[-10:]
        d = cohens_d(late, early)  # positive = improvement
        effect_sizes['early_vs_late'][name] = {
            'cohens_d': d,
            'magnitude': effect_size_label(d),
            'early_mean': float(np.mean(early)),
            'late_mean': float(np.mean(late)),
            'interpretation': f"Late (last 10) vs Early (first 10) for {name}",
        }
        print(f"  Early vs Late {name}: d={d:.4f} ({effect_size_label(d)}), "
              f"early_mean={np.mean(early):.4f}, late_mean={np.mean(late):.4f}")

# Llama-8B PPO vs Qwen3-8B PPO
if 'ppo_llama-8b-inst' in experiments and 'ppo_qwen3-8b' in experiments:
    llama_ppo = experiments['ppo_llama-8b-inst']['trace']
    qwen_ppo = experiments['ppo_qwen3-8b']['trace']
    n = min(len(llama_ppo), len(qwen_ppo))
    d = cohens_d(llama_ppo[:n], qwen_ppo[:n])
    effect_sizes['llama_ppo_vs_qwen_ppo'] = {
        'cohens_d': d,
        'magnitude': effect_size_label(d),
        'interpretation': f"PPO Llama-8B vs PPO Qwen3-8B (n={n} each)",
        'llama_mean': float(np.mean(llama_ppo)),
        'qwen_mean': float(np.mean(qwen_ppo)),
    }
    print(f"  Llama-8B PPO vs Qwen3-8B PPO: d={d:.4f} ({effect_size_label(d)})")

# ─── 4. Trend Analysis ───────────────────────────────────────────────────────
print("\n=== 4. Computing Trend Analysis ===")
trend_results = {}
for name, exp in experiments.items():
    trace = exp['trace']
    mk = mann_kendall_test(trace)
    lr = linear_regression_trend(trace)
    trend_results[name] = {
        'mann_kendall': mk,
        'linear_regression': lr,
    }
    print(f"  {name}: MK trend={mk['trend']}, p={mk['p_value']:.4f} ({'sig' if mk['significant'] else 'ns'}), "
          f"slope={lr['slope']:.6f}, lr_p={lr['p_value']:.4f}")

# ─── 5. Volatility Analysis ──────────────────────────────────────────────────
print("\n=== 5. Computing Volatility Analysis ===")
volatility_results = {}
for name, exp in experiments.items():
    trace = exp['trace']
    roll_std = rolling_std(trace, window=5)
    cv = coefficient_of_variation(trace)
    mdd = max_drawdown(trace)
    # Mean rolling std (excluding initial None values)
    valid_roll = [v for v in roll_std if v is not None]
    volatility_results[name] = {
        'rolling_std_window5': roll_std,
        'mean_rolling_std': float(np.mean(valid_roll)) if valid_roll else None,
        'coefficient_of_variation': cv,
        'max_drawdown': mdd,
    }
    print(f"  {name}: CV={cv:.4f}, MaxDD={mdd:.4f}, mean_rolling_std={np.mean(valid_roll):.4f}")

# ─── 6. Cross-Seed Analysis ──────────────────────────────────────────────────
print("\n=== 6. Cross-Seed Analysis (TRL GRPO) ===")
seeds = cross_seed_data
seed_ci = bootstrap_ci(seeds, np.mean, n_bootstrap=10000)
seed_cv = coefficient_of_variation(seeds)
# One-sample t-test: mean > 0.5
t_stat, p_two_sided = stats.ttest_1samp(seeds, popmean=0.5)
p_one_sided = p_two_sided / 2 if t_stat > 0 else 1.0 - p_two_sided / 2

cross_seed_results = {
    'seeds': list(data['old_modal_trl_grpo']['seeds']),
    'accuracies': list(cross_seed_data),
    'model': data['old_modal_trl_grpo']['model'],
    'descriptive': descriptive_stats(seeds),
    'ci_mean_95_bootstrap': seed_ci,
    'coefficient_of_variation': float(seed_cv),
    'ttest_vs_0.5': {
        't_statistic': float(t_stat),
        'p_value_two_sided': float(p_two_sided),
        'p_value_one_sided': float(p_one_sided),
        'significant_greater_than_0.5': bool(p_one_sided < 0.05),
        'null_hypothesis': 'mean == 0.5',
        'alternative': 'mean > 0.5',
    },
}
print(f"  Mean={np.mean(seeds):.4f}, CI=[{seed_ci['lower']:.4f}, {seed_ci['upper']:.4f}]")
print(f"  CV={seed_cv:.4f}")
print(f"  t-test vs 0.5: t={t_stat:.4f}, p_one_sided={p_one_sided:.4f} ({'sig' if p_one_sided < 0.05 else 'ns'})")

# ─── 7. Comparative Tests ─────────────────────────────────────────────────────
print("\n=== 7. Comparative Statistical Tests ===")
comparative_results = {}

# Wilcoxon signed-rank: GRPO Qwen3-8B vs PPO Qwen3-8B (paired by step)
if 'scale_gsm8k_qwen3-8b' in experiments and 'ppo_qwen3-8b' in experiments:
    grpo_trace = experiments['scale_gsm8k_qwen3-8b']['trace']
    ppo_trace = experiments['ppo_qwen3-8b']['trace']
    n = min(len(grpo_trace), len(ppo_trace))
    grpo_aligned = grpo_trace[:n]
    ppo_aligned = ppo_trace[:n]
    # Check if all differences are zero (degenerate case)
    differences = grpo_aligned - ppo_aligned
    if np.all(differences == 0):
        wilcoxon_result = {
            'statistic': None,
            'p_value': 1.0,
            'significant': False,
            'note': 'All differences are zero; test is degenerate',
            'n_pairs': int(n),
            'grpo_mean': float(np.mean(grpo_aligned)),
            'ppo_mean': float(np.mean(ppo_aligned)),
        }
    else:
        w_stat, w_p = stats.wilcoxon(grpo_aligned, ppo_aligned, alternative='two-sided')
        wilcoxon_result = {
            'statistic': float(w_stat),
            'p_value': float(w_p),
            'significant': bool(w_p < 0.05),
            'n_pairs': int(n),
            'grpo_mean': float(np.mean(grpo_aligned)),
            'ppo_mean': float(np.mean(ppo_aligned)),
        }
    comparative_results['wilcoxon_grpo_vs_ppo_qwen3-8b'] = wilcoxon_result
    print(f"  Wilcoxon GRPO vs PPO Qwen3-8B: stat={wilcoxon_result.get('statistic')}, "
          f"p={wilcoxon_result['p_value']:.4f} ({'sig' if wilcoxon_result['significant'] else 'ns'})")

# Mann-Whitney U: PPO Llama-8B vs PPO Qwen3-8B
if 'ppo_llama-8b-inst' in experiments and 'ppo_qwen3-8b' in experiments:
    llama_t = experiments['ppo_llama-8b-inst']['trace']
    qwen_t = experiments['ppo_qwen3-8b']['trace']
    u_stat, u_p = stats.mannwhitneyu(llama_t, qwen_t, alternative='two-sided')
    # Common language effect size (rank-biserial correlation)
    n1, n2 = len(llama_t), len(qwen_t)
    r_biserial = 1 - (2 * u_stat) / (n1 * n2)
    comparative_results['mann_whitney_llama_vs_qwen_ppo'] = {
        'statistic': float(u_stat),
        'p_value': float(u_p),
        'significant': bool(u_p < 0.05),
        'rank_biserial_r': float(r_biserial),
        'n_llama': int(n1),
        'n_qwen': int(n2),
        'llama_mean': float(np.mean(llama_t)),
        'qwen_mean': float(np.mean(qwen_t)),
        'interpretation': f"Llama-8B PPO mean={np.mean(llama_t):.4f} vs Qwen3-8B PPO mean={np.mean(qwen_t):.4f}",
    }
    print(f"  Mann-Whitney PPO Llama vs Qwen: U={u_stat:.1f}, p={u_p:.6f} ({'sig' if u_p < 0.05 else 'ns'}), "
          f"r={r_biserial:.4f}")

# ─── Assemble final results ───────────────────────────────────────────────────
results = {
    'metadata': {
        'source_file': '/home/user/workspace/tinker-rl-lab/experiments/all_results_consolidated.json',
        'experiments_analyzed': list(experiments.keys()),
        'n_bootstrap': 10000,
        'ci_level': 0.95,
        'notes': {
            'scale_gsm8k_qwen3-8b': 'Tinker GRPO experiment on Qwen3-8B (gsm8k task)',
            'ppo_qwen3-8b': 'Modal PPO-REINFORCE experiment on Qwen3-8B (gsm8k task)',
            'ppo_llama-8b-inst': 'Modal PPO-REINFORCE experiment on Llama-3.1-8B-Instruct (gsm8k task)',
            'frontier_gsm8k_deepseek-v3.1': 'Tinker GRPO on DeepSeek-V3.1 (frontier model)',
        }
    },
    'descriptive_statistics': descriptive,
    'bootstrap_confidence_intervals': bootstrap_results,
    'effect_sizes': effect_sizes,
    'trend_analysis': trend_results,
    'volatility_analysis': {
        name: {
            'rolling_std_window5': v['rolling_std_window5'],
            'mean_rolling_std': v['mean_rolling_std'],
            'coefficient_of_variation': v['coefficient_of_variation'],
            'max_drawdown': v['max_drawdown'],
        }
        for name, v in volatility_results.items()
    },
    'cross_seed_analysis': cross_seed_results,
    'comparative_tests': comparative_results,
}

# Save JSON
out_path = '/home/user/workspace/tinker-rl-lab/experiments/statistical_analysis.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON to {out_path}")

# ─── Generate Markdown Report ─────────────────────────────────────────────────
def fmt_p(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"

def significance_marker(p, alpha=0.05):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

exp_labels = {
    'scale_gsm8k_qwen3-8b': 'Qwen3-8B GRPO (Tinker)',
    'ppo_qwen3-8b': 'Qwen3-8B PPO',
    'ppo_llama-8b-inst': 'Llama-3.1-8B PPO',
    'frontier_gsm8k_deepseek-v3.1': 'DeepSeek-V3.1 GRPO (Frontier)',
}

md_lines = []
md_lines.append("# Statistical Analysis Report: Tinker RL Lab Experiments")
md_lines.append("")
md_lines.append("> **Data source:** `all_results_consolidated.json`  ")
md_lines.append("> **Bootstrap:** 10,000 resamples, 95% CI  ")
md_lines.append("> **Significance threshold:** α = 0.05  ")
md_lines.append("")
md_lines.append("---")
md_lines.append("")

# Section 1: Descriptive Statistics
md_lines.append("## 1. Descriptive Statistics")
md_lines.append("")
md_lines.append("| Experiment | N | Mean | Median | Std | Min | Max | IQR | Skewness | Kurtosis |")
md_lines.append("|:-----------|--:|-----:|-------:|----:|----:|----:|----:|---------:|---------:|")
for name, ds in descriptive.items():
    label = exp_labels.get(name, name)
    md_lines.append(
        f"| {label} | {ds['n']} | {ds['mean']:.4f} | {ds['median']:.4f} | "
        f"{ds['std']:.4f} | {ds['min']:.4f} | {ds['max']:.4f} | {ds['iqr']:.4f} | "
        f"{ds['skewness']:.3f} | {ds['kurtosis']:.3f} |"
    )
md_lines.append("")
md_lines.append("**Notes:**")
md_lines.append("- Std computed with Bessel's correction (ddof=1)")
md_lines.append("- Skewness > 0 indicates right-skewed distribution; < 0 indicates left-skewed")
md_lines.append("- Kurtosis is excess kurtosis (normal distribution = 0)")
md_lines.append("")

# Section 2: Bootstrap CIs
md_lines.append("## 2. Bootstrap Confidence Intervals (95%, 10,000 Resamples)")
md_lines.append("")
md_lines.append("| Experiment | Mean Reward | 95% CI (Mean) | Last-10 Avg | 95% CI (Last-10) |")
md_lines.append("|:-----------|------------:|:--------------|------------:|:-----------------|")
for name, br in bootstrap_results.items():
    label = exp_labels.get(name, name)
    ci_m = br['ci_mean']
    ci_l10 = br['ci_last10_avg']
    md_lines.append(
        f"| {label} | {ci_m['point_estimate']:.4f} | "
        f"[{ci_m['lower']:.4f}, {ci_m['upper']:.4f}] | "
        f"{ci_l10['point_estimate']:.4f} | "
        f"[{ci_l10['lower']:.4f}, {ci_l10['upper']:.4f}] |"
    )
md_lines.append("")

# Section 3: Effect Sizes
md_lines.append("## 3. Effect Sizes (Cohen's d)")
md_lines.append("")
md_lines.append("### 3a. Between-Method Comparisons")
md_lines.append("")
md_lines.append("| Comparison | Group A Mean | Group B Mean | Cohen's d | Magnitude |")
md_lines.append("|:-----------|-------------:|-------------:|----------:|:----------|")

if 'grpo_vs_ppo_qwen3-8b' in effect_sizes:
    e = effect_sizes['grpo_vs_ppo_qwen3-8b']
    md_lines.append(
        f"| GRPO Qwen3-8B vs PPO Qwen3-8B | {e['grpo_mean']:.4f} | {e['ppo_mean']:.4f} | "
        f"{e['cohens_d']:.4f} | {e['magnitude'].title()} |"
    )

if 'llama_ppo_vs_qwen_ppo' in effect_sizes:
    e = effect_sizes['llama_ppo_vs_qwen_ppo']
    md_lines.append(
        f"| PPO Llama-3.1-8B vs PPO Qwen3-8B | {e['llama_mean']:.4f} | {e['qwen_mean']:.4f} | "
        f"{e['cohens_d']:.4f} | {e['magnitude'].title()} |"
    )
md_lines.append("")

md_lines.append("### 3b. Early vs Late Training (First 10 vs Last 10 Steps)")
md_lines.append("")
md_lines.append("| Experiment | Early Mean | Late Mean | Cohen's d | Magnitude | Direction |")
md_lines.append("|:-----------|----------:|----------:|----------:|:----------|:----------|")
for name, e in effect_sizes.get('early_vs_late', {}).items():
    label = exp_labels.get(name, name)
    direction = "Improved" if e['cohens_d'] > 0.2 else ("Degraded" if e['cohens_d'] < -0.2 else "Stable")
    md_lines.append(
        f"| {label} | {e['early_mean']:.4f} | {e['late_mean']:.4f} | "
        f"{e['cohens_d']:.4f} | {e['magnitude'].title()} | {direction} |"
    )
md_lines.append("")
md_lines.append("*Positive Cohen's d = Late training outperforms early training*")
md_lines.append("")

# Section 4: Trend Analysis
md_lines.append("## 4. Trend Analysis")
md_lines.append("")
md_lines.append("### 4a. Mann-Kendall Trend Test")
md_lines.append("")
md_lines.append("| Experiment | Kendall τ | S Statistic | Z Score | p-value | Sig | Trend |")
md_lines.append("|:-----------|----------:|------------:|--------:|--------:|:---:|:------|")
for name, tr in trend_results.items():
    label = exp_labels.get(name, name)
    mk = tr['mann_kendall']
    md_lines.append(
        f"| {label} | {mk['tau']:.4f} | {mk['S']} | {mk['z']:.3f} | "
        f"{fmt_p(mk['p_value'])} | {significance_marker(mk['p_value'])} | {mk['trend'].title()} |"
    )
md_lines.append("")

md_lines.append("### 4b. Linear Regression Trend")
md_lines.append("")
md_lines.append("| Experiment | Slope | Intercept | R² | p-value | Sig |")
md_lines.append("|:-----------|------:|----------:|---:|--------:|:---:|")
for name, tr in trend_results.items():
    label = exp_labels.get(name, name)
    lr = tr['linear_regression']
    md_lines.append(
        f"| {label} | {lr['slope']:.6f} | {lr['intercept']:.4f} | "
        f"{lr['r_squared']:.4f} | {fmt_p(lr['p_value'])} | {significance_marker(lr['p_value'])} |"
    )
md_lines.append("")
md_lines.append("*Slope units: reward per training step. Positive slope = upward trend.*")
md_lines.append("")

# Section 5: Volatility
md_lines.append("## 5. Volatility Analysis")
md_lines.append("")
md_lines.append("| Experiment | CV | Max Drawdown | Mean Rolling Std (w=5) |")
md_lines.append("|:-----------|---:|-------------:|-----------------------:|")
for name, vol in volatility_results.items():
    label = exp_labels.get(name, name)
    cv = vol['coefficient_of_variation']
    mdd = vol['max_drawdown']
    mrs = vol['mean_rolling_std']
    md_lines.append(
        f"| {label} | {cv:.4f} | {mdd:.4f} | {mrs:.4f} |"
    )
md_lines.append("")
md_lines.append("**Definitions:**")
md_lines.append("- **CV** (Coefficient of Variation) = σ/|μ|; lower = more stable training")
md_lines.append("- **Max Drawdown** = largest peak-to-trough reward decline")
md_lines.append("- **Mean Rolling Std** = average local volatility over 5-step windows")
md_lines.append("")

# Rolling std detail table
md_lines.append("### 5a. Rolling Std (Window=5) Values")
md_lines.append("")
md_lines.append("*Values shown for steps 5 onward (first 4 steps have insufficient window)*")
md_lines.append("")
# Show in a compact format per experiment
for name, vol in volatility_results.items():
    label = exp_labels.get(name, name)
    valid = [(i+1, v) for i, v in enumerate(vol['rolling_std_window5']) if v is not None]
    step_vals = " | ".join([f"Step {s}: {v:.4f}" for s, v in valid[:10]])
    md_lines.append(f"**{label}:** {step_vals}")
    md_lines.append("")

# Section 6: Cross-Seed Analysis
md_lines.append("## 6. Cross-Seed Analysis (TRL GRPO, Qwen2.5-0.5B, 5 Seeds)")
md_lines.append("")
md_lines.append(f"**Seeds:** {cross_seed_results['seeds']}  ")
md_lines.append(f"**Accuracies:** {[round(float(a), 3) for a in cross_seed_results['accuracies']]}  ")
md_lines.append(f"**Model:** {cross_seed_results['model']} | **GPU:** L4 | **Steps:** 125")
md_lines.append("")
cs_ds = cross_seed_results['descriptive']
ci_cs = cross_seed_results['ci_mean_95_bootstrap']
ttest = cross_seed_results['ttest_vs_0.5']
md_lines.append("| Statistic | Value |")
md_lines.append("|:----------|------:|")
md_lines.append(f"| Mean | {cs_ds['mean']:.4f} |")
md_lines.append(f"| Median | {cs_ds['median']:.4f} |")
md_lines.append(f"| Std | {cs_ds['std']:.4f} |")
md_lines.append(f"| Min | {cs_ds['min']:.4f} |")
md_lines.append(f"| Max | {cs_ds['max']:.4f} |")
md_lines.append(f"| IQR | {cs_ds['iqr']:.4f} |")
md_lines.append(f"| CV | {cross_seed_results['coefficient_of_variation']:.4f} |")
md_lines.append(f"| 95% CI (Bootstrap) | [{ci_cs['lower']:.4f}, {ci_cs['upper']:.4f}] |")
md_lines.append("")
md_lines.append("### One-Sample t-test: Is Mean Significantly > 0.5?")
md_lines.append("")
md_lines.append("| Test Parameter | Value |")
md_lines.append("|:---------------|------:|")
md_lines.append(f"| H₀ | μ = 0.5 |")
md_lines.append(f"| H₁ | μ > 0.5 |")
md_lines.append(f"| t-statistic | {ttest['t_statistic']:.4f} |")
md_lines.append(f"| p-value (two-sided) | {fmt_p(ttest['p_value_two_sided'])} |")
md_lines.append(f"| p-value (one-sided) | {fmt_p(ttest['p_value_one_sided'])} |")
md_lines.append(f"| Significant (α=0.05) | {'**Yes** ✓' if ttest['significant_greater_than_0.5'] else 'No'} |")
md_lines.append("")
if ttest['significant_greater_than_0.5']:
    md_lines.append(f"> **Conclusion:** The mean accuracy ({cs_ds['mean']:.3f}) is statistically significantly "
                    f"greater than 0.5 (p = {fmt_p(ttest['p_value_one_sided'])}).")
else:
    md_lines.append(f"> **Conclusion:** Cannot reject H₀ that mean = 0.5 (p = {fmt_p(ttest['p_value_one_sided'])}).")
md_lines.append("")

# Section 7: Comparative Tests
md_lines.append("## 7. Comparative Statistical Tests")
md_lines.append("")

if 'wilcoxon_grpo_vs_ppo_qwen3-8b' in comparative_results:
    w = comparative_results['wilcoxon_grpo_vs_ppo_qwen3-8b']
    md_lines.append("### 7a. Wilcoxon Signed-Rank Test: GRPO vs PPO (Qwen3-8B, paired by step)")
    md_lines.append("")
    md_lines.append("| Parameter | Value |")
    md_lines.append("|:----------|------:|")
    md_lines.append(f"| GRPO Mean | {w['grpo_mean']:.4f} |")
    md_lines.append(f"| PPO Mean | {w['ppo_mean']:.4f} |")
    md_lines.append(f"| N pairs | {w['n_pairs']} |")
    if w.get('statistic') is not None:
        md_lines.append(f"| W statistic | {w['statistic']:.1f} |")
    md_lines.append(f"| p-value | {fmt_p(w['p_value'])} |")
    md_lines.append(f"| Significant | {'**Yes** ✓' if w['significant'] else 'No'} |")
    if 'note' in w:
        md_lines.append(f"| Note | {w['note']} |")
    md_lines.append("")
    if w['significant']:
        md_lines.append(f"> **Conclusion:** GRPO and PPO on Qwen3-8B show significantly different reward distributions (p = {fmt_p(w['p_value'])}).")
    else:
        md_lines.append(f"> **Conclusion:** No statistically significant difference detected between GRPO and PPO on Qwen3-8B (p = {fmt_p(w['p_value'])}).")
    md_lines.append("")

if 'mann_whitney_llama_vs_qwen_ppo' in comparative_results:
    mw = comparative_results['mann_whitney_llama_vs_qwen_ppo']
    md_lines.append("### 7b. Mann-Whitney U Test: PPO Llama-3.1-8B vs PPO Qwen3-8B")
    md_lines.append("")
    md_lines.append("| Parameter | Value |")
    md_lines.append("|:----------|------:|")
    md_lines.append(f"| Llama-8B Mean | {mw['llama_mean']:.4f} |")
    md_lines.append(f"| Qwen3-8B Mean | {mw['qwen_mean']:.4f} |")
    md_lines.append(f"| N (Llama) | {mw['n_llama']} |")
    md_lines.append(f"| N (Qwen) | {mw['n_qwen']} |")
    md_lines.append(f"| U statistic | {mw['statistic']:.1f} |")
    md_lines.append(f"| Rank-biserial r | {mw['rank_biserial_r']:.4f} |")
    md_lines.append(f"| p-value | {fmt_p(mw['p_value'])} |")
    md_lines.append(f"| Significant | {'**Yes** ✓' if mw['significant'] else 'No'} |")
    md_lines.append("")
    if mw['significant']:
        md_lines.append(f"> **Conclusion:** Llama-3.1-8B PPO (mean={mw['llama_mean']:.4f}) significantly outperforms "
                        f"Qwen3-8B PPO (mean={mw['qwen_mean']:.4f}) on gsm8k (p = {fmt_p(mw['p_value'])}, r = {mw['rank_biserial_r']:.4f}).")
    else:
        md_lines.append(f"> **Conclusion:** No significant difference detected between PPO models (p = {fmt_p(mw['p_value'])}).")
    md_lines.append("")

# Summary
md_lines.append("---")
md_lines.append("")
md_lines.append("## Summary of Key Findings")
md_lines.append("")
md_lines.append("| Finding | Result |")
md_lines.append("|:--------|:-------|")

# Compile key findings
for name, mk_lr in trend_results.items():
    mk = mk_lr['mann_kendall']
    label = exp_labels.get(name, name)
    sig = "Significant upward trend" if (mk['significant'] and mk['trend'] == 'increasing') else \
          "Significant downward trend" if (mk['significant'] and mk['trend'] == 'decreasing') else "No significant trend"
    md_lines.append(f"| {label} trend (MK test) | {sig} (p = {fmt_p(mk['p_value'])}) |")

if 'grpo_vs_ppo_qwen3-8b' in effect_sizes:
    e = effect_sizes['grpo_vs_ppo_qwen3-8b']
    md_lines.append(f"| GRPO vs PPO Qwen3-8B | d = {e['cohens_d']:.3f} ({e['magnitude']} effect); GRPO mean={e['grpo_mean']:.3f}, PPO mean={e['ppo_mean']:.3f} |")

if 'llama_ppo_vs_qwen_ppo' in effect_sizes:
    e = effect_sizes['llama_ppo_vs_qwen_ppo']
    md_lines.append(f"| Llama vs Qwen PPO | d = {e['cohens_d']:.3f} ({e['magnitude']} effect); Llama={e['llama_mean']:.3f}, Qwen={e['qwen_mean']:.3f} |")

cs_ttest = cross_seed_results['ttest_vs_0.5']
md_lines.append(f"| TRL GRPO cross-seed mean > 0.5 | {'Yes' if cs_ttest['significant_greater_than_0.5'] else 'No'} (p = {fmt_p(cs_ttest['p_value_one_sided'])}) |")
md_lines.append(f"| TRL GRPO cross-seed CV | {cross_seed_results['coefficient_of_variation']:.4f} (moderate seed-to-seed variability) |")
md_lines.append("")

# Caveats
md_lines.append("## Caveats and Limitations")
md_lines.append("")
md_lines.append("1. **Small sample sizes:** Reward traces have only 20–30 steps; statistical power is limited.")
md_lines.append("2. **Non-independence:** Consecutive training steps are correlated (autocorrelation), which may inflate trend test significance.")
md_lines.append("3. **Bootstrap CIs for last-10 avg** are based on n=10 samples only — treat as indicative, not definitive.")
md_lines.append("4. **GRPO identification:** `scale_gsm8k_qwen3-8b` is treated as the GRPO trace for Qwen3-8B (Tinker platform runs GRPO); no explicit step-level GRPO trace is stored in the consolidated JSON for the TRL GRPO multi-seed experiment.")
md_lines.append("5. **Multiple comparisons:** Seven+ hypothesis tests performed without correction; individual p-values should be interpreted cautiously.")
md_lines.append("")

md_text = "\n".join(md_lines)
md_path = '/home/user/workspace/tinker-rl-lab/experiments/statistical_analysis.md'
with open(md_path, 'w') as f:
    f.write(md_text)
print(f"Saved Markdown to {md_path}")
print("\n=== DONE ===")

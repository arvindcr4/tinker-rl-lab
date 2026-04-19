# Statistical Analysis Report: Tinker RL Lab Experiments

> **Data source:** `all_results_consolidated.json`  
> **Bootstrap:** 10,000 resamples, 95% CI  
> **Significance threshold:** α = 0.05  

---

## 1. Descriptive Statistics

| Experiment | N | Mean | Median | Std | Min | Max | IQR | Skewness | Kurtosis |
|:-----------|--:|-----:|-------:|----:|----:|----:|----:|---------:|---------:|
| DeepSeek-V3.1 GRPO (Frontier) | 20 | 0.8500 | 0.8750 | 0.1042 | 0.6250 | 1.0000 | 0.1250 | -0.179 | -0.559 |
| Qwen3-8B GRPO (Tinker) | 30 | 0.3208 | 0.3125 | 0.1466 | 0.0625 | 0.6250 | 0.2500 | 0.098 | -0.872 |
| Qwen3-8B PPO | 30 | 0.2833 | 0.2500 | 0.2842 | 0.0000 | 1.0000 | 0.5000 | 0.596 | -0.506 |
| Llama-3.1-8B PPO | 30 | 0.9500 | 1.0000 | 0.1211 | 0.5000 | 1.0000 | 0.0000 | -2.372 | 4.900 |

**Notes:**
- Std computed with Bessel's correction (ddof=1)
- Skewness > 0 indicates right-skewed distribution; < 0 indicates left-skewed
- Kurtosis is excess kurtosis (normal distribution = 0)

## 2. Bootstrap Confidence Intervals (95%, 10,000 Resamples)

| Experiment | Mean Reward | 95% CI (Mean) | Last-10 Avg | 95% CI (Last-10) |
|:-----------|------------:|:--------------|------------:|:-----------------|
| DeepSeek-V3.1 GRPO (Frontier) | 0.8500 | [0.8063, 0.8938] | 0.8625 | [0.8125, 0.9125] |
| Qwen3-8B GRPO (Tinker) | 0.3208 | [0.2687, 0.3729] | 0.3187 | [0.2500, 0.3875] |
| Qwen3-8B PPO | 0.2833 | [0.1833, 0.3833] | 0.3500 | [0.1750, 0.5250] |
| Llama-3.1-8B PPO | 0.9500 | [0.9000, 0.9917] | 0.9500 | [0.8750, 1.0000] |

## 3. Effect Sizes (Cohen's d)

### 3a. Between-Method Comparisons

| Comparison | Group A Mean | Group B Mean | Cohen's d | Magnitude |
|:-----------|-------------:|-------------:|----------:|:----------|
| GRPO Qwen3-8B vs PPO Qwen3-8B | 0.3208 | 0.2833 | 0.1659 | Negligible |
| PPO Llama-3.1-8B vs PPO Qwen3-8B | 0.9500 | 0.2833 | 3.0524 | Large |

### 3b. Early vs Late Training (First 10 vs Last 10 Steps)

| Experiment | Early Mean | Late Mean | Cohen's d | Magnitude | Direction |
|:-----------|----------:|----------:|----------:|:----------|:----------|
| DeepSeek-V3.1 GRPO (Frontier) | 0.8375 | 0.8625 | 0.2353 | Small | Improved |
| Qwen3-8B GRPO (Tinker) | 0.2437 | 0.3187 | 0.5373 | Medium | Improved |
| Qwen3-8B PPO | 0.3250 | 0.3500 | 0.0795 | Negligible | Stable |
| Llama-3.1-8B PPO | 0.9750 | 0.9500 | -0.2683 | Small | Degraded |

*Positive Cohen's d = Late training outperforms early training*

## 4. Trend Analysis

### 4a. Mann-Kendall Trend Test

| Experiment | Kendall τ | S Statistic | Z Score | p-value | Sig | Trend |
|:-----------|----------:|------------:|--------:|--------:|:---:|:------|
| DeepSeek-V3.1 GRPO (Frontier) | 0.0579 | 11 | 0.349 | 0.7271 | ns | Increasing |
| Qwen3-8B GRPO (Tinker) | 0.1310 | 57 | 1.009 | 0.3132 | ns | Increasing |
| Qwen3-8B PPO | -0.0115 | -5 | -0.075 | 0.9398 | ns | Decreasing |
| Llama-3.1-8B PPO | -0.0437 | -19 | -0.499 | 0.6176 | ns | Decreasing |

### 4b. Linear Regression Trend

| Experiment | Slope | Intercept | R² | p-value | Sig |
|:-----------|------:|----------:|---:|--------:|:---:|
| DeepSeek-V3.1 GRPO (Frontier) | 0.001880 | 0.8321 | 0.0114 | 0.6542 | ns |
| Qwen3-8B GRPO (Tinker) | 0.002948 | 0.2781 | 0.0314 | 0.3493 | ns |
| Qwen3-8B PPO | -0.001446 | 0.3043 | 0.0020 | 0.8142 | ns |
| Llama-3.1-8B PPO | -0.000779 | 0.9613 | 0.0032 | 0.7663 | ns |

*Slope units: reward per training step. Positive slope = upward trend.*

## 5. Volatility Analysis

| Experiment | CV | Max Drawdown | Mean Rolling Std (w=5) |
|:-----------|---:|-------------:|-----------------------:|
| DeepSeek-V3.1 GRPO (Frontier) | 0.1226 | 0.3750 | 0.1088 |
| Qwen3-8B GRPO (Tinker) | 0.4568 | 0.5000 | 0.1315 |
| Qwen3-8B PPO | 1.0029 | 1.0000 | 0.2770 |
| Llama-3.1-8B PPO | 0.1274 | 0.5000 | 0.1118 |

**Definitions:**
- **CV** (Coefficient of Variation) = σ/|μ|; lower = more stable training
- **Max Drawdown** = largest peak-to-trough reward decline
- **Mean Rolling Std** = average local volatility over 5-step windows

### 5a. Rolling Std (Window=5) Values

*Values shown for steps 5 onward (first 4 steps have insufficient window)*

**DeepSeek-V3.1 GRPO (Frontier):** Step 5: 0.1046 | Step 6: 0.1118 | Step 7: 0.1369 | Step 8: 0.0884 | Step 9: 0.1046 | Step 10: 0.1425 | Step 11: 0.1425 | Step 12: 0.1046 | Step 13: 0.1046 | Step 14: 0.1046

**Qwen3-8B GRPO (Tinker):** Step 5: 0.1926 | Step 6: 0.1896 | Step 7: 0.1712 | Step 8: 0.1135 | Step 9: 0.1135 | Step 10: 0.0815 | Step 11: 0.0559 | Step 12: 0.1296 | Step 13: 0.2044 | Step 14: 0.2044

**Qwen3-8B PPO:** Step 5: 0.4183 | Step 6: 0.4183 | Step 7: 0.4472 | Step 8: 0.4472 | Step 9: 0.4183 | Step 10: 0.2500 | Step 11: 0.2500 | Step 12: 0.2500 | Step 13: 0.2236 | Step 14: 0.2236

**Llama-3.1-8B PPO:** Step 5: 0.1118 | Step 6: 0.1118 | Step 7: 0.1118 | Step 8: 0.1118 | Step 9: 0.0000 | Step 10: 0.0000 | Step 11: 0.0000 | Step 12: 0.2236 | Step 13: 0.2236 | Step 14: 0.2236

## 6. Cross-Seed Analysis (TRL GRPO, Qwen2.5-0.5B, 5 Seeds)

**Seeds:** [42, 123, 456, 789, 1024]  
**Accuracies:** [0.735, 0.81, 0.62, 0.74, 0.765]  
**Model:** Qwen2.5-0.5B | **GPU:** L4 | **Steps:** 125

| Statistic | Value |
|:----------|------:|
| Mean | 0.7340 |
| Median | 0.7400 |
| Std | 0.0703 |
| Min | 0.6200 |
| Max | 0.8100 |
| IQR | 0.0300 |
| CV | 0.0958 |
| 95% CI (Bootstrap) | [0.6720, 0.7820] |

### One-Sample t-test: Is Mean Significantly > 0.5?

| Test Parameter | Value |
|:---------------|------:|
| H₀ | μ = 0.5 |
| H₁ | μ > 0.5 |
| t-statistic | 7.4426 |
| p-value (two-sided) | 0.0017 |
| p-value (one-sided) | < 0.001 |
| Significant (α=0.05) | **Yes** ✓ |

> **Conclusion:** The mean accuracy (0.734) is statistically significantly greater than 0.5 (p = < 0.001).

## 7. Comparative Statistical Tests

### 7a. Wilcoxon Signed-Rank Test: GRPO vs PPO (Qwen3-8B, paired by step)

| Parameter | Value |
|:----------|------:|
| GRPO Mean | 0.3208 |
| PPO Mean | 0.2833 |
| N pairs | 30 |
| W statistic | 202.0 |
| p-value | 0.5296 |
| Significant | No |

> **Conclusion:** No statistically significant difference detected between GRPO and PPO on Qwen3-8B (p = 0.5296).

### 7b. Mann-Whitney U Test: PPO Llama-3.1-8B vs PPO Qwen3-8B

| Parameter | Value |
|:----------|------:|
| Llama-8B Mean | 0.9500 |
| Qwen3-8B Mean | 0.2833 |
| N (Llama) | 30 |
| N (Qwen) | 30 |
| U statistic | 872.0 |
| Rank-biserial r | -0.9378 |
| p-value | < 0.001 |
| Significant | **Yes** ✓ |

> **Conclusion:** Llama-3.1-8B PPO (mean=0.9500) significantly outperforms Qwen3-8B PPO (mean=0.2833) on gsm8k (p = < 0.001, r = -0.9378).

---

## Summary of Key Findings

| Finding | Result |
|:--------|:-------|
| DeepSeek-V3.1 GRPO (Frontier) trend (MK test) | No significant trend (p = 0.7271) |
| Qwen3-8B GRPO (Tinker) trend (MK test) | No significant trend (p = 0.3132) |
| Qwen3-8B PPO trend (MK test) | No significant trend (p = 0.9398) |
| Llama-3.1-8B PPO trend (MK test) | No significant trend (p = 0.6176) |
| GRPO vs PPO Qwen3-8B | d = 0.166 (negligible effect); GRPO mean=0.321, PPO mean=0.283 |
| Llama vs Qwen PPO | d = 3.052 (large effect); Llama=0.950, Qwen=0.283 |
| TRL GRPO cross-seed mean > 0.5 | Yes (p = < 0.001) |
| TRL GRPO cross-seed CV | 0.0958 (moderate seed-to-seed variability) |

## Caveats and Limitations

1. **Small sample sizes:** Reward traces have only 20–30 steps; statistical power is limited.
2. **Non-independence:** Consecutive training steps are correlated (autocorrelation), which may inflate trend test significance.
3. **Bootstrap CIs for last-10 avg** are based on n=10 samples only — treat as indicative, not definitive.
4. **GRPO identification:** `scale_gsm8k_qwen3-8b` is treated as the GRPO trace for Qwen3-8B (Tinker platform runs GRPO); no explicit step-level GRPO trace is stored in the consolidated JSON for the TRL GRPO multi-seed experiment.
5. **Multiple comparisons:** Seven+ hypothesis tests performed without correction; individual p-values should be interpreted cautiously.

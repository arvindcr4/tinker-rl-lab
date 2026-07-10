# ZVF Lagged Regression Analysis

## Purpose

This analysis addresses the reviewer objection that ZVF is tautologically
correlated with reward on binary-reward tasks (since ZVF = p^G + (1-p)^G
is a deterministic function of accuracy p).

If ZVF is merely a restatement of accuracy, then after controlling for
current reward R_t, ZVF_t should have NO additional predictive power for
future reward R_{t+k}. If ZVF provides independent early-warning signal,
then the coefficient on ZVF_t should remain significant after controlling
for R_t.

## Data: 33 runs with step-level ZVF/GU telemetry

| Model | Task | Steps |
|---|---|---:|
| llama-8b-inst | tool_use | 30 |
| qwen3-32b | tool_use | 30 |
| deepseek-v3.1 | gsm8k | 20 |
| nemotron-120b | gsm8k | 20 |
| llama-8b-inst | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3.5-4b | gsm8k | 30 |
| gpt-oss-20b | gsm8k | 30 |
| kimi-k2 | gsm8k | 20 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| unknown | unknown | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |
| qwen3-8b | gsm8k | 30 |

## Lag k = 1

### Cross-Run Pooled Regression (N = 927)

| Model | R² | Adj. R² | ZVF coef | ZVF SE | ZVF p-value |
|---|---|---|---|---|---|
| Reward only | 0.295 | 0.294 | — | — | — |
| Reward + ZVF | 0.296 | 0.295 | 0.031 | 0.021 | 0.1303 |
| Reward + ZVF + Interaction | 0.301 | 0.299 | -0.030 | — | 0.3345 |

**Delta R² from adding ZVF:** 0.002
**Delta Adj. R²:** 0.001
**Incremental F-test (H0: ZVF coef = 0):** F = 2.293, p = 0.1303

### Within-Run Regression (controls for run-level confounders)

- Runs with sufficient data: 33
- Mean ZVF coefficient: 0.000
- Std ZVF coefficient: 0.171
- t-test (H0: mean ZVF coef = 0): t = 0.003, p = 0.9973
- Fraction of runs with negative ZVF coefficient: 0.455
- Mean ΔR² from adding ZVF: 0.052

### Partial Correlation Analysis

- Zero-order r(ZVF_t, R_{t+1}): -0.096
- Zero-order r(ZVF_t, R_t): -0.252
- Partial r(ZVF_t, R_{t+1} | R_t): 0.050 (p = 0.1303)

## Lag k = 3

### Cross-Run Pooled Regression (N = 861)

| Model | R² | Adj. R² | ZVF coef | ZVF SE | ZVF p-value |
|---|---|---|---|---|---|
| Reward only | 0.232 | 0.231 | — | — | — |
| Reward + ZVF | 0.232 | 0.230 | -0.004 | 0.022 | 0.8649 |
| Reward + ZVF + Interaction | 0.242 | 0.239 | -0.092 | — | 0.0074 |

**Delta R² from adding ZVF:** 0.000
**Delta Adj. R²:** -0.001
**Incremental F-test (H0: ZVF coef = 0):** F = 0.029, p = 0.8649

### Within-Run Regression (controls for run-level confounders)

- Runs with sufficient data: 33
- Mean ZVF coefficient: -0.052
- Std ZVF coefficient: 0.158
- t-test (H0: mean ZVF coef = 0): t = -1.875, p = 0.0700
- Fraction of runs with negative ZVF coefficient: 0.515
- Mean ΔR² from adding ZVF: 0.045

### Partial Correlation Analysis

- Zero-order r(ZVF_t, R_{t+3}): -0.129
- Zero-order r(ZVF_t, R_t): -0.258
- Partial r(ZVF_t, R_{t+3} | R_t): -0.006 (p = 0.8649)

## Lag k = 5

### Cross-Run Pooled Regression (N = 795)

| Model | R² | Adj. R² | ZVF coef | ZVF SE | ZVF p-value |
|---|---|---|---|---|---|
| Reward only | 0.248 | 0.247 | — | — | — |
| Reward + ZVF | 0.251 | 0.249 | 0.038 | 0.023 | 0.0964 |
| Reward + ZVF + Interaction | 0.256 | 0.254 | -0.029 | — | 0.4048 |

**Delta R² from adding ZVF:** 0.003
**Delta Adj. R²:** 0.002
**Incremental F-test (H0: ZVF coef = 0):** F = 2.771, p = 0.0964

### Within-Run Regression (controls for run-level confounders)

- Runs with sufficient data: 33
- Mean ZVF coefficient: -0.007
- Std ZVF coefficient: 0.132
- t-test (H0: mean ZVF coef = 0): t = -0.320, p = 0.7513
- Fraction of runs with negative ZVF coefficient: 0.424
- Mean ΔR² from adding ZVF: 0.029

### Partial Correlation Analysis

- Zero-order r(ZVF_t, R_{t+5}): -0.075
- Zero-order r(ZVF_t, R_t): -0.251
- Partial r(ZVF_t, R_{t+5} | R_t): 0.059 (p = 0.0964)

## Lag k = 10

### Cross-Run Pooled Regression (N = 630)

| Model | R² | Adj. R² | ZVF coef | ZVF SE | ZVF p-value |
|---|---|---|---|---|---|
| Reward only | 0.237 | 0.236 | — | — | — |
| Reward + ZVF | 0.242 | 0.240 | 0.052 | 0.026 | 0.0453 |
| Reward + ZVF + Interaction | 0.256 | 0.252 | -0.049 | — | 0.2096 |

**Delta R² from adding ZVF:** 0.005
**Delta Adj. R²:** 0.004
**Incremental F-test (H0: ZVF coef = 0):** F = 4.023, p = 0.0453

### Within-Run Regression (controls for run-level confounders)

- Runs with sufficient data: 33
- Mean ZVF coefficient: 0.049
- Std ZVF coefficient: 0.198
- t-test (H0: mean ZVF coef = 0): t = 1.423, p = 0.1644
- Fraction of runs with negative ZVF coefficient: 0.333
- Mean ΔR² from adding ZVF: 0.066

### Partial Correlation Analysis

- Zero-order r(ZVF_t, R_{t+10}): -0.061
- Zero-order r(ZVF_t, R_t): -0.264
- Partial r(ZVF_t, R_{t+10} | R_t): 0.080 (p = 0.0453)

## Verdict

**ZVF does NOT provide statistically significant incremental predictive power** beyond current reward in the cross-run pooled analysis.
Adding ZVF to the model increases R² by only 0.003 (incremental F-test p = 0.0964).

Within-run analysis: mean ZVF coefficient is NOT significantly non-zero (p = 0.7513), suggesting ZVF's predictive power may be driven by cross-run confounders rather than within-trajectory dynamics.

Partial correlation analysis: r(ZVF_t, R_{t+5} | R_t) = 0.059 (p = 0.0964), suggesting ZVF's zero-order correlation with future reward is largely explained by current reward.

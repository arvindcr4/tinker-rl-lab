# E-R2b held-out evaluation (Colab TRL adapters, Colab vLLM sampler)

Pass@k is recomputed from `per_problem_c` with the unbiased estimator in `tinker_passk.pass_at_k`, then averaged over problems. Deltas are adapter minus base on the same problem order. The 95% interval is a paired percentile bootstrap over problems (10000 resamples, seed 0). The p-value is a two-sided sign-flip test over problems (10000 flips, seed 0), `(#{|T*| >= |T|} + 1) / (n_perm + 1)`.

## Held-out

| model | n | pass@1 | pass@8 | pass@32 | truncation_rate | mean_tokens | Δpass@1 | 95% CI | p | Δpass@32 | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|
| base | 32 | 0.305781 | 0.780868 | 0.895000 | 0.992969 | 511.6242 | 0 | [0, 0] | 1.000000 | 0 | [0, 0] | 1.000000 |
| er2b_trl_g2_s123 | 32 | 0.487969 | 0.859252 | 0.935000 | 0.954688 | 508.7830 | 0.182188 | [0.158438, 0.206094] | 0.000100 | 0.040000 | [0.010000, 0.075000] | 0.041096 |
| er2b_trl_g2_s456 | 32 | 0.477187 | 0.851128 | 0.930000 | 0.950625 | 508.8795 | 0.171406 | [0.148594, 0.194531] | 0.000100 | 0.035000 | [0.005000, 0.070000] | 0.067393 |
| er2b_trl_g16_s123 | 32 | 0.347813 | 0.799293 | 0.895000 | 0.985156 | 511.1184 | 0.042031 | [0.027656, 0.056250] | 0.000100 | 0 | [-0.025000, 0.030000] | 1.000000 |
| er2b_trl_g16_s456 | 32 | 0.363438 | 0.814936 | 0.920000 | 0.982812 | 510.9575 | 0.057656 | [0.040781, 0.074375] | 0.000100 | 0.025000 | [-0.010000, 0.060000] | 0.270673 |

## Pooled arms

G=2 pools seeds 123 and 456. G=16 pools seeds 123 and 456. For each problem the arm score is the unweighted mean of the seeds' unbiased pass@k, then the same paired delta, interval, and sign-flip test are taken against base.

| arm | seeds | pass@1 | pass@8 | pass@32 | truncation_rate | mean_tokens | Δpass@1 | 95% CI | p | Δpass@32 | 95% CI | p |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|
| G=2 | 123, 456 | 0.482578 | 0.855190 | 0.932500 | 0.952657 | 508.8313 | 0.176797 | [0.154529, 0.199063] | 0.000100 | 0.037500 | [0.010000, 0.070000] | 0.023998 |
| G=16 | 123, 456 | 0.355625 | 0.807115 | 0.907500 | 0.983984 | 511.0380 | 0.049844 | [0.036250, 0.063672] | 0.000100 | 0.012500 | [-0.012500, 0.040000] | 0.481352 |

## Training reproduction

Last-10 train reward is `last10_avg` when that field is present, otherwise the mean of the last 10 `reward_trace` values (or `step_log[].reward` if the trace is absent). Final-20 ZVF is the mean of the last 20 `step_log[].zvf` values. A shorter log is averaged over the steps that exist and marked with `n`. A TRL run is incomplete when `status` is not `completed` or logged steps are below planned `steps`.

| arm | G | seed | july last10 | TRL last10 | july ZVF final20 | TRL ZVF final20 | july steps | TRL steps | TRL incomplete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G=2 s=123 | 2 | 123 | 0.900000 | 0.462500 | 0.975000 | 0.662500 | 160/160 | 160/160 | no |
| G=2 s=456 | 2 | 456 | 0.962500 | 0.425000 | 0.981250 | 0.687500 | 160/160 | 160/160 | no |
| G=16 s=123 | 16 | 123 | 0.321875 | 0.283594 | 0.193750 | 0.212500 | 20/20 | 20/20 | no |
| G=16 s=456 | 16 | 456 | 0.389062 | 0.353906 | 0.112500 | 0.118750 | 20/20 | 20/20 | no |

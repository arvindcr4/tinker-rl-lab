# Breakthroughs Hidden in the Tinker RL Data

Date: 2026-07-13

This audit connects deployment reliability, GRPO trainability, and adaptive
group-size control using the repository's existing experiment data.

## 1. Deployment reliability and GRPO trainability are the same curve

For a task with success probability `p` and group size `k`:

- `pass^k = E[p^k]`
- `pass@k = 1 - E[(1-p)^k]`
- `ZVF = E[p^k + (1-p)^k]`

Therefore:

```text
pass@k - pass^k = 1 - ZVF
```

The best-of-versus-reliability "scissor gap" is exactly the probability that a
rollout group contains both successes and failures. It is therefore also the
probability of obtaining usable contrastive GRPO signal.

The identity was checked on 505 unique `(seed, problem)` observations for group
sizes 2 through 16. The maximum numerical discrepancy was
`1.11e-16`—machine precision.

## 2. G=4 is a robust frontier optimum, not a global sweet spot

Within the 505-task uncertain-middle cohort, raw contrastive yield increases as
group size grows, but compute efficiency declines. Using
`(1 - ZVF) / sqrt(G)` as an SNR-adjusted utility proxy, G=4 is optimal.

- G=4 utility: `0.32914`
- G=5 utility: `0.32737`
- Difference: `0.00177`
- 95% task-bootstrap interval: `[0.00082, 0.00270]`
- Bootstrap selection: G=4 won `5,000 / 5,000` resamples

This result is conditional. The cohort excludes prompts near `p=0` and `p=1`,
so it does not overturn the existing result that there is no robust global G=4
sweet spot. It suggests using G=4 specifically for prompts classified near the
learning frontier.

## 3. Deployment reliability can improve while trainability deteriorates

Two task distributions with nearly identical mean success probability
(`0.674`) behave very differently when task-level variance increases:

| Distribution | Variance | pass^5 | ZVF | Contrastive yield |
|---|---:|---:|---:|---:|
| Observed uncertain middle | 0.0369 | 0.235 | 0.268 | 0.732 |
| Equal-mean polarized | 0.1140 | 0.520 | 0.588 | 0.412 |

The polarized benchmark looks much more reliable under `pass^5`, but it is
substantially harder to train with grouped contrast because more tasks become
effectively always-pass or always-fail.

Mean accuracy alone therefore cannot characterize either deployment
reliability or RL trainability. The task-difficulty distribution is a
load-bearing part of both measurements.

## 4. The adaptive-G controller spends most escalation on the easy tail

The empirical G-prime controller dataset contains the expected
`4 methods x 40 steps x 16 prompts = 2,560` observations. It fires on 1,867 of
them.

| Saturation boundary | Fires | Share | Mean recovered contrast |
|---|---:|---:|---:|
| All correct | 1,723 | 92.3% | 0.513 |
| All wrong | 144 | 7.7% | 0.634 |

The controller treats all-correct and all-wrong saturation symmetrically, but
92.3% of its escalation decisions target all-correct prompts. All-wrong prompts
also recover more contrast per escalation.

An all-wrong-only controller would remove 92.3% of the current escalation
overhead. This is an experiment proposal, not yet a performance claim: held-out
accuracy preservation must be tested directly.

## Recommended next experiment

Run a seed-paired three-arm controller ablation:

1. Current symmetric escalation.
2. All-wrong-only escalation.
3. All-wrong escalation plus all-correct retirement or distillation.

Pre-register:

- held-out accuracy;
- contrastive groups per 1,000 rollouts;
- rollout and wall-clock cost;
- all-correct versus all-wrong saturation composition;
- results by prompt-difficulty band.

Use G=4 only for prompts classified into the uncertain-middle band. Preserve the
existing no-global-sweet-spot conclusion until multi-model and held-out evidence
supports something stronger.

## Evidence and reproducibility

Primary inputs:

- `platform_hybrid/experiments/results/zvf_iter46_per_prompt_isog.tsv`
- `platform_hybrid/experiments/results/p5p8/p7_iter203_emp_per_obs.tsv`
- `platform_hybrid/experiments/results/berkeley/passk_reliability_curve.tsv`

Reproducible audit:

- `analysis/breakthroughs_2026-07-13/analyze_breakthroughs.py`
- `analysis/breakthroughs_2026-07-13/summary.json`
- `analysis/breakthroughs_2026-07-13/duality_curve.tsv`
- `analysis/breakthroughs_2026-07-13/polarization_paradox.tsv`
- `analysis/breakthroughs_2026-07-13/controller_boundary_summary.tsv`

Data checks:

- 2,525 probability-source rows reduce to 505 unique `(seed, problem)` keys.
- Repeated keys contain no conflicting `p_x` values.
- Recomputed reliability values match the existing rounded curve within
  `4.5e-6`.
- The controller dataset has all expected 2,560 rows.

The findings are algebraic and descriptive. They do not yet establish a causal
improvement in held-out accuracy.

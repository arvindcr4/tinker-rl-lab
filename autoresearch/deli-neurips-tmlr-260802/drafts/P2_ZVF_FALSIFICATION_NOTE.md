# Workshop note draft — ZVF sampling-model falsification

Date: 2026-08-02  
Source: portfolio §3.2 (`PORTFOLIO_DECISION.md`)  
Status: **zero-GPU evidence freeze** — numbers recomputed offline; full LaTeX short note not yet typeset (the 46pp `paper_P2_zvf.tex` remains demoted and must **not** be submitted as-is).

## Claim (only this)

The i.i.d. Bernoulli ZVF model
\(\mathrm{ZVF}_{\mathrm{iid}}(p,G)=p^G+(1-p)^G\)
is systematically wrong on real GRPO-style groups, **and the sign of the bias is model-dependent**.

Define per-group divergence
\(\delta = \mathrm{ZVF}_{\mathrm{iid}}(\hat p,G) - \mathrm{ZVF}_{\mathrm{obs}}\).

| Regime | Data | \(n\) | mean \(\delta\) | bootstrap 95% CI | Sign |
|---|---|---:|---:|---|---|
| Real GSM8K reasoning | Qwen3-8B, G=8, 3 seeds × 200 problems, full per-group reward vectors | 600 groups | **+0.1224** | **[+0.1115, +0.1337]** | iid **over**-predicts zero-variance |
| Synthetic arithmetic | Qwen2.5-0.5B, G∈{2,4,8,16}, 3 seeds × 40 steps (`groupsize_zvf_sweep.json`) | 459 steps | **−0.0703** | **[−0.0818, −0.0590]** | iid **under**-predicts zero-variance |

Both CIs exclude zero. Opposite signs.

### Consequence (practitioner-facing)

Iso-G rollout sizing and Bernoulli-derived prompt filters (AERO / GRESO / DAPO-style) are miscalibrated **in both directions** if they treat \(\mathrm{ZVF}_{\mathrm{iid}}\) as the sampling model. Do not pool ZVF across tasks.

### Algebra (lemma, not a result)

Under the binary i.i.d. group model, \(\mathrm{pass@}G - p^G = 1 - \mathrm{ZVF}\).  
This is an identity under the model assumptions — **not** an empirical discovery “verified to \(1.11\times 10^{-16}\) on 505 tasks.” Drop that marketing line everywhere.

## Sources (checked-in, offline)

| Artifact | Path | Role |
|---|---|---|
| GSM8K per-seed | `platform_hybrid/experiments/results/tinker_gsm8k_zvf_s{42,123,456}.json` | Primary real-task groups |
| GSM8K summary | `…/tinker_gsm8k_zvf_summary.json` | Aggregates; **seed order fixed 2026-08-02** (`zvf_per_seed` now [0.13, 0.19, 0.155] for seeds [42,123,456]) |
| Arithmetic sweep | `…/groupsize_zvf_sweep.json` | 12 runs / 480 step slots (459 with both zvf+reward) |
| **Forbidden** | `…/variance_mitigation.tsv` | Simulation (negative rewards/accuracies) — **delete from any note** |

### Per-seed GSM8K ZVF (recomputed)

| Seed | overall_zvf | overall_accuracy |
|---:|---:|---:|
| 42 | 0.130 | 0.6969 |
| 123 | 0.190 | 0.7125 |
| 456 | 0.155 | 0.6731 |
| mean | 0.1583 | 0.6942 |

## Explicit non-claims

- No causal “ZVF predicts failure” claim.
- No pooled cross-experiment Spearman/AUROC from heterogeneous cells.
- No AERO-vs-GRPO ranking from `variance_mitigation.tsv`.
- No controller advantage (P7 E3 stays out).
- No 92.3% base-rate figure from P12.

## What was deleted from the P2 publication unit (disposition)

Per `PORTFOLIO_ROSTER_DISPOSITION.md`, the 46pp P2 PDF is demoted. A future ≤8pp workshop `.tex` must start from this note’s tables and the two JSON sources only — not from a cut of `paper_P2_zvf.tex`.

Remaining editorial work for that `.tex` (no GPU):

1. Typeset this note (6–8 pp) with T1–T3 lemma from P10 proofs (copy theorems only).
2. Cite `zhang2026aero` (arXiv:2602.14338) and `le2025rlzvp` (arXiv:2509.21880) separately — no “also reported as” merge.
3. Optional: import P7 0/1867 structural-inertness + ZVF/PCD micro-jitter (not the 92.3% figure).
4. Within-GSM8K stratification warning (R02 / P8 sign flip at run level).

## Recompute commands

```bash
python3 - <<'PY'
import json, random
random.Seed = 20260802  # documentary; use Random(20260802)
# GSM8K delta_div — see session script; mean +0.1224, CI as above
# Arithmetic: groupsize_zvf_sweep.json step_log mean_reward / zvf vs G
PY
```

Bootstrap seed for the CI above: `random.Random(20260802)` on 600 GSM8K group-level \(\delta\); arithmetic used 5000 resamples of 459 step-level \(\delta\).

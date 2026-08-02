# E1 statistical reanalysis

Date: 2026-08-02

## Bottom line

All four E1 arm-versus-GRPO comparisons are **inconclusive**. The earlier DAPO
`DISAPPEARS` verdict is superseded.

The run records and paired held-out scores are unchanged. The correction is to
the analysis code:

1. The old achieved-MDE calculation used the large-sample normal approximation
   `(z_.975 + z_.80) * s / sqrt(8)`. With eight paired seeds, the exact
   noncentral paired-t calculation is required.
2. The preregistration named Benjamini-Hochberg correction across four tests,
   but the function implementing it was never called by the aggregator.

The corrected aggregator uses exact two-sided paired-t power at alpha 0.05 and
power 0.80. It also emits the four raw paired-t p-values, executes
Benjamini-Hochberg at FDR 0.05, and requires the adjusted decision for every
sign-based verdict.

## Corrected results

| Arm | Paired delta | 95% bootstrap CI | Exact MDE80 | Raw p | BH reject | Verdict |
|---|---:|---:|---:|---:|:---:|---|
| DAPO | +0.00100 | [-0.00450, +0.00675] | 0.01012 | 0.756 | No | INCONCLUSIVE |
| GSPO | +0.00500 | [-0.00125, +0.01200] | 0.01185 | 0.210 | No | INCONCLUSIVE |
| Dr.GRPO | -0.00200 | [-0.00950, +0.00725] | 0.01483 | 0.673 | No | INCONCLUSIVE |
| AERO | -0.00075 | [-0.00825, +0.00675] | 0.01319 | 0.858 | No | INCONCLUSIVE |

DAPO's 90% bootstrap interval is inside the +/-0.01 equivalence margin, but
its exact MDE80 is 0.0101159. Because that exceeds the margin, the locked rule
does not permit `DISAPPEARS`. The other three comparisons also fail to establish
either a directional difference or equivalence.

## Reproduction

```bash
python3 zvf-program/audit/aggregate_audit.py \
  --input-dir zvf-program/audit/results/full \
  --output zvf-program/audit/results/audit.json \
  --tex-output zvf-program/audit/results/audit_results.tex

python3 -m unittest zvf-program/audit/test_aggregate_audit.py
```

The machine-readable source of truth is `results/audit.json`. Historical notes
that quote DAPO as `DISAPPEARS` are stale and must not be used in a manuscript,
review response, talk, or abstract.

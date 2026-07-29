# E1 statistical re-audit

**Date:** 2026-07-28  
**Purpose:** independent verification of the post-submission E1 claims before using them in the NeurIPS 36320 rebuttal.

## Finding

The frozen repository aggregate labels DAPO `DISAPPEARS`, but that verdict is not safe to use. The aggregator computes the achieved MDE with a normal approximation:

`(z_.975 + z_.80) * s_d / sqrt(8) = 0.008667`, or **0.867 pp**.

For the eight paired held-out differences

`[-.002, -.012, -.006, .002, 0, .016, 0, .010]`,

the sample standard deviation is `0.00875051`. An exact finite-sample two-sided paired-t power calculation at alpha `.05` and power `.80` gives:

`MDE = 0.0101159`, or **1.012 pp**.

This exceeds the preregistered 1 pp equivalence margin. The preregistration says that if achieved MDE exceeds the margin, the verdict must be `INCONCLUSIVE`. The preregistered Benjamini-Hochberg step over four comparisons is also not applied by `aggregate_audit.py`.

The DAPO mean difference remains **+0.10 pp** and its paired-bootstrap 90% interval remains **[-0.35,+0.575] pp**. A finite-sample TOST calculation is positive (`max p=.01135`), but the separate preregistered MDE gate fails. The fail-closed verdict is therefore `INCONCLUSIVE` until the analysis contract and implementation are repaired and independently reviewed.

## Rebuttal boundary

- Treat DAPO, GSPO, Dr.GRPO, and AERO as inconclusive.
- Do not quote the frozen `DISAPPEARS` label or normal-approximation MDE as a confirmed result.
- Use E1 only as private post-submission feasibility evidence for executing a same-stack audit workflow.
- State that independent reviewer verification requires an anonymized artifact.
- Preserve the frozen aggregate as provenance; do not silently rewrite it.

## Sources checked

- `zvf-program/audit/preregistration.json`
- `zvf-program/audit/aggregate_audit.py`
- `zvf-program/audit/results/audit.json`
- `zvf-program/audit/results/full/*.json`
- `zvf-program/audit/results/full/manifests/*.json`


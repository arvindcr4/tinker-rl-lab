# Manuscript blueprint

<!-- MAIN_TEXT_START -->
## Title

When Can We Skip a Group? A Complete Multi-Seed Test of Contrast-Aware Rollout Allocation for RLVR

## Abstract

State the operator problem, the two-arm intervention, the complete two-task matrix, the joint cost/non-inferiority estimand, one numerical result with its interval after execution, and the math-task boundary. Do not include equations, thresholds without context, implementation paths, or claims about other algorithms.

## 1. Introduction

Motivate charged rollout cost and the risk of spending samples on groups that provide no centered reward contrast. State one research question and three contributions: the intervention, the complete paired-seed study, and the fail-closed evidence pipeline. If the external-user receipt is absent, do not call the work use-inspired.

## 2. Related work

Use a compact comparison table covering group-relative sampling, dynamic curricula, early stopping, RL reproducibility, and verifier reliability. Distinguish interventions from diagnostics and sequence-level rewards from intended quality.

## 3. Method

Define the baseline and intervention, exact sampling policy, canonical objective, charged-token accounting, fixed components, ZVF/GU telemetry, held-out evaluator, and the external-user contribution gate.

## 4. Experimental design

Show the complete task-arm matrix, paired seed set, held-out sizes, power derivation, blinded variance reassessment, provenance fields, and failure/missingness policy before any results.

## 5. Results

Lead with one generated numerical table: every task-arm row, completed seed count, charged-token mean and interval, held-out accuracy and interval, paired cost effect, paired capability effect, and verdict. Follow with mechanism telemetry and prespecified sensitivity analyses. Never replace a value with an artifact reference.

## 6. External-user evaluation

Include this section only if the external-user receipt and ethics gate are complete. Report the pre-existing workflow, participant population, primary operational outcome, uncertainty, and limitations. Otherwise omit the section and the use-inspired label.

## 7. Limitations

Limit claims to the frozen math tasks, model, stack, reward parsers, horizons, and sampling policy. Discuss false homogeneity at two samples, verifier error, task coverage, and statistical precision.

## 8. Conclusion

Repeat the joint cost-quality result and exact claim boundary in two short paragraphs.
<!-- MAIN_TEXT_END -->

## Artifact appendix

Paths, hashes, run identities, environment locks, and recovery receipts belong here. The appendix supports reproducibility but never substitutes for the numerical main table.


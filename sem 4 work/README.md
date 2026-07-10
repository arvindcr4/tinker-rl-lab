# Semester 4 Work — Solo Continuation

This folder is the professor-facing record of Arvind C R's individual continuation of the Semester 3 Group 6 project.

- Student researcher: **Arvind C R**, PES University
- Project guide: **Ramesh Prakash Guledgudd**, PES University
- Starting boundary: after Git tag `capstone-final-2026-04-25`

## What changed in Semester 4

Semester 4 retains the shared code and Semester 3 experimental foundation, but the new research direction, experiment expansion, analysis, paper writing, and P1–P8 outputs are Arvind's solo work. The current papers name Arvind as author and Ramesh Prakash Guledgudd as project guide.

The continuation adds a much larger post-capstone evidence base, including automated research iterations through iteration 206, Berkeley-course-inspired audits and prototypes, expanded statistical analyses, new GRPO diagnostics and controllers, reporting/registry work, and an applied fraud-detection study.

## Semester 4 submission

- [`submissions/neurips-workshop/`](submissions/neurips-workshop/) — the NeurIPS workshop submission, including a freshly compiled anonymous PDF and its build/provenance notes.

The NeurIPS main-track submission is not part of this semester; it is preserved under [`../sem 3 work/submissions/neurips-main-track/`](../sem%203%20work/submissions/neurips-main-track/).

## Ready-to-read papers

All PDFs below were freshly compiled from the current LaTeX sources for this separation pass:

1. [`P1-scaling-laws.pdf`](papers/P1-scaling-laws.pdf) — Scaling Laws for GRPO Post-Training of Language Models: A Cross-Library, Cross-Scale Study
2. [`P2-zero-variance-fraction.pdf`](papers/P2-zero-variance-fraction.pdf) — The Zero-Variance Fraction: A Descriptive Diagnostic for Signal Starvation in GRPO
3. [`P3-group-size.pdf`](papers/P3-group-size.pdf) — Group Size in GRPO: Contrast Density and the Bridge to DPO
4. [`P4-length-bias.pdf`](papers/P4-length-bias.pdf) — Length Bias and Held-Out Generalization in GRPO and Dr. GRPO
5. [`P5-report-the-stack.pdf`](papers/P5-report-the-stack.pdf) — Report the Stack, Not the Label: RL-for-LLM Results Are Stack-Conditioned
6. [`P6-grpo-registry.pdf`](papers/P6-grpo-registry.pdf) — GRPO-Registry: A Machine-Readable Catalog of Group-Relative RL Stacks and Their Variant Deltas
7. [`P7-zvf-controller.pdf`](papers/P7-zvf-controller.pdf) — From Diagnostic to Controller: A Signal-Starvation Theory of GRPO and an Adaptive Group-Size Intervention
8. [`P8-fraud.pdf`](papers/P8-fraud.pdf) — LLM vs. XGBoost in Credit-Card Fraud: Sensor and Scribe, Not Scorer

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the paper-to-source and evidence map, and [`PROVENANCE.md`](PROVENANCE.md) for the exact historical boundary.

## Reproduction entry points

- [`../REPRODUCE.md`](../REPRODUCE.md) — reviewer-oriented reproduction commands
- [`../ARTIFACT.md`](../ARTIFACT.md) — result-to-artifact mapping (historical Semester 3 record; the Semester 4 paper-to-evidence map is [`EXPERIMENTS.md`](EXPERIMENTS.md))
- [`../experiments/experiment_summary.md`](../experiments/experiment_summary.md) — consolidated experiment summary
- [`../experiments/results/`](../experiments/results/) — result tables, traces, and audit outputs
- [`../AUTORESEARCH_FINDINGS.jsonl`](../AUTORESEARCH_FINDINGS.jsonl) — machine-readable iteration ledger
- [`../BERKELEY_IMPROVEMENT_BRIEF.md`](../BERKELEY_IMPROVEMENT_BRIEF.md) — Berkeley-derived improvement program

The code and raw evidence remain in their canonical root locations so all imports, audit scripts, and LaTeX builds continue to work.

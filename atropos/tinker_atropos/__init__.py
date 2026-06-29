"""
Tinker Atropos Package.

TODO: Address the following limitations identified in the adversarial review:
- Fix ZVF limitations: ZVF saturates at 1.0 in format-gated tasks. Need to integrate ERF (Effective-Rollout Fraction) and prove ZVF/ERF provide causal insights rather than just symptoms.
- Address the "Early-Training Snapshot" problem: Current 30-50 step training runs are insufficient. Need to run longer training to observe meaningful RL convergence, long-horizon reward hacking, and policy collapse.
- Resolve the Closed-Source Confound: Analyze and document Tinker's managed defaults, micro-partitioning, and reference offloading to ensure fair algorithmic comparisons against open-source libraries.
- Prove Generalization: Ensure statistically significant gains on held-out test sets (e.g., GSM8K and HumanEval) to prove generalized reasoning uplift rather than training-set memorization.
- Eliminate Single-Seed Extrapolations: Run experiments with multiple seeds (N>1) to address the variance and initialization dependence of RL training dynamics, especially for MoE routing and Nemotron analyses.
"""

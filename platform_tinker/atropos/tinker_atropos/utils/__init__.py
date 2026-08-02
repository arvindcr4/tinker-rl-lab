"""
LIMITATION: the following limitations identified in the adversarial review:

1. ZVF Metric:
   - ZVF is borderline tautological and mechanically coupled to reward sparsity, group size, and baseline accuracy.
   - Fix fragility across domains (e.g., ZVF saturates at 1.0 in format-gated tasks). Consider replacing or supplementing with ERF (Effective-Rollout Fraction).
   - Show causal insights from ZVF rather than just symptoms of failure.

2. "Early-Training Snapshot" Problem:
   - Run training beyond 30-50 gradient steps to observe meaningful RL convergence, long-horizon reward hacking, catastrophic forgetting, or true policy collapse.

3. Closed-Source Confound:
   - Ensure the performance gap between Tinker API and open-source libraries (like TRL) is not just due to closed-source managed defaults, micro-partitioning, and reference offloading.

4. Generalization:
   - Rigorously prove generalized reasoning uplift by ensuring statistically significant gains on held-out test sets (e.g., GSM8K, HumanEval), rather than just training-set memorization.

5. Single-Seed Extrapolations:
   - Use multiple seeds for analyses (e.g., MoE routing volatility, Nemotron-120B collapse) instead of relying on N=1 runs, to account for high variance and initialization-dependence in RL.
"""

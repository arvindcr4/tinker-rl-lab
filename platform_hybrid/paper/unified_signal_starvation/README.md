# Unified Signal Starvation paper

This directory contains a full methods-and-evidence manuscript connecting the
repository's verified GRPO signal-starvation results to critic-based PPO and
Single-Rollout Asynchronous Optimization (SAO).

## Claim boundary

- Verified here: the GRPO pass@G/ZVF identity, the conditional group-size
  calculation, the polarization construction, and the all-correct/all-wrong
  controller counts reproduced from checked-in artifacts.
- Proposed here: Effective Gradient Mass, the cause-aware TriageRL controller,
  and the PPO/SAO evaluation contract.
- Implemented here: framework-independent PPO/SAO gates, PAM/GSR/EGM/root-ZUF
  aggregation, a JSONL trace validator, deterministic boundary tests, compact
  PPO-loop telemetry, and a machine-readable preregistration under
  `platform_hybrid/experiments/signal_starvation/`.
- Not claimed: causal held-out improvement from TriageRL, global optimality of
  group size 4, performance preservation after removing all-correct retries,
  or any PPO/SAO/GLM-5.2 training outcome from the new instrumentation.
- Venue: workshop-short methods/proposal. PPO/SAO measurement is future work
  and is not a gate on the diagnostic-plus-contract contribution.

## Reproduce evidence

```bash
python3 analysis/breakthroughs_2026-07-13/analyze_breakthroughs.py
python3 -m unittest platform_hybrid/experiments/signal_starvation/test_metrics.py
```

## Compile

From the LaTeX skill plugin root:

```bash
python3 scripts/compile_latex.py \
  /Users/arvind/Developer/agentic_repos/tinker-rl-lab/platform_hybrid/paper/unified_signal_starvation/main.tex
```

The delivered PDF is copied to:

```text
output/pdf/signal-starvation-grpo-ppo-sao.pdf
```

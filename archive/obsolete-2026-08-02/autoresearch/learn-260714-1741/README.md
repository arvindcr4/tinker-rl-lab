# Autoresearch log: unified signal starvation

## Objective

Develop a publishable paper that extends the repository's GRPO
signal-starvation thesis to PPO and SAO, including the critic-based PPO used in
the GLM-5.2 training account, without presenting proposed experiments as
completed results.

## Approach registry

| Family | Strength | Failure mode | Disposition |
|---|---|---|---|
| Reward/advantage variance only | Cheap and familiar | Cannot see PPO or SAO clipping | Rejected as incomplete |
| Clip fraction only | Already logged in PPO systems | Treats high- and low-advantage clipped tokens equally | Rejected as primary metric |
| Coefficient mass, EGM | Unifies flat advantage and token gates; zero gives an exact certificate | Positive mass can cancel in parameter space | Kept as online proxy |
| Actual score-gradient norm, GUN | Geometry-aware and directly measures the sampled actor gradient | Requires backward-pass instrumentation and is noisy | Kept as periodic audit |
| Symmetric low-signal retry | Closest to the existing GRPO controller | Spends heavily on solved saturation | Kept as causal control arm |
| Cause-aware routing | Maps failure, solution, critic lag, staleness, and hacks to different actions | Can bias the sampling distribution | Kept with base-stream floor and propensities |

## Adversarial audit

- EGM equal to zero implies zero score-function gradient; the converse is
  explicitly false.
- Nonzero gradient is not a useful or reward-improving gradient; held-out
  endpoints remain primary.
- Outcome-conditioned retries change the objective unless the selection
  probability is known and corrected.
- GLM compaction can multiply sub-traces from one episode; route once per root
  trajectory and report both root-macro and token-micro metrics.
- Invalid or hacked coding-agent rollouts are quarantined rather than treated
  as hard failures.
- Existing evidence is GRPO-only. PPO and SAO effects are preregistered
  hypotheses, not results.

## Verified inputs

- `analysis/breakthroughs_2026-07-13/summary.json`
- `analysis/breakthroughs_2026-07-13/analyze_breakthroughs.py`
- `platform_hybrid/experiments/results/zvf_iter46_per_prompt_isog.tsv`
- `platform_hybrid/experiments/results/p5p8/p7_iter203_emp_per_obs.tsv`
- `platform_hybrid/experiments/results/berkeley/passk_reliability_curve.tsv`
- Hou et al., *Single-Rollout Asynchronous Optimization for Agentic
  Reinforcement Learning*, arXiv:2607.07508.
- Z.ai, *GLM-5.2: Built for Long-Horizon Tasks*.

## Deliverables

- `platform_hybrid/paper/unified_signal_starvation/main.tex`
- `platform_hybrid/paper/unified_signal_starvation/references.bib`
- `output/pdf/signal-starvation-grpo-ppo-sao.pdf`

## Verification

- Deterministic GRPO audit rerun successfully on 2026-07-14.
- LaTeX compiled with TeX Live; all citations and cross-references resolved and
  no overfull boxes were reported.
- The 11-page letter-size PDF was rendered to PNG and visually inspected as a
  contact sheet, with individual checks of the title, formalism, controller,
  and evidence pages.
- Final PDF SHA-256:
  `5d78d10f69c92dc15e493174ce061d3e67263bdb872a9eedc15eae198fa9788b`.

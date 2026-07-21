# Identifiability novelty audit

Date: 2026-07-20  
Decision: **kill as headline theory; retain as supporting proposition**

The executable construction is valid and grounded in the frozen E1 parser, but
the intended novelty does not survive comparison with current primary work.

## Closest work

- [TinyV: Reducing False Negatives in Verification Improves RL for LLM
  Reasoning](https://arxiv.org/abs/2505.14625) identifies widespread verifier
  false negatives, analyzes their loss of informative gradient signal and
  convergence cost, and dynamically invokes a lightweight secondary verifier
  when a rule-based verifier rejects an answer. This directly occupies the
  “zero reward may be verifier failure; perform extra verification” mechanism.
- [From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in
  Mathematical Reasoning](https://arxiv.org/abs/2505.22203) reports that
  rule-based verifiers fail on equivalent answers in different formats and
  studies the resulting RL training degradation. This is especially close to
  the executable marker-mismatch regime.
- [VerifyBench: Benchmarking Reference-based Reward Systems for Large Language
  Models](https://arxiv.org/abs/2505.15801) makes verifier accuracy itself an
  explicit benchmark target and documents substantial remaining verifier
  error, including hard and structurally varied answers.
- [Reward Hacking in Rubric-Based Reinforcement
  Learning](https://arxiv.org/abs/2605.12474) separates verifier failure from
  reward-specification failure and introduces an independent diagnostic for
  detecting training/reward divergence.

## What remains distinctive

The repo contributes a compact minimax action-reversal statement, a charged
known-correct same-path calibration probe, and direct grounding in the exact E1
reward parser. I did not find that exact triplet in the closest papers. That is
too narrow to carry a main-track novelty claim: the lower bound follows from a
standard two-state hidden-decision construction, while the practical remedy is
adjacent to TinyV's dynamic secondary verification.

## Consequence for the flagship

The paper core remains cross-stack causal conformance, where this repository
has already produced exact native-versus-intended loss and gradient
discrepancies with source receipts. The proposition can motivate why reward and
runtime-path conformance must precede compute allocation, but it must not be
marketed as a new general theory of adaptive rollout control.

No GPU or pilot scope is authorized by this audit. The next preregistration must
test whether the discovered framework semantics cause reproducible short-run
training or policy-decision differences under matched compute.

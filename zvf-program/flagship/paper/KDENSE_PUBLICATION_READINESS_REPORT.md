# Publication-readiness audit: July 2026 spectral/entropy manuscript

## Scope and decision

This audit covers `spectral_entropy_paper.pdf`, the July 2026 manuscript titled
“Spectral Legendre Routing and Quantum-Inspired Givens Entropic Attention for
Zero-Variance Starvation Mitigation in Policy Optimization.” It does **not**
cover or revise the May submission.

The submitted July manuscript is not defensible as a main-track empirical
paper. Its central benchmark claims are not backed by the repository, its
forward-path mechanism cannot create a policy gradient when the centered
reward advantage is zero, and several reported zero-variance rates are
incompatible with the stated binary-group setup. The accompanying code is
useful as a mathematical and synthetic-test prototype, but that is a narrower
contribution.

The evidence-safe replacement is
`spectral_entropy_paper_kdense_revision.tex`. It is suitable as a transparent
preprint, negative-results/mechanism-audit paper, or workshop submission. A
main-track claim still requires the prospective language-model experiments
below.

## Evidence audited

- Exact July PDF uploaded to a K-Dense Pro session for a five-stage adversarial
  audit; the source May submission was explicitly excluded.
- Repository modules and the deterministic synthetic benchmark in
  `zvf-program/flagship/pilot/`.
- Stored nine-condition aggregate ledger in
  `spectral_benchmark_results.json`.
- Sixteen focused repository tests for the spectral, Givens, and benchmark
  components.
- A K-Dense reconstruction tested over 77 shape, mask, quadrature, isometry,
  gradient, and feasibility cases, followed by a 623-check primary-literature
  consistency audit.

## K-Dense publication evaluation

The completed K-Dense Pro scientific-writer session
`session_20260728_134510_bc5adeeae1c4` reviewed only the July evidence-safe
revision and its audit bundle. Its downloadable output is
`session_20260728_134510_bc5adeeae1c4-files.zip` (SHA-256
`e5fde308efd4311b4d46dcbbfe8b3c171bfad4920ffc286bcffe962a3c9bf390`).
The bundle contains a 56-page A--I report, source, bibliography, nine figures,
14 tables, 11 LaTeX-ready passages, and an independent peer review.

Its simulated as-submitted scores were: NeurIPS 3.0/10, ICLR 3.3/10,
ACL/ARR Soundness 3.0 and Excitement 1.5, TMLR reject on claims-supported,
workshop accept (poster), and arXiv suitable after blocking traceability fixes.
These are reasoned reviewer simulations, not predictions. The objections are
more useful than the digits. After all four experiment gates pass, K-Dense
estimated roughly 6.4/10 NeurIPS, 6.6/10 ICLR, 6.0-level ACL, and a comfortable
TMLR accept; novelty remains capped by the strongest measured prior-art overlap
of 0.718.

K-Dense found four additional blockers in the six-page revision:

1. The repository's nine-condition ledger existed locally but was absent from
   the uploaded audit ZIP, so the paper did not give a reader a traceable path
   to its only empirical table.
2. The reported 9.7% gradient ratio was generated at the default
   `lambda_entropy=0.1`. The independent sweep gives
   `||gradient|| = 3.7351 * lambda` across six decades, so a single percentage
   primarily reports the chosen coupling weight.
3. The discretization table mixed conventions and its `L=32` entry did not
   match the released CSV. The operative coefficient error is 0.600% at
   `L=2048`; a trapezoid endpoint fix reduces it to 0.00203% (296x).
4. The prior norm sentence used an unshipped `D={4,8,16}` sweep and omitted the
   float32 boundary case. The released audit instead covers
   `D={64,128,512,4096}`, with one float32 absolute error of `1.20e-7`.

The revision now addresses all four: it replaces the percentage table with a
lambda sweep, reports all discretization conventions, uses the audited norm
grid and float32 result, adds a claim-to-file appendix, reports the released
forward-path gradient cosines (0.9504 for Givens and 0.2620 for spectral), and
adds the seven closest missing prior-art citations.

## Decisive technical findings

1. **The missing gradient is not repaired by a representation transform.** If
   the reward-centered advantage is zero, the score-function gradient is zero
   regardless of a forward-path spectral or Givens transform. Only an explicit
   auxiliary loss or advantage term creates a gradient. That changes the
   objective and must be validated against an independent target.
2. **The reported end-to-end results are not present in the repository.** The
   available harness uses deterministic synthetic tensors. It contains no
   language-model fine-tuning, held-out GSM8K/MATH predictions, multi-seed
   comparison, or calibrated process target for this method.
3. **Several ZVF numbers cannot use the stated denominator.** With independent
   binary rewards, the homogeneous-group probability is
   `p^G + (1-p)^G`, whose minimum is `2^(1-G)`. For `G=4`, the floor is 12.5%.
   Fourteen of seventeen manuscript rates lie below their stated binary floor;
   they require a larger group, resampling/rejection, non-binary rewards, or a
   different denominator.
4. **The hard projection is algebraically redundant.** The data-dependent
   `atan2` rotation already annihilates the coordinate that is then projected.
   The projection removed about `1e-17` of the norm in the audit, rather than
   the roughly `1e-3` removed by ordinary truncation. Norm preservation is
   correct, but it is not evidence of denoising or task-relevant information
   preservation.
5. **The discrete Legendre implementation is mask-sensitive.** Changing only
   padded length produced coefficient discrepancies up to 153%. The rectangle
   rule has first-order error, while the audited trapezoidal variant reduced
   the error by roughly 296x at length 2048. The implementation also needs an
   explicit `1 <= N_noise < D` contract.
6. **The gate is under-specified and unvalidated.** Entropy should be normalized
   by `log(L_eff)` and computed at an appropriate head/token granularity. In
   the stored synthetic aggregates, the gated and ungated spectral conditions
   are numerically identical, so the artifact shows no incremental Givens
   effect.
7. **The constituents are established prior art.** Legendre sequence features,
   Givens/unitary parameterizations, and attention-entropy diagnostics all have
   close predecessors. The defensible novelty is the particular combination
   and its proposed use as a trajectory-derived auxiliary advantage, not a new
   transform or a quantum method.

## What the safe revision changes

- Retitles and reframes the work as a testable auxiliary-objective prototype.
- Removes the “quantum-inspired” label and all unsupported GSM8K, MATH,
  sample-efficiency, deployment, and superiority claims.
- Makes the causal boundary explicit:
  `A_tilde = A_reward + lambda * A_aux` is an objective modification, not
  recovery of missing reward information.
- Separates the exact continuous Parseval identity from the approximate,
  equispaced discrete coefficient rule.
- States the repository ledger's default `lambda=0.1`, demotes its analytic
  4.48x--5.48x objective-level FLOP estimate, and replaces the headline 9.7%
  ratio with the independent lambda sweep and its magnitude--fidelity tradeoff.
- States the ZVF binary floor, padding instability, redundant projection, and
  missing end-to-end evidence in the main text rather than burying them in
  limitations.
- Expands related work to the closest verified literature and narrows novelty
  to the combined use case.
- Adds preregistered hypotheses, controls, metrics, and falsification criteria.

## Main-track experiment gate

Do not restore a “mitigates starvation” claim until all of the following are
complete.

### Frozen implementation contract

- Canonical, completion-masked GRPO with clipping and reference KL where the
  baseline uses them.
- Explicit auxiliary-advantage coupling and a preregistered lambda schedule.
- Active-token coordinates or a mask-aware Gauss-Legendre/QR-orthogonal basis.
- Normalized, local entropy gate and validated `N_noise` bounds.
- Exact sampler, reward parser, optimizer, prompts, token budget, checkpoint
  rule, and evaluation set frozen before outcomes are inspected.

### Minimum factorial comparison

- One public model family at 1--2B and 7--8B scales.
- GSM8K plus a harder MATH subset and a contamination control.
- Standard GRPO; spectral auxiliary; Givens-only; combined method;
  variance-matched random-score placebo; process/verifier repair; entropy
  shaping; dynamic resampling/larger-group controls; and a baseline given back
  the method's extra compute.
- Five paired seeds for screening, followed by the preregistered paired-design
  rule `n >= ((z_(1-a/2)+z_(1-b))^2 * s_d^2 / delta^2) + 2`; five seeds detect
  only about `1.62 * s_d` at alpha 0.05 and 80% power.
- Equal tokens, rollouts, KL budget, and evaluation frequency across conditions.

### Required outcomes

- Out-of-sample alignment of the trajectory score with verifier-corrected
  success or disjoint-sample pass@k inside homogeneous-reward groups.
- Per-seed learning curves and fixed held-out accuracy.
- Score calibration, KL, output length, wall time, token count,
  homogeneous-group rate, and all failed/collapsed runs.
- Confidence intervals and paired comparisons, not only pooled means.

### Falsification rules

Reject the mitigation claim if the score lacks out-of-sample alignment, a
variance-matched placebo performs similarly, gains disappear under paired
seeds/equal compute, a dynamic-resampling or larger-group control matches it,
or the Givens component adds no benefit over the spectral auxiliary alone. A
nonzero synthetic gradient is not a success criterion.

## Cheapest decisive next experiment

Run the offline alignment gate before any training. From early, middle, and
late frozen checkpoints at both scales, collect at least 2,000 all-correct and
2,000 all-wrong homogeneous groups. Label completions independently with a
stronger verifier, human adjudication, or disjoint-sample pass@k. The score must
have a bootstrap 95% AUC interval excluding 0.5 at both scales and tasks, and
retain partial association after controlling for length and token entropy. If
it fails, stop the end-to-end programme and publish the negative result. If it
passes, proceed to placebo, equal-compute paired-seed learning, and Givens
ablation gates in that order.

## Recommended submission route

1. Release the evidence-safe revision and audit artifact as a preprint or
   workshop/methods paper.
2. Treat the end-to-end study as a new empirical paper rather than adding
   unsupported claims back into this draft.
3. Target a main-track venue only after the implementation contract, paired
   multi-seed experiment, independent score-alignment test, and placebo
   controls are complete.

The May submission was neither used nor modified in this work.

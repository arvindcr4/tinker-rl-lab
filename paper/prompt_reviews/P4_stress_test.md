# P4 Hypothesis Stress Test (Length Bias / GRPO vs Dr.GRPO)

Executed per `research_prompts/design/hypothesis-stress-test.md` (Ready-to-Copy Prompt contract).
Role: skeptical reviewer testing causal logic.

## Inputs

**Hypothesis.** In short-horizon GSM8K chain-of-thought, the runaway length-bias
signature (increasing length + reward plateau) does NOT materialize and GRPO is
practically equivalent to Dr.GRPO; but step-level trajectories still couple length
to reward in the predicted direction (8/8 paired tests, sign-test p = 0.0039).

**Proposed mechanism** (derived from the paper's own results and discussion —
`sections/length_bias.tex`, `sections/length_bias_iter136.tex`,
`sections/p4_conclusion.tex`, `sections/frontier_synthesis_length_bias.tex`):

1. *Why the runaway signature is absent.* At 30–40 training steps, completions
   are already brief and compressing: every one of the 16 runs has a negative
   length trend (mean rho(step, len) = -0.622 for GRPO on GSM8K-CoT, -0.405 on
   arithmetic; `length_bias_summary.tsv`), and the within-run coupling
   rho(len, reward) is negative in all four (algo, task) cells (-0.29 to -0.62).
   The verbosity trap requires length dispersion plus a *positive* length–reward
   coupling and a hundreds-of-steps horizon (Dr.GRPO paper's regime); none of
   these preconditions holds here, so there is no length pathology for Dr.GRPO's
   correction to remove — endpoint equivalence (held-out gains statistically
   similar, mean length drifting ~193 → ~188 tokens) is the *expected* outcome.
2. *Why step-level coupling persists anyway.* GRPO's per-response 1/L divisor is
   a formally biased gradient at every step — Cov(S/L, R) vs the sequence-level
   Cov(S, R) — whether or not the bias cumulates into runaway length. Its
   fingerprint therefore shows up in per-step velocities: GRPO's (Δr_t, ΔL_t)
   co-move (the "treadmill"), Dr.GRPO's decouple. All 8 (hypothesis × task)
   paired deltas in iter136 lie in the predicted direction; the one-sided
   binomial sign test gives (1/2)^8 = 0.0039.

**Known counterexamples.** The Dr.GRPO and MAD-GRPO papers document the runaway
bias in longer-horizon regimes (hundreds of steps, longer completions, positive
length–reward coupling), so the null is regime-scoped by construction. The
frontier cross-examination itself flagged the length-adversarial truncation test
(A3) as the missing decisive check and "the single highest-value follow-up"
(`frontier_synthesis_length_bias.tex`).

---

## 1) Weakest link

**The inferential bridge from "similar marginal held-out accuracy + no length
inflation" to "GRPO is practically equivalent to Dr.GRPO."** Every statistic
that certifies the equivalence clause is *length-marginalized*: paired outcome
tests (McNemar-style) and endpoint means (last-10 reward 0.265 vs 0.252; last-10
length 184.4 vs 189.3 tokens) cannot distinguish a policy that solves held-out
GSM8K by compressed deduction from one whose held-out successes are mediated by
its generation-length budget. The hypothesis simultaneously asserts (clause 2)
that the two algorithms' step-level length–reward mechanics *differ on every
measured cell* — yet the equivalence claim is certified only at the one
measurement altitude (marginal accuracy at unconstrained decoding) that is
insensitive to exactly that mechanistic difference. If GRPO's held-out accuracy
is length-dependent in a way Dr.GRPO's is not, "practically equivalent" is an
artifact of evaluating both policies at their natural generation cap.

## 2) Why this link is fragile

- **The paper concedes the confound itself.** The P4 discussion states the
  paired-outcome tests "are marginal tests, which can be confounded by
  length-mediated success" (`p4_conclusion.tex`), and the frontier synthesis
  (Gemini Deep Think) makes the same objection concrete: a length-hacking
  policy can post held-out successes by over-generating tokens to stochastically
  stumble into reward. The decisive behavioral probe (A3, truncation) was
  flagged but never run, despite requiring no new training.
- **The equivalence side is thin.** Held-out evaluation is single-benchmark and,
  for the frontier comparison, single-seed; the GSM8K-CoT cell has n = 3 seeds
  per algorithm. "Practically equivalent" rests on a failure to reject at very
  low power — no TOST-style equivalence bound is reported for this clause.
- **The coupling side over-states its certainty.** The p = 0.0039 sign test
  treats the 8 (hypothesis × task) cells as independent Bernoulli(1/2) draws
  under the null. They are not: the 4 hypotheses within a task are computed from
  the *same* per-step traces on the *same* 5 (or 3) seeds — H1 and H4 share the
  ΔL_t series; H2 and H3 share the L_t trajectory. The effective number of
  independent draws is closer to 2 (tasks) than 8, and 7 of the 8 cells are
  individually "null" in `length_bias_iter136_paired_tests.tsv` (only H3 on
  arithmetic reaches p < 0.05, and its permutation p is 0.063).
- **The regime boundary is asserted, not located.** The counterexamples (Dr.GRPO,
  MAD-GRPO) establish the bias exists at longer horizons; nothing in the data
  identifies *where* between 30 steps / 190 tokens and the documented failure
  regime the equivalence stops holding — so the scoped null has an untested
  boundary on the very axis (length dependence) the hypothesis is about.

## 3) Disconfirming check

**Primary (lowest cost — zero training, one evaluation sweep on already-released
checkpoints): the length-adversarial truncation test (A3).**

> Take the already-trained converged GRPO and Dr.GRPO GSM8K-CoT checkpoints
> (Qwen2.5-1.5B-Instruct, 3 seeds per algorithm) and re-evaluate both on the same
> held-out GSM8K test items (500 problems, temperature 0, identical prompts),
> sweeping only the generation cap: T_max ∈ {64, 96, 128, 160, 192, 256, 512}
> tokens — i.e., caps well below the natural mean completion length of ~184–189
> tokens. For each (algorithm, seed, cap) record held-out accuracy and compute
> the retention ratio ret_A(T) = Acc_A(T) / Acc_A(512). Compare the paired
> (per-seed) GRPO-vs-Dr.GRPO retention curves. Cost: ~7 caps × 6 checkpoints ×
> 500 problems = 21,000 greedy generations; no new training regime, no new data.

This attacks the weakest link directly: if GRPO's held-out accuracy relies on
"stumbling into correctness" via its length budget, it should crater
non-linearly under truncation while a genuinely length-invariant Dr.GRPO policy
degrades gracefully; if both degrade identically, the equivalence claim survives
a behavioral test that the marginal paired tests cannot provide.

**Companion (near-zero cost, CPU-minutes): joint permutation for the sign test.**
Re-run the iter136 permutation machinery but flip the algorithm label *jointly
per seed* — recomputing all four statistics per task under each of the sign-flip
assignments (2^5 × 2^3 exhaustive) — so the null respects the shared-trajectory,
shared-seed dependence structure instead of assuming 8 independent cells. Report
the joint p in place of (1/2)^8.

## 4) Result pattern that would force revision

Numeric triggers, pre-stated:

1. **Equivalence clause fails** if, at any cap T ≤ 160 tokens (≈ 0.85 × the
   natural mean length), GRPO's mean retention ret_GRPO(T) falls below
   Dr.GRPO's ret_DrGRPO(T) by **≥ 10 percentage points** with the paired-seed
   bootstrap 95% CI on the difference excluding zero (equivalently, pooled
   McNemar p < 0.05 on the truncated-cap item-level outcomes). Revision
   required: "practically equivalent" must be rescoped to "equivalent only at
   unconstrained decoding," and the paper's null becomes evidence *for*
   length-mediated success under GRPO at short horizons — partially reversing
   the headline.
2. **Coupling clause must be downgraded** if the structure-respecting joint
   permutation yields **p ≥ 0.05**: the 8/8 "GLOBAL_REJECT" claim drops to
   "directionally consistent but not significant," and the hypothesis's second
   clause loses its quantitative anchor (p = 0.0039 may not be reported as such).
3. **Symmetric strengthening trigger** (to keep the check falsifiable in both
   directions): if the two retention curves stay within **±3 pp** of each other
   at every cap down to 64 tokens, the equivalence clause is *behaviorally*
   confirmed — and clause 2 must then be reworded from a mechanism with
   practical stakes to "statistically detectable but behaviorally inert at this
   horizon," since a coupling fingerprint with no length-dependence consequence
   cannot carry the paper's mechanistic narrative.

---

### Data provenance checked

- `experiments/results/length_bias_iter136_paired_tests.tsv` — 8/8 deltas in
  predicted direction; 7/8 verdicts "null"; H3-arithmetic W=15, p_param=0.031,
  p_perm=0.063, d=+2.68. Cross-task file confirms global one-sided sign-test
  p = 0.0039.
- `experiments/results/length_bias_summary.tsv` — all four cells negative
  rho(step,len) (GRPO GSM8K -0.622, Dr.GRPO -0.355); last-10 reward 0.2646 vs
  0.2521; last-10 length 184.4 vs 189.3 tokens; length-bias flag rate 0.0
  everywhere.
- `sections/frontier_synthesis_length_bias.tex` — truncation test flagged as
  "the single highest-value follow-up ... one truncation sweep away from being
  executed on the checkpoints Sec. [length-bias-pillar] already produced."

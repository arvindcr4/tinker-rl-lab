# Harvard CS2824 foundations audit

Date: 2026-07-29<br>
Course: [CS 2824: Foundations of Reinforcement Learning, Spring 2026](https://harvard-cs2824-s26.github.io/)<br>
Source repository: `harvard-cs2824-s26/harvard-cs2824-s26.github.io` at commit `5dcc34e3b861da632371645fb05aebb12a40d23c` (2026-04-17)<br>
Scope: the prospective ZVF/GRPO spectral and entropic follow-up; the frozen r4-2 campaign is evidence context only

## Applied outcome

The course changes the next experiment from a signal-injection study into an assumption-audited policy-optimization study. The combined protocol now blocks training until it can answer five questions:

1. What result is being invoked, and which of its formal assumptions hold in the LLM setting?
2. Does the sampler reach independently verified rewarding trajectories, or did the intervention only make a local gradient nonzero?
3. Which policy generated each completion, how far has the learner moved from it, and is the resulting distribution shift tolerable?
4. Does the update satisfy an empirical policy-space trust region, beyond merely using a clipped surrogate?
5. Is the remaining effect larger than optimization, estimation, approximation, and verifier error?

These obligations are encoded in [`rlhfbook_followup_preregistration.json`](rlhfbook_followup_preregistration.json), [`theory_transfer_ledger.json`](theory_transfer_ledger.json), and [`offline_falsification_packet.json`](offline_falsification_packet.json). They do not authorize a run.

## Source boundary

The relevant course units are policy gradient, NPG/TRPO, global convergence and approximation, advanced policy optimization, and RLHF (BT, DPO, and REBEL), plus the guest lecture on reward regression. The course project rubric also favors a well-motivated question, a justifiably simple method, technical depth, and clear interpretation; it explicitly welcomes reproducing proofs and testing conjectures.

The lectures mix textbook results, paper presentations, and guest material. This audit therefore uses them in three different ways:

- theorem or proof results create assumption checks only within their stated domains;
- algorithm lectures motivate telemetry and baselines;
- guest slides generate hypotheses, not primary evidence or achieved-result claims.

No tabular-MDP convergence theorem is claimed for a transformer policy here. No guest-slide performance number is imported into the manuscript.

## Course-to-repo translation

| Course lesson | Repo risk exposed | Applied change |
|---|---|---|
| A policy can have zero gradient while remaining arbitrarily suboptimal when it does not visit rewarding states. | ZVF can mean missing within-group contrast or missing rewarding-trajectory coverage; a synthetic auxiliary gradient does not distinguish them. | Added `H5_coverage`; require correct-completion coverage and mixed-group yield per charged token under a token-matched exploration sweep. |
| Stationarity-to-optimality results depend on coverage or distribution-mismatch coefficients. | Prompt sampling, completion sampling, and held-out evaluation induce different distributions, but a single aggregate ZVF hides that structure. | Added a foundations ledger covering the data-generating distribution, support condition, comparator, and empirical mismatch proxy for every theorem-shaped claim. |
| NPG/TRPO constrains change in policy space through KL/Fisher geometry. | PPO/GRPO clipping can look healthy while sequence-level ratios or actual policy drift are unhealthy. | Added empirical KL stop rules, ratio-tail and ESS telemetry, the sampler-policy version, policy lag, and a Fisher-quadratic step diagnostic. |
| Approximation and statistical estimation errors enter policy-optimization guarantees separately. | A small multi-seed effect can be smaller than model-class, sampling, or verifier error yet still be described as learning. | Added `H7_error_attribution` and separate optimization, estimation, approximation, and verifier-error outputs. |
| KL-regularized preference optimization admits alternatives such as DPO/REBEL-style regression objectives. | The current comparison could incorrectly make GRPO the only meaningful control and attribute gains to the auxiliary score rather than the loss family. | Keep the centered-reward baseline and require a preregistered regression-based comparator if a squared-regression objective is introduced. Do not silently swap objectives mid-study. |
| Off-policy reward regression may be attractive for real asynchronous systems, but depends on target, support, and estimation conditions. | r4-2 is fixed replay, while future LLM training may have trainer-inference mismatch and stale rollout policies. | Added `H6_distribution_shift`; every record must identify the actual sampler policy, lag, ratios, and ESS, followed by an on-policy replication or bounded-lag validation. |
| A theory project should state an open question, use the simplest justified approach, and interpret negative outcomes. | The spectral and entropic work could grow into a controller stack before basic calibration is established. | Retained the offline alignment gate as the cheapest decisive experiment and made a failed gate a preserved result rather than a reason to add machinery. |

## Required theorem-to-experiment ledger

Every proposed theorem-derived statement must have one row with these fields before it can enter a paper draft or run amendment:

| Field | Required content |
|---|---|
| Exact source | Course unit plus primary theorem or paper, not only a lecture title. |
| Formal domain | Finite/tabular MDP, contextual bandit, function approximation, token MDP, or another explicit domain. |
| Mapping | Prompt/state, token/action, completion/trajectory, terminal reward, policy, reference policy, and comparator. |
| Data distribution | Initial/prompt distribution, occupancy distribution, rollout policy, and evaluation distribution. |
| Assumptions | Support/coverage, bounded ratios, smoothness, realizability, conditioning, reward and horizon bounds, and any independence assumptions. |
| Status | Verified, empirically proxied, unverified, or violated. |
| Observable proxy | The registered metric and threshold used when an assumption cannot be directly established. |
| Falsifier | What result would refute the transferred claim or force it back to analogy status. |

An empty or unverified assumption cell blocks a theorem-shaped claim. It does not necessarily block a clearly labeled empirical hypothesis.

## Concrete experimental changes

### Coverage before stationarity

Report `gradient_norm` alongside `correct_completion_coverage`, all-wrong/all-correct/mixed fractions, and `mixed_group_yield_per_charged_token`. The key comparison is a token-matched sweep over sampling temperature and group size. If an auxiliary signal raises gradient norm but never increases verified correct coverage or held-out quality, it fails `H5_coverage`.

The exact MDP occupancy coefficient is generally unavailable for an LLM. The registered coverage metrics are empirical proxies, not a claim that the course theorem's assumption has been proved.

### Distribution shift and trust region

Each training row must bind `sampler_policy_version` and `data_policy_lag_steps`. Report policy-ratio quantiles, maximum importance weight, effective sample size, approximate KL to the old/data policy, KL to the initial reference, and `fisher_quadratic_step`. Match prompt schedules, initial checkpoints, task mixtures, and charged-token budgets across arms; retain arm-specific completion hashes because genuinely on-policy completions should diverge with the policies.

The run amendment must set numerical stop thresholds before execution. Clipping is retained as an algorithm detail, not treated as evidence that the actual policy update stayed inside a trust region. A fixed-replay result remains off-policy replay evidence and cannot be relabeled as on-policy learning.

### Error budget

The final table must separate, without directly comparing or summing quantities that have incompatible estimands or units:

- optimization error: registered loss/gradient/convergence diagnostics at the chosen compute budget;
- estimation error: paired-seed and sampled-evaluation uncertainty;
- approximation error: held-out residual or misspecification test for the auxiliary/regression model class;
- verifier error: independent-checker disagreement and audited false-positive/false-negative rates.

Each component receives its own registered diagnostic or uncertainty statement. A common-unit bound may be reported only if a derivation maps every component to that estimand; otherwise the components remain a structured limitation analysis.

### Comparator discipline

The minimum comparison remains group-relative control, centered reward without standard-deviation normalization, spectral arm, entropic arm, and variance-matched placebo. A REBEL- or other regression-based arm is optional until its exact primary source and target are preregistered. If added, it must use the same realized prompt/corpus fingerprints, charged-token budget, measured-FLOP tolerance, and held-out evaluation.

## What remains unchanged

- The accepted r4-2 source and receipts are not modified or relabeled.
- The failed filtered-regime CV gate is not weakened.
- Synthetic gradients remain objective diagnostics, not evidence of correctness or learning.
- GPU and external execution remain unauthorized.
- The manuscript stays at registered-feasibility/negative-results scope until every prospective gate passes with matched-cost multi-seed evidence.

## Verification

From the repository root:

```bash
python3 zvf-program/experiments-next/verify_rlhfbook_followup.py
python3 -m pytest -q tests/test_rlhfbook_followup.py
```

The verifier pins both web resources, enforces the theory ledger and offline packet, verifies the frozen review-bundle digest, and reports the live-versus-accepted objective hash distinction. Its success status is a contract-lint pass, not a stage result or promotion authorization. It performs no training and makes no external changes.

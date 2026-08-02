# Contrast-aware rollout allocation: confirmatory study design

## Decision

The next paper will not compare a catalogue of GRPO-family labels. It will test one intervention against one baseline in a complete, prespecified matrix.

The safe submission is a methods/reproducibility paper about a cost-quality intervention. “Use-inspired” is conditional, not rhetorical: it becomes available only if an external education-domain workflow and outcome are prospectively documented and evaluated.

## Research question

Can a contrast-aware early-stop sampler reduce charged generated tokens by at least 20% relative to fixed-`G=8` GRPO while the lower confidence bound for held-out accuracy difference remains above `-0.01`, separately on GSM8K and MATH-500?

Both conditions must hold on both tasks. A positive result on one task cannot compensate for a missing or negative cell on the other.

## Treatment

The baseline generates eight completions for every prompt before computing the group-relative update.

The intervention first generates two completions:

1. If their verifier rewards differ, it generates six additional completions and applies the same `G=8` update as the baseline.
2. If their rewards agree, it stops sampling that prompt and records an all-wrong or all-correct homogeneous group. No auxiliary gradient, curriculum substitution, or hidden resampling is introduced.

This changes the sampling policy but not the model, prompt schedule, optimizer, reward parser, update count, checkpoint rule, evaluation harness, or canonical GRPO objective applied to expanded groups. Charged tokens include the two-completion probe and every expansion.

Qwen3-8B runs in its explicit non-thinking mode for both arms. Training samples with `temperature=0.7`, `top_p=0.8`, and `top_k=20`; fixed held-out evaluation uses deterministic non-thinking decoding. This decoder contract was frozen prospectively after a non-evidence MATH preflight showed 100% clipping under thinking mode. Completion clipping is a required receipt field, and an all-clipped preflight cannot clear the execution gate.

The objective, optimizer, precision, and adapter configuration are pinned in the protocol (amendment `A003_confirmatory_hardening`): canonical GRPO (`epsilon=0.2`, token-level importance sampling, group reward scaling, `beta=0`), `adamw_torch_fused` at `lr=1e-6` with a linear schedule, and LoRA `r=16`/`alpha=32` on all linear modules. Both arms share this exact configuration.

The probe size is a deliberate worst-case design point, justified prospectively in A003: a `G=2` probe maximizes savings and false-homogeneity risk simultaneously, so any harm from the censoring mechanism is maximized and therefore detectable by the joint non-inferiority guard. A `G=4` probe is the preregistered first follow-up ablation and is out of this paper's claim universe.

## Closed matrix

| Task | Baseline | Intervention | Paired seeds | Held-out examples |
|---|---|---|---:|---:|
| GSM8K | fixed `G=8` GRPO | `G=2` check, mixed groups expand to `G=8` | 23 | 1,000 |
| MATH-500 | fixed `G=8` GRPO | `G=2` check, mixed groups expand to `G=8` | 23 | 500 |

All four cells are primary. The initial plan is 92 training units. After eight paired seeds per cell, a blinded variance-only reassessment may increase the final count to at most 24 seeds per cell. It cannot reduce the count, inspect arm effects, change margins, or drop a task. Seeds beyond the original sixteen are derived by the frozen deterministic rule in amendment A003 (ascending primes above 293, disjoint from E1 and all frozen seeds; the 24th seed is the reserved 349).

“Corpus” is not a crossed experimental factor in this paper. Each task has one frozen, task-native training/evaluation pipeline: GSM8K train to GSM8K test, and MATH-lighteval train to the disjoint MATH-500 evaluation set. Applying the MATH training corpus to the GSM8K integer task, or vice versa, would change the task and reward contract rather than fill a missing factorial cell. The claim universe is therefore the complete two-task by two-arm matrix above; no cross-corpus transfer claim is made.

## Estimands and success rule

For each task, with seed pairing fixed before execution:

- cost effect: relative difference in charged generated tokens, intervention minus baseline;
- capability effect: held-out exact-match accuracy difference, intervention minus baseline;
- mechanism telemetry: all-wrong, all-correct, and mixed-group fractions; false-homogeneity rate at `G=2` measured by the baseline's first two samples; clip fractions; policy-ratio tails; KL; parser disagreement; and wall time.

The paper may claim joint success only if, on both tasks:

1. the upper confidence bound for relative token-cost difference is at most `-0.20`; and
2. the lower confidence bound for held-out accuracy difference is greater than `-0.01`.

The two endpoints form an intersection-union decision within each task. Holm adjustment is applied across the two tasks. Any incomplete cell, failed provenance gate, power failure, or interval crossing a boundary yields `INCONCLUSIVE`, never “equivalent” or “failed.”

## Power and replication

The capability planning standard deviation is `0.0128285396`, the worst paired held-out standard deviation in the completed E1 audit. Under amendment A003 the planning calculation is exact, not a normal approximation: a one-sided noncentral paired-t calculation at the Holm worst-case per-task alpha of `0.0125` (Holm across two tasks tests the smaller p at alpha/2) with 80% power and a one-point margin requires 19 paired seeds. A 20% variance-transfer inflation yields 23, within the 24-seed cap. The hash-bound stdlib implementation of the exact calculation is shared with the corrected E1 aggregator.

The 80% target is per endpoint per task. The joint success rule requires every endpoint to pass on both tasks, so the joint criterion is not powered above the product of the per-endpoint powers; this is disclosed as a limitation rather than adjusted away.

There is no valid prior for the new intervention's paired token-cost variance. At eight completed pairs per task, the blinded reassessment therefore estimates both held-out-difference variance and the variance of `log(tokens_intervention/tokens_baseline)` without exposing the sign or task-arm mean effect. It recomputes the paired count required for 80% power at the one-point capability margin (exact one-sided noncentral-t at the Holm worst-case alpha) and the `log(0.8)` cost boundary (one-sided paired-t on the paired log token ratio at the same alpha and power). The frozen final count is the larger requirement, capped at 24; exceeding the cap yields `STOP_UNDERPOWERED` rather than a relaxed claim.

The E1 value is a planning prior, not evidence for the new intervention or MATH-500. The blinded variance reassessment protects against transfer failure. Optimization steps, prompts, completions, checkpoints, and held-out rows are not training replications.

Non-evidence preflights may run on preregistered confirmatory seeds (amendment A003): a preflight is at most one optimizer step with eight held-out rows, is labeled `preflight-not-evidence`, and never enters the ledger; confirmatory units are fresh independent runs, and no preflight outcome reaches any analysis.

## External-user gate

The intended external population is mathematics educators or education-product operators who already review AI-generated worked solutions. Before “use-inspired” can be selected, the project needs:

- a dated partner attestation describing the pre-existing review workflow and decision;
- ethics/consent disposition before recruitment;
- a frozen sampling and blinding plan;
- a prospectively powered primary external outcome, such as acceptance without correction or review time per accepted solution;
- a signed completion receipt and numeric result table.

Without those receipts, the external study is absent and the contribution remains methodology/reproducibility. Repository authors or ML researchers role-playing users do not satisfy this gate.

## What is not claimed

- no universal GRPO, PPO, framework, or model ranking;
- no transfer to coding, tool use, subjective preference rewards, or non-math tasks;
- no stopping-controller benefit beyond the exact sampler tested;
- no capability improvement unless the held-out interval supports it;
- no use-inspired contribution without the external-user gate;
- no result from an artifact filename, private dashboard, optimization trace, or post-selected checkpoint.

## Stop/go path

1. Verify and freeze this design.
2. Implement the sampler behind unit and conformance tests.
3. Run a local dry-run with synthetic receipts only. **Complete.**
4. Obtain explicit GPU authorization before any remote execution. **Complete, with publication and external recruitment still excluded.**
5. Run and independently verify the four task-arm preflight units; these remain non-evidence. The gate requires a private Hugging Face artifact, a finished W&B run, verified provider cleanup, one shared scientific-stack fingerprint, a non-clipped completion path, and a live mixed-reward optimizer update in every task-arm cell. Each intervention task must also exercise its homogeneous early-stop branch. Missing coverage blocks confirmatory execution rather than being inferred from synthetic tests.
6. Execute the first eight paired seeds in every cell.
7. Perform the blinded variance-only reassessment; freeze the final seed count.
8. Complete every remaining cell and fixed held-out evaluation.
9. Generate the numerical table from the reconciled ledger.
10. If pursuing use-inspired status, complete the external-user gate; otherwise omit that contribution type.
11. Build the paper only after the verifier reports all primary cells complete.

# Next AI/ML submission design

This directory is the executable design for the paper that follows submission 36320. It is deliberately not an expansion of the heterogeneous May audit.

The paper asks one causal question:

> Can a contrast-aware early-stop sampler reduce charged RLVR rollout tokens while keeping held-out mathematical accuracy within a predeclared one-percentage-point regret margin?

The confirmatory scope is closed before execution:

- one model and one canonical open stack;
- two math tasks, GSM8K and MATH-500;
- two arms, fixed-`G=8` GRPO and a `G=2` contrast check that expands only mixed groups to `G=8`;
- twenty-three paired, preregistered seeds per task-arm cell (exact noncentral-t power at the Holm worst-case alpha, per amendment A003), with a blinded variance rule that may increase but never decrease the count;
- fixed held-out evaluation for every cell;
- one generated numerical main table whose values must come from the result ledger.

This fixes the prior review failure by limiting the claim universe to the complete matrix. HumanEval, MBPP, tool use, framework ranking, PPO comparisons, and broad GRPO generalization are explicitly out of scope.

The default contribution types are methodology, reproducibility, and negative results. A use-inspired label is blocked unless a real education-domain partner documents a pre-existing workflow and a prospective external-user outcome is completed. The template is not evidence and cannot unlock that claim.

## Current status

`DESIGN_FROZEN_EXECUTION_AUTHORIZED`

Local implementation and remote GPU execution are authorized by a bound receipt. External-user recruitment, publication, submission, pushing, and result promotion before complete receipts remain unauthorized. The existing E1 audit is used only as a conservative variance-planning prior. Its stale aggregate verdicts are not results for this paper.

The executable preflight stack now includes a pure sampler contract, a pinned TRL adapter, an environment gate, a credential-isolated remote runner, and an independent local receipt validator. A preflight is always labeled `preflight-not-evidence`; it cannot populate the confirmatory main table.

Transport is provenance rather than treatment: Colab, Hugging Face Jobs, Kaggle, and GCP Compute launchers bind the requested provider and hardware flavor into distinct fingerprints, while the model, sampler, objective, data, and receipt checks remain fixed. Hugging Face Jobs runs additionally pin Trackio; every provider still requires the same private Hub and W&B receipts. The GCP path uses pre-existing Secret Manager references, a 90-minute Spot A100 limit, a dedicated private receipt bucket with public access prevention, and verified deletion of its exact temporary VM. End-to-end transport is currently proven on Colab and GCP only: the HF Jobs attempt failed pre-allocation (HTTP 402, receipted) and the Kaggle attempt failed at secret lookup with a P100 substituted for the requested A100 (receipted); both launchers are implemented and dry-run-verified but have not yet transported a run.

The Colab CLI path validates OAuth, Hugging Face, and W&B credentials before allocation; installs its pinned stack through a bounded long-running `exec`; verifies the requested GPU before uploading credentials; deletes credential files immediately; runs training in a child process so secrets never enter the persistent kernel environment; streams the child result marker through the parent kernel; checks the private Hub commit and finished W&B run independently; and fails closed unless server-side session enumeration proves cleanup. If Colab still drops the streamed marker, the launcher can recover only from the request-derived private Hub repository after finding the manifest and final adapter at one resolved commit, revalidating the complete manifest, and proving that its referenced W&B run finished.

The first MATH-500 baseline preflight found two frozen MATH-lighteval training solutions that use unbraced `\boxed 2` and `\boxed 9` targets. Prospective amendment `A001_math_unbraced_boxed_targets` adds a numeric-only compatibility rule, retains all 7,500 training rows, and is hash-bound before any confirmatory row exists. The failed preflight receipt is retained and remains non-evidence.

The amended MATH baseline then showed that thinking-mode decoding clipped every training completion at 1,024 tokens and produced no reward contrast. Qwen3's model card warns against greedy evaluation in thinking mode and documents non-thinking mode as the efficiency-oriented hard switch. Prospective amendment `A002_qwen3_non_thinking_decoder` therefore freezes non-thinking chat templates, the model-card training sampler (`temperature=0.7`, `top_p=0.8`, `top_k=20`), deterministic non-thinking evaluation, and a fail-closed completion-clipping field before confirmatory execution.

A pre-confirmatory adversarial audit (2026-08-02) found the frozen protocol under-specified: optimizer and adapter hyperparameters lived only in a preflight script, the seed-extension rule was missing, the power derivation used the normal approximation the same-day E1 reanalysis repudiated and ignored the Holm worst-case alpha, the probe size had no preregistered justification, and the authorization did not anchor a protocol hash. Prospective amendment `A003_confirmatory_hardening` therefore pins the objective/optimizer/adapter configuration into the protocol, freezes 23 paired seeds (exact one-sided noncentral-t power at the Holm worst-case alpha 0.0125: 19 seeds, inflated to 23) with a deterministic extension rule to the 24-seed cap, specifies the blinded-reassessment method, justifies the single `G=2` probe size as a worst-case design point with `G=4` as the preregistered follow-up, documents preflight seed reuse, completes the results-contract telemetry fields, and binds the authorization to a canonical protocol hash.

## Verify

```bash
python3 zvf-program/next-submission/verify_design.py
python3 -m pytest -q \
  tests/test_next_submission_design.py \
  tests/test_next_submission_sampler.py \
  tests/test_next_submission_trl_adapter.py \
  tests/test_next_submission_preflight.py
python3 zvf-program/next-submission/run_preflight.py \
  --task gsm8k \
  --arm contrast_early_stop_g2_to_g8 \
  --seed 211 \
  --dry-run
# Remove --dry-run only from a clean committed tree after checking `colab sessions`.
python3 zvf-program/next-submission/run_hf_jobs_preflight.py \
  --task gsm8k \
  --arm contrast_early_stop_g2_to_g8 \
  --seed 211 \
  --flavor a100-large \
  --dry-run
python3 zvf-program/next-submission/run_kaggle_preflight.py \
  --task gsm8k \
  --arm contrast_early_stop_g2_to_g8 \
  --seed 211 \
  --accelerator NvidiaTeslaA100 \
  --dry-run
python3 zvf-program/next-submission/run_gcp_preflight.py \
  --task gsm8k \
  --arm contrast_early_stop_g2_to_g8 \
  --seed 211 \
  --dry-run
```

The verifier checks the complete matrix, the exact noncentral-t power calculation at the Holm worst-case alpha against a hash-bound implementation, the claim ledger, the contribution-type gate, the generated-table contract, the manuscript order, the execution authorization (including its canonical protocol anchor), executable source hashes, the preregistered seed values and extension reserve, the objective/optimizer/adapter configuration, the estimand boundaries and joint success rule, the analysis rules, the telemetry lists in both the protocol and the results contract, and the frozen evidence boundary.

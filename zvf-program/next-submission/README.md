# Next AI/ML submission design

This directory is the executable design for the paper that follows submission 36320. It is deliberately not an expansion of the heterogeneous May audit.

The paper asks one causal question:

> Can a contrast-aware early-stop sampler reduce charged RLVR rollout tokens while keeping held-out mathematical accuracy within a predeclared one-percentage-point regret margin?

The confirmatory scope is closed before execution:

- one model and one canonical open stack;
- two math tasks, GSM8K and MATH-500;
- two arms, fixed-`G=8` GRPO and a `G=2` contrast check that expands only mixed groups to `G=8`;
- sixteen paired, preregistered seeds per task-arm cell, with a blinded variance rule that may increase but never decrease the count;
- fixed held-out evaluation for every cell;
- one generated numerical main table whose values must come from the result ledger.

This fixes the prior review failure by limiting the claim universe to the complete matrix. HumanEval, MBPP, tool use, framework ranking, PPO comparisons, and broad GRPO generalization are explicitly out of scope.

The default contribution types are methodology, reproducibility, and negative results. A use-inspired label is blocked unless a real education-domain partner documents a pre-existing workflow and a prospective external-user outcome is completed. The template is not evidence and cannot unlock that claim.

## Current status

`DESIGN_FROZEN_EXECUTION_AUTHORIZED`

Local implementation and remote GPU execution are authorized by a bound receipt. External-user recruitment, publication, submission, pushing, and result promotion before complete receipts remain unauthorized. The existing E1 audit is used only as a conservative variance-planning prior. Its stale aggregate verdicts are not results for this paper.

The executable preflight stack now includes a pure sampler contract, a pinned TRL adapter, an environment gate, a credential-isolated remote runner, and an independent local receipt validator. A preflight is always labeled `preflight-not-evidence`; it cannot populate the confirmatory main table.

Transport is provenance rather than treatment: Colab, Hugging Face Jobs, and Kaggle launchers bind the requested provider and hardware flavor into distinct fingerprints, while the model, sampler, objective, data, and receipt checks remain fixed. Hugging Face Jobs runs additionally pin Trackio; every provider still requires the same private Hub and W&B receipts.

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
```

The verifier checks the complete matrix, power calculation, claim ledger, contribution-type gate, generated-table contract, manuscript order, execution authorization, executable source hashes, and frozen evidence boundary.

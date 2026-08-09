# Professor meeting brief — Pavlov-domain Tinker experiments

Date: 2026-08-09
Status: live baseline complete; tracked training smoke pending

## Decision and evidence boundary

Continue with `Qwen/Qwen3.6-35B-A3B` as the primary Tinker candidate because its
mixture-of-experts pricing is favorable for the available budget and it preserves
the multimodal, tool-use, code, and long-context capabilities required by the
Pavlov's List task contract. Do **not** claim a post-training improvement yet.

The current live evidence covers strict single-turn xLAM function calling only. It
does not establish usefulness across the contract's 16 domains or production
readiness for any company. GSM8K remains calibration-only.

## Live evidence available for the meeting

| Item | Current result | Evidence status |
|---|---:|---|
| Frozen base evaluation | 7/100 perfect calls; mean strict reward 0.070 | admissible baseline receipt |
| Base evaluation tokens | 42,993 prompt; 12,601 sampled | recorded in receipt |
| Base evaluation estimated cost | $0.04004 | conservative price-based estimate |
| First training smoke | stopped after 4 completed steps | inadmissible: W&B initialization defect |
| Interrupted Tinker run | `96e18cc8-84cb-5cae-9326-661f29394922:train:0` | provenance only; not a result |
| Interrupted-run checkpoints | one step-0 sampler checkpoint; no trained checkpoint | provenance only |

Authoritative baseline receipt:
[`base_eval_100.json`](../../autoresearch/orchestrator-260809-0922/base_eval_100.json).
The baseline is mirrored in the online W&B run
[`qwen36-base-xlam-eval-100`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-pavlov/runs/pavlovbasexlam100260809).

Rejected-run provenance:
[`rejected_untracked_smoke.json`](../../autoresearch/orchestrator-260809-0922/rejected_untracked_smoke.json).
Its rejected W&B provenance record is
[`REJECTED-untracked-qwen36-xlam-smoke`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-pavlov/runs/pavlovuntrackedsmoke260809).
Its sole untrained step-0 adapter is archived at
[`arvindcr4/pavlov-xlam-smoke-260809-seed809-step0-aborted`](https://huggingface.co/arvindcr4/pavlov-xlam-smoke-260809-seed809-step0-aborted).

The interrupted smoke observed rewards `0.000, 0.000, 0.625, 0.000` over its four
completed steps. These values are not used as scientific evidence because the run
was not logged to W&B and did not reach a trained checkpoint or held-out evaluation.

## Tracking invariant for every new run

A run may begin paid Tinker work only after online W&B initialization succeeds. The
W&B run must record the Tinker training-run ID, immutable configuration, seed,
dataset/split identity, budget cap, per-step loss and reward, evaluation metrics,
and links to Hugging Face checkpoints.

Every periodic and final Tinker sampler checkpoint must be exported to a private
Hugging Face model repository. A failed W&B initialization or failed HF export stops
the run fail-closed. An unlogged run or an unpublished model checkpoint is
inadmissible evidence.

## Next experiment: tracked strict-tool-use smoke

| Field | Frozen value |
|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B` |
| Dataset | `Salesforce/xlam-function-calling-60k`, deterministic seed-809 split |
| Train / held-out | 3,000 / 500 examples |
| Smoke steps | 10 |
| Batch / group | 2 prompts / 4 completions |
| LoRA rank | 32 |
| Learning rate | `2e-5` |
| Prompt / response caps | 1,200 / 128 tokens |
| Periodic checkpoint | steps 5 and 10, plus final |
| Primary metric | perfect-call rate on frozen 100-example held-out slice |
| Success criterion | exceed the frozen 7% base result without split or verifier changes |

The smoke is a systems-and-signal gate, not the final experiment. If it fails to
produce a nonzero reward-bearing training signal or does not beat the frozen base,
do not spend the remaining budget on a longer copy of the same configuration.

## Budgeted follow-up if the smoke passes

The hard user authorization is $18 on Tinker only. Operations use a $16.50 cap and
retain $1.50 for billing lag. Before the new smoke, conservative known/possible spend
is approximately $0.14: $0.040 for the base evaluation plus less than $0.10 for the
interrupted smoke. Live billing remains authoritative when it catches up.

Use successive halving rather than three blind full runs:

1. Run short tracked arms at learning rates `1e-5`, `2e-5`, and `4e-5` with identical
   data, seed, sampling, reward, and evaluation.
2. Select by frozen held-out perfect-call rate, breaking ties with strict mean reward
   and then lower estimated cost.
3. Extend only the winning arm, checkpointing and evaluating at fixed intervals.
4. Preserve a final held-out slice that is not consulted during arm selection.

## What this experiment can and cannot answer

If successful, the xLAM study shows whether inexpensive GRPO can improve exact API
tool-call formation on a real function-calling dataset. It directly informs the
tool-use, browser, enterprise, and multi-domain portions of the broader contract, but
it cannot by itself cover code repair, computer vision/control, finance state,
science environments, security, chip design, artifact creation, or long-horizon work.

The next campaign phase therefore needs a mixed stateful curriculum and held-out
domain evaluations. The 53-company mapping is a coverage contract, not evidence that
one xLAM-trained adapter is useful to all 53 companies.

## Suggested meeting summary

> We replaced a GSM8K-only plan with a 53-company, 16-domain capability contract. The
> first live measurement is a frozen exact tool-use baseline: Qwen3.6 gets 7/100 on
> held-out xLAM at about four cents. We caught and stopped an unlogged smoke run, made
> W&B and Hugging Face receipts mandatory, and will use a tracked 10-step gate before
> spending the rest of the $18 budget. Any improvement claim will be tied to a frozen
> held-out split; all-company usefulness remains a later multi-domain question.

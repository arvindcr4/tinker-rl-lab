# Learnings — Confirmed Findings & Dead Ends

New agents MUST read this file before writing a variant. Avoid re-exploring
dead ends; avoid re-stating confirmed findings.

Updated by `coordinator.py wave ingest`.

## Confirmed (seeded from the paper, Runs 1–44)

- **Capacity threshold between 3B and 4B** — Qwen3-3B and Llama-3B stall on GSM8K
  (all-incorrect saturation), while Qwen3-4B saturates to all-correct groups. This
  is the observation we're trying to rescue or reinforce.
- **LoRA rank does not affect zero-loss rate** — ranks 8, 16, 64 all show the same
  zero-loss dynamics. Rank is probably not a rescue knob on its own.
- **3B failure is from all-incorrect saturation** (zero-advantage), not from high
  variance. Rescue must introduce at least one non-zero reward signal per group.
- **4B zero-loss is productive** (all-correct), opposite of 3B. Any proposed reward
  shape change must not collapse 4B's existing signal.
- **Binary 0/1 reward on GSM8K \\boxed{} extraction** (see `grpo_gsm8k_base.py:54`)
  is the current baseline. Graded rewards have not been tested.

## Dead ends (seeded from worklog.md)

(none yet — this section populates as the loop runs)

## Environment / proxy pivot (2026-04-11)

- **Tinker does not support Qwen3-0.6B / Qwen3-1.7B** — original plan used these as
  the fast Phase 1 proxy. `BadRequestError: base_model ... is not supported`.
- **Llama-3.2-1B / 3B tokenizer fails to load** under Python 3.14 + transformers 5.4
  (`ModuleNotFoundError: LlamaConfig`). Blocks using Llama as either proxy or target
  until the env is fixed.
- **Phase 1 proxy is now Qwen/Qwen3.5-4B (base)**. Smoke test @ 5 steps, group_size=4,
  temperature=0.8 on GSM8K: 0% accuracy, 100% zero_reward, 100% zero_loss — cleanly
  reproduces the zero-advantage saturation failure we are trying to rescue. 139s
  wall clock, ~$0.50 est per 5-step smoke. Full 100-step run estimated $4–$5.
- **Phase 2 target remains Llama-3.2-3B** once the tokenizer env bug is fixed, OR we
  can fall back to tokenizer-free prompting if Tinker accepts token IDs directly.
- **Smoke result baseline for comparison**: first5_avg=0.0, last10_avg=0.0, peak=0.0.
  ANY variant beating 0.0 by 10% (i.e. >0.10 last10_avg) is a Wave-1 winner.

## Open questions (to be tested)

- Does group_size >16 introduce enough diversity to escape all-incorrect saturation?
- Does higher sampling temperature (>1.0) introduce exploration without breaking
  format?
- Does easy→hard curriculum on prompt length bootstrap the 3B model?
- Does graded partial-credit reward (number of correct digits, presence of
  \\boxed{}, step-count) break zero-advantage?
- Does advantage rank-normalization help when all rewards are the same value?

## Confirmed-optimal (lock these)

(none yet — this section populates when a knob is confirmed across 4+ seeds)


## Wave wave_001 ingest (2026-04-11) — INFRA FAILURE, NOT HYPOTHESIS FAILURE

All 8 Wave-1 variants (v001-v008) failed with Tinker HTTP 402 billing block on
org `arvindcr_4@gmail.com_default_org`. None of H01–H08 were actually tested to
completion. Hypotheses REMAIN OPEN and will be re-queued once billing is
restored at https://tinker-console.thinkingmachines.ai/billing/balance.

### Infra dead end
- **Tinker billing block (2026-04-11)** — every Wave-1 run died on the first
  `sample` or `optim_step` call. Not a recipe problem. Loop cannot proceed on
  Tinker until the user pays / rotates keys.

### Informative step-1 signals (pre-billing-block)

Even though no run completed, two variants got past step 1 and gave us a tiny
peek at the baseline failure mode:

- **v003 (group_size=128)**: step 1 reward=0.012 (1.2% acc) — at least one
  non-zero reward in the 128-sample group. Suggests raw sample diversity alone
  can occasionally break the all-zero saturation on Qwen3.5-4B base. Not
  statistically meaningful (n=1 step) but directionally consistent with H03.
- **v004 (group_size=16, batch=4)**: step 1 reward=0.047 (4.7% acc) — larger
  per-step signal than v003, despite having fewer samples per prompt. Hint
  that prompt diversity matters as much as group size. Reinforces the value
  of H04.
- **v001 (group_size=32)**: survived ~519s before failing (longer than others),
  implying it ran several sample→advantage→backward cycles before hitting the
  wall. No reward number captured in the agent report, but it did not crash at
  step 0.

These are n=1 observations — treat as "do not discard these hypotheses", not
as findings. Re-run all 8 when billing is restored.

### Action items before Wave 2 can run
1. User: resolve Tinker billing at https://tinker-console.thinkingmachines.ai/billing/balance
2. User: rotate the leaked `tml-ig7SDcYd3...` / `tml-lpYVuVb7Zy4...` keys if not
   already done (they were neutralized in source but still live in git history)
3. Loop: re-queue H01–H08 (`coordinator.py queue add`) and re-run wave_001
4. Then proceed to H09–H20 as originally planned

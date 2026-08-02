# Experimental Design Comparison: Tinker RL vs. 2025/2026 Conference Practice

Last evidence refresh: **2026-07-11**.

This document turns the original fleet-generated checklist into an actionable,
source-backed experiment plan. It uses official ICML, ICLR, NeurIPS, and PMLR
pages where available. Claims that could not be tied to an identifiable paper
were removed from the priority list.

## Executive decision

The most important next step is **not** a new alignment algorithm. It is a
measurement and comparison layer that can tell whether an apparent gain is:

1. real on held-out prompts;
2. stable across prompt/sampling and training seeds;
3. produced by informative GRPO groups rather than zero-advantage batches;
4. independent of response-length inflation; and
5. worth its token and wall-clock cost.

The repository now implements the first measurement layer in
`zvf-program/experiments-next/`:

- `build_pool.py` records rollout token counts, prompt latency, retries,
  failures, goodput, and MTBF when measurable;
- `analyze_rollout_quality.py` reports zero-variance prompts, group reward
  variance, active advantages, prompt-clustered bootstrap intervals, and
  length-confounding diagnostics;
- `aggregate_seed_audits.py` aggregates at least three distinct evaluation
  seeds while explicitly refusing to call them training-seed replicates.
- `passk_eval.py` now emits problem-clustered intervals and prompt
  fingerprints; `compare_passk_results.py` refuses unverified base/post-RL
  pairing and reports paired pass@k deltas.

## Evidence corrections to the original comparison

| Original claim | Evidence-backed correction |
|---|---|
| “Advantage Collapse Rate (ACR)” is a standard metric. | Use the directly measurable **zero-variance prompt/group rate**, all-correct/all-incorrect rates, and active-advantage fraction. ICLR 2026 RL-ZVP focuses specifically on extracting signal from zero-variance prompts. |
| MathVerify is a process reward model. | Math-Verify is an outcome/answer verifier. A process or faithfulness evaluation must inspect the reasoning itself. ICML 2026 introduces Causal Importance of Reasoning (CIR) and Sufficiency of Reasoning (SR) for this purpose. |
| Longer reasoning is automatically better. | Length is an outcome and a confounder. ICLR 2026 step-entropy work and NeurIPS 2025 S-GRPO both report accuracy jointly with sequence-length reductions. |
| Three evaluation seeds prove training stability. | They only measure prompt-selection and sampling uncertainty for a frozen checkpoint. Publication claims about optimization stability still require independent **training seeds**. |
| Simultaneous multi-task batching should be implemented immediately. | First establish trustworthy single-task measurements. ICML 2026 MT-GRPO shows why naive multi-task GRPO can let tasks dominate each other and recommends worst-task accuracy and task-wise zero-advantage tracking. |
| Every named 2026 method in the old draft is accepted conference work. | Not established. In particular, this refresh does not treat uncited names such as FiMi-RM, Sensitivity-LoRA, NitroBox, AutoHet, PARGS, or MRPO as evidence. NeurIPS 2026 main-paper notifications are scheduled for [2026-09-24](https://neurips.cc/Conferences/2026/Dates), so those submissions are not cited as accepted work here. |

## Result from the first real Qwen3-8B pool

Source: [p350 rollout pool](zvf-program/experiments-next/results/pool_qwen3-8b_train_n512_r32_s42_p350.json)
(350 prompts, 32 rollouts per prompt, 11,200 rollouts, frozen base checkpoint).

The new prompt-clustered audit reports:

| Metric | Estimate | 95% prompt-clustered bootstrap CI |
|---|---:|---:|
| pass@1 | 0.3030 | [0.2787, 0.3281] |
| zero-variance prompt rate | 0.1086 | [0.0743, 0.1429] |
| active-advantage fraction | 0.8914 | [0.8571, 0.9229] |
| mean group reward variance | 0.1576 | [0.1492, 0.1662] |

All zero-variance prompts in this pool are all-incorrect; none are all-correct.
This means zero-variance groups are measurable waste, but at roughly 11% they
are **not the dominant bottleneck** for this checkpoint. Do not make RL-ZVP or
dynamic pruning the headline until post-RL checkpoints show a materially higher
rate.

The separately completed held-out baseline reports pass@1/8/32 of
**0.3044 / 0.7974 / 0.9100** over 200 GSM8K test problems. Problem-clustered
95% intervals are **[0.2752, 0.3313] / [0.7526, 0.8392] / [0.8650, 0.9500]**.
The large gap between pass@1 and pass@32 makes a matched post-RL pass@k curve
essential: it separates distribution sharpening from expansion of the
capability frontier.

The historical pool did not record token counts, so no honest length-bias
estimate can be recovered from it. New sampling records token counts; a fresh
fully instrumented pool is required for a full-coverage length audit.

## Prioritized experiment improvements

### P0 — required before new headline runs

1. **Paired base vs. post-RL held-out pass@k**
   - Same prompt IDs, tokenizer, temperature, top-p, maximum tokens, and sample
     count for both checkpoints.
   - Report pass@1, pass@8, and pass@32 with prompt-clustered intervals.
   - Primary interpretation: pass@1-only gain is sharpening; pass@32 gain is
     evidence of capability expansion.

2. **GRPO signal-quality panel for every checkpoint**
   - Zero-variance, all-correct, and all-incorrect prompt rates.
   - Mean within-group reward variance and active-advantage fraction.
   - Token-normalized goodput and retry/failure counts.
   - Do not call zero-variance prompts “collapsed advantages” without reporting
     the exact operational definition.

3. **Length-confounding panel**
   - Mean correct and incorrect response lengths.
   - Point-biserial reward/length correlation.
   - Length-predictive AUC and accuracy by equal-count length quartile.
   - Accuracy and efficiency must be reported together.

4. **Independent evaluation seeds**
   - Minimum seeds: 42, 43, and 44 for each frozen checkpoint.
   - Aggregate mean, sample standard deviation, and bootstrap interval.
   - Keep this separate from independent training-seed evidence.

### P1 — compute-matched causal ablations

5. **SFT warm-up ablation**
   - Arms: base→RL, short SFT→RL, and SFT-only.
   - Match total sampled/training tokens, not merely optimizer steps.
   - Evaluate held-out accuracy, pass@k, reasoning length, and wall-clock cost.
   - ICML 2025 long-CoT analysis finds SFT is not strictly necessary but can
     simplify and accelerate training; ICML 2026 CIR/SR work finds small SFT
     can improve reasoning faithfulness.

6. **Group-size Pareto frontier**
   - Pre-register G ∈ {4, 8, 16, 32}; use the same prompt pool and total token
     budget.
   - Report accuracy, zero-variance rate, active advantages per 1M tokens,
     output tokens/s, peak memory, and failure/retry rate.
   - Avoid tables reconstructed from fallback numbers; missing runs remain
     missing.

7. **Reasoning faithfulness probe**
   - Start with a low-cost evaluation before training a PRM: truncate or
     perturb reasoning and measure answer changes (CIR-like), then ask an
     independent verifier whether the reasoning alone determines the answer
     (SR-like).
   - Only add process/generative rewards if this probe finds a real gap.

### P2 — only after P0/P1 are complete

8. **Multi-task GRPO**
   - Track task-wise zero-variance rates and worst-task accuracy.
   - Compare uniform sampling against adaptive task weighting under matched
     tokens; average accuracy alone is insufficient.

9. **Generative reward reasoning / unverifiable data**
   - Reward Reasoning Models and JEPO justify this direction, but it changes the
     scientific question and adds judge-model confounds.
   - Keep it as a separate study after the verifiable GSM8K/MATH protocol is
     stable.

## Verified supporting papers

- [VinePPO: Refining Credit Assignment in RL Training of LLMs — ICML 2025](https://proceedings.mlr.press/v267/kazemnejad25a.html): compares training and test accuracy and reports wall-clock efficiency.
- [Demystifying Long Chain-of-Thought Reasoning — ICML 2025](https://proceedings.mlr.press/v267/yang25ae.html): SFT/no-SFT, compute, reward shaping, and OOD reasoning evidence.
- [S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models — NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/44f09d1973a88529b40029e6c3693ded-Abstract-Conference.html): reports accuracy jointly with 40.4–61.1% shorter sequences.
- [Reward Reasoning Models — NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dd35bb9efff094897fb6688a57675212-Abstract-Conference.html): uses deliberate reasoning before producing rewards.
- [Beyond Verifiable Rewards: JEPO — NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6bd67a424dc59481e1e5a5061ffc8dfe-Abstract-Conference.html): extends RL to semi-verifiable and unverifiable long-form data.
- [No Prompt Left Behind / RL-ZVP — ICLR 2026](https://iclr.cc/virtual/2026/poster/10007755): directly studies zero-variance prompts.
- [Making Slow Thinking Faster — ICLR 2026](https://iclr.cc/virtual/2026/poster/10008526): measures step entropy, sequence compression, and accuracy.
- [Quantile Advantage Estimation — ICLR 2026](https://iclr.cc/virtual/2026/poster/10009065): connects baseline design to entropy collapse/explosion and sparse advantages.
- [Multi-Task GRPO — ICML 2026](https://icml.cc/virtual/2026/poster/60544): emphasizes worst-task accuracy, adaptive task weights, and task-specific zero advantages.
- [Outcome-Based Rewards Do Not Guarantee Faithful and Verifiable Reasoning — ICML 2026](https://icml.cc/virtual/2026/poster/63576): introduces CIR and SR and tests SFT plus auxiliary reasoning rewards.
- [GRPO's Loss, Dynamics, and Success Amplification — ICML 2026](https://icml.cc/virtual/2026/poster/60548): analyzes reward normalization and KL regularization through probability of success.
- [Unbiased Dynamic Pruning for Efficient Group-Based Policy Optimization — ICML 2026](https://icml.cc/virtual/2026/poster/65323): reports unbiased pruning, token density, speed, and accuracy.

## Stop rules

- Do not launch a larger model until the P0 panel works end-to-end on the
  current Qwen3-8B checkpoint.
- Do not claim training stability from evaluation-seed sweeps.
- Do not report a held-out gain without a matched base checkpoint.
- Do not report a group-size winner unless total tokens and evaluation prompts
  are matched.
- Do not cite an unverified venue/year from the previous fleet draft.

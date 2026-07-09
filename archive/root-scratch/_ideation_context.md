# GRPO Post-Training — Ideation Brief (4 Pillars)

**Project:** "A Unified Benchmark for RL Post-Training of Language Models" (PES MTech capstone).
GRPO/PPO/DPO post-training of LLMs. We want **novel research questions + inventions** (new
methods / metrics / algorithms) grounded in the four "pillar" experiments below.

## The 4 pillars + our LATEST MEASURED results (native GPU, L40S, this week)

**Pillar 1 — PPO vs GRPO, same-stack (de-confounded).** Qwen2.5-0.5B, arithmetic verifiable reward,
identical framework/model/budget; ONLY the advantage estimator differs (GRPO group-relative baseline
vs PPO learned value head). 5 seeds × 40 steps.
- Result: GRPO held-out **0.983 ± 0.003**, PPO **0.982 ± 0.005**; paired Δ = **+0.001, t=0.34, p=0.75**
  → **no significant difference once the stack is controlled** (the paper's headline PPO-vs-GRPO gap was a stack artifact).

**Pillar 2 — Zero-Variance Fraction (ZVF).** Fraction of GRPO groups with zero reward variance
(all-correct or all-wrong → zero advantage → no learning signal). Logged per step with confounders
(batch mean reward, policy entropy, advantage variance).

**Pillar 3 — Trainability / group size.** G ∈ {2,4,8,16}, 3 seeds, held-out accuracy.
- Result: held-out **0.968 → 0.985 → 0.985 → 0.988** as G grows; mean **ZVF 0.845 → 0.754 → 0.690 → 0.631**
  → larger groups monotonically **reduce ZVF** and slightly improve held-out.

**Pillar 4 — Held-out generalization (Dr.GRPO vs GRPO).** GSM8K chain-of-thought (long-output regime
where Dr.GRPO's length-bias fix applies), Qwen2.5-1.5B-Instruct, pre→post held-out McNemar test. (running)

## What we want from you
1. Frame **sharp, novel research questions** around each pillar (things the current results *provoke*).
2. Propose **inventions**: a new metric, a new advantage/credit-assignment scheme, a new curriculum, or a
   new diagnostic — something publishable, not incremental.
3. Identify where these ideas sit vs current SOTA (2025–2026) and what would falsify them.
4. For each idea, give a **minimal experiment** we can run on a single GPU (0.5B–1.5B model) to test it.

The full paper (methods, tables, prior framing) is attached as `main.pdf`.

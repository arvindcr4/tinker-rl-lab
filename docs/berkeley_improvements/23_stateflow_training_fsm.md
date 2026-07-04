# 23 — StateFlow training-dynamics state machine (F24 L3, Chi Wang)

**Lecture mined:** F24 **L3 — Chi Wang, "Agentic frameworks (AutoGen / StateFlow)"** — a
lecture *not previously in the ledger* (covered were L1, L2, L5, L6, L8, L9, L10, L11, L12).

**Key paper (verified 2026-07-04 via arXiv abs page):**
- **Yiran Wu, Tianwei Yue, Shaokun Zhang, Chi Wang, Qingyun Wu — _StateFlow: Enhancing
  LLM Task-Solving through State-Driven Workflows_**, arXiv:2403.11322, 2024. Core claim:
  conceptualize complex task-solving as a **finite state machine** — *process* is grounded
  by discrete **states + condition-based transitions**, while sub-task work happens as
  *actions inside a state* (13–28% higher success at 3–5× less cost vs ReAct on
  InterCode-SQL / ALFWorld).

**Target:** A2 (eval methodology) + A3 (post-training science).

## The port

StateFlow grounds an agent's *process* in a state machine. We port that abstraction from
the inference loop onto the **GRPO training trajectory**: one run is a state machine over
training steps, with three **latched** states driven by observable step-level signals in
`experiments/results/groupsize_zvf_sweep.json` (same-stack sweep, 4 G × 3 seed × 40 step;
per-step `{zvf, mean_reward, entropy, advantage_variance, grad_norm}`):

```
EXPLORE  --(mean_reward ≥ 0.5·R_T)-->  CONSOLIDATE  --(mean_reward ≥ 0.9·R_T)-->  CONVERGE
```

`R_T` = mean reward over the last 10 steps (the run's own plateau); latching enforces the
StateFlow DAG (no backward transitions). This turns a training curve into a **diagnosable,
condition-transitioned process** — the exact object StateFlow argues you should reason over
instead of a monolithic rollout.

## Measured result — **4/5 DECISIVE → DECISIVE overall**
Script `scripts/berkeley/stateflow_training_fsm.py`; outputs
`experiments/results/berkeley/stateflow_*.tsv` + `stateflow_summary.json`.

| # | Hypothesis | Result | Verdict |
|---|---|---|---|
| **H1** | **State validity** — rule-FSM states are recoverable unsupervised | median step-match **0.925** vs a DP-optimal 3-segment piecewise-constant fit on standardized (reward − entropy); mean boundary offset **1.46 steps** | **DECISIVE** |
| **H2** | **Transition ordering** — entropy ↓ & reward ↑ across EXPLORE→CONSOLIDATE→CONVERGE | **12/12** runs strictly monotone on both axes | **DECISIVE** |
| **H3** | **State-aware early stop** — gradient-aware stop is the correct StateFlow terminal state | grad-aware stop (stop at adv_var<0.5) **retains 1.001** of terminal reward at **44%** compute saving; reward-only stop saves **72%** but drops to 0.982 | **DECISIVE** (efficiency); *caveat*: a fixed step-21 stop ties it (1.0012) — see H4 |
| **H4** | **G modulates the schedule** — does group size reshape state boundaries? | Spearman ρ(log₂G, conv-entry) = **−0.14**; per-G conv-entry step = {G2:10.0, G4:11.0, G8:10.3, G16:9.3} — **flat** | **NULL** (clean) |
| **H5** | **Convergence–gradient LAG** — reward-CONVERGE ≠ gradient death | grad_death − conv_entry **> 0 in 12/12** runs, **median lag = 12 steps**; all 12 learning loci (max reward gain) fall outside CONVERGE; mean CONVERGE adv_var = 0.68 (still live) | **DECISIVE** |

## Interpretation (the sharp bit)

**H5 is the discovery, and it retro-explains H3 and H4.** Reward plateaus ~11 steps
*before* the GRPO gradient actually dies: the group advantage variance stays ≈1.0 (full
group disagreement, live gradient) for a median **12 steps** after the reward-defined
CONVERGE state is entered. So:

- A **reward-plateau early-stop is premature** — it discards ~12 steps that still carry
  learning signal (this is exactly why the reward-only stop in H3 retains only 0.982). The
  correct StateFlow terminal state is **gradient death, not reward convergence**; that stop
  is lossless (retain 1.001) and still saves 44% compute.
- This **bridges row-20 (CoT-decoding tension)**: row 20 found the GRPO gradient lives in
  the *low-confidence* band that CoT-decoding discards. Row 23 gives the temporal dual —
  the gradient lives in the *post-reward-plateau* band that a reward-based stopping rule
  discards. Same mechanism (group disagreement persists past the "confident/converged"
  surface), one projected onto the confidence axis, one onto the training-time axis.
- **H4's clean NULL** (group size does not move the schedule; conv-entry ≈ step 10 for all
  G) is consistent with the pillar's Estimator-/stack-equivalence theme: G reshapes
  *variance*, not the *state schedule*. Because the schedule is so regular, a fixed-step
  stop is competitive with the adaptive one (H3 caveat) — **the value of state-awareness
  scales with schedule irregularity**, which is low on this fast arithmetic stack. On
  noisier stacks (larger models, sparser reward) the adaptive gradient-aware rule should
  separate from fixed; that is the falsifiable follow-up.

## Go / no-go

**GO as a one-sentence eval-methodology + post-training diagnostic** (A2/A3): report a
GRPO run as a 3-state StateFlow machine and *define convergence by gradient death (adv_var
collapse), not reward plateau* — the two differ by a median 12 steps and a reward-based
stop is systematically premature. Paper-facing: a P3 sentence + the row-20 bridge.
**No new section** — a stabilizer + a clean group-size NULL.

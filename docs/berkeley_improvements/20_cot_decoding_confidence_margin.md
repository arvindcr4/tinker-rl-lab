# 20 — CoT-Decoding Confidence Margin as a GRPO Group-Level Diagnostic

**Source lecture.** F24 "LLM Agents" L1 — Denny Zhou (Google DeepMind), *LLM
Reasoning*. Key paper: **Xuezhi Wang & Denny Zhou, "Chain-of-Thought Reasoning
Without Prompting," arXiv:2402.10200 (submitted 15 Feb 2024), NeurIPS 2024**
(verified via arXiv abstract + neurips.cc/media/neurips-2024 slide deck, 2026-07-04).

**Target.** A5 (inference-time reasoning) → Pillar 2 (ZVF) + Pillar 3 (group size).

## The course idea

CoT-decoding replaces greedy decoding with **top-k branching at the first step**:
several of the alternative continuations contain a chain of thought the greedy
path omits. The paper's central object is an **answer-confidence margin**

>  Δ = mean over answer tokens of ( p(top-1 token) − p(top-2 token) ),

and its headline empirical claim is that **Δ reliably selects correct reasoning
paths — better than sequence probability or answer frequency.** Confident answers
(large Δ) are disproportionately the correct, CoT-bearing ones.

## Mapping onto our RL stack

A GRPO group of `G` binary rollouts exposes a group-level analog of Δ: the
**decisiveness margin**

>  M_t = | 2 · mean_reward_t − 1 |   ∈ [0, 1],

where 0 = a maximally uncertain group (half the rollouts correct) and 1 = a
unanimous group. This is exactly the group-level image of Wang & Zhou's
answer-confidence: how peaked the group's answer distribution is. Crucially,
`M_t = 1 ⟺ ZVF` (a unanimous group yields zero within-group advantage), so
CoT-decoding's most-confident regime is GRPO's **dead-gradient** regime. That
tension is the payoff of this port.

Data: `experiments/results/groupsize_zvf_sweep.json` — same-stack sweep, 4 group
sizes {2,4,8,16} × 3 seeds × 40 steps, per-step `zvf / mean_reward / entropy /
advantage_variance / grad_norm` (Qwen2.5-0.5B, arithmetic-correctness).

## Hypotheses & measured results

| id | claim (CoT-decoding → GRPO) | metric | result | verdict |
| --- | --- | --- | --- | --- |
| H1 | M is a genuine confidence axis (Δ-analog); ZVF = P(M=1) | median within-run ρ(M_t, zvf_t) | **0.929** | **DECISIVE** |
| H2 | training surfaces confident answering (Δ↑ as CoT is learned) | runs with slope(M)>0 **and** slope(entropy)<0 | **12/12** | **DECISIVE** |
| H3 | concentration → correctness (Δ predicts correct) | Spearman(mean_entropy, heldout_acc) | −0.007 | NULL |
| H4 | **RL/CoT-decoding tension**: the learning frontier is the LOW-confidence region CoT-decoding discards | runs with M(peak-adv-var step) < M(terminal); median ρ(M, adv_var) | **12/12; ρ=−0.425** | **DECISIVE** |
| H5 | larger G sits more on the frontier at convergence (lower terminal M) | terminal M vs G | reversed (0.954→0.979) | NULL |

**Verdict: 3/5 DECISIVE → SUGGESTIVE.**

### What the decisive results say

- **H1.** M and ZVF move together within every run (median ρ=0.93): the group
  decisiveness margin is a real, measurable confidence signal in our stack, and
  ZVF is precisely its saturation event (M→1). CoT-decoding's Δ has a faithful
  group-level counterpart here.
- **H2.** In all 12 runs, margin rises and entropy falls over training: RL
  post-training *manufactures* the high-confidence answering that CoT-decoding
  surfaces at inference time. The two routes to confident CoT (decode-time
  branching vs. reward optimization) converge on the same end state.
- **H4 (novel).** The step of maximum learning signal (peak advantage-variance)
  sits at *lower* margin than the terminal state in **12/12** runs, and margin is
  negatively coupled to advantage-variance within-run (median ρ=−0.425).
  **The gradient lives in the low-confidence band CoT-decoding is built to throw
  away.** An inference-time selector that keeps only high-Δ paths would, applied
  as an RL data filter, *delete exactly the groups GRPO learns from.* This is the
  training-time dual of the frontier-synthesis observation that static G
  "over-samples the learning frontier while starving the tails" — the frontier is
  the low-confidence region, and confidence-based selection is anti-correlated
  with learning yield.

### Honest nulls

- **H3** has no power: held-out accuracy is compressed to 0.97–0.995 across all
  12 runs, so no run-level covariate can separate them. Reported as NULL, not
  spun.
- **H5 reverses**: terminal margin *increases* with G (0.954 → 0.979). The
  well-known ZVF-decreases-with-G effect (0.838 → 0.631 in this file) is a
  *whole-training* phenomenon driven by early steps, where larger groups more
  often catch some contrast. At **convergence** the sign flips: larger groups
  more reliably produce unanimous-correct groups, so terminal margin rises. The
  frontier benefit of large G is therefore a *transient*, mid-training effect,
  not a terminal one — a nuance worth stating in the Pillar-3 narrative.

## Go / No-Go

**GO as a one-sentence Pillar-2/3 sharpening + a diagnostic, not a new section.**
The paper-facing contribution is H4: it gives an inference-time (CoT-decoding)
name and mechanism to why confidence-based rollout selection is the *wrong*
compute-allocation prior for GRPO — the learning frontier is definitionally
low-confidence. This complements row 11 (ZVF-as-contrastive-yield) and the
Iso-Yield thread by importing an independent inference-time framework that
predicts the same frontier. No new environment or training run required; the
diagnostic runs in <1 s on existing per-step logs.

## Reproduce

```
python3 scripts/berkeley/cot_decoding_confidence_margin.py
# -> experiments/results/berkeley/cot_decoding_{h1..h5,summary}.{tsv,json}
```

## Citation (verified 2026-07-04)

Xuezhi Wang, Denny Zhou. *Chain-of-Thought Reasoning Without Prompting.*
arXiv:2402.10200, Feb 2024. NeurIPS 2024. (arXiv abs + NeurIPS 2024 slide deck
96654 confirm title/authors/venue.)

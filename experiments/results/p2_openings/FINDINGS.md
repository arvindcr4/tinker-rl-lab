# P2 opening — zero-compute experiment on real n2 tensors (2026-07-06)

Data: `experiments/results/n2_reward_tensor_resume/{grpo,aero,areal,gift}_s0_tensors.jsonl` (40 steps each, 16 prompts × 8 samples, binary verifiable reward). Result: `collapse_baseline_analysis.json`.

## Findings
1. **Gradient waste is massive:** mean ZVF (fraction of prompt-groups with zero within-group reward variance → zero advantage → zero gradient) = **0.72–0.77** across all four methods. ~3/4 of every batch contributes nothing.
2. **Collapse is easy-driven, not chance-driven:** ~65–71% of prompts are *all-correct* (already solved) vs only ~5% *all-wrong*. The wasted budget is spent re-solving easy prompts.
3. **Naive cross-prompt baseline = difficulty confounder:** assigning advantage to collapsed groups vs the global mean gives |adv|≈0.21 — but that signal is just "reward easiness / penalize hardness," not a learning signal (Deep Think's exact objection; RLOO already does cross-prompt baselining).
4. **Difficulty-isocline baseline correctly recovers ≈0** (|adv|≈0.012–0.02): a collapsed easy prompt's difficulty-neighbors are also easy, so no artificial signal is injected — but that also means **no recoverable gradient exists** in these groups.

## Implication (honest redirect of P2)
The proposed "Cross-Prompt Latent Contrastive Advantage" **does not recover useful gradient** in binary-reward GRPO, because collapse is dominated by already-solved easy prompts with no within-group signal. Re-baselining is the wrong lever. The data points to **difficulty-targeted sampling / adaptive curriculum**: detect solved-easy prompts and reallocate the ~72% wasted rollout budget to mid-difficulty prompts (where variance — and gradient — is highest). This is closer to AERO's "generate variance" direction but framed as a **prompt-selection curriculum** rather than a rollout-count tweak — a cleaner, testable angle.

## Next (needs a Tinker run)
Test the curriculum lever live: baseline GRPO vs "drop groups with pass-rate ∈ {0,1}, resample mid-difficulty" on Qwen 0.5B/1.5B GSM8K, compare reward-per-FLOP and final held-out accuracy.


## Independent verification (kimi, 2026-07-06) — verdict: SUSPECT
Mechanics recompute correctly, but the CONCLUSION is underpowered/overclaimed. Details in `reports/esa_phase1/verification_kimi_codex.md`. The **multi-seed campaign** (`experiments/openings/campaign.py`, W&B group `campaign`) is running to fix exactly this (3 seeds, matched baseline vs curriculum).
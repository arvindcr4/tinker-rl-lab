# P1_scaling: requests for shared files (coordinator applies)

Branch: worktree-agent-a055228624fa283bd. I did not edit `sections/_shared_*.tex`, `references.bib`, `tikz/*`, `group_size_reconcile.tex` or `related_work_v2.tex`.

## R1. `sections/_shared_methods.tex`, Sec. 2.3 Hyperparameter Mapping (finding 14). Values contradict the released configs

The current text reads "we sample G = 16 completions per prompt on medium-scale checkpoints (4B–32B, T = 30 steps) and G = 8 completions per prompt on frontier checkpoints (120B–1T, T = 20 steps) at sampling temperature 0.7, maximum generation length 512 tokens, learning rate 10^-5–2×10^-5". The released artifacts say something different:

- Trace JSONs (`platform_hybrid/experiments/tinker-runs/results/*_gsm8k_*.json`) have `group`=8, `lr`=3e-5 and `rank`=32 for the 4B–32B runs and gpt-oss-20B. They have `group`=4, `lr`=1e-5 and `rank`=16 for DeepSeek-V3.1, Nemotron-120B, Qwen3-235B and Kimi-K2.
- The W&B configs (inventory `wandb_runs.json`, project tinker-rl-lab-world-class) agree and add `batch`=2 prompts per step.
- The reward granularity matches: 1/16 steps on the medium runs (8×2 rollouts) and 1/8 on the frontier runs (4×2).
- `tinker-runs/scripts/world_class_experiments.py` uses `temperature=0.8, top_p=0.95, max_tokens=512`. (Its frontier `batch=1` disagrees with W&B and with the reward granularity, so treat the W&B config as authoritative.)

Requested replacement, old → new:

OLD: `we sample $G = 16$ completions per prompt on medium-scale checkpoints (4B--32B, $T = 30$ steps) and $G = 8$ completions per prompt on frontier checkpoints (120B--1T, $T = 20$ steps) at sampling temperature 0.7, maximum generation length 512 tokens, learning rate $10^{-5}$--$2\times10^{-5}$`

NEW: `we sample $G = 8$ completions for each of 2 prompts per step (16 rollouts) on the 4B--32B checkpoints and gpt-oss-20B (LoRA rank 32, learning rate $3\times10^{-5}$, $T = 30$ steps) and $G = 4$ completions for each of 2 prompts (8 rollouts) on the frontier checkpoints (LoRA rank 16, learning rate $10^{-5}$, $T = 20$ steps), at sampling temperature 0.8, top-$p$ 0.95, maximum generation length 512 tokens`

Before applying, check the KL coefficient claim β∈{0,0.01,0.04}: I found no KL term in `world_class_experiments.py`.

## R2. `sections/_shared_methods.tex`, Sec. 3.1 Models: "LoRA adapters (r = 32)" for frontier models

The frontier runs used rank 16 (JSON and W&B). OLD `LoRA adapters ($r = 32$)` → NEW `LoRA adapters ($r = 16$ for the frontier runs, $r = 32$ otherwise)`.

## R3. `sections/_shared_methods.tex`, Table 5 (legacy 20-test BH table), finding 16

Rank 1 ("Dense vs MoE", p = 2.4e-21) and the ‡ rows come from per-step pseudo-replication. Ideally, move Table 5 to an appendix and put the restricted Tier-A/B family in the main text with test statistic, df and n. At minimum, append this to the caption: `Rank~1 used the withdrawn 6-vs-6 split with Nemotron-120B mislabelled dense; with the corrected 7-vs-5 split and one point per anchor the exact one-sided permutation $p$ is $141/792 = 0.178$ (Sec.~\ref{sec:p1-rawtrace}).`

## R4. `tikz/taxonomy.pdf`, `tikz/reward_flow.pdf`, `tikz/pipeline.pdf` are missing (findings 36, 41–43)

Figures 1–3 of P1 render as "[missing figure artifact]" placeholders because only the `.tex` sources exist in `tikz/`. Please build them centrally. When doing so: enlarge the node fonts, use "~ Partial" consistently for HumanEval (not "◦"), and use G rather than N for group size in `reward_flow`.

**Status: done** (branch `fix/shared-tikz-figures`). The three PDFs are now built and tracked, node fonts are enlarged, HumanEval and Function Calling both show "~ Partial" to match the caption, and `reward_flow` already used G and is now included at `\textwidth`. Findings 36 and 41–43 are fixed.

## R5. `references.bib`: no changes needed for P1

The duplicate Kaplan and Nimmaturi entries, the Kimi capitalisation, OpenHands and the self-citation (findings 65–69) all render correctly in the current build.

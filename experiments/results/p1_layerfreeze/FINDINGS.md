# P1 white-box — per-layer adaptation profile under GRPO-style updates (Colab L4, 2026-07-06)

Ran on Colab L4 GPU (Tinker can't — needs per-layer gradient access). Qwen2.5-1.5B-Instruct, LoRA on q/k/v/o, 5 GRPO-style steps, advantage-weighted loss, per-layer LoRA grad-norm recorded. `experiments/openings/p1_layer_profile.py`.

## Findings (28 layers)
1. **Adaptation is PREDICTABLE from step 1:** the top-k highest-gradient layers at step 1 == the top-k over all steps (`step1_predicts_final_topk_overlap = 1.0`). => you can decide what to freeze after ONE step. This is the core premise of "predictive layer-freezing" (differentiates from SALF's during-training selection).
2. **But only MODERATELY concentrated:** top 25% of layers carry ~48% of total gradient norm (`concentration_top25pct_share = 0.476`), not the >90% a "freeze almost everything" story would need. Layer 0 dominates (grad-norm 63 vs 6-17 elsewhere); a secondary mid-late band (layers 17-23) also adapts.
3. **Honest implication:** aggressive freezing would lose real signal (half the gradient is outside the top quarter), but *predictive* freezing of the reliably-cold layers (predictable from step 1) is well supported. The paper's lever is **predictability**, not extreme sparsity.

## Caveats
Small proof-of-concept: 1.5B model, 5 steps, tiny hardcoded arithmetic set. Establishes the MECHANISM (predictability) on a real GPU; a scaled run (bigger model, GSM8K, more steps, multiple seeds) is needed to quantify the freeze-able fraction and the FLOP saving.

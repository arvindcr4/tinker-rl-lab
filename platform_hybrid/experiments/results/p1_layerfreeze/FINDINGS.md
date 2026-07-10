# P1 white-box — per-layer adaptation profile under GRPO-style updates (Colab L4, 2026-07-06)

Ran on Colab L4 GPU (Tinker can't — needs per-layer gradient access). Qwen2.5-1.5B-Instruct, LoRA on q/k/v/o, 5 GRPO-style steps, advantage-weighted loss, per-layer LoRA grad-norm recorded. `platform_hybrid/experiments/openings/p1_layer_profile.py`.

## Findings (28 layers)
1. **Adaptation is PREDICTABLE from step 1:** the top-k highest-gradient layers at step 1 == the top-k over all steps (`step1_predicts_final_topk_overlap = 1.0`). => you can decide what to freeze after ONE step. This is the core premise of "predictive layer-freezing" (differentiates from SALF's during-training selection).
2. **But only MODERATELY concentrated:** top 25% of layers carry ~48% of total gradient norm (`concentration_top25pct_share = 0.476`), not the >90% a "freeze almost everything" story would need. Layer 0 dominates (grad-norm 63 vs 6-17 elsewhere); a secondary mid-late band (layers 17-23) also adapts.
3. **Honest implication:** aggressive freezing would lose real signal (half the gradient is outside the top quarter), but *predictive* freezing of the reliably-cold layers (predictable from step 1) is well supported. The paper's lever is **predictability**, not extreme sparsity.

## Caveats
Small proof-of-concept: 1.5B model, 5 steps, tiny hardcoded arithmetic set. Establishes the MECHANISM (predictability) on a real GPU; a scaled run (bigger model, GSM8K, more steps, multiple seeds) is needed to quantify the freeze-able fraction and the FLOP saving.

## SCALED re-test (3B, real GSM8K, 2 seeds, 2026-07-06) — REVERSES the predictability claim
`p1_layer_profile_scaled.py` on Colab L4 (persistent session + ADC), Qwen2.5-3B-Instruct, 24 real GSM8K problems, 10 steps, G=4, 36 layers, seeds {0,1}. Result: `scaled_result.json`.
- **step1→final top-k overlap = 0.111** (BOTH seeds, std 0) — i.e. **~chance (1 of 9 layers, = layer 0 only)**. This DOES NOT replicate the 1.5B toy result (overlap=1.0). The "which layers matter is predictable from step 1" premise — the basis for *predictive* layer-freezing and P1's strongest positive — **collapses under scaling**. Mechanistically: layer 0 is always hot, but the mid-late band (layers ~17-25) that dominates by end-of-training is NOT identifiable at step 1; the specialization emerges over steps.
- **concentration_top25% = 0.39** (±0.013) — moderate concentration roughly HOLDS across scales (was 0.48 at 1.5B).
- **Verdict:** P1's predictability finding was an artifact of the toy 1.5B/hardcoded-arithmetic setup. Honest P1 now = "adaptation is moderately concentrated (layer-0 + a mid-late band) but NOT predictable from step 1." This is the 3rd single-run positive (after curriculum, P3) that a rigorous re-test turns into a null — the scaled-retest discipline is doing its job. Implication: P5 (provenance) stands alone as the flagship; P1's freeze-ability angle needs the emerged-band (not step-1) signal.

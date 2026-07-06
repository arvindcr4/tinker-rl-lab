# P1 actual layer-freeze test (Colab L4, Qwen2.5-1.5B, 2026-07-06)

LoRA on hot band only (layer 0 + 12-27, 61% of full params) FROZEN cold, vs full-LoRA. 2 seeds, GSM8K. `freeze_flop_result.json`.

| arm | trainable params | mean held-out gain |
|---|---|---|
| full (all 28 layers) | 2.18M | +0.0625 (s0 +0.125, s1 0.0) |
| frozen (hot band, 17 layers) | 1.32M (0.61x) | +0.000 (both seeds) |

## Finding — aggressive freezing loses the benefit
Freezing ~40% of layers (39% param saving) drops the mean held-out gain from +0.0625 to +0.000 — the frozen arm never improved. Consistent with P1's honest reframing: concentration is only MODERATE (top-25% carry ~40-48%), so you cannot freeze the cold layers without discarding real signal. The freeze lever does not work at this scale. Underpowered (n=8 held-out, 2 seeds) — both arms noisy, but frozen is strictly <= full.

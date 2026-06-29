# E2 production run (Qwen3-4B-Instruct-2507, 3 seeds)

Logged to W&B `zvf-colab-experiments` (run name `E2_lora_vs_fullft_4b`).
Pillar 4. Held-out on 50 synthetic-arithmetic problems.
Per-seed trajectories in `e2_lora_vs_fullft_4b.json`.

**Headline:** 3-seed mean heldout_delta: LoRA +0.160 (std 0.020) vs full-FT +0.100 (std 0.020); LoRA-full gap +0.060; mean ZVF: LoRA 0.954 vs full-FT 0.758

- Steps per arm: 40
- Group size G: 6
- Batch: 2
- LoRA rank/alpha/dropout: 16/32/0
- LoRA targets: q_proj, k_proj, v_proj, o_proj
- LR: LoRA=0.0001, full-FT=1e-06
- Heldout N: 50 (matched between arms; seed-reset each run)
- Seed reset per arm via `random.seed(s); torch.manual_seed(s)` so heldout set is reproducible

## Caveat (honest scope)

- Synthetic arithmetic, not GSM8K — directional evidence on the LoRA↔full axis.
- Tinker side is LoRA-only; this script's full-FT arm is the comparison Tinker can't run.
- 3 seeds is the minimum for std-dev reporting; consider 5+ for any production claim.
- 4B is the smallest model where full-FT is memory-tight on A100 40GB; larger models
  require LoRA-only or gradient-accumulation tricks.

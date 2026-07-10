# experiments/10x_structural_ceiling/configs/ — INDEX

**Purpose:** DATA DUMP — 30 run YAMLs for the structural-ceiling matrix, consumed by `../grpo_10x_runner.py` / `../round2_runner.py` (each sets model, task, GRPO hyperparams, steps).

**Naming convention:** `block_<letter>_<task>_<model/variant>.yaml`, one per matrix cell:
- `block_a_*` multi-seed GSM8K (seed4/5); `block_b_*` family isolation (gemma2_9b, mistral_7b, phi3_medium); `block_c_*` size ladder (qwen 0.6b/1.7b/4b/14b); `block_d_*ppo`, `block_e_*dpo` algorithm baselines; `block_f_tooluse_{constrained,unconstrained}`; `block_g_*group{4,32,64}` group-size; `block_h_{math,humaneval}` benchmark transfer; `block_i_*lr_{1e5,1e4,3e4}` LR sweep; `block_j_tooluse_{gemma2_9b,llama8b}` cross-family tool-use.
- `round2_*.yaml` = round-2 extensions (phase easy/mid/hard, dense_code, reinforce_gsm8k, 300-step continuations).

**How to pick a file:** map the block letter + variant from `../EXPERIMENT_MATRIX.md` to the filename.

**Find it fast:**
- group-saturation onset configs → `block_g_gsm8k_group*.yaml`
- REINFORCE / difficulty-binning → `round2_*.yaml`

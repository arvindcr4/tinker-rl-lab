# experiments/framework_config_dumps/ — INDEX

**Purpose:** Reproducibility supplement for `paper/sections/framework_configs_appendix.tex` — one exact hyperparameter YAML per framework for the canonical Qwen3-8B + GSM8K matched run (G=8, lr=1e-5, seed=42, 30 steps). Readers verify fairness by diffing the four pairwise.

**Key files:**
- `trl_qwen3_8b_gsm8k.yaml` — HuggingFace TRL GRPOTrainer (on-device ref model, kl.beta 0.04, token-level IS).
- `tinker_qwen3_8b_gsm8k.yaml` — Tinker (managed); 11 fields serialized `null # managed_by_tinker` (uncontrolled; why the "byte-identical" claim was retracted). Uses Qwen3-8B-**Base**.
- `openrlhf_qwen3_8b_gsm8k.yaml` — OpenRLHF (vLLM+Ray, kl.beta 0.02, sequence-level IS).
- `verl_qwen3_8b_gsm8k.yaml` — Bytedance veRL (sharded Ray ref, kl.beta 0.04, token-level IS).
- `README.md` — the 44-field partition (25 identical / 11 Tinker-managed / 8 documented-deviations), the Base-vs-Instruct confound, and `to_yaml()` regeneration commands.

**Find it fast:**
- which fields differ across frameworks → `README.md` "field partition"
- what Tinker hides → the `null # managed_by_tinker` lines in `tinker_qwen3_8b_gsm8k.yaml`

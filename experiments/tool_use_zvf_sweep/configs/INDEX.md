# experiments/tool_use_zvf_sweep/configs/ — INDEX

**Purpose:** DATA DUMP — auto-generated Tinker sweep YAMLs (by `../gen_configs.py`; do not hand-edit). Each sets env (group_size=16, batch=64, 50 steps), the OpenAI-compatible rollout server, and Tinker LoRA/lr/checkpoint config.

**Naming convention:** `tool_use_qwen3_<4b|8b>_<v1|v2>_s<42|43|44>.yaml` — model × reward-design × seed. `v1` = sparse reward, `v2` = partial-credit reward. `smoke_qwen3_4b_v{1,2}.yaml` = quick smoke tests. Model→tokenizer mapping is in `../manifest.tsv` (4b → Qwen3.5-4B, 8b → Qwen3-8B).

**How to pick a file:** choose model + reward version + seed; or read `../manifest.tsv`.

**Find it fast:**
- quick sanity run → `smoke_qwen3_4b_v1.yaml`

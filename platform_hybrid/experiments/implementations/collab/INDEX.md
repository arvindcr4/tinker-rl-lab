# experiments/implementations/collab/ — INDEX

**Purpose:** Teammate (collaborator) Colab/QLoRA scripts for tool-/function-calling fine-tuning on small Qwen2.5 models. Ordered pipelines: SFT → GRPO → eval. Written for T4/Colab (`!pip install` headers).

**Key files:**
- `Qwen2.5-0.5B_tool_call_finetune.py` / `Qwen2.5-0.5B_tool_call_eval.py` — 0.5B QLoRA tool-call fine-tune + eval.
- `Qwen2.5-1.5B_tool_call_sft.py` (STEP 1, glaive-function-calling-v2) → `Qwen2.5-1.5B_tool_call_grpo.py` (STEP 2, GRPO from SFT adapter) → `Qwen2.5-1.5B_tool_call_eval.py` (STEP 3, SFT-vs-GRPO comparison).
- `Qwen2.5-3B_multiturn_sft.py` / `_grpo.py` / `_eval.py` — advanced multi-turn tool-call chains (toolbench-v1); GRPO variant is "TRUE multi-turn".

**Find it fast:**
- single-model SFT→GRPO→eval flow → the three `Qwen2.5-1.5B_tool_call_*` files in order
- multi-turn tool chains → the `Qwen2.5-3B_multiturn_*` files

# Deep Dive: `platform_hybrid/experiments/implementations/collab/Qwen2.5-3B_multiturn_eval.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_hybrid/experiments/implementations/collab/Qwen2.5-3B_multiturn_eval.py` (363 lines)

## Overview
`Qwen2.5-3B_multiturn_eval.py` is an evaluation/measurement script that quantifies outcomes and produces evidence. It turns raw run outputs into comparable metrics and receipts rather than anecdotes.
It leans on **config, peft, regex, torch, transformers, wandb** to do its work.

## Key Components
- `load_model()` -- function
- `infer()` -- function
- `extract_json()` -- function
- `is_valid_tool_call()` -- function
- `is_clean_output()` -- function
- `get_tool_name()` -- function
- `get_args()` -- function
- `score_chain()` -- function
- `run_scenario()` -- Run a full multi-turn scenario.  FIX 1: wrap-up message injected immediately when tool_responses        are exhausted (tool_idx >= len), bef

## Concepts & Decisions
### Comparability over raw numbers
- **What**: Results only matter relative to a shared frozen protocol; evaluation exists to keep every framework measured against the same yardstick.

### Configuration as declarative data (YAML/JSON/TOML)
- **What**: Knobs live in YAML/JSON/TOML files or tables rather than code, so a run's intent is inspectable and diffable without reading the program.
- **Why used here**: A single frozen `CanonicalSpec` + preregistration files is the repo's whole comparability contract -- config-as-data is what makes runs hashable and testable.
- **When**: Anywhere parameters should be changeable without editing code, or compared across runs.
- **Trade-offs**: Config can drift from what the code actually reads; validation (pydantic) is what catches a key that no longer means what it says.

### Parameter-Efficient Fine-Tuning (LoRA & friends)
- **What**: PEFT freezes the base weights and trains small low-rank adapter matrices (LoRA r=16 here) plus a handful of config classes (`LoraConfig`).
- **Why used here**: LoRA makes the canonical 30-step GRPO run affordable and makes checkpoints tiny and shareable -- the repo freezes `peft` into every framework path.
- **When**: When you want to adapt a large model without retraining it or shipping full copies.
- **Trade-offs**: Adapters cap what the model can learn and add a merge step before inference; k-bit quantized base weights complicate offloading.

### Text processing with regular expressions
- **What**: `re` matches/extracts patterns in text -- parsing logs, sanitizing identifiers, or validating formats that don't warrant a full parser.
- **Why used here**: Receipts, path probes, and name munging are string-shaped; regex is the compact tool for targeted extraction.
- **When**: Small, well-defined text patterns where a parser is overkill.
- **Trade-offs**: Regex is opaque and easy to get subtly wrong; complex grammars should graduate to a real parser.

### PyTorch tensor computation & autograd
- **What**: PyTorch is the numeric engine: `torch.Tensor` holds batched GPU/CPU arrays and `torch.autograd` builds the computation graph so gradients flow from a loss back to every parameter.
- **Why used here**: TRL, transformers, vLLM and this repo's RL loops are all built on PyTorch, so using it directly avoids impedance mismatch between framework and training code.
- **When**: Anywhere gradients must reach model weights -- training, RL rollouts, LoRA adaptation, or evaluation under a different dtype.
- **Trade-offs**: Eager execution is easy to debug but slower than compiled graphs; `torch.compile`/export recover speed at the cost of traceability.

### Hugging Face Transformers (pretrained models & tokenizers)
- **What**: The `transformers` library loads pretrained checkpoints (here Qwen3-8B) and their tokenizers behind a uniform `AutoModelForCausalLM`/`AutoTokenizer` interface.
- **Why used here**: It gives one stable API over many architectures plus hosted checkpoints, which is why it is the shared backbone across every framework in this repo.
- **When**: Any task that starts from an existing LLM and adds training, serving, or eval.
- **Trade-offs**: The abstraction hides internals; subtle differences between architectures can surprise you when you rely on undocumented behavior.

### Experiment tracking with Weights & Biases
- **What**: W&B records metrics, hyperparameters, and artifacts to a hosted or local run timeline, giving every training run a shareable dashboard and history.
- **Why used here**: The repo treats receipts/evidence as first-class outputs, and W&B is one of the three independent channels (HF + W&B + GCS) whose agreement is the trust signal.
- **When**: When a run's value is in its history -- comparing sweeps, auditing, or sharing results without sending weights.
- **Trade-offs**: Adds a network dependency and an external account; local-only runs must opt out or write a local fallback.


## Related Code
- sibling `platform_hybrid/experiments/implementations/collab/Qwen2.5-0.5B_tool_call_eval.py`
- sibling `platform_hybrid/experiments/implementations/collab/Qwen2.5-0.5B_tool_call_finetune.py`
- sibling `platform_hybrid/experiments/implementations/collab/Qwen2.5-1.5B_tool_call_eval.py`
- sibling `platform_hybrid/experiments/implementations/collab/Qwen2.5-1.5B_tool_call_grpo.py`
- sibling `platform_hybrid/experiments/implementations/collab/Qwen2.5-1.5B_tool_call_sft.py`
- sibling `platform_hybrid/experiments/implementations/collab/Qwen2.5-3B_multiturn_grpo.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*

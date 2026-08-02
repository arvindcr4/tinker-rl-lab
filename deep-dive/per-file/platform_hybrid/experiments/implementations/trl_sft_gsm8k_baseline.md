# Deep Dive: `platform_hybrid/experiments/implementations/trl_sft_gsm8k_baseline.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_hybrid/experiments/implementations/trl_sft_gsm8k_baseline.py` (202 lines)

## Overview
`trl_sft_gsm8k_baseline.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **argparse, peft, torch, transformers, trl, wandb** to do its work.
*Self-description:* "TRL SFT Baseline for GSM8K =========================== Matched baseline for GRPO comparison (addresses criticism #8).  This trains the SAME model (Qwen3-8B base"

## Key Components
- `load_gsm8k_sft_dataset()` -- Load GSM8K train set formatted for SFT.  Each example becomes a (question, chain-of-thought + answer) pair, matching the format the GRPO env
- `parse_args()` -- function
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

### Command-line argument parsing
- **What**: `argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with help text and error handling for free.
- **Why used here**: Every platform entry point must be runnable by humans and by shelling-out code, so a stable, documented CLI is the contract between them.
- **When**: When a script is invoked by people, CI, or other processes and needs explicit knobs.
- **Trade-offs**: Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and auto-generated help.

### Parameter-Efficient Fine-Tuning (LoRA & friends)
- **What**: PEFT freezes the base weights and trains small low-rank adapter matrices (LoRA r=16 here) plus a handful of config classes (`LoraConfig`).
- **Why used here**: LoRA makes the canonical 30-step GRPO run affordable and makes checkpoints tiny and shareable -- the repo freezes `peft` into every framework path.
- **When**: When you want to adapt a large model without retraining it or shipping full copies.
- **Trade-offs**: Adapters cap what the model can learn and add a merge step before inference; k-bit quantized base weights complicate offloading.

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

### HF TRL training library (PPO/GRPO-style RLHF)
- **What**: `trl` implements RLHF-style trainers (PPO, and the GRPO variant used here) that coordinate policy, reference model, reward computation, and rollout generation.
- **Why used here**: It removes the burden of writing the RL loop from scratch and is one of the five frameworks whose equivalence this repo proves.
- **When**: When you need a maintained, well-tested RLHF loop and accept its configuration model.
- **Trade-offs**: The library's opinionated defaults can fight custom research setups; equivalence testing exists precisely because each framework behaves slightly differently.

### Experiment tracking with Weights & Biases
- **What**: W&B records metrics, hyperparameters, and artifacts to a hosted or local run timeline, giving every training run a shareable dashboard and history.
- **Why used here**: The repo treats receipts/evidence as first-class outputs, and W&B is one of the three independent channels (HF + W&B + GCS) whose agreement is the trust signal.
- **When**: When a run's value is in its history -- comparing sweeps, auditing, or sharing results without sending weights.
- **Trade-offs**: Adds a network dependency and an external account; local-only runs must opt out or write a local fallback.


## Related Code
- sibling `platform_hybrid/experiments/implementations/cleanrl_ppo_math.py`
- sibling `platform_hybrid/experiments/implementations/d3rlpy_offline.py`
- sibling `platform_hybrid/experiments/implementations/p1_scaled_layer_freeze.py`
- sibling `platform_hybrid/experiments/implementations/p2p3_token_budget_curriculum.py`
- sibling `platform_hybrid/experiments/implementations/p4_length_bias_kl_mask.py`
- sibling `platform_hybrid/experiments/implementations/p7_zvf_pid.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*

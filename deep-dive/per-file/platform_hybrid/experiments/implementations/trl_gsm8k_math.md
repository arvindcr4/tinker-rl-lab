# Deep Dive: `platform_hybrid/experiments/implementations/trl_gsm8k_math.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_hybrid/experiments/implementations/trl_gsm8k_math.py` (221 lines)

## Overview
`trl_gsm8k_math.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **protocol, regex, torch, transformers, trl, wandb** to do its work.
*Self-description:* "TRL GRPO GSM8K Word Problems ============================= Port of Tinker Math RL (GSM8K) to TRL.  GSM8K is a dataset of grade school math word problems. Unlike"

## Key Components
- `extract_gsm8k_answer()` -- Extract numeric answer from GSM8K-style response.  GSM8K answers are typically formatted as: "#### 42" or just a number at the end.
- `verify_gsm8k_answer()` -- Verify if predicted answer matches expected (with tolerance).
- `gsm8k_reward_function()` -- Verifiable reward function for GSM8K.  Reward structure: - reward=1.0: Correct final answer - reward=0.5: Has chain-of-thought but wrong ans
- `load_gsm8k_dataset()` -- Load GSM8K dataset with proper formatting.
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

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
- `utils.seed` &rarr; local `utils/seed.py`
- sibling `platform_hybrid/experiments/implementations/cleanrl_ppo_math.py`
- sibling `platform_hybrid/experiments/implementations/d3rlpy_offline.py`
- sibling `platform_hybrid/experiments/implementations/p1_scaled_layer_freeze.py`
- sibling `platform_hybrid/experiments/implementations/p2p3_token_budget_curriculum.py`
- sibling `platform_hybrid/experiments/implementations/p4_length_bias_kl_mask.py`
- sibling `platform_hybrid/experiments/implementations/p7_zvf_pid.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*

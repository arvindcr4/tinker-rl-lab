# Deep Dive: `zvf-program/colab-experiments/e1_evaluate_checkpoint.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `zvf-program/colab-experiments/e1_evaluate_checkpoint.py` (441 lines)

## Overview
`e1_evaluate_checkpoint.py` is an evaluation/measurement script that quantifies outcomes and produces evidence. It turns raw run outputs into comparable metrics and receipts rather than anecdotes.
It leans on **argparse, config, peft, protocol, torch, transformers, wandb** to do its work.
*Self-description:* "Resume E1 from a completed HF training checkpoint and finish evaluation."

## Key Components
- `recovered_rollout_count()` -- Recover the generated-rollout count from immutable trainer telemetry.
- `parse_args()` -- function
- `utc_now()` -- function
- `validate_progress()` -- function
- `rewind_unhashed_suffix()` -- Discard an unverifiable recovery suffix so it can be replayed with hashes.
- `load_remote_progress()` -- function
- `upload_progress()` -- function
- `copy_final_adapter()` -- function
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Comparability over raw numbers
- **What**: Results only matter relative to a shared frozen protocol; evaluation exists to keep every framework measured against the same yardstick.

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

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Command-line argument parsing
- **What**: `argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with help text and error handling for free.
- **Why used here**: Every platform entry point must be runnable by humans and by shelling-out code, so a stable, documented CLI is the contract between them.
- **When**: When a script is invoked by people, CI, or other processes and needs explicit knobs.
- **Trade-offs**: Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and auto-generated help.


## Related Code
- sibling `zvf-program/colab-experiments/_smoke_e2.py`
- sibling `zvf-program/colab-experiments/_smoke_e2_v2.py`
- sibling `zvf-program/colab-experiments/e1_grad_signal.py`
- sibling `zvf-program/colab-experiments/e1_grpo_confirmatory.py`
- sibling `zvf-program/colab-experiments/e2_lora_vs_fullft.py`
- sibling `zvf-program/colab-experiments/e2_lora_vs_fullft_4b.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*

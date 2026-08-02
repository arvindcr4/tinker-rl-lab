# Deep Dive: `platform_hybrid/experiments/modal/modal_new_experiments.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_hybrid/experiments/modal/modal_new_experiments.py` (646 lines)

## Overview
`modal_new_experiments.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **config, numpy, parallel, peft, regex, torch, transformers, wandb** to do its work.
*Self-description:* "Modal H100 experiments:  1. PPO training on Qwen3.5-4B (new model, compare with GRPO results) 2. Held-out GSM8K evaluation on our best GRPO checkpoints 3. Multi"

## Key Components
- `run_ppo_qwen35_4b()` [app.function(image=gpu_image, gpu='H100', timeout=3600, volumes={'/results': results_vol}, secrets=[modal.Secret.from_dict({'WANDB_API_KEY': WANDB_KEY, 'HF_TOKEN': HF_TOKEN, 'WANDB_PROJECT': 'tinker-rl-lab-world-class'})])] -- PPO training on Qwen3.5-4B for comparison with GRPO Tinker results.
- `run_grpo_multiseed_qwen3_8b()` [app.function(image=gpu_image, gpu='H100', timeout=3600, volumes={'/results': results_vol}, secrets=[modal.Secret.from_dict({'WANDB_API_KEY': WANDB_KEY, 'HF_TOKEN': HF_TOKEN, 'WANDB_PROJECT': 'tinker-rl-lab-world-class'})])] -- GRPO training on Qwen3-8B with different seed for variance estimation.
- `run_held_out_eval()` [app.function(image=gpu_image, gpu='H100', timeout=3600, volumes={'/results': results_vol}, secrets=[modal.Secret.from_dict({'WANDB_API_KEY': WANDB_KEY, 'HF_TOKEN': HF_TOKEN, 'WANDB_PROJECT': 'tinker-rl-lab-world-class'})])] -- Run held-out GSM8K evaluation using test split on base models.
- `main()` [app.local_entrypoint()] -- Launch all experiments in parallel.

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

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

### Text processing with regular expressions
- **What**: `re` matches/extracts patterns in text -- parsing logs, sanitizing identifiers, or validating formats that don't warrant a full parser.
- **Why used here**: Receipts, path probes, and name munging are string-shaped; regex is the compact tool for targeted extraction.
- **When**: Small, well-defined text patterns where a parser is overkill.
- **Trade-offs**: Regex is opaque and easy to get subtly wrong; complex grammars should graduate to a real parser.

### Parallelism (threads / processes / futures)
- **What**: `concurrent.futures`/threading/multiprocessing run independent work concurrently, cutting wall-clock for fan-out tasks.
- **Why used here**: Rollout generation and multi-backend dispatch are embarrassingly parallel, so pools/futures are a cheap win.
- **When**: Many independent units of work that share no mutable state.
- **Trade-offs**: Threads don't speed up CPU-bound Python (GIL); processes add copy/serialization cost and need picklable arguments.

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

### Numeric arrays with NumPy
- **What**: NumPy gives dense N-d arrays and vectorized math (reductions, broadcasting) that run at C speed.
- **Why used here**: Reward computation and metrics are array operations; vectorizing over a batch is both faster and more readable than Python loops.
- **When**: Any batched numeric transform -- rewards, accuracy, aggregations across rollouts.
- **Trade-offs**: NumPy and torch each own their memory; converting between them copies unless you share storage carefully.


## Related Code
- sibling `platform_hybrid/experiments/modal/modal_drgrpo_gsm8k_cot.py`
- sibling `platform_hybrid/experiments/modal/modal_drgrpo_vs_grpo.py`
- sibling `platform_hybrid/experiments/modal/modal_groupsize_zvf_sweep.py`
- sibling `platform_hybrid/experiments/modal/modal_grpo_openrlhf.py`
- sibling `platform_hybrid/experiments/modal/modal_grpo_skyrl.py`
- sibling `platform_hybrid/experiments/modal/modal_grpo_tinker.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*

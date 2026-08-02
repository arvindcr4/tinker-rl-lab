# Deep Dive: `platform_hybrid/experiments/modal/modal_grpo_openrlhf.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_hybrid/experiments/modal/modal_grpo_openrlhf.py` (292 lines)

## Overview
`modal_grpo_openrlhf.py` is a training path that runs gradient-based optimization (GRPO/PPO-style) over model weights. Rollouts are generated, scored by a reward model, and their feedback is backpropagated through a policy updated toward higher reward.
It leans on **config, numpy, parallel, regex, subprocess, torch, transformers, wandb, web** to do its work.
*Self-description:* "Modal GRPO Campaign — OpenRLHF launcher.  Runs the Task 4 framework-gap config on a single Modal H100 via pypi openrlhf:      Qwen3-8B, seed=42, G=8 (n_samples_"

## Key Components
- `run_openrlhf_qwen3_8b()` [app.function(image=image, gpu='H100', timeout=7200, secrets=[modal.Secret.from_dict({'HF_TOKEN': HF_TOKEN, 'WANDB_API_KEY': WANDB_KEY, 'WANDB_PROJECT': 'tinker-rl-lab-world-class'})])] -- Execute the OpenRLHF GRPO GSM8K-500 run on a single H100.
- `main()` [app.local_entrypoint()] -- function

## Concepts & Decisions
### Why the loop is GRPO and not full RLHF
- **What**: GRPO is the reward-model-free-online variant this protocol froze: it relies on group-relative advantage, cutting the value critic and memory.

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

### Web service / API layer
- **What**: A web framework (FastAPI/Flask) exposes endpoints so external actors can query or drive the system over HTTP with structured request/response schemas.
- **Why used here**: Demos and live checkpoints are served over the web so reviewers can interact without shipping weights or code.
- **When**: When something must be reachable by a browser or another service at a URL.
- **Trade-offs**: A server is a long-running, externally reachable surface -- more security and liveness concerns than a plain script.

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

### Numeric arrays with NumPy
- **What**: NumPy gives dense N-d arrays and vectorized math (reductions, broadcasting) that run at C speed.
- **Why used here**: Reward computation and metrics are array operations; vectorizing over a batch is both faster and more readable than Python loops.
- **When**: Any batched numeric transform -- rewards, accuracy, aggregations across rollouts.
- **Trade-offs**: NumPy and torch each own their memory; converting between them copies unless you share storage carefully.

### Process orchestration (subprocess)
- **What**: `subprocess.run`/`Popen` spawns and captures external commands, letting Python drive shell steps, remote CLIs, and other tools as child processes.
- **Why used here**: Remote backends provision a box then shell out to a driver command -- subprocess is the seam between 'plan' and 'actually run elsewhere'.
- **When**: When work is naturally a separate executable: `modal run`, `gcloud`, ssh commands, secondary scripts.
- **Trade-offs**: Argument quoting/escaping and env leakage are footguns; you lose in-process debugging across the boundary.


## Related Code
- sibling `platform_hybrid/experiments/modal/modal_drgrpo_gsm8k_cot.py`
- sibling `platform_hybrid/experiments/modal/modal_drgrpo_vs_grpo.py`
- sibling `platform_hybrid/experiments/modal/modal_groupsize_zvf_sweep.py`
- sibling `platform_hybrid/experiments/modal/modal_grpo_skyrl.py`
- sibling `platform_hybrid/experiments/modal/modal_grpo_tinker.py`
- sibling `platform_hybrid/experiments/modal/modal_grpo_trl.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*

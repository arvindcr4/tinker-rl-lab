# Deep Dive: `platform_modal/scripts/modal_run_experiments.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_modal/scripts/modal_run_experiments.py` (540 lines)

## Overview
`modal_run_experiments.py` is an entry point that parses intent from the command line and dispatches to the underlying machinery. It translates `--framework/--backend` flags into a plan, then executes or dry-runs it, acting as the seam between human intent and framework-specific code.
It leans on **config, numpy, peft, regex, torch, transformers, trl** to do its work.
*Self-description:* "Modal GPU Runner for TinkerRL Lab Experiments ============================================== Runs all benchmark experiments on Modal's GPU infrastructure. Uses "

## Key Components
- `run_grpo_math()` [app.function(image=llm_image, gpu='L4', timeout=3600, volumes={RESULTS_DIR: results_vol}, retries=1)] -- Run TRL GRPO on arithmetic task with Qwen2.5-0.5B.
- `run_sb3_math()` [app.function(image=rl_image, gpu='T4', timeout=600, volumes={RESULTS_DIR: results_vol})] -- Run Stable Baselines3 PPO on arithmetic environment.
- `run_cleanrl_math()` [app.function(image=rl_image, gpu='T4', timeout=600, volumes={RESULTS_DIR: results_vol})] -- Run CleanRL-style PPO on arithmetic environment.
- `run_tianshou_math()` [app.function(image=rl_image, gpu='T4', timeout=600, volumes={RESULTS_DIR: results_vol})] -- Run PPO on arithmetic (Tianshou-style architecture).
- `main()` [app.local_entrypoint()] -- function

## Concepts & Decisions
### One entry, many substrates
- **What**: Every backend eventually re-enters the local dispatch, so the CLI is both the human interface and the remote-on-box interface.

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

### Text processing with regular expressions
- **What**: `re` matches/extracts patterns in text -- parsing logs, sanitizing identifiers, or validating formats that don't warrant a full parser.
- **Why used here**: Receipts, path probes, and name munging are string-shaped; regex is the compact tool for targeted extraction.
- **When**: Small, well-defined text patterns where a parser is overkill.
- **Trade-offs**: Regex is opaque and easy to get subtly wrong; complex grammars should graduate to a real parser.

### HF TRL training library (PPO/GRPO-style RLHF)
- **What**: `trl` implements RLHF-style trainers (PPO, and the GRPO variant used here) that coordinate policy, reference model, reward computation, and rollout generation.
- **Why used here**: It removes the burden of writing the RL loop from scratch and is one of the five frameworks whose equivalence this repo proves.
- **When**: When you need a maintained, well-tested RLHF loop and accept its configuration model.
- **Trade-offs**: The library's opinionated defaults can fight custom research setups; equivalence testing exists precisely because each framework behaves slightly differently.

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
- sibling `platform_modal/scripts/_reviewer_points_extract.py`
- sibling `platform_modal/scripts/anonymize.sh`
- sibling `platform_modal/scripts/build_submission.py`
- sibling `platform_modal/scripts/build_university_submission.py`
- sibling `platform_modal/scripts/contamination_check.py`
- sibling `platform_modal/scripts/ed25519-sign.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*

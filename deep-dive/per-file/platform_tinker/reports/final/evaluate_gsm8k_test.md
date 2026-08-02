# Deep Dive: `platform_tinker/reports/final/evaluate_gsm8k_test.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_tinker/reports/final/evaluate_gsm8k_test.py` (363 lines)

## Overview
`evaluate_gsm8k_test.py` is a test/verification module that pins invariants of surrounding code so regressions are caught without manual checking. Instead of asserting on trained results, these tests assert structural and dispatch invariants (config validity, dry-run plans, framework threading).
It leans on **argparse, config, numpy, protocol, regex, torch, transformers** to do its work.
*Self-description:* "GSM8K Test Set Evaluation Script ================================= This script evaluates trained checkpoints on the held-out GSM8K test set.  Usage:     python "

## Key Components
- `bootstrap_accuracy_ci()` -- Bootstrap a 95% CI for exact-match accuracy from Bernoulli outcomes.
- `setup_argparse()` -- function
- `extract_answer()` -- Extract final numeric answer from response.  Handles multiple output formats: - GSM8K standard: #### <number> - LaTeX boxed: \boxed{<number>
- `normalize_number()` -- Normalize numbers for comparison.
- `load_model()` -- Load model and tokenizer based on arguments.
- `generate_with_tinker()` -- Generate using Tinker API.
- `generate_with_hf()` -- Generate using HuggingFace model.
- `set_seed()` -- function
- `extract_question_and_answer()` -- Extract question and ground truth from GSM8K example.
- `evaluate_model()` -- Main evaluation loop.
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Why tests here are invariants, not runs
- **What**: Each test asserts a fact that must stay true (dispatch threads the right framework, plans point at real files) -- the closest CI can get to verifying a GPU experiment without one.

### Command-line argument parsing
- **What**: `argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with help text and error handling for free.
- **Why used here**: Every platform entry point must be runnable by humans and by shelling-out code, so a stable, documented CLI is the contract between them.
- **When**: When a script is invoked by people, CI, or other processes and needs explicit knobs.
- **Trade-offs**: Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and auto-generated help.

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


## Related Code
- sibling `platform_tinker/reports/final/build_group6_final_report.sh`
- sibling `platform_tinker/reports/final/create_results_dashboard.py`
- sibling `platform_tinker/reports/final/create_submission_ppt.py`
- sibling `platform_tinker/reports/final/generate_figures.py`
- sibling `platform_tinker/reports/final/prepare_blind_review_package.py`
- sibling `platform_tinker/reports/final/run_heldout_parallel.sh`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*

# Deep Dive: `zvf-program/colab-experiments/e2_lora_vs_fullft_4b_v2.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `zvf-program/colab-experiments/e2_lora_vs_fullft_4b_v2.py` (244 lines)

## Overview
`e2_lora_vs_fullft_4b_v2.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **config, peft, regex, subprocess, torch, transformers** to do its work.
*Self-description:* "E2 production v2: LoRA vs full-FT on Qwen3-4B-Instruct-2507, HARDER task.  Differences from the v0.9 E2 run (which saturated immediately): - 5 seeds (0-4) inste"

## Key Components
- `problem()` -- 3-digit + 3-digit addition: middle-difficulty task.  v1 was a+b in [11,60] -> saturated to ceiling in 1-3 steps (no signal). v2 first attemp
- `prompt_of()` -- function
- `parse()` -- Qwen3-4B-2507 is non-thinking mode (no think blocks), but be defensive.
- `gen_group()` -- function
- `seq_logprob()` -- function
- `heldout_acc()` [torch.no_grad()] -- function
- `run()` -- function
- `aggregate()` -- function

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

### Text processing with regular expressions
- **What**: `re` matches/extracts patterns in text -- parsing logs, sanitizing identifiers, or validating formats that don't warrant a full parser.
- **Why used here**: Receipts, path probes, and name munging are string-shaped; regex is the compact tool for targeted extraction.
- **When**: Small, well-defined text patterns where a parser is overkill.
- **Trade-offs**: Regex is opaque and easy to get subtly wrong; complex grammars should graduate to a real parser.

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

### Process orchestration (subprocess)
- **What**: `subprocess.run`/`Popen` spawns and captures external commands, letting Python drive shell steps, remote CLIs, and other tools as child processes.
- **Why used here**: Remote backends provision a box then shell out to a driver command -- subprocess is the seam between 'plan' and 'actually run elsewhere'.
- **When**: When work is naturally a separate executable: `modal run`, `gcloud`, ssh commands, secondary scripts.
- **Trade-offs**: Argument quoting/escaping and env leakage are footguns; you lose in-process debugging across the boundary.


## Related Code
- sibling `zvf-program/colab-experiments/_smoke_e2.py`
- sibling `zvf-program/colab-experiments/_smoke_e2_v2.py`
- sibling `zvf-program/colab-experiments/e1_evaluate_checkpoint.py`
- sibling `zvf-program/colab-experiments/e1_grad_signal.py`
- sibling `zvf-program/colab-experiments/e1_grpo_confirmatory.py`
- sibling `zvf-program/colab-experiments/e2_lora_vs_fullft.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*

# experiments/tinker-runs/logs/ — INDEX

**Purpose:** DATA DUMP — plaintext console logs (`.log`) captured from the Tinker GRPO training runs. One file per run; skim for per-step reward / crash traces.

**Naming convention:** `grpo_100_{math,synth,xlam}.log` = 100-step data-source runs; `grpo_exp_{a,b,c,d}.log` = the A–D ablation experiments; `gsm8k_8B_s{042,137,256,512,999}.log` = Qwen3-8B GSM8K per-seed; `gsm8k_8B_rank{8,16,64}.log` = LoRA-rank ablation; `gsm8k_8B_100step.log`, `gsm8k_4B_s137.log`, `grpo_run.log` = misc.

**How to pick a file:** match the run tag (model / seed sNNN / rank / data source) to the filename; cross-reference the same tag in `../results/*.json`.

**Find it fast:**
- console output for a run → `gsm8k_<size>_s<seed>.log` or `grpo_exp_<x>.log`

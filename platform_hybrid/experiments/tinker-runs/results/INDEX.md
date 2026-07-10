# experiments/tinker-runs/results/ — INDEX

**Purpose:** DATA DUMP — per-run result JSONs from the Tinker-API GRPO campaigns. Each file is a dict with `experiment, model, task, seed, rank, lr, group, steps, run_id, checkpoint, reward trace, ...` (feeds the top-level `master_results` aggregation).

**Naming convention:** `<theme>_<task>_<model>.json`:
- `scale_gsm8k_*` = model-size scaling (qwen3-8b, qwen3.5-4b/27b, qwen3-32b, llama-8b-inst).
- `frontier_gsm8k_*` = frontier models (deepseek-v3.1, nemotron-120b, qwen3-235b).
- `moe_gsm8k_*` = MoE base vs instruct (qwen3-30b-a3b).
- `arch_gsm8k_*` = architecture probes (gpt-oss-20b, kimi-k2).
- `cross_tool_*` = tool-use cross-task (both 0%). `llama33_70b_seeds.json` = multi-seed.
- `campaign_*.json`, `campaign_v2_*.json`, `tinker_parallel_*.json` = timestamped batch dumps; `wave6_ablations.json` = the temp/rank/batch sweep results.

**How to pick a file:** identify the run in `../../experiment_summary.md`, then match `<theme>_<task>_<model>`. Small (~0.5 KB) files are partial/interrupted runs; large ones are full traces.

**Find it fast:**
- Wave-6 ablation numbers → `wave6_ablations.json`
- a scaling run → `scale_gsm8k_<model>.json`

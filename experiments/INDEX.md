# experiments/ — INDEX

**Purpose:** All RL/SFT/distillation experiments for the Tinker RL Lab (GRPO/PPO/DPO on GSM8K, tool-use, code), plus the aggregation, statistics, and paper-rendering pipeline built to answer an adversarial review. Central research question: when does RL fine-tuning help vs. hit a "structural ceiling".

**Key files (top level):**
- `experiment_summary.md` — master human-readable table of all 44 runs (GRPO/PPO per model, findings). Start here.
- `master_results.json` / `master_results.csv` — authoritative consolidated corpus (schema v3.0, 62 rows). `all_results_consolidated.json` is an earlier snapshot.
- `aggregate_results.py` — rebuilds master_results.{json,csv} from every `**/results/*.json`.
- `compute_statistics.py` + `render_stat_rigor_tex.py` — bootstrap CIs, Cohen's d, Bonferroni; drive paper stat tables (`statistical_analysis.*`, `stat_rigor_tables.json`).
- `survival_analysis.py`, `stratified_heldout.py`, `base_instruct_paired.py`, `group_size_token_normalized.py`, `variance_mitigation_integration.py`, `tool_use_reward_analysis.py`, `bfclv4_tool_use.py`, `tinker_direct_eval.py` — targeted reviewer-response analyses (each docstring names the W#/Q# it addresses).
- `power_analysis_cohens_d.py`, `fit_saturation_model.py`, `analyze_lora_sparsity.py`, `plot_*.py`, `create_presentation.py` — power analysis, curve fits, figures, pptx.
- `modal_runner.py`, `modal_batch_runner.py`, `run_tinker.sh` — launchers.
- `CHANGELOG.md` (row-count provenance), `worklog.md`, `RALPH_PLAN*.md`, `README.md`.

**Subfolders:** (each has its own INDEX.md)
- `modal/` — GPU (Modal H100) experiment scripts; the 4 "pillar" de-confound experiments live here.
- `tinker-runs/` — Tinker-API GRPO campaign scripts + logs + per-run result JSONs.
- `10x_structural_ceiling/` — the structural-ceiling study (block A–J matrix + configs + RESULTS.md).
- `tool_use_zvf_sweep/` — Tinker tool-use ZVF sweep (config gen, cost, runner).
- `implementations/` — same recipes ported to TRL/SB3/CleanRL/Tianshou/etc; `collab/` = teammate Colab scripts.
- `results/` — data dump: metrics/eval JSON+JSONL+TSV for every experiment.
- `axolotl/` — Axolotl SFT/GRPO/Dr.GRPO YAML baselines.
- `framework_config_dumps/` — exact per-framework hyperparameter YAMLs (TRL/Tinker/OpenRLHF/veRL).
- `notebooks/` — 6 tutorial notebooks for the core Tinker recipes.
- `collab-results/` — teammate notebooks + pptx deliverables.
- `_archive/` — superseded/duplicate result rows (audit trail).

**Find it fast:**
- headline results/findings → `experiment_summary.md`
- pillar (de-confound) experiments → `modal/INDEX.md`
- structural-ceiling thesis → `10x_structural_ceiling/RESULTS.md`
- raw metrics/eval data → `results/INDEX.md`

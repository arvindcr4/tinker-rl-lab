# experiments/results/ — INDEX

**Purpose:** Central DATA DUMP — raw training/eval outputs for every experiment, consumed by the top-level aggregators and paper renderers. ~34 files: large `.json`/`.jsonl` traces + small `.tsv` analysis tables + a few figures/scripts.

**Naming convention:** file name ≈ the experiment/analysis that produced it. `*_metrics.jsonl`/`*_checkpoints.jsonl` = per-step training logs (arithmetic, distillation_on/off). `<pillar>.json` = pillar outputs (`samestack_ppo_grpo.json`, `groupsize_zvf_sweep.json`, `drgrpo_gsm8k_cot*.json`, `drgrpo_vs_grpo.json`). `heldout_gsm8k.json` / `llama_heldout_gsm8k.json` = held-out eval. `tinker_gsm8k_zvf_s{42,123,456}.json` (+ `_summary`) = per-seed ZVF traces. `tinker_direct_eval*.json`, `framework_comparison.{json,pdf,png}`, `modal_results_all.json`. `*.tsv` = compact analysis tables emitted by the top-level `*.py` scripts (e.g. `variance_mitigation.tsv`, `zvf_partial_correlations.tsv`, `base_instruct_paired.tsv`, `heldout_stratified.tsv`, `survival_analysis.tsv`, `group_size_token_normalized.tsv`, `statistical_rigor_report.tsv`).

**How to pick a file:** find the experiment in `../experiment_summary.md`, then match its name here; or grep the producing script's docstring (each top-level `.py` names its output path).

**Key files:**
- `aggregate_framework_comparison.py` + `plot_framework_comparison.py` — build/plot `framework_comparison.*` from the Task-4 framework runs.

**Find it fast:**
- what produced a TSV → grep the filename in `../*.py` docstrings
- pillar result → `samestack_ppo_grpo.json` / `groupsize_zvf_sweep.json` / `drgrpo_gsm8k_cot.json`

# experiments/10x_structural_ceiling/ — INDEX

**Purpose:** The "structural ceiling" study — systematically mapping when GRPO helps (format/structure tasks: JSON, answer-boxing) vs. fails (semantic reasoning: math, code). ~50 Tinker runs across a block A–J matrix; introduces the **Group Saturation / Zero-Variance-Fraction (ZVF) diagnostic**.

**Key files:**
- `EXPERIMENT_MATRIX.md` — the block A–J plan (seeds, family isolation, size ladder, PPO/DPO, constrained decoding, G-sweep, benchmark transfer, LR sweep, tool-use). Read first.
- `RESULTS.md` — findings (ceiling hierarchy tool-use > GSM8K > MATH-500 >> HumanEval; cross-family; group saturation).
- `grpo_10x_runner.py` — unified GRPO runner (GSM8K/MATH/HumanEval/tool-use) with per-step saturation diagnostic; driven by `configs/*.yaml`.
- `round2_runner.py` — extends runner: REINFORCE, difficulty binning (GU phase diagram), dense code reward, 300-step `--resume`.
- `group_saturation_diagnostic.py` — per-step ZVF measurement + stall correlation.
- `gsm8k_dpo.py` — Block E DPO baseline. `prebin_gsm8k.py` (+ `phase_bins.json`) — difficulty pre-binning.
- `analyze_results.py` — pulls W&B, makes scaling curves / heatmaps / phase diagrams / LaTeX tables.
- `run_all.sh`, `run_parallel.sh`, `run_round2*.sh`, `run_incomplete.sh`, `run_dpo_baseline.sh` — launchers.

**Subfolders:**
- `configs/` — 30 per-block run YAMLs (`block_<a-j>_*`, `round2_*`). See its INDEX.md.

**Find it fast:**
- the experimental plan → `EXPERIMENT_MATRIX.md`
- the findings → `RESULTS.md`
- run a block → `grpo_10x_runner.py --config configs/block_<x>_*.yaml`

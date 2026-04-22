# Figure Provenance

This document traces each figure in the paper to its generation script, input data, and the exact claims made in the caption.

**Quick Verification:**
```bash
python3 scripts/regenerate_figures.py --check
```

---

## Figure Summary

| Figure | Title | Script | Input Data | Caption Claims |
|--------|-------|--------|------------|---------------|
| Fig 2 | Library Taxonomy | `tikz/taxonomy.pdf` | Manual | Library categories |
| Fig 3 | Verifiable Reward Flow | `tikz/reward_flow.pdf` | Manual | Reward paradigm description |
| Fig 4 | Experiment Pipeline | `tikz/pipeline.pdf` | Manual | Workflow description |
| Fig 5 | Learning Curves | `scripts/regenerate_figures.py::fig_learning_curves` | `master_results.json` | Arithmetic task, RL libraries |
| Fig 6 | Performance Profiles | `scripts/regenerate_figures.py::fig_performance_profiles` | `master_results.json` | IQM, median, optimality gap |
| Fig 7 | Reward-Scale Context | `scripts/regenerate_figures.py::fig_scaling` | `master_results.json` | Model size vs reward |
| Fig 8 | Sensitivity Heatmap | `scripts/regenerate_figures.py::fig_sensitivity` | `master_results.json` | Model × task grid |
| Fig 9 | Comparison Bars | `scripts/regenerate_figures.py::fig_comparison_bars` | `master_results.json` | Training family comparison |
| Fig 10 | Wave6 Sensitivity | `scripts/wave6_sensitivity.py` | `experiments/.../wave6_*.json` | Temperature, LoRA rank, batch |
| Fig 11 | Framework Comparison | `scripts/regenerate_figures.py::fig_framework` | `framework_comparison.json` | Stack-sensitivity evidence |
| Fig 12 | PPO vs GRPO | `scripts/regenerate_figures.py::fig_ppo_grpo` | `master_results.json` | Artifact-sensitive comparison |
| Fig 13 | KL Proxy | `scripts/regenerate_figures.py::fig_kl_proxy` | Step-level logs | Stability proxies |
| Fig 14 | ZVF Heatmap | `scripts/regenerate_figures.py::fig_zvf_heatmap` | `master_results.json` | Per-step ZVF |
| Fig 15 | ZVF Correlation | `scripts/regenerate_figures.py::fig_zvf_correlation` | `master_results.json` | ZVF vs performance |
| Fig 16 | Group Size Ablation | `scripts/regenerate_figures.py::fig_group_size` | `master_results.json` | G ∈ {2,4,8,16} |
| Fig 17 | Scaling Law Fits | `scripts/regenerate_figures.py::fig_scaling_law` | `master_results.json` | Exponential fits |
| Fig 18 | Scaling Parameters | `scripts/regenerate_figures.py::fig_scaling_params` | `master_results.json` | Rmax and k vs size |
| Fig 19 | Effect Sizes Forest | `scripts/regenerate_figures.py::fig_effect_sizes` | Statistical analysis | Cohen's d |
| Fig 20 | Results Dashboard | `scripts/create_results_dashboard.py` | Multiple | Panel A-D summary |

---

## Detailed Figure Analysis

### Figure 5: Learning Curves

**Script:** `scripts/regenerate_figures.py::fig_learning_curves`

**Input rows from `master_results.json`:**
- `scale_gsm8k_qwen3-8b` (GRPO, Tinker)
- `scale_gsm8k_qwen3-8b_base` (GRPO, Tinker)
- `ppo_qwen3-8b` (PPO, Modal H100)
- `trl_grpo_qwen2.5-0.5b` (TRL-GRPO, Modal L4)
- `sb3_ppo_*`, `cleanrl_ppo_*`, `tianshou_ppo_*` (Classical baselines)

**Caption in paper:**
> "Learning curves for the arithmetic sanity task across RL libraries."

**⚠️ CAPTION NOTE:** 
The figure title says "arithmetic" but the actual input data includes GSM8K, not just arithmetic. The caption should read:
> "Learning curves for the arithmetic sanity task and GSM8K across RL libraries."

This is a caption inaccuracy that does not affect scientific claims. The figure correctly shows training reward traces.

**Verification:**
```bash
python3 scripts/regenerate_figures.py --check fig_learning_curves
```

---

### Figure 6: Performance Profiles

**Script:** `scripts/regenerate_figures.py::fig_performance_profiles`

**Input:** Training reward distributions using `rliable` library

**Caption claims:**
- IQM (interquartile mean)
- Median
- Optimality gap

**Note:** Performance profiles summarize the toy arithmetic environment only; they are NOT held-out LLM capability profiles.

---

### Figure 7: Reward-Scale Context

**Script:** `scripts/regenerate_figures.py::fig_scaling`

**Input:** Peak and last-10 reward vs model parameter count

**Caption:**
> "Descriptive reward/accuracy context under the rollout/evaluation parser"

**⚠️ CAPTION NOTE:** 
"Accuracy" refers to training reward, not held-out accuracy. The figure is descriptive context, not a benchmark diagnostic.

---

### Figure 8: Sensitivity Heatmap

**Script:** `scripts/regenerate_figures.py::fig_sensitivity`

**Input:** `master_results.json` cross-model × task grid

**Caption:**
> "Final online reward by model × task"

**Verification:**
```bash
python3 scripts/regenerate_figures.py --check fig_sensitivity
```

---

### Figure 10: Wave6 Sensitivity

**Script:** `scripts/wave6_sensitivity.py`

**Input:** `experiments/tinker-runs/results/wave6_ablations.json`

**Caption:**
> "Wave~6 Qwen-8B / GSM8K sensitivity ablations"

**Variables:**
- Temperature: T ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
- LoRA rank: r ∈ {4, 8, 16, 32, 64}
- Batch size: B ∈ {1, 2, 4, 8}

---

### Figure 11: Framework Comparison

**Script:** `scripts/regenerate_figures.py::fig_framework`

**Input:** `framework_comparison.json`

**Caption:**
> "Stack-sensitivity probe. The visible GRPO configuration is nominally matched..."

**Note:** This figure is evidence that GRPO config is under-specified, NOT a causal framework ranking.

---

### Figure 12: PPO vs GRPO

**Script:** `scripts/regenerate_figures.py::fig_ppo_grpo`

**Input:** `master_results.json` Qwen and Llama PPO/GRPO rows

**Caption:**
> "Artifact-specific step-level reward comparison between GRPO and PPO"

**⚠️ IMPORTANT NOTE:**
The Qwen PPO comparison is artifact-sensitive:
- Ledger value: 0.225
- Statistics summary: 0.350

This figure is used for reporting hygiene, NOT as directional GRPO > PPO evidence.

---

### Figure 16: Group Size Ablation

**Script:** `scripts/regenerate_figures.py::fig_group_size`

**Input:** `master_results.json` with G ∈ {2, 4, 8, 16}

**Caption:**
> "Empirical GRPO group-size ablation on Qwen3-8B / GSM8K"

**Finding:** G=8 empirical sweet spot (single-seed, 30 steps)

**⚠️ LIMITATION:** Single-seed observation; multi-seed validation needed before calling G=8 optimal.

---

### Figure 17-18: Scaling Law Fits

**Script:** `scripts/regenerate_figures.py::fig_scaling_law`, `fig_scaling_params`

**Input:** Exponential saturation fits to reward traces

**Caption:**
> "Exponential saturation fits to reward traces"

**⚠️ IMPORTANT:**
- Mean R² = 0.210 (weak fit)
- Descriptive only, NOT a validated scaling law
- Short traces (20-30 steps)

---

### Figure 20: Results Dashboard

**Script:** `scripts/create_results_dashboard.py`

**Panels:**
- A: Tool-call format reward (SFT → GRPO improvement)
- B: Held-out GSM8K (82.0% → 83.3%, p=0.26)
- C: ZVF diagnostic behavior
- D: PPO/GRPO reversal across model families

---

## Figure Generation Commands

### Regenerate all figures
```bash
cd paper
python3 ../scripts/regenerate_figures.py
```

### Check figure inputs without regenerating
```bash
python3 scripts/regenerate_figures.py --check
```

### Verify specific figure
```bash
python3 scripts/regenerate_figures.py --figure learning_curves --verify
```

---

## What Figures Are NOT

| Figure | NOT a claim of... |
|--------|-------------------|
| Fig 5 | Benchmark ranking |
| Fig 6 | LLM capability profiles |
| Fig 7 | Validated scaling law |
| Fig 8 | Held-out performance matrix |
| Fig 10 | Optimal hyperparameters |
| Fig 11 | Framework superiority |
| Fig 12 | GRPO > PPO |
| Fig 16 | G=8 is globally optimal |
| Fig 17-18 | Validated scaling laws |

---

## Verification Checklist

- [ ] All input JSON files exist and contain referenced rows
- [ ] Generated PDFs match captions (except known caption issues above)
- [ ] No synthetic/fallback data used in main paper figures
- [ ] Old figure scripts with fallback generation are archived

---

## Archiving Note

The `paper/figures/` directory may contain older scripts with synthetic fallback generation. These are archived/internal and should NOT be used for verification. Use the scripts listed in the table above.

---

## Updates

| Date | Change |
|------|--------|
| 2026-04-22 | Initial version; note caption discrepancy for Fig 5 |

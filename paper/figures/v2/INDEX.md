# paper/figures/v2/ — INDEX

**Purpose:** Submission-quality figure set the paper actually embeds. Generated deterministically (seed=7) from `experiments/master_results.json` by `../../../scripts/regenerate_figures.py`. Each figure is 300dpi PNG + editable vector PDF, Okabe–Ito colorblind-safe palette, serif type matched to NeurIPS body. See `README.md` for full design rules + data provenance.

**Key files:**
- `README.md` — design rules, figure inventory table, reproduction + provenance
- `learning_curves.*` — GRPO/PPO-REINFORCE reward curves (GSM8K & tool-use)
- `comparison_bars.*` — mean±std final reward by training family
- `scaling.*`, `scaling_law_figure.png`, `scaling_params_figure.png` — params vs reward scaling
- `ppo_vs_grpo.*` — paired GSM8K curves (Qwen3-8B, Llama-3.1-8B)
- `sensitivity_heatmap.*` — model×task reward grid
- `kl_proxy.*`, `zvf_heatmap.png`, `zvf_correlation.*`, `effect_sizes_forest.png` — ZVF / KL / effect-size plots
- `group_size_ablation.*` — GRPO group-size sweep G∈{2,4,8,16,32}
- `framework_comparison.*` — Tinker GRPO vs legacy RL frameworks
- `reward_stability.png`, `performance_profiles.png`, `old_trl_seeds.png` — stability / profiles / TRL seeds

**Find it fast:**
- to know which stem maps to which figure → `README.md` inventory table
- to regenerate all 16 artifacts → `python scripts/regenerate_figures.py` (idempotent)

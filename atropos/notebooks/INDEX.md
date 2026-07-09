# atropos/notebooks/ — INDEX

**Purpose:** Jupyter notebooks presenting experiment results (WandB curves + analysis), plus the generator that builds them and Colab setup helpers.

**Key files:**
- `experiment_overview.ipynb` — big-picture summary; start here (large, embeds all plots).
- `gsm8k_qwen_8b.ipynb`, `gsm8k_qwen_30b_moe.ipynb`, `gsm8k_llama_8b.ipynb`, `gsm8k_llama_3b.ipynb` — GSM8K runs across model families (3b = negative result).
- `math_qwen_8b.ipynb`, `math_llama_8b.ipynb` — harder MATH benchmark.
- `math_curriculum.ipynb`, `moe_routing_ablation.ipynb`, `bootstrap_threshold.ipynb` — E7/E6/E5 analyses.
- `tool_use_qwen_0_5b.ipynb`, `tool_use_qwen_8b.ipynb`, `humaneval_qwen_8b.ipynb` — tool-use / code tasks.
- `systematic_scaling_study.ipynb` — cross-model scaling analysis.
- `generate_notebooks.py` — parses trainer logs and (re)builds these notebooks + HTML for sharing.
- `run_experiments_colab.ipynb`, `colab_ssh_setup.ipynb` — run experiments / SSH into Colab.
- `TRANSCRIPT.md` — presentation order and talking points for the HTML exports.

**Find it fast:**
- to see overall findings → `experiment_overview.ipynb`
- to regenerate notebooks from logs → `generate_notebooks.py`
- to run on Colab → `run_experiments_colab.ipynb` / `colab_ssh_setup.ipynb`

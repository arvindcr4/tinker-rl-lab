# experiments/modal/ — INDEX

**Purpose:** Modal (H100/A10G) GPU scripts. Home of the 4 **PILLAR** experiments that de-confound the paper's headline claims by running competing algorithms in the SAME minimal stack/model/task/compute (only the algorithm differs). Run with `modal run experiments/modal/<file>`.

**★ PILLAR scripts (start here):**
- `modal_samestack_ppo_grpo.py` — **Pillar 1**: PPO vs GRPO, same stack (Qwen2.5-0.5B arithmetic), only advantage estimator differs (group-relative vs learned value head). 5 seeds → paired test.
- `modal_groupsize_zvf_sweep.py` — **Pillars 2+3**: GRPO group-size (G) sweep with per-step ZVF / entropy / advantage-variance instrumentation + held-out eval (replaces hardcoded fallback rows).
- `modal_drgrpo_gsm8k_cot.py` — **Pillar 4**: Dr.GRPO vs GRPO on GSM8K chain-of-thought (long-output regime) + pre→post held-out McNemar generalization test.
- `modal_drgrpo_vs_grpo.py` — sibling A/B of pillar 4 on the short arithmetic probe (isolates the two Dr.GRPO fixes).

_Note: no `local_*.py` native ports exist in this repo despite prior references; the pillar scripts above are the canonical implementations._

**Framework-gap launchers (Task 4, Qwen3-8B/GSM8K G=8 lr=1e-5):**
- `modal_grpo_trl.py`, `modal_grpo_verl.py`, `modal_grpo_openrlhf.py` — GRPO on each open framework for the framework comparison.

**Other experiment scripts:**
- `modal_parallel_runner.py` — main multi-experiment runner (PPO/heldout/humaneval/KL) w/ W&B + HF Hub checkpoints.
- `modal_ppo_campaign.py`, `modal_ppo_fix.py`, `modal_new_experiments.py` — PPO campaigns / OOM fixes / new-model runs.
- `modal_heldout_eval.py`, `modal_llama_heldout_eval.py` — held-out GSM8K eval of top checkpoints (Llama gated-model variant).
- `relaunch_kl.py` — isolated KL-tracking rerun after the gradient-bug fix.
- `FIXES_APPLIED.md` — the KL no_grad / timeout / HumanEval-loading fixes.

**Subfolders:**
- `results/` — `modal_parallel_results.json` (PPO/heldout/KL outputs; see its INDEX.md).

**Find it fast:**
- PPO vs GRPO de-confounded → `modal_samestack_ppo_grpo.py`
- ZVF + group size → `modal_groupsize_zvf_sweep.py`
- Dr.GRPO / generalization → `modal_drgrpo_gsm8k_cot.py`

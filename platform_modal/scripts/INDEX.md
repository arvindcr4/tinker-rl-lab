# platform_modal/scripts/ — INDEX

**Purpose:** ~25 standalone utilities supporting experiments, figures, evaluation, statistics, and NeurIPS submission packaging (not framework integrations themselves).

**Key files by theme:**
- **Run experiments / sweeps:** `modal_run_experiments.py` (Modal GPU runner, ungated Qwen2.5), `hyperparameter_sweep_grpo.py`, `run_seeds.sh` (multi-seed), `smoke_test.sh` (<10 min reviewer test).
- **Figures:** `make_paper_figures.py`, `make_real_figures.py`, `generate_figures.py`, `regenerate_figures.py`, `regenerate_measured_figures.py`, `regenerate_missing_figures.py`, `plot_wandb_zvf.py`, `plot_erf_wandb.py`.
- **Evaluation harnesses:** `eval_harmbench.py` (safety), `run_mt_bench.py` (FastChat judge), `contamination_check.py` (train/test overlap).
- **Statistics / ZVF diagnostic:** `statistical_rigor_report.py`, `hyperparam_sensitivity.py`, `partial_correlation_zvf.py`, `zvf_compute_cross_framework.py`.
- **Submission / audit / rebuttal:** `build_submission.py`, `anonymize.sh`, `integration_audit.py`, `weave_addendum.py`, `reviewer_response_score.sh` + `_reviewer_points_extract.py`.

**Find it fast:**
- to run all experiments on GPU → `modal_run_experiments.py`
- to rebuild paper figures → `regenerate_figures.py` / `make_real_figures.py`
- to compute the ZVF metric across frameworks → `zvf_compute_cross_framework.py`
- to package the NeurIPS submission → `build_submission.py` + `anonymize.sh`

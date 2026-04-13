# Artifact Description: TinkerRL-Bench

## A Unified Benchmark for RL Post-Training of Language Models

This document follows ACM guidelines for artifact evaluation, targeting three ACM Reproducibility Badges:

- **Artifacts Available**: Permanently archived on GitHub and Hugging Face Hub
- **Artifacts Evaluated — Functional**: Documented, consistent, complete, and exercisable
- **Artifacts Evaluated — Reusable**: Well-structured for extension and repurposing

---

## 1. Abstract

This artifact contains the complete implementation of TinkerRL-Bench, a unified benchmark for RL post-training of language models. The artifact includes:

- 11 RL training implementations across 7 libraries
- Standardized evaluation scripts with statistical analysis (rliable)
- Reproducible environments via Docker and pinned dependencies
- Multi-seed experiment runners and result aggregation tools
- Hugging Face Hub integration for model checkpoint sharing

## 2. Artifact Identification

| Property | Value |
|----------|-------|
| **Title** | TinkerRL-Bench: A Unified Benchmark for RL Post-Training of Language Models |
| **Type** | Software + Data |
| **Format** | Git repository |
| **Repository** | https://github.com/arvindcr4/tinker-rl-lab |
| **License** | Apache 2.0 |
| **DOI** | _To be assigned upon archival_ |
| **Languages** | Python 3.10, Bash, LaTeX |

## 3. Description

### 3.1 How Delivered

The artifact is delivered as a Git repository containing:

```
tinker-rl-lab/
├── experiments/           # RL implementations (11 files across 7 libraries)
│   ├── implementations/   # Training scripts
│   ├── notebooks/         # Analysis notebooks
│   └── results/           # Output metrics
├── utils/                 # Shared utilities
│   ├── seed.py            # Deterministic seed management
│   └── stats.py           # Statistical analysis (rliable, bootstrap CI)
├── huggingface/           # HF Hub integration
│   ├── MODEL_CARD_TEMPLATE.md
│   └── upload_to_hub.py
├── scripts/               # Automation
│   ├── run_seeds.sh       # Multi-seed runner
│   └── anonymize.sh       # Double-blind anonymization
├── paper/                 # Manuscript sources
│   ├── main.tex           # NeurIPS format
│   ├── acm_main.tex       # ACM format
│   └── references.bib     # Shared bibliography
├── Dockerfile             # Reproducible container
├── requirements.txt       # Pinned dependencies
├── REPRODUCE.md           # Reproduction commands
├── COMPUTE.md             # Resource documentation
└── LICENSE                # Apache 2.0
```

### 3.2 Hardware Dependencies

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **GPU** | NVIDIA A100 40GB | NVIDIA A100 80GB |
| **VRAM** | 24 GB (for 1B models) | 80 GB (for 14B+ models) |
| **RAM** | 32 GB | 64 GB |
| **Disk** | 50 GB | 200 GB |
| **CUDA** | 12.1+ | 12.4+ |

### 3.3 Software Dependencies

All dependencies are pinned in `requirements.txt`. The Dockerfile provides a fully self-contained environment:

- **Base**: NVIDIA CUDA 12.4 + Ubuntu 22.04
- **Python**: 3.10
- **Key libraries**: PyTorch 2.5.1, Transformers 4.46.3, TRL 0.12.2, Stable-Baselines3 2.4.0, CleanRL 1.2.0, Tianshou 1.1.0, PufferLib 1.0.0, rl-games 1.6.1, d3rlpy 2.6.0
- **Analysis**: rliable 1.1.0, scipy 1.14.1, matplotlib 3.9.2

### 3.4 Benchmarks

| Task | Dataset | Metric | Expected Result |
|------|---------|--------|-----------------|
| Math RL (Arithmetic) | Generated | Accuracy | ~100% in 20 steps |
| Math RL (GSM8K) | GSM8K | Accuracy | See paper |
| Chat SFT | NoRobots | Loss convergence | < 1.0 |
| Preference (Shorter) | Generated | Win rate | > 50% |
| Distillation | OpenThoughts3 | KL divergence | Decreasing |

## 4. Installation & Setup

### Option A: Docker (Recommended)

```bash
# Build the container
docker build -t tinkerrl-bench .

# Run with GPU access
docker run --gpus all -it tinkerrl-bench bash
```

### Option B: Local Installation

```bash
# Create virtual environment
python3.10 -m venv tinkerrl-env
source tinkerrl-env/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Environment Variables

```bash
export TINKER_API_KEY="your-key-here"       # Required for Tinker experiments
export HF_TOKEN="your-hf-token"             # Optional: for HF Hub upload
export WANDB_API_KEY="your-wandb-key"       # Optional: for experiment tracking
```

## 5. Experiment Workflow

### 5.1 Single Experiment

```bash
python experiments/implementations/trl_grpo_math.py --seed 42
```

### 5.2 Multi-Seed Evaluation (Recommended)

```bash
# Run 5 seeds for GRPO arithmetic
./scripts/run_seeds.sh "python experiments/implementations/trl_grpo_math.py"

# Run all experiments with all seeds
for exp in experiments/implementations/*.py; do
    ./scripts/run_seeds.sh "python $exp"
done
```

### 5.3 Statistical Analysis

```bash
# Generate rliable analysis, bootstrap CIs, significance tests
python utils/stats.py --results-dir results/ --rliable --output analysis/
```

### 5.4 Upload to Hugging Face Hub

```bash
python huggingface/upload_to_hub.py \
    --model-path checkpoints/grpo-math-seed42 \
    --repo-name tinkerrl-bench/grpo-math \
    --card-template huggingface/MODEL_CARD_TEMPLATE.md
```

## 6. Evaluation Criteria Mapping

### 6.1 Artifacts Available

| Criterion | Evidence |
|-----------|----------|
| Permanently available | GitHub repository (public) + Hugging Face Hub |
| Relevant to study | All code, data scripts, and configs for paper results |
| Value beyond text | Statistical analysis tools, Docker env, model cards |

### 6.2 Artifacts Evaluated — Functional

| Criterion | Evidence |
|-----------|----------|
| **Documented** | README.md, REPRODUCE.md, ARTIFACT.md, inline code comments |
| **Consistent** | All implementations produce the results reported in the paper |
| **Complete** | All 11 implementations, analysis tools, and configs included |
| **Exercisable** | `scripts/run_seeds.sh` runs all experiments end-to-end |
| **Verification** | `utils/stats.py` produces statistical validation of results |

### 6.3 Artifacts Evaluated — Reusable

| Criterion | Evidence |
|-----------|----------|
| Well-structured | Modular design: `experiments/`, `utils/`, `scripts/`, `huggingface/` |
| Documented beyond minimum | Model card templates, CCS codes, statistical methodology |
| Standards adherence | ACM badge criteria, NeurIPS checklist, HF model card spec |
| Extensible | New RL libraries can be added by following implementation template |

## 7. Step-by-Step Reproduction Guide

To reproduce all main results from the paper:

```bash
# 1. Set up environment
docker build -t tinkerrl-bench .
docker run --gpus all -it tinkerrl-bench bash

# 2. Set API keys
export TINKER_API_KEY="your-key"

# 3. Run all experiments (5 seeds each)
for exp in experiments/implementations/*.py; do
    echo "Running: $exp"
    ./scripts/run_seeds.sh "python $exp"
done

# 4. Generate statistical analysis
python utils/stats.py \
    --results-dir results/ \
    --rliable \
    --bootstrap-samples 10000 \
    --output analysis/

# 5. Generate paper tables and figures
python utils/stats.py \
    --results-dir results/ \
    --latex \
    --output paper/tables/

# 6. Total time: ~446 A100 GPU-hours
```

## 8. Expected Results

After running all experiments, the `analysis/` directory will contain:

- `aggregate_metrics.json` — IQM, median, optimality gap per algorithm
- `pairwise_tests.csv` — Welch's t-test and Mann-Whitney U results
- `learning_curves.pdf` — Learning curves with 95% CI bands
- `performance_profiles.pdf` — rliable performance profiles
- `latex_tables/` — Ready-to-include LaTeX tables for the paper

Results should match Table 2 in the paper within the reported confidence intervals.

## 9. Artifact Archival

| Repository | Purpose | Persistence |
|------------|---------|-------------|
| GitHub | Source code, scripts, documentation | Permanent (public repo) |
| Hugging Face Hub | Model checkpoints, datasets | Permanent (Hub storage) |
| _Zenodo_ | _Snapshot DOI (to be created)_ | _Permanent (DOI)_ |

## 10. Contact

For questions about the artifact, please open a GitHub issue or contact the authors at the email addresses provided in the paper.

---

_This artifact description follows the [ACM Artifact Review and Badging v1.1](https://www.acm.org/publications/policies/artifact-review-and-badging-current) guidelines._

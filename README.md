# Tinker RL Lab

A unified benchmark for RL post-training of language models, spanning 11 implementations across 7 RL libraries with standardized evaluation protocols.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-purple.svg)](https://neurips.cc/Conferences/2026)
[![ACM](https://img.shields.io/badge/ACM-Artifact_Available-green.svg)](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
[![ACM](https://img.shields.io/badge/ACM-Artifact_Functional-green.svg)](ARTIFACT.md)
[![ACM](https://img.shields.io/badge/ACM-Artifact_Reusable-darkgreen.svg)](ARTIFACT.md)

## Overview

This repository consolidates multiple research projects focused on:
- **Tinker Platform Experiments**: RL training, SFT, preference learning, and distillation
- **Atropos Integration**: Connecting Atropos environments with Tinker API
- **Cross-Library Benchmark**: Standardized comparison across TRL, SB3, CleanRL, Tianshou, PufferLib, rl_games, d3rlpy

## Key Results

| Experiment | Method | Result (mean ± SE, 5 seeds) |
|-----------|--------|----------------------------|
| Math RL (Arithmetic) | GRPO | 69.5% → 100% accuracy in ~20 steps |
| Math RL (GSM8K) | GRPO | See scaling analysis |
| Chat SFT | SFT | Converged on NoRobots |
| Preference Shorter | DPO | Effectively shapes response style |
| Distillation (Off-Policy) | SFT | Knowledge transfer verified |
| Distillation (On-Policy) | IS Loss | KL minimization to teacher |

## Repository Structure

```
tinker-rl-lab/
├── grpo-results/         # GRPO RL experiment results and metrics
├── agentic-rl-finetuning/ # Agentic RL fine-tuning research
├── capstone-literature-survey/ # Literature Survey: RL for LLMs (GRPO Scaling)
├── experiments/           # Tinker RL Cookbook experiments
│   ├── notebooks/        # Jupyter notebooks for each experiment
│   ├── implementations/  # RL implementations (PPO, DPO, GRPO, etc.)
│   ├── results/          # Training metrics
│   └── logs/             # Training output logs
│
├── atropos/              # Tinker-Atropos integration
│   ├── tinker_atropos/   # Core package
│   │   ├── environments/ # GSM8K, Math, LogP steering
│   │   ├── trainer.py    # Tinker trainer
│   │   └── config.py     # Configuration management
│   ├── configs/          # YAML configurations
│   └── notebooks/        # Analysis notebooks
│
├── utils/                # Shared utilities
│   ├── seed.py           # Seed management for reproducibility
│   └── stats.py          # Statistical analysis (rliable, bootstrap CI)
│
├── huggingface/          # HuggingFace Hub integration
│   ├── MODEL_CARD_TEMPLATE.md
│   └── upload_to_hub.py  # Upload models/datasets to HF Hub
│
├── scripts/              # Automation scripts
│   ├── run_seeds.sh      # Multi-seed experiment runner
│   ├── anonymize.sh      # Anonymization for double-blind review
│   ├── hyperparam_sensitivity.py  # Hyperparameter sweep analysis
│   ├── contamination_check.py     # Data contamination detection
│   └── generate_figures.py        # Publication-quality figures
│
├── paper/                # Paper sources
│   ├── main.tex          # NeurIPS 2026 LaTeX source
│   ├── acm_main.tex      # ACM (sigconf) LaTeX source
│   └── references.bib    # Shared ACM-compliant bibliography
│
├── Dockerfile            # Reproducible environment
├── requirements.txt      # Pinned dependencies
├── pyproject.toml        # pip-installable package config
├── REPRODUCE.md          # Exact reproduction commands
├── COMPUTE.md            # Compute resource documentation
├── BASELINES.md          # Floor/ceiling baselines documentation
├── BENCHMARKS_COMPARISON.md  # Comparison vs existing RL benchmarks
├── CONTRIBUTING.md       # Contribution guidelines
├── NEURIPS_CHECKLIST.md  # NeurIPS paper checklist responses
├── ACM_CHECKLIST.md      # ACM submission checklist
├── ARTIFACT.md           # ACM artifact evaluation description
├── LIMITATIONS_AND_IMPACT.md  # Limitations & broader impact
├── .github/workflows/ci.yml  # GitHub Actions CI
└── LICENSE               # Apache 2.0
```

## Quick Start

### Option A: Docker (Recommended for Reproducibility)

```bash
docker build -t tinker-rl-lab .
docker run --gpus all -it tinker-rl-lab bash
```

### Option B: Local Installation

```bash
python3 -m venv tinker-env
source tinker-env/bin/activate
pip install -e .              # Install as package
# Or: pip install -r requirements.txt
```

### Running Experiments

```bash
export TINKER_API_KEY="your-key-here"

# Single seed
python experiments/implementations/trl_grpo_math.py --seed 42

# Multi-seed (recommended)
./scripts/run_seeds.sh "python experiments/implementations/trl_grpo_math.py"

# Statistical analysis
python utils/stats.py --results-dir results/ --rliable
```

See [REPRODUCE.md](REPRODUCE.md) for complete reproduction instructions.

## Experiments

### Implementations Across Libraries

| Library | File | Algorithm | Category |
|---------|------|-----------|----------|
| TRL (HuggingFace) | `trl_grpo_math.py` | GRPO | LLM-native |
| TRL | `trl_gsm8k_math.py` | GRPO | LLM-native |
| TRL | `trl_chat_sft.py` | SFT | LLM-native |
| TRL | `trl_dpo_shorter.py` | DPO | LLM-native |
| TRL | `trl_distillation.py` | SFT + KL | LLM-native |
| Stable Baselines3 | `sb3_ppo_math.py` | PPO | Classic RL |
| CleanRL | `cleanrl_ppo_math.py` | PPO | Research |
| Tianshou | `tianshou_ppo_math.py` | PPO | Modular |
| PufferLib | `pufferlib_math.py` | PPO | High-perf |
| rl_games (NVIDIA) | `rl_games_math.py` | PPO | High-perf |
| d3rlpy | `d3rlpy_offline.py` | CQL | Offline |

### Scaling Analysis

| Model | Parameters | Config |
|-------|-----------|--------|
| Llama-3.2-1B | 1B | `experiments/` |
| Llama-3.2-3B | 3B | `atropos/configs/gsm8k_llama_3b.yaml` |
| Llama-3.1-8B | 8B | `atropos/configs/gsm8k_llama_8b.yaml` |
| Qwen3-4B | 4B | `atropos/configs/gsm8k_qwen_4b.yaml` |
| Qwen3-8B | 8B | `atropos/configs/gsm8k_qwen_8b.yaml` |
| Qwen3-14B | 14B | `atropos/configs/gsm8k_qwen_14b.yaml` |
| Qwen3-30B-A3B (MoE) | 30B | `atropos/configs/gsm8k_qwen_30b_moe.yaml` |

### Consolidated Projects

This repository consolidates all PES LLM Research projects:

| Original Repo | Description | Created |
|--------------|-------------|---------|
| [tinker-experiments](https://github.com/arvindcr4/tinker-experiments) | Tinker RL Cookbook experiments | Jan 2026 |
| [tinker-atropos](https://github.com/arvindcr4/tinker-atropos) | Atropos + Tinker integration | Mar 2026 |
| [rl](https://github.com/arvindcr4/rl) | RL Gym tasks and documentation | Aug 2025 |
| [rl_master](https://github.com/arvindcr4/rl_master) | Task execution and MCP tools | Nov 2025 |
| [agentic-rl-finetuning](https://github.com/pes-llm-research/agentic-rl-finetuning) | Agentic RL fine-tuning | Mar 2026 |
| [capstone-literature-survey](https://github.com/arvindcr4/capstone-literature-survey) | GRPO Scaling Literature Survey | Mar 2026 |

## Reproducibility

This repository follows both NeurIPS and ACM reproducibility guidelines:

- **Seed management**: Deterministic across Python, NumPy, PyTorch, CUDA (`utils/seed.py`)
- **Pinned dependencies**: Exact versions in `requirements.txt`
- **Docker**: Full environment in `Dockerfile`
- **Multi-seed evaluation**: 5 seeds per experiment (42, 123, 456, 789, 1024)
- **Statistical rigor**: Bootstrap CIs, Welch's t-test, rliable metrics (`utils/stats.py`)
- **Compute documentation**: Full GPU-hours breakdown in `COMPUTE.md`
- **ACM Artifact Badges**: Targeting Artifacts Available, Evaluated—Functional, and Evaluated—Reusable ([details](ARTIFACT.md))

## Documentation

### Reproducibility & Artifacts
- [REPRODUCE.md](REPRODUCE.md) — Exact commands to reproduce all results
- [COMPUTE.md](COMPUTE.md) — Compute resources and costs
- [BASELINES.md](BASELINES.md) — Floor/ceiling baselines and statistical methodology
- [ARTIFACT.md](ARTIFACT.md) — ACM artifact evaluation description
- [LIMITATIONS_AND_IMPACT.md](LIMITATIONS_AND_IMPACT.md) — Limitations and broader impact
- [BENCHMARKS_COMPARISON.md](BENCHMARKS_COMPARISON.md) — Comparison with existing RL benchmarks
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute

### Submission Checklists
- [NEURIPS_CHECKLIST.md](NEURIPS_CHECKLIST.md) — NeurIPS paper checklist responses
- [ACM_CHECKLIST.md](ACM_CHECKLIST.md) — ACM submission checklist (CCS, badges, TAPS)

### Analysis Scripts
- `scripts/hyperparam_sensitivity.py` — Hyperparameter sensitivity sweep
- `scripts/contamination_check.py` — Data contamination detection
- `scripts/generate_figures.py` — Publication-quality figure generation

### External
- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai)
- [Atropos GitHub](https://github.com/NousResearch/atropos)
- [TRL Documentation](https://huggingface.co/docs/trl)

## Citation

### NeurIPS Format
```bibtex
@inproceedings{tinkerrl2026neurips,
  title={A Unified Benchmark for RL Post-Training of Language Models},
  author={PES LLM Research Team},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

### ACM Format
```bibtex
@inproceedings{tinkerrl2026acm,
  author    = {PES LLM Research Team},
  title     = {A Unified Benchmark for RL Post-Training of Language Models},
  booktitle = {Proceedings of [Conference]},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/nnnnnnn.nnnnnnn}
}
```

## References

- Henderson et al., [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560) (2018)
- Pineau et al., [Improving Reproducibility in ML Research](https://arxiv.org/abs/2003.12206) (2020)
- Colas et al., [A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms](https://arxiv.org/abs/1904.06979) (2019)
- Patterson et al., [Empirical Design in Reinforcement Learning](https://arxiv.org/abs/2304.01315) (2024)
- Jordan et al., [Position: Benchmarking is Limited in RL Research](https://arxiv.org/abs/2406.16241) (2024)

## Authors

PES LLM Research Team

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

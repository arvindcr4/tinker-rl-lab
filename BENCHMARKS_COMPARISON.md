# TinkerRL vs Existing RL Benchmarks

## Comparison Table

| Feature | TinkerRL | OpenAI Gym/Gymnasium | Stable Baselines3 Zoo | CleanRL | rl-baselines3-zoo | Dopamine | RLlib |
|---|---|---|---|---|---|---|---|
| **Focus** | Multi-library reproducibility | Environment standard | SB3 training recipes | Single-file implementations | Pre-trained agents | Reproducible research | Scalable production |
| **# Libraries** | 8+ (TRL, SB3, CleanRL, Tianshou, PufferLib, rl_games, d3rlpy) | N/A (env only) | 1 (SB3) | 1 (CleanRL) | 1 (SB3) | 1 (custom) | 1 (RLlib) |
| **LLM-RL** | ✅ GRPO, DPO, SFT, Distillation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Classic RL** | ✅ PPO across libraries | ✅ Environments | ✅ Multiple algorithms | ✅ PPO, DQN, etc. | ✅ Pre-trained | ✅ DQN, Rainbow, etc. | ✅ Multiple algorithms |
| **Offline RL** | ✅ CQL, IQL, BC via d3rlpy | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Limited |
| **Verifiable Rewards** | ✅ Binary correctness | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Seed Management** | ✅ Centralized utility | ❌ | Partial | ✅ Per-file | ❌ | ✅ | Partial |
| **Statistical Rigor** | ✅ IQM, bootstrap CI, performance profiles | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Reproducibility Docs** | ✅ REPRODUCE.md, Dockerfile, compute budget | ❌ | Partial | ✅ | Partial | ✅ | ❌ |
| **ACM Artifact Ready** | ✅ Badges, ARTIFACT.md | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **NeurIPS DB Ready** | ✅ Checklist, paper template | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **HuggingFace Integration** | ✅ Model cards, upload scripts | ❌ | ✅ Hub models | ❌ | ✅ Hub models | ❌ | ❌ |
| **Containerized** | ✅ Dockerfile | ❌ | ❌ | ✅ Docker support | ❌ | ❌ | ✅ Docker support |
| **pip-installable** | ✅ pyproject.toml | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **CI/CD** | ✅ GitHub Actions | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |

## Key Differentiators

### 1. Multi-Library Comparative Benchmarking
TinkerRL is the **first benchmark to run the same RL task across 8+ different libraries** with standardized evaluation. This enables direct comparison of:
- Training efficiency (samples to convergence)
- Wall-clock performance
- Code complexity and maintainability
- Reproducibility across frameworks

### 2. LLM-RL Bridge
TinkerRL uniquely spans both **classic RL** (SB3, CleanRL, Tianshou) and **LLM fine-tuning with RL** (TRL GRPO, DPO, SFT). This bridges two communities that rarely share benchmarks.

### 3. Verifiable Reward Paradigm
Following DeepSeek-R1 and Tinker (2024), TinkerRL uses **binary verifiable rewards** (correct/incorrect) rather than shaped rewards. This:
- Eliminates reward hacking
- Enables exact accuracy measurement
- Provides a cleaner signal for comparing algorithms

### 4. Statistical Best Practices Built-In
TinkerRL includes utilities for:
- Interquartile Mean (IQM) aggregation
- Bootstrap confidence intervals
- Performance profiles
- Multi-seed experiment runners

These follow recommendations from Agarwal et al. (2021) and are not available out-of-the-box in other benchmarks.

### 5. Publication-Ready Infrastructure
TinkerRL ships with:
- NeurIPS and ACM paper templates
- Artifact evaluation documentation
- Reproducibility checklists
- Figure generation scripts
- Anonymization scripts

## Limitations Compared to Existing Benchmarks

| Limitation | Details |
|---|---|
| **Task diversity** | Currently focused on arithmetic/math tasks; Gymnasium covers thousands of environments |
| **Scale** | Designed for research-scale; RLlib handles production distributed training |
| **Algorithm variety** | Focuses on PPO variants; Dopamine covers DQN/Rainbow/IQN |
| **Community size** | New project; SB3 and Gymnasium have large communities |
| **Pre-trained models** | Limited; SB3 Zoo and RLlib have extensive model zoos |

## References

- Brockman, G., et al. (2016). OpenAI Gym. arXiv:1606.01540.
- Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR.
- Huang, S., et al. (2022). CleanRL: High-quality Single-file Implementations of Deep RL Algorithms. JMLR.
- Weng, J., et al. (2022). Tianshou: A Highly Modularized Deep RL Library. JMLR.
- Agarwal, R., et al. (2021). Deep RL at the Edge of the Statistical Precipice. NeurIPS.
- Castro, P.S., et al. (2018). Dopamine: A Research Framework for Deep RL. arXiv.
- Liang, E., et al. (2018). RLlib: Abstractions for Distributed RL. ICML.

# Tinker RL Experiments - Multi-Library Implementations

Complete implementations of Tinker RL experiments ported to various RL libraries.

## Original Tinker Experiments

| Experiment | Description | Method |
|------------|-------------|--------|
| Math RL (Arithmetic) | Train LLM to add numbers | GRPO with verifiable rewards |
| Math RL (GSM8K) | Train on word problems | GRPO with chain-of-thought |
| Chat SL | Supervised fine-tuning | SFT on NoRobots dataset |
| Preference Shorter | Train for concise responses | DPO pairwise preference |
| Distillation (Off-Policy) | SFT on teacher outputs | Behavior cloning |
| Distillation (On-Policy) | KL divergence to teacher | Online KL minimization |

## Implementations

### TRL (HuggingFace) - Best for LLMs
| File | Experiment | Algorithm |
|------|------------|-----------|
| `trl_grpo_math.py` | Math RL (Arithmetic) | GRPOTrainer |
| `trl_gsm8k_math.py` | Math RL (GSM8K) | GRPOTrainer |
| `trl_chat_sft.py` | Chat SL | SFTTrainer |
| `trl_dpo_shorter.py` | Preference Shorter | DPOTrainer |
| `trl_distillation.py` | Distillation | SFT + KL loss |

### Classic RL Libraries
| File | Library | Use Case |
|------|---------|----------|
| `sb3_ppo_math.py` | Stable Baselines3 | General RL, modular |
| `cleanrl_ppo_math.py` | CleanRL | Research, single-file |
| `tianshou_ppo_math.py` | Tianshou | Modular PyTorch RL |

### High-Performance RL
| File | Library | Use Case |
|------|---------|----------|
| `pufferlib_math.py` | PufferLib | High-throughput training |
| `rl_games_math.py` | rl_games (NVIDIA) | GPU-optimized, Isaac Gym |
| `d3rlpy_offline.py` | d3rlpy | Offline RL |

## Quick Start

```bash
cd implementations/

# Install all dependencies
pip install -r requirements.txt

# Run all experiments
./run_all.sh

# Or run individual experiments
python trl_grpo_math.py      # Recommended starting point
python sb3_ppo_math.py       # Classic RL baseline
```

## Key Findings

### Library Recommendations

| Task Type | Recommended Library | Why |
|-----------|--------------------|----|
| **LLM Math RL** | TRL GRPOTrainer | Native verifiable rewards |
| **LLM Preference** | TRL DPOTrainer | Built-in pairwise learning |
| **LLM Distillation** | TRL SFTTrainer | Easy teacher-student setup |
| **Classic RL** | Stable Baselines3 | Well-documented, modular |
| **Research** | CleanRL | Transparent, single-file |
| **High Throughput** | PufferLib / rl_games | GPU-optimized |
| **Offline RL** | d3rlpy | CQL, IQL algorithms |

### Hyperparameter Mapping

| Tinker Param | SB3/CleanRL | TRL |
|--------------|-------------|-----|
| `learning_rate=1e-4` | `learning_rate=1e-4` | `learning_rate=1e-4` |
| `group_size=4` | N/A | `num_generations=4` |
| `clip_range=0.2` | `clip_range=0.2` | `eps_clip=0.2` |
| `lora_rank=32` | N/A | `lora_r=32` |

### Important Notes

1. **GRPO is best for LLM math** - Native verifiable reward support
2. **DPO degrades over iterations** - Limit to 1-2 rounds
3. **KL early stopping** - Use `target_kl=0.01` in PPO
4. **SB3/CleanRL are for RL agents, not LLMs** - Shown for pattern reference

## File Structure

```
implementations/
├── README.md                # This file
├── requirements.txt         # All dependencies
├── run_all.sh              # Run all experiments
│
├── # TRL (LLM training)
├── trl_grpo_math.py        # GRPO arithmetic
├── trl_gsm8k_math.py       # GRPO word problems
├── trl_chat_sft.py         # SFT on NoRobots
├── trl_dpo_shorter.py      # DPO preference learning
├── trl_distillation.py     # On/off-policy distillation
│
├── # Classic RL
├── sb3_ppo_math.py         # Stable Baselines3
├── cleanrl_ppo_math.py     # CleanRL
├── tianshou_ppo_math.py    # Tianshou
│
├── # High-performance RL
├── pufferlib_math.py       # PufferLib config
├── rl_games_math.py        # NVIDIA rl_games config
└── d3rlpy_offline.py       # Offline RL
```

## Reward Functions

All implementations use the same verifiable reward structure:

```python
def reward_function(completion, expected_answer):
    predicted = extract_answer(completion)

    if predicted is None:
        return -0.1  # Invalid format
    elif predicted == expected_answer:
        return 1.0   # Correct
    else:
        return 0.0   # Wrong but valid format
```

## References

- [Tinker Documentation](https://thinkingmachines.ai/tinker)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [CleanRL](https://docs.cleanrl.dev/)
- [Tianshou](https://tianshou.readthedocs.io/)
- [PufferLib](https://github.com/PufferAI/PufferLib)
- [rl_games](https://github.com/Denys88/rl_games)
- [d3rlpy](https://d3rlpy.readthedocs.io/)

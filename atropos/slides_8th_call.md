# Tinker RL Project — 8th Guidance Call
## Group 6 | PES University | 1 March 2026

---

## Slide 1: Progress Since 7th Call

### Completed
- **SOTA Literature Survey**: Mapped current state-of-the-art for GRPO on small models
- **Tinker-Atropos Setup**: Cloned, configured, and running experiments via cloud API
- **Experiment 1 Running**: Qwen3-8B GRPO on GSM8K (50 steps, in progress)
- **4 Experiment Configs Created**: Scaling ladder across model sizes

### Addressing Feedback
- "What is the current SOTA?" → Added SOTA comparison slide (Slide 3)
- "What are we improving?" → Defined contribution angle (Slide 4)
- "Try bigger models" → Qwen3-8B (running), Qwen3-30B-A3B MoE (queued)

---

## Slide 2: Experiment Status

| Experiment | Model | Size | Status | GSM8K Accuracy |
|---|---|---|---|---|
| Math RL (Arithmetic) | Llama-3.2-1B | 1B | DONE (prev) | 100% (arithmetic) |
| GSM8K Word Problems | Llama-3.2-1B | 1B | DONE (prev) | ~63% |
| **GSM8K Scaling** | **Qwen3-8B** | **8B** | **RUNNING** | **TBD** |
| GSM8K Scaling | Qwen3-30B-A3B | 30B (3B active) | QUEUED | TBD |
| GSM8K Scaling | Llama-3.2-3B | 3B | QUEUED | TBD |
| GSM8K Scaling | Llama-3.1-8B | 8B | QUEUED (needs HF) | TBD |

**Training Details (Qwen3-8B)**:
- Method: GRPO, LoRA rank 32, lr=4e-5
- Batch size: 128, Group size: 16
- Platform: Tinker cloud API (GPU provisioned remotely)
- Early metrics (step 0-7): Loss dropping 205→22, reward ~5-8%

---

## Slide 3: Current SOTA — Where We Stand

### SOTA: Small Model RL for Math Reasoning (2025-2026)

| Model | Size | Method | Best Result | Source |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Distillation from 671B | MATH-500: 83.9% | DeepSeek, Jan 2025 |
| DeepScaleR-1.5B | 1.5B | GRPO on distilled model | Beats o1-Preview (AIME) | Feb 2025 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | Distillation from 671B | MATH-500: 92.8% | DeepSeek, Jan 2025 |
| DeepSeek-R1-Distill-Llama-8B | 8B | Distillation from 671B | MATH-500: 89.1% | DeepSeek, Jan 2025 |
| SimpleRL-Zoo | 7B | Zero RL (GRPO) | Substantial gains (8K examples) | COLM 2025 |
| TinyZero | 3B | GRPO | Emergent "aha moments" | Feb 2025 |
| DeepCoder-14B | 14B | RL + code rewards | 60.6% LiveCodeBench (≈o3-mini) | Apr 2025 |
| **Our result** | **1B** | **GRPO (no distillation)** | **GSM8K: ~63%** | **This project** |

### Key Observation
SOTA models (DeepScaleR, DeepSeek-R1-Distill) start from a **distilled checkpoint** from a 671B teacher model.
Our approach: **pure GRPO from a general-purpose base model** — no distillation, no massive teacher required.

---

## Slide 4: Our Contribution — What We're Improving

### Gap in Literature
1. Most GRPO results are on **Qwen-family** models only — Llama behavior under GRPO is underexplored
2. No systematic **scaling study** comparing GRPO across model families (Llama vs Qwen) at multiple sizes
3. Existing work requires **expensive infrastructure** — our Tinker-based approach is cloud-accessible
4. **Reproducibility**: Most papers lack runnable notebooks

### Our Contribution
> **Systematic GRPO scaling analysis across model families and sizes (1B→8B→30B) on math reasoning, with reproducible cloud-based experiments on the Tinker platform**

### Specific Experiments
| Dimension | What we compare |
|---|---|
| **Model size** | 1B → 3B → 8B → 30B (MoE) |
| **Model family** | Llama vs Qwen under identical GRPO recipe |
| **Training paradigm** | Pure GRPO vs Distillation+GRPO |
| **Infrastructure** | Cloud API (Tinker) — no GPU management |
| **Reproducibility** | Jupyter notebooks + YAML configs for every run |

---

## Slide 5: Technical Setup

### Architecture: Tinker + Atropos

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  GSM8K Env       │────▶│  Atropos API      │────▶│  Tinker Trainer     │
│  (rollouts +     │     │  (coordination)   │     │  (cloud GPU)       │
│   scoring)       │◀────│                    │◀────│  LoRA + GRPO       │
└─────────────────┘     └──────────────────┘     └────────────────────┘
      ▲                                                    │
      │              ┌──────────────────┐                  │
      └──────────────│  Tinker Inference │◀─────────────────┘
                     │  (sampling)       │  updated weights
                     └──────────────────┘  each step
```

- **GRPO**: No critic/reward model needed (1x memory vs PPO's 4x)
- **LoRA**: Rank 32, ~0.1% of parameters trained
- **Cloud API**: All GPU compute via Tinker — runs from CPU-only laptop
- **Verifiable rewards**: Binary (correct/incorrect) per GSM8K answer

### Hyperparameters (Consistent Across All Experiments)
| Parameter | Value |
|---|---|
| LoRA rank | 32 |
| Learning rate | 3-5 × 10⁻⁵ |
| Batch size | 128 |
| Group size | 16 |
| Max token length | 512 |
| Training steps | 50 |
| Loss function | Importance Sampling |

---

## Slide 6: Timeline & Next Steps

### Upcoming
| Date | Milestone |
|---|---|
| Mar 1 | Qwen3-8B experiment running |
| Mar 3-5 | Complete Qwen3-30B-A3B MoE experiment |
| Mar 5-7 | Llama-3B and Llama-8B experiments (after HF login) |
| Mar 7 | 9th Guidance — present scaling results |
| Mar 14-21 | Ablation studies (lr, rank, group_size) |
| Mar 28 | **3rd Submission — Interim Report** |
| Apr 11 | Final Report + Conference Paper |

### For Interim Report (Mar 28)
- Chapter 3: Methodology (GRPO on Tinker, experiment design)
- Scaling ladder results with charts
- SOTA comparison table
- Ablation analysis

### Guidance Questions
1. Should we add a **code generation** experiment (DeepCoder recipe) or focus on math scaling depth?
2. Is the **Llama vs Qwen comparison** angle strong enough for conference paper?
3. Any preference on conference venue? (NeurIPS workshop, EMNLP, ACL findings?)
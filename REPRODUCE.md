# Reproducing TinkerRL Lab Experiments

This document provides exact commands to reproduce every experiment reported in the paper.

## Prerequisites

### Option A: Docker (Recommended)

```bash
# Build the container
docker build -t tinker-rl-lab .

# Run with GPU access
docker run --gpus all -v $(pwd)/results:/workspace/tinker-rl-lab/results -it tinker-rl-lab bash
```

### Option B: Local Installation

```bash
# Requires Python 3.10+, CUDA 12.4+
python3 -m venv tinker-env
source tinker-env/bin/activate
pip install -r requirements.txt
```

### Environment Variables

```bash
export TINKER_API_KEY="<your-key>"   # Required for Tinker experiments
export WANDB_API_KEY="<your-key>"     # Optional, for experiment tracking
export SEED=42                         # Default seed
```

---

## Multi-Seed Runs

All experiments must be run with multiple seeds for statistical validity.
We use seeds: `[42, 123, 456, 789, 1024]`.

The `run_seeds.sh` script automates this:

```bash
# Run any experiment with 5 seeds
./scripts/run_seeds.sh "python experiments/implementations/trl_grpo_math.py" 42 123 456 789 1024

# Aggregate results
python utils/stats.py --results-dir results/ --experiment trl_grpo_math
```

---

## Experiment 1: Math RL (Arithmetic) — GRPO

### TRL Implementation (Primary)

```bash
for SEED in 42 123 456 789 1024; do
    python experiments/implementations/trl_grpo_math.py \
        --seed $SEED \
        --output-dir results/trl_grpo_math/seed_${SEED}
done
```

### Cross-Library Baselines

```bash
# Stable Baselines3
for SEED in 42 123 456 789 1024; do
    python experiments/implementations/sb3_ppo_math.py --seed $SEED
done

# CleanRL
for SEED in 42 123 456 789 1024; do
    python experiments/implementations/cleanrl_ppo_math.py --seed $SEED
done

# Tianshou
for SEED in 42 123 456 789 1024; do
    python experiments/implementations/tianshou_ppo_math.py --seed $SEED
done
```

---

## Experiment 2: Math RL (GSM8K) — GRPO

```bash
for SEED in 42 123 456 789 1024; do
    python experiments/implementations/trl_gsm8k_math.py \
        --seed $SEED \
        --output-dir results/trl_gsm8k_math/seed_${SEED}
done
```

---

## Experiment 3: Chat Supervised Fine-Tuning

```bash
for SEED in 42 123 456 789 1024; do
    python experiments/implementations/trl_chat_sft.py \
        --seed $SEED \
        --output-dir results/trl_chat_sft/seed_${SEED}
done
```

---

## Experiment 4: Preference Learning (DPO — Shorter)

```bash
for SEED in 42 123 456 789 1024; do
    python experiments/implementations/trl_dpo_shorter.py \
        --seed $SEED \
        --output-dir results/trl_dpo_shorter/seed_${SEED}
done
```

---

## Experiment 5: Knowledge Distillation

```bash
for SEED in 42 123 456 789 1024; do
    python experiments/implementations/trl_distillation.py \
        --seed $SEED \
        --output-dir results/trl_distillation/seed_${SEED}
done
```

---

## Experiment 6: Atropos + Tinker Integration (GSM8K)

```bash
# Terminal 1: Start Atropos API
run-api

# Terminal 2: Start training with specific seed
for SEED in 42 123 456 789 1024; do
    SEED=$SEED python atropos/launch_training.py \
        --config atropos/configs/gsm8k_llama_8b.yaml \
        --seed $SEED
done

# Terminal 3: Start environment
python atropos/tinker_atropos/environments/gsm8k_tinker.py serve \
    --config atropos/configs/gsm8k_llama_8b.yaml
```

---

## Experiment 7: Scaling Analysis (3B → 30B)

```bash
for MODEL_CONFIG in gsm8k_llama_3b gsm8k_llama_8b gsm8k_qwen_4b gsm8k_qwen_8b gsm8k_qwen_14b gsm8k_qwen_30b_moe; do
    for SEED in 42 123 456 789 1024; do
        SEED=$SEED python atropos/launch_training.py \
            --config atropos/configs/${MODEL_CONFIG}.yaml \
            --seed $SEED
    done
done
```

---

## Statistical Analysis

After running all experiments:

```bash
# Generate tables and figures with confidence intervals
python utils/stats.py \
    --results-dir results/ \
    --output-dir paper/figures/ \
    --format latex

# Generate rliable-style aggregate metrics
python utils/stats.py \
    --results-dir results/ \
    --rliable \
    --output-dir paper/figures/
```

---

## Compute Requirements

| Experiment | GPU | Time per Seed | Total (5 seeds) |
|-----------|-----|---------------|-----------------|
| Math RL (Arithmetic) | 1x A100 40GB | ~30 min | ~2.5 hrs |
| Math RL (GSM8K) | 1x A100 80GB | ~4 hrs | ~20 hrs |
| Chat SFT | 1x A100 40GB | ~2 hrs | ~10 hrs |
| DPO Shorter | 1x A100 40GB | ~1 hr | ~5 hrs |
| Distillation | 1x A100 80GB | ~3 hrs | ~15 hrs |
| Scaling (3B-30B) | 1-4x A100 80GB | ~2-12 hrs each | ~150 hrs |

**Total estimated compute: ~200 GPU-hours on A100s**

---

## Verifying Results

To verify that results match those reported in the paper:

```bash
# Compare your results against reported values
python utils/verify_results.py \
    --results-dir results/ \
    --expected-results paper/expected_results.json \
    --tolerance 0.02
```

Results should fall within ±2% of reported values (accounting for hardware differences and remaining non-determinism in CUDA operations).

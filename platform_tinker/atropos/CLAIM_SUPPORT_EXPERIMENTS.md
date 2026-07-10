# Claim-Support Experiment Matrix

This matrix is the minimum credible package for defending a claim stronger than:

> "We optimized GSM8K."

It is designed to answer the reviewer objection that GSM8K saturation alone does not establish robust mathematical reasoning.

## What This Package Tests

1. `Matched-family scaling`
Why: separates a real scaling trend from a one-off 3B vs 8B anecdote.

2. `Replication`
Why: shows the result is not a lucky single trajectory.

3. `Prompt-prefix ablation`
Why: checks whether the worked-example prefix is carrying the result.

4. `Bootstrap-threshold control`
Why: distinguishes model capacity from initial-signal / task-difficulty effects.

5. `Harder in-domain math`
Why: tests whether GSM8K gains survive on a more difficult symbolic-math benchmark.

6. `Out-of-distribution benchmark evaluation`
Why: tests whether the trained checkpoint generalizes beyond one saturated benchmark.

## Training Blocks

### 1. Matched Qwen GSM8K scaling
- [gsm8k_qwen_4b.yaml](/home/arvind/tinker-rl-lab/atropos/configs/gsm8k_qwen_4b.yaml)
- [gsm8k_qwen_8b.yaml](/home/arvind/tinker-rl-lab/atropos/configs/gsm8k_qwen_8b.yaml)
- [gsm8k_qwen_14b.yaml](/home/arvind/tinker-rl-lab/atropos/configs/gsm8k_qwen_14b.yaml)

Run:
```bash
./run_claim_support_train.sh scaling
```

### 2. Replication / prompt / bootstrap controls
- [gsm8k_qwen_8b_seed1.yaml](/home/arvind/tinker-rl-lab/atropos/configs/gsm8k_qwen_8b_seed1.yaml)
- [gsm8k_qwen_8b_seed2.yaml](/home/arvind/tinker-rl-lab/atropos/configs/gsm8k_qwen_8b_seed2.yaml)
- [gsm8k_qwen_8b_seed3.yaml](/home/arvind/tinker-rl-lab/atropos/configs/gsm8k_qwen_8b_seed3.yaml)
- [gsm8k_qwen_8b_no_prefix.yaml](/home/arvind/tinker-rl-lab/atropos/configs/gsm8k_qwen_8b_no_prefix.yaml)
- [bootstrap_threshold_easy.yaml](/home/arvind/tinker-rl-lab/atropos/configs/bootstrap_threshold_easy.yaml)
- [bootstrap_threshold_hardest.yaml](/home/arvind/tinker-rl-lab/atropos/configs/bootstrap_threshold_hardest.yaml)

Run:
```bash
./run_claim_support_train.sh controls
```

### 3. Harder math benchmark training
- [math_qwen_4b.yaml](/home/arvind/tinker-rl-lab/atropos/configs/math_qwen_4b.yaml)
- [math_qwen_8b.yaml](/home/arvind/tinker-rl-lab/atropos/configs/math_qwen_8b.yaml)
- [math_qwen_14b.yaml](/home/arvind/tinker-rl-lab/atropos/configs/math_qwen_14b.yaml)

Run:
```bash
./run_claim_support_train.sh hard_math
```

### Run everything
```bash
./run_claim_support_train.sh all
```

## Out-of-Distribution Evaluation

Use [eval_reasoning_suite.py](/home/arvind/tinker-rl-lab/atropos/eval_reasoning_suite.py) to evaluate a base model or saved checkpoint on:

- `gsm8k`
- `gsm1k`
- `gsm_symbolic_main`
- `gsm_symbolic_p1`
- `gsm_symbolic_p2`
- `math`
- `olympiadbench`

The intended use is:

1. Train a checkpoint on GSM8K.
2. Serve that checkpoint with [serve.py](/home/arvind/tinker-rl-lab/atropos/serve.py).
3. Run deterministic eval on multiple benchmarks.

### End-to-end example
```bash
./run_claim_support_evals.sh \
  --config configs/gsm8k_qwen_8b.yaml \
  --weights tinker://.../sampler_weights/step_50 \
  --no-prefix
```

### Direct eval without helper script
```bash
python serve.py --model Qwen/Qwen3-8B --weights tinker://... --port 8031

python eval_reasoning_suite.py \
  --model Qwen/Qwen3-8B \
  --base-url http://127.0.0.1:8031/v1 \
  --benchmarks gsm8k gsm1k gsm_symbolic_main math olympiadbench \
  --no-prefix \
  --output-dir logs/reasoning_eval_qwen8b_step50
```

## How These Results Support Claims

You can defend progressively stronger statements depending on what the results show.

### Weak but safe
- "GRPO + LoRA strongly optimizes GSM8K under deterministic exact-match evaluation."

### Medium
Requirements:
- multi-seed GSM8K stability
- no-prefix still improves materially
- MATH also improves

Claim:
- "The gains are not reducible to a single seed or a fixed worked-example prefix, and transfer to a harder symbolic-math benchmark."

### Stronger
Requirements:
- GSM8K-trained checkpoint improves on `gsm1k` or `gsm_symbolic_*`
- GSM8K-trained checkpoint also improves on `math` and/or `olympiadbench`

Claim:
- "The post-training gains extend beyond within-benchmark GSM8K optimization and show evidence of broader mathematical generalization."

## What Not To Claim Even After Running This

Still avoid:
- "robust mathematical reasoning" as a blanket claim from GSM8K alone
- "agentic RL" from these experiments
- "pure RL with no supervision" if the prompt prefix is enabled
- "capacity threshold" unless the bootstrap controls and intermediate scale points agree
